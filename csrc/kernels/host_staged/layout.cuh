#pragma once

#include "common.cuh"

namespace pcie_ep::host_staged {

// S2/S4 baseline: one thread computes stable source-token and expert ordinals.
// Preserve this serial reference path; parallel layout is a later optimization.
__global__ void compute_route_layout(
    DeviceState state, const std::int64_t* topk_indices, const float* topk_weights, int num_tokens
) {
    if (threadIdx.x || blockIdx.x)
        return;

    // A new epoch distinguishes reused descriptors from the previous dispatch.
    ++*state.epoch;
    if (*state.epoch >= 0x100000000ull) {
        state.errors[kErrorCodeField] = kErrorEpochExhausted;
        store_release_system(
            signal_value_ptr(state.mapped_source_buffers[state.local_rank], state.layout.abort_offset), *state.epoch);
        return;
    }
    *state.num_remote_tokens = 0;
    for (int index = 0; index < state.layout.num_ranks + state.layout.num_experts; ++index)
        state.rank_expert_counts[index] = 0;
    for (int index = 0; index < kNumErrorFields; ++index) {
        state.errors[index] = 0;
        state.statistics[index] = 0;
    }
    state.transport_timestamps_ns[kTransportStartNsField] = state.transport_timestamps_ns[kTransportEndNsField] = 0;
    for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
        if (load_acquire_system(signal_value_ptr(state.mapped_source_buffers[rank], state.layout.abort_offset))) {
            state.errors[kErrorCodeField] = kErrorRuntimeAlreadyAborted;
            return;
        }
    }

    const int num_experts_per_rank = state.layout.num_experts / state.layout.num_ranks;
    for (int token_index = 0; token_index < num_tokens; ++token_index) {
        TokenMetadata metadata{};
        metadata.token_index = token_index;
        for (int rank = 0; rank < kNumMaxRanks; ++rank)
            metadata.rank_ordinals[rank] = -1;

        // Expert choices must be distinct. -1 denotes a dropped expert branch.
        for (int topk_index = 0; topk_index < state.layout.num_topk; ++topk_index) {
            const std::int64_t expert_index = topk_indices[token_index * state.layout.num_topk + topk_index];
            metadata.expert_indices[topk_index] = static_cast<int>(expert_index);
            metadata.expert_ordinals[topk_index] = -1;
            metadata.weights[topk_index] = topk_weights ?
                topk_weights[token_index * state.layout.num_topk + topk_index] : 1.0f;

            bool invalid = expert_index < -1 || expert_index >= state.layout.num_experts;
            for (int previous = 0; previous < topk_index; ++previous) {
                if (expert_index >= 0 && expert_index == topk_indices[token_index * state.layout.num_topk + previous])
                    invalid = true;
            }
            if (invalid) {
                state.errors[kErrorCodeField] = kErrorInvalidExpertRouting;
                state.errors[kErrorContextField] = token_index;
                state.errors[kErrorDetailField] = topk_index;
                metadata.expert_indices[topk_index] = -1;
                continue;
            }
            if (expert_index >= 0) {
                const int destination_rank = expert_index / num_experts_per_rank;
                metadata.destination_mask |= 1ull << destination_rank;
                metadata.expert_ordinals[topk_index] =
                    state.rank_expert_counts[state.layout.num_ranks + expert_index]++;
            }
        }

        // Sender deduplication: one ordinal per destination rank, and only one
        // remote-token-list entry even if several remote ranks need the token.
        for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
            if (metadata.destination_mask & (1ull << rank))
                metadata.rank_ordinals[rank] = state.rank_expert_counts[rank]++;
        }
        state.route_plan[token_index] = metadata;
        if (metadata.destination_mask & ~(1ull << state.local_rank))
            state.remote_token_indices[(*state.num_remote_tokens)++] = token_index;
    }
    if (state.errors[kErrorCodeField]) {
        store_release_system(
            signal_value_ptr(state.mapped_source_buffers[state.local_rank], state.layout.abort_offset), *state.epoch);
    }
}

__global__ void publish_counts(DeviceState state) {
    if (state.errors[kErrorCodeField])
        return;

    __shared__ int failed;
    const int thread_index = threadIdx.x;
    const std::uint64_t epoch = *state.epoch;
    const std::uint64_t start_ns = global_time_ns();

    // Counts have their own consumption handshake. Reusing a mailbox is safe
    // only after every peer consumed the preceding epoch, including zero work.
    if (!thread_index) {
        failed = 0;
        if (epoch > 1) {
            for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
                if (rank != state.local_rank) {
                    while (load_acquire_system(signal_value_ptr(
                        state.mapped_source_buffers[state.local_rank], state.layout.counts_consumed_offset, rank)) != epoch - 1) {
                        if (should_abort(state, start_ns, kErrorCountsMailboxWait)) {
                            failed = 1;
                            break;
                        }
                    }
                    if (failed)
                        break;
                }
            }
        }
    }
    __syncthreads();
    if (failed)
        return;

    auto* published_counts = reinterpret_cast<int*>(
        state.mapped_source_buffers[state.local_rank] + state.layout.counts_offset);
    for (int index = thread_index; index < state.layout.num_ranks + state.layout.num_experts; index += blockDim.x)
        published_counts[index] = state.rank_expert_counts[index];

    // Every writer fences its own stores; the leader publishes only after the
    // whole CTA has joined. A leader-only fence would not cover other writers.
    __threadfence_system();
    __syncthreads();
    if (!thread_index) {
        store_release_system(
            signal_value_ptr(state.mapped_source_buffers[state.local_rank], state.layout.counts_ready_offset), epoch);
    }
}

__global__ void compute_receive_prefixes(DeviceState state, int expert_alignment) {
    if (state.errors[kErrorCodeField] || threadIdx.x)
        return;

    const std::uint64_t start_ns = global_time_ns();
    const std::uint64_t epoch = *state.epoch;
    const int num_experts_per_rank = state.layout.num_experts / state.layout.num_ranks;
    state.source_offsets[0] = 0;
    for (int expert = 0; expert < num_experts_per_rank; ++expert)
        state.expert_counts[expert] = 0;

    // Stable order is source rank first, then source token order. Counts remain
    // on the GPU; no actual receive count is needed by the Python launch path.
    for (int source_rank = 0; source_rank < state.layout.num_ranks; ++source_rank) {
        while (load_acquire_system(signal_value_ptr(
            state.mapped_source_buffers[source_rank], state.layout.counts_ready_offset)) != epoch) {
            if (should_abort(state, start_ns, kErrorCountsReadyWait))
                return;
        }
        auto* source_counts = reinterpret_cast<int*>(state.mapped_source_buffers[source_rank] + state.layout.counts_offset);
        state.source_offsets[source_rank + 1] = state.source_offsets[source_rank] + source_counts[state.local_rank];
        for (int expert = 0; expert < num_experts_per_rank; ++expert) {
            state.source_expert_prefix[source_rank * num_experts_per_rank + expert] = state.expert_counts[expert];
            state.expert_counts[expert] += source_counts[
                state.layout.num_ranks + state.local_rank * num_experts_per_rank + expert];
        }
        if (source_rank != state.local_rank) {
            store_release_system(signal_value_ptr(
                state.mapped_source_buffers[source_rank], state.layout.counts_consumed_offset, state.local_rank), epoch);
        }
    }
    state.expert_offsets[0] = 0;
    for (int expert = 0; expert < num_experts_per_rank; ++expert) {
        state.expert_offsets[expert + 1] = state.expert_offsets[expert] +
            ((state.expert_counts[expert] + expert_alignment - 1) / expert_alignment) * expert_alignment;
    }
    *state.unique_count = state.source_offsets[state.layout.num_ranks];
    *state.expanded_count = 0;
    for (int expert = 0; expert < num_experts_per_rank; ++expert) {
        *state.expanded_count += state.expert_counts[expert];
        state.expert_prefix[expert] = state.expert_offsets[expert] + state.expert_counts[expert];
    }
}

__device__ inline void save_unique_metadata(
    DeviceState& state, TokenMetadata& metadata, int source_rank, int output_row
) {
    const int num_experts_per_rank = state.layout.num_experts / state.layout.num_ranks;
    state.unique_source_metadata[output_row * kNumSourceMetadataFields] = source_rank;
    state.unique_source_metadata[output_row * kNumSourceMetadataFields + 1] = metadata.token_index;
    state.unique_source_metadata[output_row * kNumSourceMetadataFields + 2] = -1;
    for (int topk_index = 0; topk_index < state.layout.num_topk; ++topk_index) {
        const int expert_index = metadata.expert_indices[topk_index];
        const bool local_expert = expert_index >= state.local_rank * num_experts_per_rank &&
            expert_index < (state.local_rank + 1) * num_experts_per_rank;
        state.unique_topk_indices[output_row * state.layout.num_topk + topk_index] =
            local_expert ? expert_index - state.local_rank * num_experts_per_rank : -1;
        state.unique_topk_weights[output_row * state.layout.num_topk + topk_index] =
            local_expert ? metadata.weights[topk_index] : 0;
        state.unique_expert_ordinals[output_row * state.layout.num_topk + topk_index] = metadata.expert_ordinals[topk_index];
    }
}

__global__ void copy_local_tokens(DeviceState state, const std::uint16_t* input, int num_tokens) {
    if (state.errors[kErrorCodeField])
        return;
    const int token_index = blockIdx.x;
    if (token_index >= num_tokens)
        return;
    TokenMetadata metadata = state.route_plan[token_index];
    if (!(metadata.destination_mask & (1ull << state.local_rank)))
        return;
    const int output_row = state.source_offsets[state.local_rank] + metadata.rank_ordinals[state.local_rank];
    for (int hidden_index = threadIdx.x; hidden_index < state.layout.num_hidden; hidden_index += blockDim.x) {
        state.unique_payload[std::size_t(output_row) * state.layout.num_hidden + hidden_index] =
            input[std::size_t(token_index) * state.layout.num_hidden + hidden_index];
    }
    if (!threadIdx.x)
        save_unique_metadata(state, metadata, state.local_rank, output_row);
}

}  // namespace pcie_ep::host_staged
