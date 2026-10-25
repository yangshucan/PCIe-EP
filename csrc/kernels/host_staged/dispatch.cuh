#pragma once

#include "layout.cuh"

namespace pcie_ep::host_staged {

__global__ void record_transport_start(DeviceState state) {
    if (!threadIdx.x)
        state.transport_timestamps_ns[kTransportStartNsField] = global_time_ns();
}

// S1/S2 reserve one host slot per possible source token. Clear inactive metadata
// as well: receivers scan capacity and must never see a prior epoch's mask.
__global__ void write_batch_metadata(DeviceState state, int num_tokens) {
    if (state.errors[kErrorCodeField])
        return;
    const int token_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_index >= state.layout.num_max_tokens_per_rank)
        return;
    TokenMetadata metadata{};
    if (token_index < num_tokens)
        metadata = state.route_plan[token_index];
    reinterpret_cast<TokenMetadata*>(
        state.mapped_source_buffers[state.local_rank] + state.layout.batch_metadata_offset)[token_index] = metadata;
    __threadfence_system();
}

__global__ void write_batch_payload(DeviceState state, const std::uint16_t* input, int num_tokens) {
    if (state.errors[kErrorCodeField])
        return;
    const int token_index = blockIdx.x;
    if (token_index >= num_tokens)
        return;
    TokenMetadata metadata = state.route_plan[token_index];
    if (!(metadata.destination_mask & ~(1ull << state.local_rank)))
        return;

    auto* source_segment = state.mapped_source_buffers[state.local_rank];
    auto* host_payload = reinterpret_cast<std::uint16_t*>(source_segment + state.layout.batch_payload_offset) +
        std::size_t(token_index) * state.layout.num_hidden;
    for (int hidden_index = threadIdx.x; hidden_index < state.layout.num_hidden; hidden_index += blockDim.x) {
        host_payload[hidden_index] = input[std::size_t(token_index) * state.layout.num_hidden + hidden_index];
    }
    if (!threadIdx.x)
        reinterpret_cast<TokenMetadata*>(source_segment + state.layout.batch_metadata_offset)[token_index] = metadata;
    __threadfence_system();
}

// This finite kernel follows all metadata/payload writers in the same stream.
// Unlike the bounded ring, the full batch can be published before receiving.
__global__ void publish_batch(DeviceState state, int num_tokens) {
    if (state.errors[kErrorCodeField] || threadIdx.x)
        return;
    auto* source_segment = state.mapped_source_buffers[state.local_rank];
    __threadfence_system();
    store_release_system(signal_value_ptr(source_segment, state.layout.batch_ready_offset), *state.epoch);
}

__global__ void transport_batch(DeviceState state) {
    if (state.errors[kErrorCodeField])
        return;

    __shared__ int action;
    __shared__ int output_row;
    __shared__ TokenMetadata metadata;
    const int thread_index = threadIdx.x;
    const std::uint64_t start_ns = global_time_ns();
    const std::uint64_t epoch = *state.epoch;

    for (int source_rank = 0; source_rank < state.layout.num_ranks; ++source_rank) {
        if (source_rank != state.local_rank) {
            if (!thread_index) {
                action = 0;
                while (load_acquire_system(signal_value_ptr(
                    state.mapped_source_buffers[source_rank], state.layout.batch_ready_offset)) != epoch) {
                    if (should_abort(state, start_ns, kErrorBatchReadyWait)) {
                        action = -1;
                        break;
                    }
                }
            }
            __syncthreads();
            if (action < 0)
                return;

            for (int token_index = 0; token_index < state.layout.num_max_tokens_per_rank; ++token_index) {
                if (!thread_index) {
                    metadata = reinterpret_cast<TokenMetadata*>(
                        state.mapped_source_buffers[source_rank] + state.layout.batch_metadata_offset)[token_index];
                    action = (metadata.destination_mask & (1ull << state.local_rank)) ? 1 : 0;
                    output_row = action ? state.source_offsets[source_rank] + metadata.rank_ordinals[state.local_rank] : 0;
                }
                __syncthreads();
                if (action) {
                    auto* host_payload = reinterpret_cast<std::uint16_t*>(
                        state.mapped_source_buffers[source_rank] + state.layout.batch_payload_offset) +
                        std::size_t(token_index) * state.layout.num_hidden;
                    for (int hidden_index = thread_index; hidden_index < state.layout.num_hidden; hidden_index += blockDim.x) {
                        state.unique_payload[std::size_t(output_row) * state.layout.num_hidden + hidden_index] =
                            host_payload[hidden_index];
                    }
                    if (!thread_index) {
                        save_unique_metadata(state, metadata, source_rank, output_row);
                        state.statistics[kNumHostPayloadReadsField]++;
                    }
                }
                __syncthreads();
            }
            if (!thread_index) {
                store_release_system(signal_value_ptr(
                    state.mapped_source_buffers[source_rank], state.layout.batch_ack_offset, state.local_rank), epoch);
            }
        }
    }

    if (!thread_index) {
        for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
            if (rank != state.local_rank) {
                while (load_acquire_system(signal_value_ptr(
                    state.mapped_source_buffers[state.local_rank], state.layout.batch_ack_offset, rank)) != epoch) {
                    if (should_abort(state, start_ns, kErrorBatchAckWait))
                        return;
                }
            }
        }
        state.statistics[kNumHostPayloadWritesField] = *state.num_remote_tokens;
        state.transport_timestamps_ns[kTransportEndNsField] = global_time_ns();
    }
}

// S3/S4: one CTA owns both directions. A blocked sender still services receive
// work and returns credits, preventing finite host windows from forming a
// send-only wait cycle. Every thread participates in every CTA barrier below.
__global__ void transport_ring(DeviceState state, const std::uint16_t* input) {
    if (state.errors[kErrorCodeField])
        return;

    __shared__ int action;
    __shared__ int output_row;
    __shared__ int done;
    __shared__ int payload_slot;
    __shared__ TokenMetadata metadata;

    const int thread_index = threadIdx.x;
    int num_consumed[kNumMaxRanks] = {};
    int num_sent[kNumMaxRanks] = {};
    int num_produced = 0;
    const std::uint64_t start_ns = global_time_ns();
    const std::uint64_t epoch = *state.epoch;
    auto* source_segment = state.mapped_source_buffers[state.local_rank];
    if (!thread_index)
        state.transport_timestamps_ns[kTransportStartNsField] = global_time_ns();

    while (true) {
        // Receive at most one descriptor per source in each progress round.
        // Acquire READY before reading the referenced payload and metadata.
        for (int source_rank = 0; source_rank < state.layout.num_ranks; ++source_rank) {
            if (source_rank != state.local_rank) {
                if (!thread_index) {
                    action = 0;
                    const int num_required = state.source_offsets[source_rank + 1] - state.source_offsets[source_rank];
                    if (num_consumed[source_rank] < num_required) {
                        const int descriptor_index = num_consumed[source_rank];
                        auto* descriptor = reinterpret_cast<Descriptor*>(
                            state.mapped_source_buffers[source_rank] + state.layout.descriptors_offset) +
                            state.local_rank * state.layout.num_descriptor_slots + descriptor_index % state.layout.num_descriptor_slots;
                        const std::uint64_t expected_sequence = (epoch << 32) | std::uint64_t(descriptor_index + 1);
                        if (load_acquire_system(&descriptor->ready_sequence) == expected_sequence) {
                            metadata = reinterpret_cast<TokenMetadata*>(
                                state.mapped_source_buffers[source_rank] + state.layout.ring_metadata_offset)[descriptor->payload_slot];
                            output_row = state.source_offsets[source_rank] + descriptor->rank_ordinal;
                            payload_slot = descriptor->payload_slot;
                            action = 1;
                            if (descriptor->payload_sequence != ((epoch << 32) | std::uint64_t(metadata.token_index + 1))) {
                                state.errors[kErrorCodeField] = kErrorPayloadSequenceMismatch;
                                store_release_system(signal_value_ptr(source_segment, state.layout.abort_offset), epoch);
                                action = -1;
                            }
                        }
                    }
                }
                __syncthreads();
                if (action < 0)
                    return;

                // Receiver deduplication: one mapped-host read into unique HBM
                // per source token, regardless of the number of local experts.
                if (action) {
                    auto* host_payload = reinterpret_cast<std::uint16_t*>(
                        state.mapped_source_buffers[source_rank] + state.layout.ring_payload_offset) +
                        std::size_t(payload_slot) * state.layout.num_hidden;
                    for (int hidden_index = thread_index; hidden_index < state.layout.num_hidden; hidden_index += blockDim.x) {
                        state.unique_payload[std::size_t(output_row) * state.layout.num_hidden + hidden_index] =
                            host_payload[hidden_index];
                    }
                    if (!thread_index) {
                        save_unique_metadata(state, metadata, source_rank, output_row);
                        state.statistics[kNumHostPayloadReadsField]++;
                    }
                }
                __threadfence_system();
                __syncthreads();

                // First release the source payload reference, then return this
                // descriptor's credit. The producer checks these independently.
                if (!thread_index && action) {
                    const int descriptor_index = num_consumed[source_rank];
                    auto* descriptor = reinterpret_cast<Descriptor*>(
                        state.mapped_source_buffers[source_rank] + state.layout.descriptors_offset) +
                        state.local_rank * state.layout.num_descriptor_slots + descriptor_index % state.layout.num_descriptor_slots;
                    store_release_system(signal_value_ptr(
                        state.mapped_source_buffers[source_rank], state.layout.payload_ack_offset,
                        payload_slot * state.layout.num_ranks + state.local_rank), descriptor->payload_sequence);
                    store_release_system(&descriptor->consumed_sequence, (epoch << 32) | std::uint64_t(descriptor_index + 1));
                    num_consumed[source_rank]++;
                }
                __syncthreads();
            }
        }

        // Probe a single outgoing token. No credit is reserved unless every
        // target descriptor and the reused payload slot are available.
        if (!thread_index) {
            action = 0;
            if (num_produced < *state.num_remote_tokens) {
                const int token_index = state.remote_token_indices[num_produced];
                TokenMetadata candidate = state.route_plan[token_index];
                const int candidate_slot = num_produced % state.layout.num_payload_slots;
                bool available = true;

                // A multicast slot cannot be overwritten until every target of
                // its previous token acknowledges that it finished host reads.
                if (num_produced >= state.layout.num_payload_slots) {
                    const int previous_token = state.remote_token_indices[num_produced - state.layout.num_payload_slots];
                    const std::uint64_t previous_sequence = (epoch << 32) | std::uint64_t(previous_token + 1);
                    const std::uint64_t previous_targets = state.route_plan[previous_token].destination_mask &
                        ~(1ull << state.local_rank);
                    for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
                        if (previous_targets & (1ull << rank)) {
                            if (load_acquire_system(signal_value_ptr(
                                source_segment, state.layout.payload_ack_offset,
                                candidate_slot * state.layout.num_ranks + rank)) != previous_sequence)
                                available = false;
                        }
                    }
                }
                for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
                    if (rank != state.local_rank && (candidate.destination_mask & (1ull << rank)) &&
                        num_sent[rank] >= state.layout.num_descriptor_slots) {
                        auto* descriptor = reinterpret_cast<Descriptor*>(source_segment + state.layout.descriptors_offset) +
                            rank * state.layout.num_descriptor_slots + num_sent[rank] % state.layout.num_descriptor_slots;
                        if (load_acquire_system(&descriptor->consumed_sequence) !=
                            ((epoch << 32) | std::uint64_t(num_sent[rank] - state.layout.num_descriptor_slots + 1)))
                            available = false;
                    }
                }
                if (available) {
                    metadata = candidate;
                    payload_slot = candidate_slot;
                    action = 1;
                }
            }
        }
        __syncthreads();

        // One host write serves all destination GPUs. The per-target descriptor
        // publication follows all writers' system fences and the CTA join.
        if (action) {
            auto* host_payload = reinterpret_cast<std::uint16_t*>(source_segment + state.layout.ring_payload_offset) +
                std::size_t(payload_slot) * state.layout.num_hidden;
            for (int hidden_index = thread_index; hidden_index < state.layout.num_hidden; hidden_index += blockDim.x) {
                host_payload[hidden_index] = input[std::size_t(metadata.token_index) * state.layout.num_hidden + hidden_index];
            }
            if (!thread_index)
                reinterpret_cast<TokenMetadata*>(source_segment + state.layout.ring_metadata_offset)[payload_slot] = metadata;
        }
        __threadfence_system();
        __syncthreads();

        if (!thread_index && action) {
            const std::uint64_t payload_sequence = (epoch << 32) | std::uint64_t(metadata.token_index + 1);
            for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
                if (rank != state.local_rank && (metadata.destination_mask & (1ull << rank))) {
                    const int descriptor_index = num_sent[rank];
                    auto* descriptor = reinterpret_cast<Descriptor*>(source_segment + state.layout.descriptors_offset) +
                        rank * state.layout.num_descriptor_slots + descriptor_index % state.layout.num_descriptor_slots;
                    descriptor->payload_sequence = payload_sequence;
                    descriptor->payload_slot = payload_slot;
                    descriptor->token_index = metadata.token_index;
                    descriptor->rank_ordinal = metadata.rank_ordinals[rank];
                    store_release_system(&descriptor->ready_sequence, (epoch << 32) | std::uint64_t(descriptor_index + 1));
                    num_sent[rank]++;
                    state.statistics[kNumDescriptorsPublishedField]++;
                }
            }
            num_produced++;
            state.statistics[kNumHostPayloadWritesField]++;
        }

        // Completion includes outgoing ACK drain, not only local receive work.
        // This also makes the first slots of the next epoch safe to reuse.
        if (!thread_index) {
            done = num_produced == *state.num_remote_tokens;
            for (int source_rank = 0; source_rank < state.layout.num_ranks; ++source_rank) {
                if (source_rank != state.local_rank)
                    done &= num_consumed[source_rank] == state.source_offsets[source_rank + 1] - state.source_offsets[source_rank];
            }
            if (done) {
                const int first_pending = max(0, num_produced - state.layout.num_payload_slots);
                for (int produced_index = first_pending; produced_index < num_produced; ++produced_index) {
                    const int token_index = state.remote_token_indices[produced_index];
                    const std::uint64_t targets = state.route_plan[token_index].destination_mask & ~(1ull << state.local_rank);
                    for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
                        if (targets & (1ull << rank)) {
                            if (load_acquire_system(signal_value_ptr(
                                source_segment, state.layout.payload_ack_offset,
                                (produced_index % state.layout.num_payload_slots) * state.layout.num_ranks + rank)) !=
                                ((epoch << 32) | std::uint64_t(token_index + 1)))
                                done = 0;
                        }
                    }
                }
            }
            if (!done && should_abort(state, start_ns, kErrorRingProgressWait))
                done = -1;
        }
        __syncthreads();
        if (done)
            break;
    }
    if (!thread_index)
        state.transport_timestamps_ns[kTransportEndNsField] = global_time_ns();
}

// Independent S2/S4 epilogue. Duplicate expert branches read the unique HBM
// token; no expert branch repeats a mapped-host read. Padding is written only
// within each expert's unused aligned tail and does not overlap valid rows.
__global__ void scatter_expert_inputs(
    DeviceState state, std::uint16_t* expanded_payload, std::int64_t* expanded_expert_indices,
    float* expanded_weights, std::int64_t* expanded_source_metadata, int capacity, bool zero_padding
) {
    if (state.errors[kErrorCodeField])
        return;
    const int unique_row = blockIdx.x;
    const int num_unique_tokens = static_cast<int>(*state.unique_count);
    const int thread_index = threadIdx.x;
    const int num_experts_per_rank = state.layout.num_experts / state.layout.num_ranks;

    if (unique_row < num_unique_tokens) {
        const int source_rank = static_cast<int>(state.unique_source_metadata[unique_row * kNumSourceMetadataFields]);
        const int token_index = static_cast<int>(state.unique_source_metadata[unique_row * kNumSourceMetadataFields + 1]);
        for (int topk_index = 0; topk_index < state.layout.num_topk; ++topk_index) {
            const int expert_index = static_cast<int>(state.unique_topk_indices[unique_row * state.layout.num_topk + topk_index]);
            if (expert_index >= 0) {
                const int output_row = state.expert_offsets[expert_index] +
                    state.source_expert_prefix[source_rank * num_experts_per_rank + expert_index] +
                    state.unique_expert_ordinals[unique_row * state.layout.num_topk + topk_index];
                if (output_row >= capacity) {
                    if (!thread_index) {
                        state.errors[kErrorCodeField] = kErrorExpandedCapacity;
                        store_release_system(signal_value_ptr(
                            state.mapped_source_buffers[state.local_rank], state.layout.abort_offset), *state.epoch);
                    }
                    return;
                }
                for (int hidden_index = thread_index; hidden_index < state.layout.num_hidden; hidden_index += blockDim.x) {
                    expanded_payload[std::size_t(output_row) * state.layout.num_hidden + hidden_index] =
                        state.unique_payload[std::size_t(unique_row) * state.layout.num_hidden + hidden_index];
                }
                if (!thread_index) {
                    expanded_expert_indices[output_row] = expert_index;
                    expanded_weights[output_row] = state.unique_topk_weights[unique_row * state.layout.num_topk + topk_index];
                    expanded_source_metadata[output_row * kNumSourceMetadataFields] = source_rank;
                    expanded_source_metadata[output_row * kNumSourceMetadataFields + 1] = token_index;
                    expanded_source_metadata[output_row * kNumSourceMetadataFields + 2] = topk_index;
                }
            }
        }
    }
    if (zero_padding && unique_row < num_experts_per_rank) {
        const int start_row = state.expert_offsets[unique_row] + state.expert_counts[unique_row];
        const int end_row = state.expert_offsets[unique_row + 1];
        for (int output_row = start_row; output_row < end_row; ++output_row) {
            for (int hidden_index = thread_index; hidden_index < state.layout.num_hidden; hidden_index += blockDim.x)
                expanded_payload[std::size_t(output_row) * state.layout.num_hidden + hidden_index] = 0;
            if (!thread_index) {
                expanded_expert_indices[output_row] = -1;
                expanded_weights[output_row] = 0;
                expanded_source_metadata[output_row * kNumSourceMetadataFields] =
                    expanded_source_metadata[output_row * kNumSourceMetadataFields + 1] =
                    expanded_source_metadata[output_row * kNumSourceMetadataFields + 2] = -1;
            }
        }
    }
}

}  // namespace pcie_ep::host_staged
