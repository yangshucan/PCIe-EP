#pragma once

#include "common.cuh"

namespace pcie_ep::host_staged {

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

}  // namespace pcie_ep::host_staged
