#pragma once

#include "layout.cuh"

namespace pcie_ep::host_staged {

// Separate expansion remains a supported runtime option.
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
