#pragma once

#include "advanced_common.cuh"
#include "layout.cuh"
#include <cub/block/block_scan.cuh>

namespace pcie_ep::host_staged {

__global__ void advanced_begin_epoch(AdvancedDeviceState state) {
    if (threadIdx.x || blockIdx.x)
        return;
    for (int index = 0; index < kNumErrorFields; ++index)
        state.base.errors[index] = 0;
    for (int index = 0; index < kNumStatisticsFields; ++index)
        state.base.statistics[index] = 0;
    for (int index = 0; index < state.channels * kNumStatisticsFields; ++index)
        state.channel_statistics[index] = 0;
    for (int index = 0; index < state.channels * 2; ++index)
        state.channel_timestamps_ns[index] = 0;
    state.base.transport_timestamps_ns[0] = state.base.transport_timestamps_ns[1] = 0;
    ++*state.base.epoch;
    if (*state.base.epoch >= 0x100000000ull) {
        advanced_record_error(state, kErrorEpochExhausted);
        store_release_system(signal_value_ptr(state.base.mapped_source_buffers[state.base.local_rank],
            state.shared.channel_abort_offset), *state.base.epoch);
        return;
    }
    // Abort is terminal for this runtime, including Graph replay: resetting
    // the output diagnostics must never revive an already failed protocol.
    if (advanced_any_abort(state)) {
        advanced_record_error(state, kErrorRuntimeAlreadyAborted);
        store_release_system(signal_value_ptr(state.base.mapped_source_buffers[state.base.local_rank],
            state.shared.channel_abort_offset), *state.base.epoch);
    }
}

__global__ void advanced_compute_receive_prefixes(AdvancedDeviceState state, int alignment) {
    if (threadIdx.x || blockIdx.x || state.base.errors[kErrorCodeField])
        return;
    auto& base = state.base;
    const int ranks = base.layout.num_ranks;
    const int experts_per_rank = base.layout.num_experts / ranks;
    const auto epoch = *base.epoch;
    const auto start = global_time_ns();
    base.source_offsets[0] = 0;
    for (int expert = 0; expert < experts_per_rank; ++expert)
        base.expert_counts[expert] = 0;
    for (int source = 0; source < ranks; ++source) {
        auto* segment = base.mapped_source_buffers[source];
        while (load_acquire_system(signal_value_ptr(segment, base.layout.counts_ready_offset)) != epoch) {
            if (advanced_failed(state, start)) {
                advanced_record_error(state, kErrorCountsReadyWait);
                store_release_system(signal_value_ptr(base.mapped_source_buffers[base.local_rank],
                    state.shared.channel_abort_offset), epoch);
                return;
            }
        }
        auto* counts = reinterpret_cast<int*>(segment + base.layout.counts_offset);
        auto* channel_counts = reinterpret_cast<int*>(segment + state.shared.channel_counts_offset);
        base.source_offsets[source + 1] = base.source_offsets[source] + counts[base.local_rank];
        for (int expert = 0; expert < experts_per_rank; ++expert) {
            base.source_expert_prefix[source * experts_per_rank + expert] = base.expert_counts[expert];
            base.expert_counts[expert] += counts[ranks + base.local_rank * experts_per_rank + expert];
        }
        for (int channel = 0; channel < state.channels; ++channel)
            state.expected_receive_counts[channel * ranks + source] = channel_counts[channel * ranks + base.local_rank];
        if (source != base.local_rank)
            store_release_system(signal_value_ptr(segment, base.layout.counts_consumed_offset, base.local_rank), epoch);
    }
    base.expert_offsets[0] = 0;
    *base.expanded_count = 0;
    for (int expert = 0; expert < experts_per_rank; ++expert) {
        base.expert_offsets[expert + 1] = base.expert_offsets[expert] +
            ((base.expert_counts[expert] + alignment - 1) / alignment) * alignment;
        *base.expanded_count += base.expert_counts[expert];
        base.expert_prefix[expert] = base.expert_offsets[expert] + base.expert_counts[expert];
    }
    *base.unique_count = base.source_offsets[ranks];
}

__device__ inline void advanced_store_expanded_metadata(
    AdvancedDeviceState& state, TokenMetadata& metadata, int source, int branch, int row, int expert
) {
    state.expanded_expert_indices[row] = expert;
    state.expanded_weights[row] = metadata.weights[branch];
    state.expanded_source_metadata[row * 3] = source;
    state.expanded_source_metadata[row * 3 + 1] = metadata.token_index;
    state.expanded_source_metadata[row * 3 + 2] = branch;
}

__global__ void advanced_copy_local_tokens(AdvancedDeviceState state, int num_tokens) {
    if (state.base.errors[kErrorCodeField] || int(blockIdx.x) >= num_tokens)
        return;
    const int token = blockIdx.x;
    TokenMetadata metadata = state.base.route_plan[token];
    if (!(metadata.destination_mask & (1ull << state.base.local_rank)))
        return;
    for (int branch = 0; branch < state.base.layout.num_topk; ++branch)
        metadata.weights[branch] = state.topk_weights ? state.topk_weights[token * state.base.layout.num_topk + branch] : 1.0f;
    const int row = state.base.source_offsets[state.base.local_rank] + metadata.rank_ordinals[state.base.local_rank];
    if (!state.fuse_receive_expand) {
        for (int hidden = threadIdx.x; hidden < state.base.layout.num_hidden; hidden += blockDim.x)
            state.base.unique_payload[std::size_t(row) * state.base.layout.num_hidden + hidden] =
                state.input_payload[std::size_t(token) * state.base.layout.num_hidden + hidden];
        if (!threadIdx.x)
            save_unique_metadata(state.base, metadata, state.base.local_rank, row);
        return;
    }
    const int experts_per_rank = state.base.layout.num_experts / state.base.layout.num_ranks;
    for (int branch = 0; branch < state.base.layout.num_topk; ++branch) {
        const int expert = metadata.expert_indices[branch] - state.base.local_rank * experts_per_rank;
        if (expert < 0 || expert >= experts_per_rank)
            continue;
        const int destination = state.base.expert_offsets[expert] +
            state.base.source_expert_prefix[state.base.local_rank * experts_per_rank + expert] + metadata.expert_ordinals[branch];
        for (int hidden = threadIdx.x; hidden < state.base.layout.num_hidden; hidden += blockDim.x)
            state.expanded_payload[std::size_t(destination) * state.base.layout.num_hidden + hidden] =
                state.input_payload[std::size_t(token) * state.base.layout.num_hidden + hidden];
        if (!threadIdx.x)
            advanced_store_expanded_metadata(state, metadata, state.base.local_rank, branch, destination, expert);
    }
}

__global__ void advanced_zero_padding(AdvancedDeviceState state) {
    const int expert = blockIdx.x;
    if (state.base.errors[kErrorCodeField])
        return;
    for (int row = state.base.expert_offsets[expert] + state.base.expert_counts[expert];
         row < state.base.expert_offsets[expert + 1]; ++row) {
        for (int hidden = threadIdx.x; hidden < state.base.layout.num_hidden; hidden += blockDim.x)
            state.expanded_payload[std::size_t(row) * state.base.layout.num_hidden + hidden] = 0;
        if (!threadIdx.x) {
            state.expanded_expert_indices[row] = -1;
            state.expanded_weights[row] = 0;
            for (int field = 0; field < 3; ++field)
                state.expanded_source_metadata[row * 3 + field] = -1;
        }
    }
}

__global__ void advanced_reduce_statistics(AdvancedDeviceState state) {
    if (threadIdx.x || blockIdx.x)
        return;
    std::int64_t start = 0, end = 0;
    for (int channel = 0; channel < state.channels; ++channel) {
        const auto begin = state.channel_timestamps_ns[channel * 2];
        if (begin && (!start || begin < start))
            start = begin;
        const auto finish = state.channel_timestamps_ns[channel * 2 + 1];
        if (finish > end)
            end = finish;
        for (int index = 0; index < kNumStatisticsFields; ++index)
            state.base.statistics[index] += state.channel_statistics[channel * kNumStatisticsFields + index];
    }
    state.base.transport_timestamps_ns[0] = start;
    state.base.transport_timestamps_ns[1] = end;
}

// Optional S6 layout. It retains the serial kernels above as the reference
// implementation, while preserving source-token order in every prefix scan.
__global__ void advanced_route_flags(AdvancedDeviceState state, int num_tokens) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens || state.base.errors[kErrorCodeField])
        return;
    TokenMetadata metadata{};
    metadata.token_index = token;
    for (int rank = 0; rank < kNumMaxRanks; ++rank)
        metadata.rank_ordinals[rank] = -1;
    const int experts_per_rank = state.base.layout.num_experts / state.base.layout.num_ranks;
    for (int branch = 0; branch < state.base.layout.num_topk; ++branch) {
        const auto expert = state.topk_indices[token * state.base.layout.num_topk + branch];
        bool invalid = expert < -1 || expert >= state.base.layout.num_experts;
        for (int previous = 0; previous < branch; ++previous)
            invalid |= expert >= 0 && expert == state.topk_indices[token * state.base.layout.num_topk + previous];
        metadata.expert_indices[branch] = invalid ? -1 : int(expert);
        metadata.expert_ordinals[branch] = -1;
        if (invalid)
            advanced_record_error(state, kErrorInvalidExpertRouting);
        else if (expert >= 0)
            metadata.destination_mask |= 1ull << (expert / experts_per_rank);
    }
    state.base.route_plan[token] = metadata;
}

__global__ void advanced_stable_ordinals(AdvancedDeviceState state, int num_tokens) {
    using Scan = cub::BlockScan<int, kNumThreads>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ int tile_prefix;
    const int bin = blockIdx.x;
    const int ranks = state.base.layout.num_ranks;
    if (state.base.errors[kErrorCodeField]) {
        if (!bin && !threadIdx.x)
            store_release_system(signal_value_ptr(state.base.mapped_source_buffers[state.base.local_rank],
                state.shared.channel_abort_offset), *state.base.epoch);
        return;
    }
    if (!threadIdx.x)
        tile_prefix = 0;
    __syncthreads();
    for (int begin = 0; begin < num_tokens; begin += kNumThreads) {
        const int token = begin + threadIdx.x;
        int hit = 0, selected_branch = -1;
        if (token < num_tokens) {
            if (bin < ranks) {
                hit = bool(state.base.route_plan[token].destination_mask & (1ull << bin));
            } else {
                for (int branch = 0; branch < state.base.layout.num_topk; ++branch) {
                    if (state.base.route_plan[token].expert_indices[branch] == bin - ranks) {
                        hit = 1;
                        selected_branch = branch;
                    }
                }
            }
        }
        int ordinal, aggregate;
        Scan(scan_storage).ExclusiveSum(hit, ordinal, aggregate);
        if (hit) {
            if (bin < ranks)
                state.base.route_plan[token].rank_ordinals[bin] = tile_prefix + ordinal;
            else
                state.base.route_plan[token].expert_ordinals[selected_branch] = tile_prefix + ordinal;
        }
        __syncthreads();
        if (!threadIdx.x)
            tile_prefix += aggregate;
        __syncthreads();
    }
    if (!threadIdx.x)
        state.base.rank_expert_counts[bin] = tile_prefix;
}

__global__ void advanced_prepare_channels_parallel(AdvancedDeviceState state, int num_tokens, bool reuse_layout) {
    using Scan = cub::BlockScan<int, kNumThreads>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ int rank_counts[kNumMaxRanks];
    __shared__ int endpoint_counts[kNumMaxEndpoints];
    __shared__ int remote_prefix;
    __shared__ int failed;
    const int channel = blockIdx.x;
    const int ranks = state.base.layout.num_ranks;
    auto* own = state.base.mapped_source_buffers[state.base.local_rank];
    const auto epoch = *state.base.epoch;
    const auto start = global_time_ns();
    if (!threadIdx.x) {
        // Channel CTAs can discover a mailbox timeout independently. Keep
        // the early-exit verdict uniform even if another CTA reports first.
        failed = state.base.errors[kErrorCodeField] != 0;
        remote_prefix = 0;
        if (!failed && epoch > 1) {
            for (int rank = 0; rank < ranks; ++rank) {
                if (rank == state.base.local_rank)
                    continue;
                while (load_acquire_system(signal_value_ptr(own, state.base.layout.counts_consumed_offset, rank)) != epoch - 1) {
                    if (advanced_failed(state, start)) {
                        advanced_record_error(state, kErrorCountsMailboxWait);
                        failed = 1;
                        break;
                    }
                }
                if (failed)
                    break;
            }
        }
        if (failed)
            store_release_system(signal_value_ptr(own, state.shared.channel_abort_offset, channel), epoch);
    }
    for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x)
        rank_counts[rank] = 0;
    for (int endpoint = threadIdx.x; endpoint < state.shared.num_endpoints; endpoint += blockDim.x)
        endpoint_counts[endpoint] = 0;
    __syncthreads();
    if (failed)
        return;
    if (!reuse_layout) {
        const int channel_tokens = num_tokens > channel ? (num_tokens - channel + state.channels - 1) / state.channels : 0;
        for (int begin = 0; begin < channel_tokens; begin += kNumThreads) {
            const int ordinal = begin + threadIdx.x;
            const int token = channel + ordinal * state.channels;
            std::uint64_t endpoints = 0;
            if (ordinal < channel_tokens) {
                const auto targets = state.base.route_plan[token].destination_mask;
                endpoints = advanced_endpoint_mask(state, targets);
                for (int rank = 0; rank < ranks; ++rank)
                    if (targets & (1ull << rank))
                        atomicAdd(rank_counts + rank, 1);
                for (int endpoint = 0; endpoint < state.shared.num_endpoints; ++endpoint)
                    if (endpoints & (1ull << endpoint))
                        atomicAdd(endpoint_counts + endpoint, 1);
            }
            int remote_ordinal, aggregate;
            Scan(scan_storage).ExclusiveSum(int(endpoints != 0), remote_ordinal, aggregate);
            if (endpoints)
                state.channel_remote_indices[channel * state.base.layout.num_max_tokens_per_rank + remote_prefix + remote_ordinal] = token;
            __syncthreads();
            if (!threadIdx.x)
                remote_prefix += aggregate;
            __syncthreads();
        }
        for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x)
            state.channel_rank_counts[channel * ranks + rank] = rank_counts[rank];
        for (int endpoint = threadIdx.x; endpoint < state.shared.num_endpoints; endpoint += blockDim.x)
            state.channel_endpoint_counts[channel * state.shared.num_endpoints + endpoint] = endpoint_counts[endpoint];
        if (!threadIdx.x)
            state.channel_num_remote_tokens[channel] = remote_prefix;
        __syncthreads();
    }
    auto* published_ranks = reinterpret_cast<int*>(own + state.shared.channel_counts_offset);
    auto* published_endpoints = reinterpret_cast<int*>(own + state.shared.endpoint_counts_offset);
    for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x)
        published_ranks[channel * ranks + rank] = state.channel_rank_counts[channel * ranks + rank];
    for (int endpoint = threadIdx.x; endpoint < state.shared.num_endpoints; endpoint += blockDim.x)
        published_endpoints[channel * state.shared.num_endpoints + endpoint] =
            state.channel_endpoint_counts[channel * state.shared.num_endpoints + endpoint];
    __threadfence_system();
}

}  // namespace pcie_ep::host_staged
