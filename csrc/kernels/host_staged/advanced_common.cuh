#pragma once

#include "common.cuh"
#include "quantization.cuh"

namespace pcie_ep::host_staged {

constexpr int kNumMaxNumaNodes = 8;
constexpr int kNumMaxEndpoints = kNumMaxRanks + kNumMaxNumaNodes;
constexpr int kNumMaxLocalStages = 3;

// Source endpoints are either a destination GPU or P + destination NUMA node.
// The latter has a CPU relay owner. All mapped signals remain single-writer.
struct AdvancedSharedMemoryLayout {
    SharedMemoryLayout base;
    std::size_t size_bytes;
    std::size_t channel_counts_offset;
    std::size_t endpoint_counts_offset;
    std::size_t source_metadata_offset;
    std::size_t source_payload_offset;
    std::size_t source_scale_offset;
    std::size_t source_ack_offset;
    std::size_t source_descriptors_offset;
    std::size_t replica_metadata_offset;
    std::size_t replica_payload_offset;
    std::size_t replica_scale_offset;
    std::size_t replica_ack_offset;
    std::size_t replica_descriptors_offset;
    std::size_t relay_drained_offset;
    std::size_t payload_stride_bytes;
    std::size_t channel_abort_offset;
    std::size_t relay_failed_offset;
    int channels;
    int stages;
    int num_numa_nodes;
    int num_endpoints;
    int replica_slots;
    int payload_dtype;
};

inline AdvancedSharedMemoryLayout make_advanced_shared_memory_layout(
    int num_ranks, int num_max_tokens_per_rank, int hidden, int topk, int experts,
    int payload_slots, int descriptor_slots, int channels, int stages,
    int num_numa_nodes, int replica_slots, int payload_dtype
) {
    AdvancedSharedMemoryLayout result{};
    result.base = make_shared_memory_layout(num_ranks, num_max_tokens_per_rank, hidden,
        topk, experts, payload_slots, descriptor_slots);
    result.channels = channels;
    result.stages = stages;
    result.num_numa_nodes = num_numa_nodes;
    result.num_endpoints = num_ranks + num_numa_nodes;
    result.replica_slots = replica_slots;
    result.payload_dtype = payload_dtype;
    result.payload_stride_bytes = align_up(std::size_t(hidden) * (payload_dtype ? 1 : 2), kNumSharedMemoryPageBytes);
    std::size_t cursor = result.base.size_bytes;
    const auto reserve = [&](std::size_t bytes) {
        const auto offset = cursor;
        cursor += align_up(bytes);
        return offset;
    };
    const std::size_t source_slots = std::size_t(channels) * payload_slots;
    const std::size_t replicas = std::size_t(channels) * num_numa_nodes * replica_slots;
    result.channel_counts_offset = reserve(std::size_t(channels) * num_ranks * sizeof(int));
    result.endpoint_counts_offset = reserve(std::size_t(channels) * result.num_endpoints * sizeof(int));
    result.source_metadata_offset = reserve(source_slots * sizeof(TokenMetadata));
    cursor = align_up(cursor, kNumSharedMemoryPageBytes);
    result.source_payload_offset = reserve(source_slots * result.payload_stride_bytes);
    result.source_scale_offset = reserve(source_slots * sizeof(float));
    result.source_ack_offset = reserve(source_slots * result.num_endpoints * sizeof(Signal));
    result.source_descriptors_offset = reserve(std::size_t(channels) * result.num_endpoints * descriptor_slots * sizeof(Descriptor));
    result.replica_metadata_offset = reserve(replicas * sizeof(TokenMetadata));
    cursor = align_up(cursor, kNumSharedMemoryPageBytes);
    result.replica_payload_offset = reserve(replicas * result.payload_stride_bytes);
    result.replica_scale_offset = reserve(replicas * sizeof(float));
    result.replica_ack_offset = reserve(replicas * num_ranks * sizeof(Signal));
    result.replica_descriptors_offset = reserve(std::size_t(channels) * num_numa_nodes * num_ranks * descriptor_slots * sizeof(Descriptor));
    result.relay_drained_offset = reserve(std::size_t(channels) * num_numa_nodes * sizeof(Signal));
    result.channel_abort_offset = reserve(std::size_t(channels) * sizeof(Signal));
    result.relay_failed_offset = reserve(std::size_t(channels) * num_numa_nodes * sizeof(Signal));
    result.size_bytes = align_up(cursor, kNumSharedMemoryPageBytes);
    return result;
}

struct AdvancedDeviceState {
    DeviceState base;
    AdvancedSharedMemoryLayout shared;
    int channels;
    int stages;
    int num_numa_nodes;
    int numa_mode;
    int payload_dtype;
    int fuse_receive_expand;
    int source_fusion;
    int warp_specialization;
    int rank_numa[kNumMaxRanks];
    const std::uint16_t* input_payload;
    const std::int64_t* topk_indices;
    const float* topk_weights;
    const std::uint8_t* fp8_payload;
    const float* fp8_scales;
    int* channel_remote_indices;
    int* channel_num_remote_tokens;
    int* channel_rank_counts;
    int* channel_endpoint_counts;
    int* expected_receive_counts;
    std::uint16_t* expanded_payload;
    std::int64_t* expanded_expert_indices;
    float* expanded_weights;
    std::int64_t* expanded_source_metadata;
    int expanded_capacity;
    std::int64_t* channel_statistics;
    std::int64_t* channel_timestamps_ns;
    std::uint64_t metadata_delay_ns;
    std::uint64_t tx_delay_ns;
    std::uint64_t rx_delay_ns;
    std::uint64_t scatter_delay_ns;
};

__device__ inline int local_load(int* address) {
    return cuda::atomic_ref<int, cuda::thread_scope_block>(*address).load(cuda::memory_order_acquire);
}

__device__ inline void local_store(int* address, int value) {
    cuda::atomic_ref<int, cuda::thread_scope_block>(*address).store(value, cuda::memory_order_release);
}

__device__ inline void local_arrive(int* address) {
    cuda::atomic_ref<int, cuda::thread_scope_block>(*address).fetch_add(1, cuda::memory_order_acq_rel);
}

__device__ inline void test_delay(std::uint64_t duration_ns) {
    if (!duration_ns)
        return;
    const auto start = global_time_ns();
    while (global_time_ns() - start < duration_ns)
        __nanosleep(64);
}

__device__ inline std::uint64_t advanced_sequence(std::uint64_t epoch, int index) {
    return (epoch << 32) | std::uint64_t(index + 1);
}

__device__ inline int advanced_endpoint(const AdvancedDeviceState& state, int destination_rank) {
    const int source_node = state.rank_numa[state.base.local_rank];
    const int destination_node = state.rank_numa[destination_rank];
    return state.numa_mode && source_node != destination_node ?
        state.base.layout.num_ranks + destination_node : destination_rank;
}

__device__ inline std::uint64_t advanced_endpoint_mask(
    const AdvancedDeviceState& state, std::uint64_t rank_mask
) {
    std::uint64_t result = 0;
    for (int rank = 0; rank < state.base.layout.num_ranks; ++rank) {
        if (rank != state.base.local_rank && (rank_mask & (1ull << rank)))
            result |= 1ull << advanced_endpoint(state, rank);
    }
    return result;
}

__device__ inline Descriptor* source_descriptor(
    const AdvancedDeviceState& state, int source, int channel, int endpoint, int ordinal
) {
    return reinterpret_cast<Descriptor*>(state.base.mapped_source_buffers[source] + state.shared.source_descriptors_offset) +
        (channel * state.shared.num_endpoints + endpoint) * state.base.layout.num_descriptor_slots +
        ordinal % state.base.layout.num_descriptor_slots;
}

__device__ inline bool use_replica(const AdvancedDeviceState& state, int source) {
    return state.numa_mode && state.rank_numa[source] != state.rank_numa[state.base.local_rank];
}

__device__ inline Descriptor* receive_descriptor(
    const AdvancedDeviceState& state, int source, int channel, int ordinal
) {
    if (!use_replica(state, source))
        return source_descriptor(state, source, channel, state.base.local_rank, ordinal);
    const int node = state.rank_numa[state.base.local_rank];
    return reinterpret_cast<Descriptor*>(state.base.mapped_source_buffers[source] + state.shared.replica_descriptors_offset) +
        ((channel * state.num_numa_nodes + node) * state.base.layout.num_ranks + state.base.local_rank) *
        state.base.layout.num_descriptor_slots + ordinal % state.base.layout.num_descriptor_slots;
}

__device__ inline std::size_t receive_slot_index(
    const AdvancedDeviceState& state, int source, int channel, int slot
) {
    return use_replica(state, source) ?
        (std::size_t(channel) * state.num_numa_nodes + state.rank_numa[state.base.local_rank]) * state.shared.replica_slots + slot :
        std::size_t(channel) * state.base.layout.num_payload_slots + slot;
}

__device__ inline TokenMetadata* receive_metadata(
    const AdvancedDeviceState& state, int source, int channel, int slot
) {
    return reinterpret_cast<TokenMetadata*>(state.base.mapped_source_buffers[source] +
        (use_replica(state, source) ? state.shared.replica_metadata_offset : state.shared.source_metadata_offset)) +
        receive_slot_index(state, source, channel, slot);
}

__device__ inline std::uint8_t* receive_payload(
    const AdvancedDeviceState& state, int source, int channel, int slot
) {
    return reinterpret_cast<std::uint8_t*>(state.base.mapped_source_buffers[source] +
        (use_replica(state, source) ? state.shared.replica_payload_offset : state.shared.source_payload_offset)) +
        receive_slot_index(state, source, channel, slot) * state.shared.payload_stride_bytes;
}

__device__ inline float* receive_scale(
    const AdvancedDeviceState& state, int source, int channel, int slot
) {
    return reinterpret_cast<float*>(state.base.mapped_source_buffers[source] +
        (use_replica(state, source) ? state.shared.replica_scale_offset : state.shared.source_scale_offset)) +
        receive_slot_index(state, source, channel, slot);
}

__device__ inline std::uint64_t* receive_ack(
    const AdvancedDeviceState& state, int source, int channel, int slot
) {
    const auto index = receive_slot_index(state, source, channel, slot);
    return signal_value_ptr(state.base.mapped_source_buffers[source],
        use_replica(state, source) ? state.shared.replica_ack_offset : state.shared.source_ack_offset,
        index * (use_replica(state, source) ? state.base.layout.num_ranks : state.shared.num_endpoints) + state.base.local_rank);
}

__device__ inline void advanced_record_error(AdvancedDeviceState& state, int code) {
    if (atomicCAS(reinterpret_cast<unsigned long long*>(state.base.errors), 0ull,
                  static_cast<unsigned long long>(code)) == 0ull) {
        state.base.errors[kErrorContextField] = state.base.local_rank;
        state.base.errors[kErrorDetailField] = *state.base.epoch;
    }
}

// Only a channel's TX-control leader writes its mapped channel-abort signal.
// CPU relay failures have their own per-worker single-writer signal array.
__device__ inline bool advanced_any_abort(AdvancedDeviceState& state) {
    for (int rank = 0; rank < state.base.layout.num_ranks; ++rank) {
        if (load_acquire_system(signal_value_ptr(state.base.mapped_source_buffers[rank],
                state.base.layout.abort_offset)))
            return true;
        for (int channel = 0; channel < state.channels; ++channel) {
            if (load_acquire_system(signal_value_ptr(state.base.mapped_source_buffers[rank],
                state.shared.channel_abort_offset, channel)))
                return true;
        }
        if (state.numa_mode) {
            for (int index = 0; index < state.channels * state.num_numa_nodes; ++index) {
                if (load_acquire_system(signal_value_ptr(state.base.mapped_source_buffers[rank],
                    state.shared.relay_failed_offset, index)))
                    return true;
            }
        }
    }
    return false;
}

__device__ inline bool advanced_failed(AdvancedDeviceState& state, std::uint64_t start_ns) {
    return advanced_any_abort(state) || global_time_ns() - start_ns > state.base.timeout_ns;
}

}  // namespace pcie_ep::host_staged
