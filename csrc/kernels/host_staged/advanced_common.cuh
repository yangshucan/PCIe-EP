#pragma once

#include "common.cuh"
#include "quantization.cuh"

namespace pcie_ep::host_staged {

constexpr int kNumMaxNumaNodes = 8;
constexpr int kNumMaxEndpoints = kNumMaxRanks + kNumMaxNumaNodes;
constexpr int kNumMaxLocalStages = 3;
constexpr std::uint64_t kReceivePrefixFailed = ~std::uint64_t(0);

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
    // Per-output, device-local claim and generation-valued READY. Never RMW
    // mapped host memory, and never share these between live buffers.
    std::uint64_t* receive_prefix_state;
    std::int64_t* prefix_tx_timestamps_ns;
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
    std::uint64_t prefix_delay_ns;
    std::uint64_t control_abort_after_ns;
    int counts_ack_skip_source;
#if PCIE_EP_ENABLE_PROFILING
    std::int64_t* profiling;
#endif
};

#if PCIE_EP_ENABLE_PROFILING
// Per-channel rows follow the pipelined kernel's 11 warp roles. Each row has
// one lane-0 writer and is flushed only on role exit. Durations overlap across
// roles; they must never be added as if they were exclusive kernel phases.
constexpr int kNumTransportProfileRoles = 11;
constexpr int kNumTransportProfileFields = 32;
constexpr int kTransportProfileSampleStride = 32;
enum TransportProfileField : int {
    kProfileEnterNs = 0, kProfileExitNs, kProfileTotalNs,
    kProfileIterations, kProfileWorkSamples, kProfileWorkNs, kProfileFenceNs, kProfileWorkAndFenceNs,
    kProfileFirstPublicationNs, kProfileLastPublicationNs,
    kProfileFailedNs, kProfileFailedCalls, kProfileFailedSignalLoads,
    kProfileDrainedNs, kProfileDrainedCalls, kProfileDrainedSamples,
    kProfileDrainedAckLoads, kProfileDrainedRelayLoads,
    kProfileDescriptorProbeNs, kProfileDescriptorProbeCalls, kProfileDescriptorProbeSamples,
    kProfileDescriptorNotReady, kProfileLocalStageBusy,
    kProfileTxFinishedNs, kProfileRxFinishedNs, kProfilePrefixWaitNs,
    kProfileAvailableNs, kProfileAvailableCalls, kProfileItems,
    kProfilePublications, kProfileSampleStride, kProfileAvailableSamples,
};
// Role 0 uses this otherwise RX-only event slot for guardian verification.
constexpr int kProfileAbortObservedNs = kProfileRxFinishedNs;

__device__ __forceinline__ std::uint64_t transport_profile_clock() {
    std::uint64_t timestamp;
    // A compiler ordering boundary only: no GPU fence/barrier is introduced.
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) :: "memory");
    return timestamp;
}

struct TransportProfile {
    std::uint64_t values[kNumTransportProfileFields] = {};

    __device__ __forceinline__ void record_work(std::uint64_t begin, std::uint64_t copied, std::uint64_t fenced) {
        ++values[kProfileWorkSamples];
        values[kProfileWorkNs] += copied - begin;
        values[kProfileFenceNs] += fenced - copied;
        values[kProfileWorkAndFenceNs] += fenced - begin;
    }

    __device__ __forceinline__ void save(AdvancedDeviceState& state, int channel, int role) {
        values[kProfileExitNs] = transport_profile_clock();
        values[kProfileTotalNs] = values[kProfileExitNs] - values[kProfileEnterNs];
        values[kProfileSampleStride] = kTransportProfileSampleStride;
        auto* output = state.profiling + (channel * kNumTransportProfileRoles + role) * kNumTransportProfileFields;
        #pragma unroll
        for (int field = 0; field < kNumTransportProfileFields; ++field)
            output[field] = values[field];
    }
};
#endif

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
__device__ inline bool advanced_any_abort(AdvancedDeviceState& state
#if PCIE_EP_ENABLE_PROFILING
    , std::uint64_t* signal_loads = nullptr
#endif
) {
    for (int rank = 0; rank < state.base.layout.num_ranks; ++rank) {
#if PCIE_EP_ENABLE_PROFILING
        if (signal_loads)
            ++*signal_loads;
#endif
        if (load_acquire_system(signal_value_ptr(state.base.mapped_source_buffers[rank],
                state.base.layout.abort_offset)))
            return true;
        for (int channel = 0; channel < state.channels; ++channel) {
#if PCIE_EP_ENABLE_PROFILING
            if (signal_loads)
                ++*signal_loads;
#endif
            if (load_acquire_system(signal_value_ptr(state.base.mapped_source_buffers[rank],
                state.shared.channel_abort_offset, channel)))
                return true;
        }
        if (state.numa_mode) {
            for (int index = 0; index < state.channels * state.num_numa_nodes; ++index) {
#if PCIE_EP_ENABLE_PROFILING
                if (signal_loads)
                    ++*signal_loads;
#endif
                if (load_acquire_system(signal_value_ptr(state.base.mapped_source_buffers[rank],
                    state.shared.relay_failed_offset, index)))
                    return true;
            }
        }
    }
    return false;
}

__device__ inline bool advanced_failed(AdvancedDeviceState& state, std::uint64_t start_ns
#if PCIE_EP_ENABLE_PROFILING
    , std::uint64_t* signal_loads = nullptr
#endif
) {
    return advanced_any_abort(state
#if PCIE_EP_ENABLE_PROFILING
        , signal_loads
#endif
    ) || global_time_ns() - start_ns > state.base.timeout_ns;
}

// All lanes participate. Each lane issues independent system-scope loads;
// the vote joins them before the leader publishes a device-local verdict.
__device__ inline bool advanced_any_abort_warp(AdvancedDeviceState& state, int lane
#if PCIE_EP_ENABLE_PROFILING
    , std::uint64_t* signal_loads = nullptr
#endif
) {
    const int flags_per_rank = 1 + state.channels +
        (state.numa_mode ? state.channels * state.num_numa_nodes : 0);
    const int num_flags = state.base.layout.num_ranks * flags_per_rank;
    bool aborted = false;
#if PCIE_EP_ENABLE_PROFILING
    unsigned loads = 0;
#endif
    for (int index = lane; index < num_flags; index += 32) {
        auto* segment = state.base.mapped_source_buffers[index / flags_per_rank];
        const int flag = index % flags_per_rank;
        auto* signal = flag == 0 ? signal_value_ptr(segment, state.base.layout.abort_offset) :
            flag <= state.channels ? signal_value_ptr(segment, state.shared.channel_abort_offset, flag - 1) :
            signal_value_ptr(segment, state.shared.relay_failed_offset, flag - 1 - state.channels);
        aborted |= load_acquire_system(signal) != 0;
#if PCIE_EP_ENABLE_PROFILING
        ++loads;
#endif
    }
#if PCIE_EP_ENABLE_PROFILING
    if (signal_loads)
        *signal_loads = __reduce_add_sync(0xffffffffu, loads);
#endif
    return __any_sync(0xffffffffu, aborted);
}

struct AdvancedHealthCheck {
    std::uint64_t start_ns;
    std::uint64_t next_scan_ns;
    std::uint64_t interval_ns;

    __device__ AdvancedHealthCheck(const AdvancedDeviceState& state, std::uint64_t start)
        : start_ns(start) {
        constexpr std::uint64_t kMaxIntervalNs = 1000000;
        const auto timeout_interval = state.base.timeout_ns / 8;
        interval_ns = timeout_interval < kMaxIntervalNs ? timeout_interval : kMaxIntervalNs;
        next_scan_ns = start + interval_ns;
    }

    // Local timeout is checked on every bounded control iteration. Remote
    // scans use wall time, so a faster idle loop cannot increase flag traffic.
    __device__ inline bool poll(AdvancedDeviceState& state, int lane
#if PCIE_EP_ENABLE_PROFILING
        , TransportProfile* profile = nullptr
#endif
    ) {
        int action = 0;
        if (!lane) {
            const auto now = global_time_ns();
            action = now - start_ns > state.base.timeout_ns ? 2 : now >= next_scan_ns;
        }
        action = __shfl_sync(0xffffffffu, action, 0);
        if (action != 1)
            return action == 2;
#if PCIE_EP_ENABLE_PROFILING
        const auto begin = !lane && profile ? transport_profile_clock() : 0;
        std::uint64_t loads = 0;
        const bool failed = advanced_any_abort_warp(state, lane, &loads);
#else
        const bool failed = advanced_any_abort_warp(state, lane);
#endif
        if (!lane) {
            const auto completed = global_time_ns();
            // Restart at completion; never catch up missed scan deadlines.
            next_scan_ns = completed + interval_ns;
#if PCIE_EP_ENABLE_PROFILING
            if (profile) {
                profile->values[kProfileFailedNs] += transport_profile_clock() - begin;
                ++profile->values[kProfileFailedCalls];
                profile->values[kProfileFailedSignalLoads] += loads;
            }
#endif
        }
        return failed;
    }
};

}  // namespace pcie_ep::host_staged
