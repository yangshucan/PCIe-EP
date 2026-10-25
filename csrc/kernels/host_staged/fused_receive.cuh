#pragma once

#include "advanced_dispatch.cuh"

namespace pcie_ep::host_staged {

constexpr int kNumFusedReceiveWorkerWarps = 3;
constexpr int kNumFusedReceiveWorkerThreads = kNumFusedReceiveWorkerWarps * 32;
constexpr int kNumVectorizedTransportThreads = 8 * 32;
constexpr int kNumPipelinedTransportThreads = 11 * 32;
constexpr int kNumReceiveExpandDiagnosticFields = 2;

enum ReceiveExpandDiagnosticField : int {
    kReceiveExpandOverlapCount = 0,
    kReceiveExpandScatterCompletionTicket = 1,
};

// The shared-memory ABI is unchanged. For these kernels, a local stage's
// atomic ready value is ticket + 1; zero means FREE. A boolean READY followed
// by a separate plain ticket read could observe an old READY and a new ticket
// while control is still preparing the new metadata. Generation-valued READY
// admits workers only after the matching release, without that ABA window.
__device__ inline bool fused_receive_initialize(
    AdvancedDeviceState& state, AdvancedLocalQueues& queues,
    std::int64_t* receive_expand_overlap
) {
    int entry_failed = 0;
    if (!threadIdx.x) {
        entry_failed = cuda::atomic_ref<std::int64_t, cuda::thread_scope_device>(
            state.base.errors[kErrorCodeField]).load(cuda::memory_order_acquire) != 0;
        queues.abort = 0;
        queues.tx_finished = 0;
        queues.rx_finished = 0;
        for (int index = 0; index < state.stages; ++index) {
            queues.tx[index].ready = 0;
            queues.rx[index].ready = 0;
        }
        if (receive_expand_overlap) {
            auto* diagnostic = receive_expand_overlap + blockIdx.x * kNumReceiveExpandDiagnosticFields;
            diagnostic[kReceiveExpandOverlapCount] = 0;
            diagnostic[kReceiveExpandScatterCompletionTicket] = 0;
        }
    }
    // The reduction latches one entry verdict for the complete CTA. Do not
    // branch on mutable queues.abort after this barrier: a fast role could
    // update it before another warp evaluates its entry branch.
    return __syncthreads_or(entry_failed) == 0;
}

__device__ inline int fused_receive_stage_action(
    AdvancedLocalQueues& queues, AdvancedLocalStage& stage,
    int ticket, int lane, bool wait_for_host_readers
) {
    int action = 0;
    if (!lane) {
        if (local_load(&queues.abort)) {
            action = -1;
        } else if (local_load(&stage.ready) == ticket + 1 &&
                   (!wait_for_host_readers ||
                    local_load(&stage.host_readers_done) == kNumFusedReceiveWorkerWarps)) {
            action = 1;
        } else if (local_load(&queues.rx_finished)) {
            action = -1;
        }
    }
    return __shfl_sync(0xffffffffu, action, 0);
}

__device__ inline bool fused_receive_wait_for_readers(
    AdvancedLocalQueues& queues, AdvancedLocalStage& stage, int lane
) {
    int readers_ready = 0;
    do {
        if (!lane) {
            readers_ready = local_load(&queues.abort) ? -1 :
                (local_load(&stage.host_readers_done) == kNumFusedReceiveWorkerWarps ? 1 : 0);
        }
        readers_ready = __shfl_sync(0xffffffffu, readers_ready, 0);
        if (!readers_ready)
            __nanosleep(32);
    } while (!readers_ready);
    return readers_ready > 0;
}

__device__ inline void fused_receive_scatter_delay(
    AdvancedDeviceState& state, int worker_index
) {
    const int hidden = state.base.layout.num_hidden;
    const int work_units = hidden % 8 == 0 ? hidden / 8 : hidden;
    // Delay the last worker that actually owns a vector/scalar. For H=7168
    // this is worker 95 (scatter warp 2, lane 31); small H never delays an
    // idle lane. The delay remains observable in valid output bytes.
    if (worker_index == min(kNumFusedReceiveWorkerThreads - 1, work_units - 1))
        test_delay(state.scatter_delay_ns);
}

__device__ inline void scatter_rx_tile_vectorized(
    AdvancedDeviceState& state, AdvancedLocalStage& stage,
    const std::uint16_t* tile, int worker_index, int* aborted
) {
    const int hidden = state.base.layout.num_hidden;
    const int experts_per_rank = state.base.layout.num_experts / state.base.layout.num_ranks;
    for (int branch = 0; branch < state.base.layout.num_topk; ++branch) {
        const int expert = stage.metadata.expert_indices[branch] - state.base.local_rank * experts_per_rank;
        if (expert < 0 || expert >= experts_per_rank)
            continue;
        const int row = state.base.expert_offsets[expert] +
            state.base.source_expert_prefix[stage.source_rank * experts_per_rank + expert] +
            stage.metadata.expert_ordinals[branch];
        if (row >= state.expanded_capacity) {
            if (!worker_index) {
                advanced_record_error(state, kErrorExpandedCapacity);
                local_store(aborted, 1);
            }
            return;
        }
        auto* output = state.expanded_payload + std::size_t(row) * hidden;
        if (hidden % 8 == 0) {
            // Both tile and row starts are 16-byte aligned in this branch.
            // This changes only the local shared -> expert-HBM copy.
            for (int vector = worker_index; vector < hidden / 8; vector += kNumFusedReceiveWorkerThreads)
                reinterpret_cast<uint4*>(output)[vector] = reinterpret_cast<const uint4*>(tile)[vector];
        } else {
            for (int column = worker_index; column < hidden; column += kNumFusedReceiveWorkerThreads)
                output[column] = tile[column];
        }
        if (!worker_index)
            advanced_store_expanded_metadata(state, stage.metadata, stage.source_rank, branch, row, expert);
    }
}

__device__ inline void fused_receive_copy_host_payload(
    AdvancedDeviceState& state, AdvancedLocalQueues& queues, AdvancedLocalStage& stage,
    int channel, int ticket, int warp, int lane, std::uint16_t* tiles
) {
    const int hidden = state.base.layout.num_hidden;
    const int units = state.payload_dtype ? hidden : (hidden % 8 ? hidden : hidden / 8);
    if (warp * 32 + lane == min(kNumFusedReceiveWorkerThreads - 1, units - 1))
        test_delay(state.rx_delay_ns);
    auto* tile = tiles + std::size_t(ticket % state.stages) * hidden;
    copy_rx_payload(state, stage, channel, tile, warp * 32 + lane, kNumFusedReceiveWorkerThreads);
    __threadfence_system();
    __syncwarp();
    if (!lane)
        local_arrive(&stage.host_readers_done);
}

// Vector-copy ablation: preserve copy -> reader join -> scatter -> next ticket
// in the same three RX warps. Only shared -> expert copying is vectorized.
__device__ void fused_receive_vectorized_worker(
    AdvancedDeviceState state, AdvancedLocalQueues& queues,
    int channel, int warp, int lane, std::uint16_t* tiles
) {
    int ticket = 0;
    while (true) {
        auto& stage = queues.rx[ticket % state.stages];
        const int action = fused_receive_stage_action(queues, stage, ticket, lane, false);
        if (action < 0)
            break;
        if (action) {
            __syncwarp();  // Stage acquire -> all metadata/payload readers.
            fused_receive_copy_host_payload(state, queues, stage, channel, ticket, warp, lane, tiles);
            if (fused_receive_wait_for_readers(queues, stage, lane)) {
                __syncwarp();  // Reader-done acquire -> all shared-tile readers.
                fused_receive_scatter_delay(state, warp * 32 + lane);
                scatter_rx_tile_vectorized(state, stage,
                    tiles + std::size_t(ticket % state.stages) * state.base.layout.num_hidden,
                    warp * 32 + lane, &queues.abort);
            }
            __threadfence_system();
            __syncwarp();
            if (!lane)
                local_arrive(&stage.workers_done);
            ++ticket;
        }
        __syncwarp();
    }
}

// Producers retire their host read independently of the scatter consumers.
// They never increment workers_done: only consumers may release a local tile.
__device__ void fused_receive_producer(
    AdvancedDeviceState state, AdvancedLocalQueues& queues, int channel,
    int warp, int lane, std::uint16_t* tiles, std::int64_t* receive_expand_overlap
) {
    int ticket = 0;
    while (true) {
        auto& stage = queues.rx[ticket % state.stages];
        const int action = fused_receive_stage_action(queues, stage, ticket, lane, false);
        if (action < 0)
            break;
        if (action) {
            __syncwarp();  // Leader acquire -> all stage-reading lanes.
            if (!warp && !lane && ticket > 0 && receive_expand_overlap) {
                auto* diagnostic = receive_expand_overlap + channel * kNumReceiveExpandDiagnosticFields;
                const auto completed = cuda::atomic_ref<std::int64_t, cuda::thread_scope_device>(
                    diagnostic[kReceiveExpandScatterCompletionTicket]).load(cuda::memory_order_acquire);
                // A conservative proof point: the preceding ticket's
                // designated scatter warp has not published completion yet.
                // This is not a stall timer or proof of a complete next read.
                if (completed < ticket)
                    diagnostic[kReceiveExpandOverlapCount]++;
            }
            fused_receive_copy_host_payload(state, queues, stage, channel, ticket, warp, lane, tiles);
            ++ticket;
        }
        __syncwarp();
    }
}

// Consumers touch only shared metadata/tile and current-GPU output/workspace.
// The original mapped host payload may already be ACKed and overwritten.
__device__ void fused_receive_consumer(
    AdvancedDeviceState state, AdvancedLocalQueues& queues, int channel,
    int warp, int lane, std::uint16_t* tiles, std::int64_t* receive_expand_overlap
) {
    int ticket = 0;
    while (true) {
        auto& stage = queues.rx[ticket % state.stages];
        const int action = fused_receive_stage_action(queues, stage, ticket, lane, true);
        if (action < 0)
            break;
        if (action) {
            // Both matching-generation READY and all host-reader arrivals
            // were acquired by the leader. shfl alone orders no memory.
            __syncwarp();
            fused_receive_scatter_delay(state, warp * 32 + lane);
            scatter_rx_tile_vectorized(state, stage,
                tiles + std::size_t(ticket % state.stages) * state.base.layout.num_hidden,
                warp * 32 + lane, &queues.abort);
            __threadfence_system();
            __syncwarp();
            if (!lane) {
                if (warp == kNumFusedReceiveWorkerWarps - 1 && receive_expand_overlap) {
                    auto* diagnostic = receive_expand_overlap + channel * kNumReceiveExpandDiagnosticFields;
                    cuda::atomic_ref<std::int64_t, cuda::thread_scope_device>(
                        diagnostic[kReceiveExpandScatterCompletionTicket]).store(ticket + 1, cuda::memory_order_release);
                }
                local_arrive(&stage.workers_done);
            }
            ++ticket;
        }
        __syncwarp();
    }
}

// Host selection requires specialization + expert expansion. Unique receive
// and scalar-reference requests continue to launch the original kernel.
__global__ void advanced_transport_vectorized(
    AdvancedDeviceState state, std::int64_t* receive_expand_overlap
) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& queues = *reinterpret_cast<AdvancedLocalQueues*>(storage);
    auto* tiles = reinterpret_cast<std::uint16_t*>(storage + kNumAdvancedQueueBytes);
    if (!fused_receive_initialize(state, queues, receive_expand_overlap))
        return;
    const int channel = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp == 0)
        advanced_tx_control(state, queues, channel, lane);
    else if (warp == 1)
        advanced_rx_control(state, queues, channel, lane);
    else if (warp < 5)
        advanced_tx_worker(state, queues, channel, warp - 2, lane);
    else
        fused_receive_vectorized_worker(state, queues, channel, warp - 5, lane, tiles);
    __syncthreads();
    if (!threadIdx.x)
        state.channel_timestamps_ns[channel * 2 + 1] = global_time_ns();
}

__global__ void advanced_transport_pipelined(
    AdvancedDeviceState state, std::int64_t* receive_expand_overlap
) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& queues = *reinterpret_cast<AdvancedLocalQueues*>(storage);
    auto* tiles = reinterpret_cast<std::uint16_t*>(storage + kNumAdvancedQueueBytes);
    if (!fused_receive_initialize(state, queues, receive_expand_overlap))
        return;
    const int channel = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp == 0)
        advanced_tx_control(state, queues, channel, lane);
    else if (warp == 1)
        advanced_rx_control(state, queues, channel, lane);
    else if (warp < 5)
        advanced_tx_worker(state, queues, channel, warp - 2, lane);
    else if (warp < 8)
        fused_receive_producer(state, queues, channel, warp - 5, lane, tiles, receive_expand_overlap);
    else
        fused_receive_consumer(state, queues, channel, warp - 8, lane, tiles, receive_expand_overlap);
    // RX control exits only after host ACK and all scatter arrivals. TX
    // control stays alive to publish aborts while either direction is active.
    // Every role therefore reaches this final CTA join on completion/abort.
    __syncthreads();
    if (!threadIdx.x)
        state.channel_timestamps_ns[channel * 2 + 1] = global_time_ns();
}

}  // namespace pcie_ep::host_staged
