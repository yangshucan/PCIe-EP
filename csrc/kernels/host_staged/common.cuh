#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace pcie_ep::host_staged {

constexpr int kNumMaxRanks = 16;
constexpr int kNumMaxTopk = 32;
constexpr int kNumMaxExperts = 4096;
constexpr int kNumThreads = 256;
constexpr int kNumErrorFields = 8;
constexpr int kNumStatisticsFields = 8;
constexpr int kNumSourceMetadataFields = 3;
constexpr int kNumTransportTimestampFields = 2;
constexpr int kNumBf16Bytes = 2;
constexpr std::size_t kNumSignalAlignmentBytes = 128;
constexpr std::size_t kNumSharedMemoryPageBytes = 4096;

// GPU diagnostic arrays are part of the Python result ABI. Keep all numeric
// codes, field positions, and the eight-element capacities unchanged.
enum ErrorCode : int {
    kErrorSuccess = 0,
    kErrorInvalidExpertRouting = 2,
    kErrorCountsMailboxWait = 11,
    kErrorCountsReadyWait = 12,
    kErrorBatchReadyWait = 13,
    kErrorBatchAckWait = 14,
    kErrorRingProgressWait = 15,
    kErrorExpandedCapacity = 16,
    kErrorPayloadSequenceMismatch = 18,
    kErrorRuntimeAlreadyAborted = 19,
    kErrorEpochExhausted = 20,
};

enum ErrorField : int {
    kErrorCodeField = 0,
    // Invalid routing: source token index / top-k index.
    // Wait failure: local rank / epoch. Unused tail fields remain zero.
    kErrorContextField = 1,
    kErrorDetailField = 2,
};

enum StatisticField : int {
    kNumHostPayloadWritesField = 0,
    kNumHostPayloadReadsField = 1,
    kNumDescriptorsPublishedField = 2,
};

enum TransportTimestampField : int {
    kTransportStartNsField = 0,
    kTransportEndNsField = 1,
};

// Each signal has a single writer. The other participants only acquire-load it;
// no cross-device read-modify-write operation targets mapped CPU memory.
struct alignas(kNumSignalAlignmentBytes) Signal {
    std::uint64_t value;
    char padding[120];
};

// One source token owns one payload even if its top-k touches several ranks.
// rank_ordinals implements rank deduplication; expert_ordinals preserves each
// expert branch for the later, entirely local HBM expansion.
struct TokenMetadata {
    std::uint64_t destination_mask;
    int token_index;
    int rank_ordinals[kNumMaxRanks];
    int expert_ordinals[kNumMaxTopk];
    int expert_indices[kNumMaxTopk];
    float weights[kNumMaxTopk];
};

// Descriptor credit and payload ACK are separate resources. Consuming this
// descriptor does not release a multicast payload until every target ACKs it.
struct alignas(kNumSignalAlignmentBytes) Descriptor {
    std::uint64_t ready_sequence;
    std::uint64_t consumed_sequence;
    std::uint64_t payload_sequence;
    int payload_slot;
    int token_index;
    int rank_ordinal;
    char padding[92];
};

// Offsets refer to one source's shared CPU segment. Preserve this field order
// and the allocation order below: S1/S2 and S3/S4 share the same memory ABI.
struct SharedMemoryLayout {
    std::size_t size_bytes;
    std::size_t counts_ready_offset;
    std::size_t counts_consumed_offset;
    std::size_t batch_ready_offset;
    std::size_t batch_ack_offset;
    std::size_t abort_offset;
    std::size_t counts_offset;
    std::size_t batch_metadata_offset;
    std::size_t batch_payload_offset;
    std::size_t ring_metadata_offset;
    std::size_t ring_payload_offset;
    std::size_t payload_ack_offset;
    std::size_t descriptors_offset;

    int num_ranks;
    int num_max_tokens_per_rank;
    int num_hidden;
    int num_topk;
    int num_experts;
    int num_payload_slots;
    int num_descriptor_slots;
};

static_assert(sizeof(int) == 4 && sizeof(std::size_t) == 8, "The staged ABI requires a 64-bit host");
static_assert(sizeof(Signal) == 128 && alignof(Signal) == 128, "Signal ABI changed");
static_assert(offsetof(Signal, value) == 0 && offsetof(Signal, padding) == 8, "Signal fields moved");
static_assert(sizeof(TokenMetadata) == 464 && alignof(TokenMetadata) == 8, "Token metadata ABI changed");
static_assert(offsetof(TokenMetadata, destination_mask) == 0, "Token destination mask moved");
static_assert(offsetof(TokenMetadata, token_index) == 8, "Token index moved");
static_assert(offsetof(TokenMetadata, rank_ordinals) == 12, "Token rank ordinals moved");
static_assert(offsetof(TokenMetadata, expert_ordinals) == 76, "Token expert ordinals moved");
static_assert(offsetof(TokenMetadata, expert_indices) == 204, "Token expert indices moved");
static_assert(offsetof(TokenMetadata, weights) == 332, "Token weights moved");
static_assert(sizeof(Descriptor) == 128 && alignof(Descriptor) == 128, "Descriptor ABI changed");
static_assert(offsetof(Descriptor, ready_sequence) == 0, "Descriptor READY moved");
static_assert(offsetof(Descriptor, consumed_sequence) == 8, "Descriptor credit moved");
static_assert(offsetof(Descriptor, payload_sequence) == 16, "Descriptor payload sequence moved");
static_assert(offsetof(Descriptor, payload_slot) == 24, "Descriptor payload slot moved");
static_assert(offsetof(Descriptor, token_index) == 28, "Descriptor token index moved");
static_assert(offsetof(Descriptor, rank_ordinal) == 32, "Descriptor rank ordinal moved");
static_assert(offsetof(Descriptor, padding) == 36, "Descriptor padding moved");
static_assert(sizeof(SharedMemoryLayout) == 136 && alignof(SharedMemoryLayout) == 8, "Layout ABI changed");
static_assert(offsetof(SharedMemoryLayout, num_ranks) == 104, "Layout dimensions moved");

inline std::size_t align_up(std::size_t value, std::size_t alignment = kNumSignalAlignmentBytes) {
    return (value + alignment - 1) / alignment * alignment;
}

inline SharedMemoryLayout make_shared_memory_layout(
    int num_ranks, int num_max_tokens_per_rank, int num_hidden, int num_topk,
    int num_experts, int num_payload_slots, int num_descriptor_slots, int stage
) {
    SharedMemoryLayout result{};
    result.num_ranks = num_ranks;
    result.num_max_tokens_per_rank = num_max_tokens_per_rank;
    result.num_hidden = num_hidden;
    result.num_topk = num_topk;
    result.num_experts = num_experts;
    result.num_payload_slots = num_payload_slots;
    result.num_descriptor_slots = num_descriptor_slots;

    std::size_t cursor = 0;
    const auto reserve = [&](std::size_t size_bytes) {
        const auto offset = cursor;
        cursor += align_up(size_bytes);
        return offset;
    };
    result.counts_ready_offset = reserve(sizeof(Signal));
    result.counts_consumed_offset = reserve(num_ranks * sizeof(Signal));
    result.batch_ready_offset = reserve(sizeof(Signal));
    result.batch_ack_offset = reserve(num_ranks * sizeof(Signal));
    result.abort_offset = reserve(sizeof(Signal));
    result.counts_offset = reserve((num_ranks + num_experts) * sizeof(int));
    result.batch_metadata_offset = reserve(stage <= 2 ? num_max_tokens_per_rank * sizeof(TokenMetadata) : 0);
    result.batch_payload_offset = reserve(
        stage <= 2 ? std::size_t(num_max_tokens_per_rank) * num_hidden * kNumBf16Bytes : 0);
    result.ring_metadata_offset = reserve(num_payload_slots * sizeof(TokenMetadata));
    result.ring_payload_offset = reserve(std::size_t(num_payload_slots) * num_hidden * kNumBf16Bytes);
    result.payload_ack_offset = reserve(std::size_t(num_payload_slots) * num_ranks * sizeof(Signal));
    result.descriptors_offset = reserve(std::size_t(num_ranks) * num_descriptor_slots * sizeof(Descriptor));
    result.size_bytes = align_up(cursor, kNumSharedMemoryPageBytes);
    return result;
}

// The host passes only this rank's CUDA pointers and mapped CPU-segment aliases.
// mapped_source_buffers never contains another GPU's device-memory pointer.
struct DeviceState {
    SharedMemoryLayout layout;
    char* mapped_source_buffers[kNumMaxRanks];
    TokenMetadata* route_plan;
    int* rank_expert_counts;
    int* remote_token_indices;
    int* num_remote_tokens;
    std::uint64_t* epoch;

    int* source_offsets;
    int* source_expert_prefix;
    int* expert_counts;
    int* expert_offsets;
    std::int64_t* unique_count;
    std::int64_t* expanded_count;
    int* expert_prefix;
    std::int64_t* errors;
    std::int64_t* statistics;
    std::int64_t* transport_timestamps_ns;

    std::uint16_t* unique_payload;
    std::int64_t* unique_topk_indices;
    float* unique_topk_weights;
    std::int64_t* unique_source_metadata;
    int* unique_expert_ordinals;
    int local_rank;
    std::uint64_t timeout_ns;
};

static_assert(std::is_standard_layout_v<DeviceState>, "Device state must remain a plain launch argument");
static_assert(sizeof(DeviceState) == 440 && alignof(DeviceState) == 8, "Device launch ABI changed");

__device__ inline std::uint64_t load_acquire_system(std::uint64_t* address) {
    return cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(*address).load(cuda::memory_order_acquire);
}

__device__ inline void store_release_system(std::uint64_t* address, std::uint64_t value) {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(*address).store(value, cuda::memory_order_release);
}

__device__ inline std::uint64_t global_time_ns() {
    std::uint64_t timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    return timestamp;
}

__device__ inline std::uint64_t* signal_value_ptr(char* segment, std::size_t offset, int index = 0) {
    return &reinterpret_cast<Signal*>(segment + offset)[index].value;
}

// Called by a progress leader. On failure, stop the runtime rather than forging
// ACKs that would let a producer overwrite memory still used by a peer.
__device__ inline bool should_abort(DeviceState& state, std::uint64_t start_ns, int error_code) {
    for (int rank = 0; rank < state.layout.num_ranks; ++rank) {
        if (load_acquire_system(signal_value_ptr(state.mapped_source_buffers[rank], state.layout.abort_offset))) {
            state.errors[kErrorCodeField] = error_code;
            return true;
        }
    }
    if (global_time_ns() - start_ns > state.timeout_ns) {
        state.errors[kErrorCodeField] = error_code;
        state.errors[kErrorContextField] = state.local_rank;
        state.errors[kErrorDetailField] = *state.epoch;
        store_release_system(
            signal_value_ptr(state.mapped_source_buffers[state.local_rank], state.layout.abort_offset), *state.epoch);
        return true;
    }
    return false;
}

}  // namespace pcie_ep::host_staged
