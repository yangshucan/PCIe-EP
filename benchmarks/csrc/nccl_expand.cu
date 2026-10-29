#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace nccl_shm_benchmark {

constexpr int kNumThreads = 256;
constexpr int kNumWarpThreads = 32;
constexpr int kNumSourceMetadataFields = 3;

// The NCCL message carries the original routes and weights along with each
// source token. No destination GPU consults precomputed host routing tables.
__global__ void pack_tokens(
    const std::uint8_t* payload, const std::int64_t* indices, const float* weights,
    std::uint8_t* packed, int num_hidden, int num_topk, int row_bytes
) {
    const int token = blockIdx.x;
    auto* output = packed + std::size_t(token) * row_bytes;
    const auto* input = reinterpret_cast<const uint4*>(payload + std::size_t(token) * num_hidden * 2);
    auto* output_vectors = reinterpret_cast<uint4*>(output);
    for (int vector = threadIdx.x; vector < num_hidden / 8; vector += blockDim.x)
        output_vectors[vector] = input[vector];

    auto* output_indices = reinterpret_cast<std::int64_t*>(output + num_hidden * 2);
    auto* output_weights = reinterpret_cast<float*>(output + num_hidden * 2 + num_topk * 8);
    for (int branch = threadIdx.x; branch < num_topk; branch += blockDim.x) {
        output_indices[branch] = indices[std::size_t(token) * num_topk + branch];
        output_weights[branch] = weights[std::size_t(token) * num_topk + branch];
    }
}

// Every expert/tile CTA computes stable token ordinals. Counting branches
// rather than a Boolean also preserves repeated expert IDs within one token.
__global__ void count_expert_tiles(
    const std::uint8_t* packed, int* positions, int* tile_counts,
    int num_tokens, int num_hidden, int num_topk, int num_tiles,
    int row_bytes, int first_expert, int num_local_experts
) {
    using BlockScan = cub::BlockScan<int, kNumThreads>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    const int expert = blockIdx.x;
    const int tile = blockIdx.y;
    const int token = tile * kNumThreads + threadIdx.x;
    const auto* indices = reinterpret_cast<const std::int64_t*>(
        packed + std::size_t(token) * row_bytes + num_hidden * 2);

    int hits = 0;
    if (token < num_tokens) {
        for (int branch = 0; branch < num_topk; ++branch) {
            const auto destination = indices[branch];
            hits += destination == first_expert + expert;
            // Exactly one CTA initializes nonlocal branches. Local branches
            // have one matching expert writer, so these stores cannot race.
            if (!expert && (destination < first_expert || destination >= first_expert + num_local_experts))
                positions[std::size_t(token) * num_topk + branch] = -1;
        }
    }
    int ordinal, tile_count;
    BlockScan(scan_storage).ExclusiveSum(hits, ordinal, tile_count);
    if (!threadIdx.x)
        tile_counts[expert * num_tiles + tile] = tile_count;
    if (token < num_tokens) {
        for (int branch = 0; branch < num_topk; ++branch) {
            if (indices[branch] == first_expert + expert)
                positions[std::size_t(token) * num_topk + branch] = ordinal++;
        }
    }
}

// One lane owns each expert's ordered tile prefix. A warp scan concatenates
// the expert segments without an atomic allocator or host count readback.
__global__ void prefix_expert_tiles(
    const int* tile_counts, int* tile_offsets, int* expert_counts,
    int* expert_offsets, int num_tiles, int num_local_experts
) {
    using WarpScan = cub::WarpScan<int>;
    __shared__ typename WarpScan::TempStorage scan_storage;
    const int expert = threadIdx.x;
    int count = 0;
    if (expert < num_local_experts) {
        for (int tile = 0; tile < num_tiles; ++tile) {
            const int index = expert * num_tiles + tile;
            tile_offsets[index] = count;
            count += tile_counts[index];
        }
        expert_counts[expert] = count;
    }
    int offset, total_count;
    WarpScan(scan_storage).ExclusiveSum(count, offset, total_count);
    if (expert < num_local_experts)
        expert_offsets[expert] = offset;
    if (!expert)
        expert_offsets[num_local_experts] = total_count;
}

// Each token CTA reads the received HBM row and expands only branches owned
// by this rank. Source/token/branch order comes from the preceding prefixes.
__global__ void scatter_expert_tokens(
    const std::uint8_t* packed, const int* positions, const int* tile_offsets,
    const int* expert_offsets, std::uint8_t* payload, std::int64_t* indices,
    float* weights, std::int64_t* source_metadata, int num_tokens_per_rank,
    int num_hidden, int num_topk, int num_tiles, int row_bytes, int first_expert
) {
    const int token = blockIdx.x;
    const auto* input = packed + std::size_t(token) * row_bytes;
    const auto* input_vectors = reinterpret_cast<const uint4*>(input);
    const auto* input_indices = reinterpret_cast<const std::int64_t*>(input + num_hidden * 2);
    const auto* input_weights = reinterpret_cast<const float*>(input + num_hidden * 2 + num_topk * 8);
    for (int branch = 0; branch < num_topk; ++branch) {
        const int position = positions[std::size_t(token) * num_topk + branch];
        if (position < 0)
            continue;
        const int expert = static_cast<int>(input_indices[branch]) - first_expert;
        const int row = expert_offsets[expert] + tile_offsets[expert * num_tiles + token / kNumThreads] + position;
        auto* output = reinterpret_cast<uint4*>(payload + std::size_t(row) * num_hidden * 2);
        for (int vector = threadIdx.x; vector < num_hidden / 8; vector += blockDim.x)
            output[vector] = input_vectors[vector];
        if (!threadIdx.x) {
            indices[row] = expert;
            weights[row] = input_weights[branch];
            source_metadata[std::size_t(row) * kNumSourceMetadataFields] = token / num_tokens_per_rank;
            source_metadata[std::size_t(row) * kNumSourceMetadataFields + 1] = token % num_tokens_per_rank;
            source_metadata[std::size_t(row) * kNumSourceMetadataFields + 2] = branch;
        }
    }
}

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype, const torch::Device& device, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be on a CUDA device");
    TORCH_CHECK(tensor.device() == device, name, " must be on the same CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unexpected dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_shape(const torch::Tensor& tensor, std::initializer_list<std::int64_t> dimensions, const char* name) {
    TORCH_CHECK(tensor.sizes() == at::IntArrayRef(dimensions.begin(), dimensions.size()), name, " has an unexpected shape");
}

void check_dimensions(int world, int tokens, int hidden, int topk, int experts, int rank) {
    TORCH_CHECK(world > 0 && tokens > 0 && hidden > 0 && topk > 0, "dimensions must be positive");
    TORCH_CHECK(experts > 0 && experts % world == 0, "experts must divide evenly across ranks");
    TORCH_CHECK(experts / world <= kNumWarpThreads, "at most 32 local experts are supported");
    TORCH_CHECK(rank >= 0 && rank < world, "rank is out of range");
    TORCH_CHECK(hidden % 8 == 0 && topk % 4 == 0, "wire rows must preserve 16-byte vector alignment");
    TORCH_CHECK(std::int64_t(world) * tokens * topk <= std::numeric_limits<int>::max(), "expanded rows exceed int32 capacity");
    TORCH_CHECK(std::int64_t(hidden) * 2 + std::int64_t(topk) * 12 <= std::numeric_limits<int>::max(), "wire row is too large");
}

void pack(
    const torch::Tensor& payload, const torch::Tensor& indices,
    const torch::Tensor& weights, const torch::Tensor& packed
) {
    TORCH_CHECK(payload.dim() == 2 && indices.dim() == 2, "payload and indices must be matrices");
    TORCH_CHECK(payload.size(0) > 0 && payload.size(0) <= std::numeric_limits<int>::max() &&
                payload.size(1) <= std::numeric_limits<int>::max(), "invalid payload dimensions");
    TORCH_CHECK(indices.size(1) > 0 && indices.size(1) <= std::numeric_limits<int>::max(), "invalid top-k dimension");
    const int tokens = payload.size(0), hidden = payload.size(1), topk = indices.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 8 == 0 && topk % 4 == 0, "wire rows must preserve 16-byte vector alignment");
    const std::int64_t row_bytes = std::int64_t(hidden) * 2 + std::int64_t(topk) * 12;
    TORCH_CHECK(row_bytes <= std::numeric_limits<int>::max(), "wire row is too large");
    const auto device = payload.device();
    check_tensor(payload, at::kBFloat16, device, "payload");
    check_tensor(indices, at::kLong, device, "indices");
    check_tensor(weights, at::kFloat, device, "weights");
    check_tensor(packed, at::kByte, device, "packed");
    check_shape(indices, {tokens, topk}, "indices");
    check_shape(weights, {tokens, topk}, "weights");
    check_shape(packed, {tokens, row_bytes}, "packed");
    const c10::cuda::CUDAGuard guard(device);
    pack_tokens<<<tokens, kNumThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const std::uint8_t*>(payload.data_ptr()), indices.data_ptr<std::int64_t>(),
        weights.data_ptr<float>(), packed.data_ptr<std::uint8_t>(), hidden, topk, row_bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand(
    const torch::Tensor& packed, const torch::Tensor& positions,
    const torch::Tensor& tile_counts, const torch::Tensor& tile_offsets,
    const torch::Tensor& expert_counts, const torch::Tensor& expert_offsets,
    const torch::Tensor& payload, const torch::Tensor& indices,
    const torch::Tensor& weights, const torch::Tensor& source_metadata,
    int world, int tokens, int hidden, int topk, int experts, int rank
) {
    check_dimensions(world, tokens, hidden, topk, experts, rank);
    const int num_tokens = world * tokens, num_local_experts = experts / world;
    const int num_tiles = (num_tokens + kNumThreads - 1) / kNumThreads;
    const int row_bytes = hidden * 2 + topk * 12;
    const std::int64_t capacity = std::int64_t(num_tokens) * topk;
    const auto device = packed.device();
    check_tensor(packed, at::kByte, device, "packed");
    check_tensor(positions, at::kInt, device, "positions");
    check_tensor(tile_counts, at::kInt, device, "tile_counts");
    check_tensor(tile_offsets, at::kInt, device, "tile_offsets");
    check_tensor(expert_counts, at::kInt, device, "expert_counts");
    check_tensor(expert_offsets, at::kInt, device, "expert_offsets");
    check_tensor(payload, at::kBFloat16, device, "payload");
    check_tensor(indices, at::kLong, device, "indices");
    check_tensor(weights, at::kFloat, device, "weights");
    check_tensor(source_metadata, at::kLong, device, "source_metadata");
    check_shape(packed, {num_tokens, row_bytes}, "packed");
    check_shape(positions, {num_tokens, topk}, "positions");
    check_shape(tile_counts, {num_local_experts, num_tiles}, "tile_counts");
    check_shape(tile_offsets, {num_local_experts, num_tiles}, "tile_offsets");
    check_shape(expert_counts, {num_local_experts}, "expert_counts");
    check_shape(expert_offsets, {num_local_experts + 1}, "expert_offsets");
    check_shape(payload, {capacity, hidden}, "payload");
    check_shape(indices, {capacity}, "indices");
    check_shape(weights, {capacity}, "weights");
    check_shape(source_metadata, {capacity, kNumSourceMetadataFields}, "source_metadata");
    const c10::cuda::CUDAGuard guard(device);
    const auto stream = at::cuda::getCurrentCUDAStream();
    count_expert_tiles<<<dim3(num_local_experts, num_tiles), kNumThreads, 0, stream>>>(
        packed.data_ptr<std::uint8_t>(), positions.data_ptr<int>(), tile_counts.data_ptr<int>(),
        num_tokens, hidden, topk, num_tiles, row_bytes, rank * num_local_experts, num_local_experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    prefix_expert_tiles<<<1, kNumWarpThreads, 0, stream>>>(
        tile_counts.data_ptr<int>(), tile_offsets.data_ptr<int>(), expert_counts.data_ptr<int>(),
        expert_offsets.data_ptr<int>(), num_tiles, num_local_experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_expert_tokens<<<num_tokens, kNumThreads, 0, stream>>>(
        packed.data_ptr<std::uint8_t>(), positions.data_ptr<int>(), tile_offsets.data_ptr<int>(),
        expert_offsets.data_ptr<int>(), reinterpret_cast<std::uint8_t*>(payload.data_ptr()),
        indices.data_ptr<std::int64_t>(), weights.data_ptr<float>(), source_metadata.data_ptr<std::int64_t>(),
        tokens, hidden, topk, num_tiles, row_bytes, rank * num_local_experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace nccl_shm_benchmark

PYBIND11_MODULE(TORCH_EXTENSION_NAME, extension) {
    extension.def("pack", &nccl_shm_benchmark::pack);
    extension.def("expand", &nccl_shm_benchmark::expand);
}
