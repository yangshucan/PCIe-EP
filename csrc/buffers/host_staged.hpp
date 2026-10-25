#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "../comm/host_shared_memory.hpp"
#include "../kernels/host_staged/layout.cuh"
#include "../kernels/host_staged/dispatch.cuh"

namespace pcie_ep::host_staged {

class Runtime {
    using Tensor = torch::Tensor;
    static constexpr int kNumMaxExpertAlignment = 256;

    struct OutputSet {
        // Unique receive rows and their per-top-k metadata.
        Tensor unique_payload;
        Tensor unique_topk_indices;
        Tensor unique_topk_weights;
        Tensor unique_source_metadata;
        Tensor unique_expert_ordinals;

        // Optional expert-major expansion, populated after transport.
        Tensor expanded_payload;
        Tensor expanded_expert_indices;
        Tensor expanded_weights;
        Tensor expanded_source_metadata;

        // GPU counts, layout, errors, and transport diagnostics.
        Tensor unique_count;
        Tensor expanded_count;
        Tensor expert_prefix;
        Tensor expert_counts;
        Tensor expert_offsets;
        Tensor source_offsets;
        Tensor source_expert_prefix;
        Tensor errors;
        Tensor statistics;
        Tensor transport_timestamps_ns;
    };

    // Every shared segment is CPU memory; all device allocations below belong
    // exclusively to this process's CUDA device.
    SharedMemoryLayout layout_;
    int local_rank_;
    int device_;
    int prepared_stage_;
    std::vector<MappedHostSegment> source_segments_;
    DeviceState device_state_{};

    Tensor route_plan_;
    Tensor rank_expert_counts_;
    Tensor remote_token_indices_;
    Tensor num_remote_tokens_;
    Tensor epoch_;

    // Retained Python tensors reserve their pool entry. Dispatch never
    // allocates a replacement output set.
    std::array<OutputSet, 2> output_pool_;
    bool closed_ = false;

public:
    Runtime(std::string directory, int local_rank, int num_ranks,
            int num_max_tokens_per_rank, int hidden, int num_topk,
            int num_experts, int num_payload_slots, int num_descriptor_slots,
            int timeout_ms, int stage):
        layout_(make_shared_memory_layout(
            num_ranks, num_max_tokens_per_rank, hidden, num_topk, num_experts,
            num_payload_slots, num_descriptor_slots, stage)),
        local_rank_(local_rank),
        prepared_stage_(stage) {
        validate_dimensions();
        C10_CUDA_CHECK(cudaGetDevice(&device_));
        device_state_.layout = layout_;
        device_state_.local_rank = local_rank_;
        device_state_.timeout_ns = uint64_t(timeout_ms) * 1000000;
        map_source_segments(directory);
        allocate_route_workspace();
        allocate_output_pool();
    }

    py::dict dispatch(Tensor payload, Tensor topk_indices, py::object weights,
                      int stage, int alignment, bool expand, bool zero_padding) {
        const auto topk_weights = validate_dispatch_inputs(
            payload, topk_indices, weights, stage, alignment);
        auto& output = select_output_set();
        const auto state = bind_output_state(output);
        launch_dispatch(state, output, payload, topk_indices, topk_weights,
                        stage, alignment, expand, zero_padding);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return assemble_dispatch_result(output, expand);
    }

    py::list pool_debug() {
        py::list result;
        for (auto& output : output_pool_) {
            py::list row;
            // Keep the original diagnostic order and field set unchanged.
            for (auto* tensor : {&output.unique_payload, &output.expanded_payload,
                                 &output.unique_topk_indices, &output.unique_topk_weights,
                                 &output.unique_source_metadata, &output.expanded_expert_indices,
                                 &output.expanded_weights, &output.expanded_source_metadata,
                                 &output.unique_count, &output.expert_counts,
                                 &output.expert_offsets, &output.source_offsets,
                                 &output.errors, &output.statistics,
                                 &output.transport_timestamps_ns}) {
                const auto count = tensor->use_count();
                auto object = py::cast(*tensor);
                row.append(py::make_tuple(count, Py_REFCNT(object.ptr())));
            }
            result.append(row);
        }
        return result;
    }

    void close() {
        if (closed_) {
            return;
        }
        c10::cuda::CUDAGuard device_guard(device_);
        unregister_source_segments(source_segments_, layout_.size_bytes);
        closed_ = true;
    }

    ~Runtime() {
        if (!closed_) {
            cudaSetDevice(device_);
            cudaDeviceSynchronize();
            release_source_segments_on_destruction(source_segments_, layout_.size_bytes);
        }
    }

private:
    void validate_dimensions() const {
        TORCH_CHECK(layout_.num_ranks >= 1 && layout_.num_ranks <= kNumMaxRanks &&
                    layout_.num_topk >= 1 && layout_.num_topk <= kNumMaxTopk &&
                    layout_.num_experts >= layout_.num_ranks &&
                    layout_.num_experts <= kNumMaxExperts &&
                    layout_.num_experts % layout_.num_ranks == 0 &&
                    layout_.num_hidden > 0 && layout_.num_max_tokens_per_rank > 0 &&
                    layout_.num_payload_slots > 0 && layout_.num_descriptor_slots > 0,
                    "unsupported dimensions");
    }

    void map_source_segments(const std::string& directory) {
        source_segments_ = register_source_segments(directory, layout_);
        for (int source_rank = 0; source_rank < layout_.num_ranks; ++source_rank) {
            device_state_.mapped_source_buffers[source_rank] =
                source_segments_[source_rank].device_address;
        }
    }

    void allocate_route_workspace() {
        const auto options = torch::TensorOptions().device(torch::kCUDA, device_);
        const auto int32_options = options.dtype(torch::kInt32);
        const auto int64_options = options.dtype(torch::kInt64);
        const auto route_bytes = size_t(layout_.num_max_tokens_per_rank) * sizeof(TokenMetadata);
        route_plan_ = torch::empty({int64_t(align_up(route_bytes, 8) / 8)}, int64_options);
        rank_expert_counts_ = torch::empty(
            {layout_.num_ranks + layout_.num_experts}, int32_options);
        remote_token_indices_ = torch::empty({layout_.num_max_tokens_per_rank}, int32_options);
        num_remote_tokens_ = torch::empty({1}, int32_options);
        epoch_ = torch::zeros({1}, int64_options);

        device_state_.route_plan = static_cast<TokenMetadata*>(route_plan_.data_ptr());
        device_state_.rank_expert_counts = rank_expert_counts_.data_ptr<int>();
        device_state_.remote_token_indices = remote_token_indices_.data_ptr<int>();
        device_state_.num_remote_tokens = num_remote_tokens_.data_ptr<int>();
        device_state_.epoch = reinterpret_cast<uint64_t*>(epoch_.data_ptr<int64_t>());
    }

    void allocate_output_pool() {
        const auto options = torch::TensorOptions().device(torch::kCUDA, device_);
        const auto int32_options = options.dtype(torch::kInt32);
        const auto int64_options = options.dtype(torch::kInt64);
        const auto bf16_options = options.dtype(torch::kBFloat16);
        const auto float32_options = options.dtype(torch::kFloat32);
        const int64_t num_unique_rows =
            int64_t(layout_.num_ranks) * layout_.num_max_tokens_per_rank;
        const int64_t num_local_experts = layout_.num_experts / layout_.num_ranks;
        // Preserve the worst-case expansion and alignment=256 padding.
        const int64_t num_expanded_rows =
            num_unique_rows * std::min(layout_.num_topk, int(num_local_experts)) +
            num_local_experts * (kNumMaxExpertAlignment - 1);

        for (auto& output : output_pool_) {
            output.unique_payload = torch::empty(
                {num_unique_rows, layout_.num_hidden}, bf16_options);
            output.unique_topk_indices = torch::empty(
                {num_unique_rows, layout_.num_topk}, int64_options);
            output.unique_topk_weights = torch::empty(
                {num_unique_rows, layout_.num_topk}, float32_options);
            output.unique_source_metadata = torch::empty(
                {num_unique_rows, kNumSourceMetadataFields}, int64_options);
            output.unique_expert_ordinals = torch::empty(
                {num_unique_rows, layout_.num_topk}, int32_options);
            output.expanded_payload = torch::empty(
                {num_expanded_rows, layout_.num_hidden}, bf16_options);
            output.expanded_expert_indices = torch::empty({num_expanded_rows}, int64_options);
            output.expanded_weights = torch::empty({num_expanded_rows}, float32_options);
            output.expanded_source_metadata = torch::empty(
                {num_expanded_rows, kNumSourceMetadataFields}, int64_options);
            output.unique_count = torch::empty({1}, int64_options);
            output.expanded_count = torch::empty({1}, int64_options);
            output.expert_prefix = torch::empty({num_local_experts}, int32_options);
            output.expert_counts = torch::empty({num_local_experts}, int32_options);
            output.expert_offsets = torch::empty({num_local_experts + 1}, int32_options);
            output.source_offsets = torch::empty({layout_.num_ranks + 1}, int32_options);
            output.source_expert_prefix = torch::empty(
                {layout_.num_ranks, num_local_experts}, int32_options);
            output.errors = torch::empty({kNumErrorFields}, int64_options);
            output.statistics = torch::empty({kNumStatisticsFields}, int64_options);
            output.transport_timestamps_ns = torch::empty({kNumTransportTimestampFields}, int64_options);
        }
    }

    Tensor validate_dispatch_inputs(const Tensor& payload, const Tensor& topk_indices,
                                    const py::object& weights, int stage,
                                    int alignment) const {
        TORCH_CHECK(!closed_, "closed runtime");
        TORCH_CHECK(stage == prepared_stage_, "stage differs from prepared layout");
        TORCH_CHECK(payload.is_cuda() && payload.get_device() == device_ &&
                    payload.scalar_type() == torch::kBFloat16 && payload.dim() == 2 &&
                    payload.size(1) == layout_.num_hidden &&
                    payload.size(0) <= layout_.num_max_tokens_per_rank && payload.is_contiguous(),
                    "bad x");
        TORCH_CHECK(topk_indices.is_cuda() && topk_indices.get_device() == device_ &&
                    topk_indices.scalar_type() == torch::kInt64 &&
                    topk_indices.sizes() == torch::IntArrayRef({payload.size(0), layout_.num_topk}) &&
                    topk_indices.is_contiguous(), "bad topk idx");
        TORCH_CHECK(alignment >= 1 && alignment <= kNumMaxExpertAlignment,
                    "alignment must be in [1,256]");
        TORCH_CHECK(stage >= 1 && stage <= 4, "bad stage");

        Tensor topk_weights;
        if (!weights.is_none()) {
            topk_weights = weights.cast<Tensor>();
            TORCH_CHECK(topk_weights.is_cuda() && topk_weights.get_device() == device_ &&
                        topk_weights.scalar_type() == torch::kFloat32 &&
                        topk_weights.sizes() == topk_indices.sizes() &&
                        topk_weights.is_contiguous(), "bad weights");
        }
        return topk_weights;
    }

    static bool tensor_has_no_external_reference(const Tensor& tensor) {
        auto* python_slot = tensor.unsafeGetTensorImpl()->pyobj_slot();
        // A cached PyObject changes TensorImpl use_count even after a caller
        // releases its result. Preserve the tested PyObject uniqueness rule.
        return python_slot->load_pyobj() ? python_slot->has_unique_reference()
                                         : tensor.use_count() == 1;
    }

    OutputSet& select_output_set() {
        for (auto& output : output_pool_) {
            bool available = true;
            for (auto* tensor : {&output.unique_payload, &output.expanded_payload,
                                 &output.unique_topk_indices, &output.unique_topk_weights,
                                 &output.unique_source_metadata, &output.expanded_expert_indices,
                                 &output.expanded_weights, &output.expanded_source_metadata,
                                 &output.unique_count, &output.expanded_count,
                                 &output.expert_prefix, &output.expert_counts,
                                 &output.expert_offsets, &output.source_offsets,
                                 &output.errors, &output.statistics,
                                 &output.transport_timestamps_ns}) {
                available &= tensor_has_no_external_reference(*tensor);
            }
            if (available) {
                return output;
            }
        }
        TORCH_CHECK(false, "preallocated output pool exhausted; release prior results or use a separate buffer");
    }

    DeviceState bind_output_state(OutputSet& output) const {
        auto state = device_state_;
        state.source_offsets = output.source_offsets.data_ptr<int>();
        state.source_expert_prefix = output.source_expert_prefix.data_ptr<int>();
        state.expert_counts = output.expert_counts.data_ptr<int>();
        state.expert_offsets = output.expert_offsets.data_ptr<int>();
        state.unique_count = output.unique_count.data_ptr<int64_t>();
        state.expanded_count = output.expanded_count.data_ptr<int64_t>();
        state.expert_prefix = output.expert_prefix.data_ptr<int>();
        state.errors = output.errors.data_ptr<int64_t>();
        state.statistics = output.statistics.data_ptr<int64_t>();
        state.transport_timestamps_ns = output.transport_timestamps_ns.data_ptr<int64_t>();
        state.unique_payload = static_cast<uint16_t*>(output.unique_payload.data_ptr());
        state.unique_topk_indices = output.unique_topk_indices.data_ptr<int64_t>();
        state.unique_topk_weights = output.unique_topk_weights.data_ptr<float>();
        state.unique_source_metadata = output.unique_source_metadata.data_ptr<int64_t>();
        state.unique_expert_ordinals = output.unique_expert_ordinals.data_ptr<int>();
        return state;
    }

    void launch_dispatch(DeviceState state, OutputSet& output, const Tensor& payload,
                         const Tensor& topk_indices, const Tensor& topk_weights,
                         int stage, int alignment, bool expand, bool zero_padding) const {
        const auto stream = at::cuda::getCurrentCUDAStream(device_);
        const int num_tokens = payload.size(0);
        const auto* input_payload = static_cast<uint16_t*>(payload.data_ptr());
        const auto* input_indices = topk_indices.data_ptr<int64_t>();
        const auto* input_weights = topk_weights.defined() ? topk_weights.data_ptr<float>() : nullptr;

        // Preserve every original kernel boundary, launch shape, and ordering.
        compute_route_layout<<<1, 1, 0, stream>>>(state, input_indices, input_weights, num_tokens);
        publish_counts<<<1, kNumThreads, 0, stream>>>(state);
        compute_receive_prefixes<<<1, 1, 0, stream>>>(state, alignment);
        if (num_tokens) {
            copy_local_tokens<<<num_tokens, kNumThreads, 0, stream>>>(state, input_payload, num_tokens);
        }
        if (stage <= 2) {
            record_transport_start<<<1, 1, 0, stream>>>(state);
            write_batch_metadata<<<(layout_.num_max_tokens_per_rank + kNumThreads - 1) / kNumThreads,
                                    kNumThreads, 0, stream>>>(state, num_tokens);
            if (num_tokens) {
                write_batch_payload<<<num_tokens, kNumThreads, 0, stream>>>(state, input_payload, num_tokens);
            }
            publish_batch<<<1, 1, 0, stream>>>(state, num_tokens);
            transport_batch<<<1, kNumThreads, 0, stream>>>(state);
        } else {
            transport_ring<<<1, kNumThreads, 0, stream>>>(state, input_payload);
        }
        if (expand) {
            const int num_blocks = std::max(
                layout_.num_ranks * layout_.num_max_tokens_per_rank,
                layout_.num_experts / layout_.num_ranks);
            scatter_expert_inputs<<<num_blocks, kNumThreads, 0, stream>>>(
                state, static_cast<uint16_t*>(output.expanded_payload.data_ptr()),
                output.expanded_expert_indices.data_ptr<int64_t>(),
                output.expanded_weights.data_ptr<float>(),
                output.expanded_source_metadata.data_ptr<int64_t>(),
                int(output.expanded_payload.size(0)), zero_padding);
        }
    }

    static py::dict assemble_dispatch_result(OutputSet& output, bool expand) {
        py::dict result;
        result["recv_x"] = expand ? output.expanded_payload : output.unique_payload;
        result["recv_topk_idx"] = expand ? output.expanded_expert_indices : output.unique_topk_indices;
        result["recv_topk_weights"] = expand ? output.expanded_weights : output.unique_topk_weights;
        result["source_meta"] = expand ? output.expanded_source_metadata : output.unique_source_metadata;
        result["unique_count"] = output.unique_count;
        result["expanded_count"] = output.expanded_count;
        result["expert_prefix"] = output.expert_prefix;
        result["expert_counts"] = output.expert_counts;
        result["expert_offsets"] = output.expert_offsets;
        result["source_offsets"] = output.source_offsets;
        result["error"] = output.errors;
        result["stats"] = output.statistics;
        result["transport_timestamps_ns"] = output.transport_timestamps_ns;
        return result;
    }
};

}  // namespace pcie_ep::host_staged
