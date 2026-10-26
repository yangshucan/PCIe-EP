#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <array>

#include "../kernels/host_staged/dispatch.cuh"
#include "../comm/numa_relay.hpp"
#include "../kernels/host_staged/advanced_layout.cuh"
#include "../kernels/host_staged/advanced_dispatch.cuh"
#include "../kernels/host_staged/fused_receive.cuh"

#include <atomic>
#include <memory>
#include <mutex>

namespace pcie_ep::host_staged {

// Independent immutable route-plan storage. Holding this opaque object keeps
// its prepared GPU plan out of the reusable plan pool; outputs are separate.
struct AdvancedLayout {
    std::uint64_t owner;
    int tokens = -1;
    torch::Tensor route_plan;
    torch::Tensor rank_expert_counts;
    torch::Tensor channel_remote_indices;
    torch::Tensor channel_num_remote_tokens;
    torch::Tensor channel_rank_counts;
    torch::Tensor channel_endpoint_counts;

    AdvancedLayout(std::uint64_t owner_id, int device, const AdvancedSharedMemoryLayout& shared): owner(owner_id) {
        const auto options = torch::TensorOptions().device(torch::kCUDA, device);
        const auto i32 = options.dtype(torch::kInt32), i64 = options.dtype(torch::kInt64);
        route_plan = torch::empty({int64_t(align_up(size_t(shared.base.num_max_tokens_per_rank) * sizeof(TokenMetadata), 8) / 8)}, i64);
        rank_expert_counts = torch::empty({shared.base.num_ranks + shared.base.num_experts}, i32);
        channel_remote_indices = torch::empty({shared.channels, shared.base.num_max_tokens_per_rank}, i32);
        channel_num_remote_tokens = torch::empty({shared.channels}, i32);
        channel_rank_counts = torch::empty({shared.channels, shared.base.num_ranks}, i32);
        channel_endpoint_counts = torch::empty({shared.channels, shared.num_endpoints}, i32);
    }
};

class AdvancedRuntime {
    using Tensor = torch::Tensor;
    struct OutputSet {
        // Unique receive rows and their per-top-k metadata.
        Tensor unique_payload;
        Tensor unique_topk_indices;
        Tensor unique_topk_weights;
        Tensor unique_source_metadata;
        Tensor unique_expert_ordinals;

        // Optional expert-major expansion, including the fused receive path.
        Tensor expanded_payload;
        Tensor expanded_expert_indices;
        Tensor expanded_weights;
        Tensor expanded_source_metadata;

        // Output-local GPU counts, prefixes, and transport diagnostics.
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
        Tensor expected_receive_counts;
        Tensor channel_statistics;
        Tensor channel_timestamps_ns;
        Tensor receive_expand_overlap;
    };

    AdvancedConfig config_;
    AdvancedSharedMemoryLayout shared_;
    int rank_;
    int device_ = -1;
    std::uint64_t owner_id_;
    inline static std::atomic<std::uint64_t> next_owner_id_{1};
    std::vector<MappedHostSegment> segments_;
    AdvancedDeviceState state_{};
    Tensor epoch_;
    Tensor fp8_payload_;
    Tensor fp8_scales_;
    std::vector<OutputSet> outputs_;
    std::vector<std::shared_ptr<AdvancedLayout>> plans_;
    std::shared_ptr<AdvancedLayout> graph_plan_;
    std::unique_ptr<NumaRelay> relay_;
    size_t shared_bytes_ = 0;
    bool closed_ = false;
    bool closing_ = false;

public:
    AdvancedRuntime(std::string directory, int rank, int p, int t, int h, int k, int e, py::dict options):
        config_(p, options), shared_(config_.layout(p, t, h, k, e)), rank_(rank),
        owner_id_(next_owner_id_.fetch_add(1, std::memory_order_relaxed)) {
        TORCH_CHECK(p >= 1 && p <= kNumMaxRanks && rank >= 0 && rank < p && t > 0 && h > 0 &&
                    k > 0 && k <= kNumMaxTopk && e >= p && e <= kNumMaxExperts && e % p == 0,
                    "unsupported advanced dimensions");
        C10_CUDA_CHECK(cudaGetDevice(&device_));
        try {
            segments_ = register_advanced_segments(directory, shared_);
            initialize_state();
            allocate_workspace();
            configure_shared_memory();
            if (config_.relay) {
                relay_ = std::make_unique<NumaRelay>(config_, shared_, rank_, segments_);
            }
        } catch (...) {
            if (relay_) {
                relay_->stop_and_join();
                relay_.reset();
            }
            release_advanced_segments(segments_, shared_.size_bytes, false);
            throw;
        }
    }

    py::dict dispatch(Tensor x, Tensor indices, py::object weight, int stage, int alignment,
                      bool expand, bool zero_padding, py::object cached_layout = py::none(), bool graph_pool = false) {
        Tensor weights;
        validate(x, indices, weight, stage, alignment, weights);
        auto& output = select_output(graph_pool);
        const bool reuse = !cached_layout.is_none();
        std::shared_ptr<AdvancedLayout> plan;
        if (reuse) {
            plan = cached_layout.cast<std::shared_ptr<AdvancedLayout>>();
            TORCH_CHECK(plan && plan->owner == owner_id_ && plan->tokens == x.size(0), "cached route layout belongs to another buffer or input shape");
        } else if (graph_pool) {
            plan = graph_plan_;
        } else {
            for (auto& candidate : plans_) {
                if (candidate.use_count() == 1) {
                    plan = candidate;
                    break;
                }
            }
            TORCH_CHECK(plan, "preallocated route plan pool exhausted; release old handles/cache plans");
            plan->tokens = x.size(0);
        }
        auto state = bind_state(output, *plan, x, indices, weights, expand);
        launch_dispatch(state, output, int(x.size(0)), alignment, expand, zero_padding, reuse);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const bool optimized_receive = state.fuse_receive_expand &&
            config_.receive_expand_mode != AdvancedConfig::ReceiveExpandMode::Scalar;
        auto result = make_result(output, expand, optimized_receive);
        result["_layout"] = graph_pool && !reuse ? py::none() : py::cast(plan);
        return result;
    }

    void set_test_delays(std::uint64_t metadata_ns, std::uint64_t tx_ns, std::uint64_t rx_ns, std::uint64_t scatter_ns) {
        config_.metadata_delay_ns = metadata_ns;
        config_.tx_delay_ns = tx_ns;
        config_.rx_delay_ns = rx_ns;
        config_.scatter_delay_ns = scatter_ns;
    }
    void set_relay_delay_ns(std::uint64_t delay_ns) {
        TORCH_CHECK(relay_, "NUMA relay is not enabled");
        relay_->set_delay(delay_ns);
    }

    py::dict relay_diagnostics() const {
        TORCH_CHECK(!closed_ && !closing_, "advanced runtime is closed or closing");
        py::dict result = relay_ ? relay_->diagnostics() : py::dict();
        result["enabled"] = config_.relay;
        result["rank"] = rank_;
        result["rank_numa_nodes"] = config_.rank_numa_nodes;
        result["physical_numa_nodes"] = config_.physical_numa_nodes;
        result["payload_stride_bytes"] = shared_.payload_stride_bytes;
        result["segment_bytes"] = shared_.size_bytes;
        result["wire_dtype"] = config_.payload_dtype ? "fp8_e4m3" : "bf16";
        result["cpu_handshake"] = "std::atomic_ref<uint64_t> lock-free acquire/release load/store";
        auto* own = static_cast<char*>(segments_[rank_].host_address);
        result["source_payload_placement"] = query_numa_pages(own + shared_.source_payload_offset,
            size_t(config_.channels) * shared_.base.num_payload_slots * shared_.payload_stride_bytes);
        py::list replicas;
        if (config_.relay) {
            for (int channel = 0; channel < config_.channels; ++channel) {
                for (int domain = 0; domain < shared_.num_numa_nodes; ++domain) {
                    const size_t slot = (size_t(channel) * shared_.num_numa_nodes + domain) * shared_.replica_slots;
                    auto record = query_numa_pages(own + shared_.replica_payload_offset + slot * shared_.payload_stride_bytes,
                        size_t(shared_.replica_slots) * shared_.payload_stride_bytes);
                    record["channel"] = channel;
                    record["domain"] = domain;
                    record["target_numa_node"] = config_.physical_numa_nodes[domain];
                    replicas.append(record);
                }
            }
        }
        result["replica_payload_placement"] = replicas;
        result["control_placement"] = "source NUMA; only replica payload pages are target-bound";
        return result;
    }

    py::list pool_debug() {
        py::list result;
        for (auto& output : outputs_) {
            py::list row;
            for (auto* tensor : output_tensors(output)) {
                const auto count = tensor->use_count();
                auto object = py::cast(*tensor);
                row.append(py::make_tuple(count, Py_REFCNT(object.ptr())));
            }
            result.append(row);
        }
        return result;
    }

    // Python must first stop new submissions, drain all rank completions and
    // reach its CPU collective close barrier. Workers join before unmapping.
    void close() {
        if (closed_) {
            return;
        }
        closing_ = true;
        c10::cuda::CUDAGuard guard(device_);
        if (relay_) {
            relay_->stop_and_join();
            relay_.reset();
        }
        release_advanced_segments(segments_, shared_.size_bytes, true);
        closed_ = true;
    }

    ~AdvancedRuntime() {
        if (closed_ || segments_.empty()) {
            return;
        }
        cudaSetDevice(device_);
        if (relay_) {
            relay_->stop_and_join();
            relay_.reset();
        }
        // No collective is safe in a destructor. Publish a failure after CPU
        // workers relinquish their signal ownership, then drain only this GPU.
        // Retain mappings if explicit collective close was skipped: another
        // process may still be using its alias to this backing storage.
        auto* own = static_cast<char*>(segments_[rank_].host_address);
        for (int channel = 0; channel < config_.channels; ++channel)
            for (int node = 0; node < shared_.num_numa_nodes; ++node)
                host_release(host_signal(own, shared_.relay_failed_offset, channel * shared_.num_numa_nodes + node), 199);
        cudaDeviceSynchronize();
    }

private:
    void initialize_state() {
        state_.base.layout = shared_.base;
        state_.base.local_rank = rank_;
        state_.base.timeout_ns = std::uint64_t(config_.timeout_ms) * 1000000;
        state_.shared = shared_;
        state_.channels = config_.channels;
        state_.stages = config_.stages;
        state_.num_numa_nodes = shared_.num_numa_nodes;
        state_.numa_mode = config_.relay;
        state_.payload_dtype = config_.payload_dtype;
        state_.source_fusion = config_.source_fusion;
        state_.warp_specialization = config_.warp_specialization;
        for (int rank = 0; rank < shared_.base.num_ranks; ++rank) {
            state_.base.mapped_source_buffers[rank] = segments_[rank].device_address;
            state_.rank_numa[rank] = config_.rank_numa_domains[rank];
        }
    }

    void allocate_workspace() {
        const auto options = torch::TensorOptions().device(torch::kCUDA, device_);
        const auto i32 = options.dtype(torch::kInt32), i64 = options.dtype(torch::kInt64);
        const auto bf16 = options.dtype(torch::kBFloat16), fp32 = options.dtype(torch::kFloat32);
        epoch_ = torch::zeros({1}, i64);
        state_.base.epoch = reinterpret_cast<std::uint64_t*>(epoch_.data_ptr<int64_t>());
        if (config_.payload_dtype) {
            fp8_payload_ = torch::empty({shared_.base.num_max_tokens_per_rank, shared_.base.num_hidden}, options.dtype(torch::kUInt8));
            fp8_scales_ = torch::empty({shared_.base.num_max_tokens_per_rank}, fp32);
            state_.fp8_payload = static_cast<std::uint8_t*>(fp8_payload_.data_ptr());
            state_.fp8_scales = fp8_scales_.data_ptr<float>();
        }
        const int p = shared_.base.num_ranks, t = shared_.base.num_max_tokens_per_rank;
        const int h = shared_.base.num_hidden, k = shared_.base.num_topk, local_experts = shared_.base.num_experts / p;
        const int64_t n = int64_t(p) * t, capacity = n * std::min(k, local_experts) + int64_t(local_experts) * 255;
        const int pool_count = config_.output_pool_size + int(config_.enable_graph);
        outputs_.resize(pool_count);
        for (auto& output : outputs_) {
            output.unique_payload = torch::empty({n, h}, bf16);
            output.unique_topk_indices = torch::empty({n, k}, i64);
            output.unique_topk_weights = torch::empty({n, k}, fp32);
            output.unique_source_metadata = torch::empty({n, kNumSourceMetadataFields}, i64);
            output.unique_expert_ordinals = torch::empty({n, k}, i32);
            output.expanded_payload = torch::empty({capacity, h}, bf16);
            output.expanded_expert_indices = torch::empty({capacity}, i64);
            output.expanded_weights = torch::empty({capacity}, fp32);
            output.expanded_source_metadata = torch::empty({capacity, kNumSourceMetadataFields}, i64);
            output.unique_count = torch::empty({1}, i64);
            output.expanded_count = torch::empty({1}, i64);
            output.expert_prefix = torch::empty({local_experts}, i32);
            output.expert_counts = torch::empty({local_experts}, i32);
            output.expert_offsets = torch::empty({local_experts + 1}, i32);
            output.source_offsets = torch::empty({p + 1}, i32);
            output.source_expert_prefix = torch::empty({p, local_experts}, i32);
            output.errors = torch::empty({kNumErrorFields}, i64);
            output.statistics = torch::empty({kNumStatisticsFields}, i64);
            output.transport_timestamps_ns = torch::empty({kNumTransportTimestampFields}, i64);
            output.expected_receive_counts = torch::empty({config_.channels, p}, i32);
            output.channel_statistics = torch::empty({config_.channels, kNumStatisticsFields}, i64);
            output.channel_timestamps_ns = torch::empty({config_.channels, kNumTransportTimestampFields}, i64);
            output.receive_expand_overlap = torch::empty({config_.channels, 2}, i64);
        }
        for (int i = 0; i < config_.output_pool_size; ++i) {
            plans_.push_back(std::make_shared<AdvancedLayout>(owner_id_, device_, shared_));
        }
        if (config_.enable_graph) {
            graph_plan_ = std::make_shared<AdvancedLayout>(owner_id_, device_, shared_);
        }
    }

    void configure_shared_memory() {
        shared_bytes_ = advanced_shared_memory_bytes(shared_.base.num_hidden, config_.stages, config_.fuse_receive_expand);
        cudaDeviceProp properties{};
        C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_));
        TORCH_CHECK(shared_bytes_ <= size_t(properties.sharedMemPerBlockOptin), "advanced shared-memory requirement exceeds device limit");
        const bool scalar_mode = config_.receive_expand_mode == AdvancedConfig::ReceiveExpandMode::Scalar;
        const size_t reference_bytes = advanced_shared_memory_bytes(
            shared_.base.num_hidden, config_.stages, scalar_mode && config_.fuse_receive_expand);
        // Attributes belong to the device/context's kernel, not this buffer.
        // Serialize initialization and only raise caps needed by live buffers.
        static std::mutex shared_attribute_mutex;
        const std::lock_guard<std::mutex> lock(shared_attribute_mutex);
        const auto raise_shared_limit = [](auto kernel, size_t required_bytes) {
            cudaFuncAttributes attributes{};
            C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
            if (required_bytes > size_t(attributes.maxDynamicSharedSizeBytes)) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(required_bytes)));
            }
        };
        raise_shared_limit(advanced_transport, reference_bytes);
        if (config_.receive_expand_mode == AdvancedConfig::ReceiveExpandMode::Vectorized) {
            raise_shared_limit(advanced_transport_vectorized, shared_bytes_);
        } else if (config_.receive_expand_mode == AdvancedConfig::ReceiveExpandMode::Pipelined) {
            raise_shared_limit(advanced_transport_pipelined, shared_bytes_);
        }
    }

    void validate(const Tensor& x, const Tensor& idx, const py::object& weight, int stage, int alignment, Tensor& weights) const {
        TORCH_CHECK(!closed_ && !closing_, "advanced runtime is closed or closing");
        TORCH_CHECK(stage == config_.stage && stage >= 6 && stage <= 8, "advanced stage differs from prepared configuration");
        TORCH_CHECK(x.is_cuda() && x.get_device() == device_ && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 &&
            x.size(1) == shared_.base.num_hidden && x.size(0) <= shared_.base.num_max_tokens_per_rank && x.is_contiguous(), "bad advanced input");
        TORCH_CHECK(idx.is_cuda() && idx.get_device() == device_ && idx.scalar_type() == torch::kInt64 &&
            idx.sizes() == torch::IntArrayRef({x.size(0), shared_.base.num_topk}) && idx.is_contiguous(), "bad advanced routing");
        TORCH_CHECK(alignment >= 1 && alignment <= 256, "alignment must be in [1,256]");
        if (!weight.is_none()) {
            weights = weight.cast<Tensor>();
            TORCH_CHECK(weights.is_cuda() && weights.get_device() == device_ && weights.scalar_type() == torch::kFloat32 &&
                weights.sizes() == idx.sizes() && weights.is_contiguous(), "bad advanced weights");
        }
    }

    static std::array<Tensor*, 20> output_tensors(OutputSet& output) {
        return {&output.unique_payload, &output.unique_topk_indices,
                &output.unique_topk_weights, &output.unique_source_metadata,
                &output.expanded_payload, &output.expanded_expert_indices,
                &output.expanded_weights, &output.expanded_source_metadata,
                &output.unique_count, &output.expanded_count,
                &output.expert_prefix, &output.expert_counts,
                &output.expert_offsets, &output.source_offsets,
                &output.errors, &output.statistics, &output.transport_timestamps_ns,
                &output.channel_statistics, &output.channel_timestamps_ns,
                &output.receive_expand_overlap};
    }
    OutputSet& select_output(bool graph_pool) {
        if (graph_pool) {
            TORCH_CHECK(config_.enable_graph, "Graph output pool was not prepared");
            return outputs_[config_.output_pool_size];
        }
        for (int i = 0; i < config_.output_pool_size; ++i) {
            bool available = true;
            for (auto* tensor : output_tensors(outputs_[i])) {
                auto* slot = tensor->unsafeGetTensorImpl()->pyobj_slot();
                available &= slot->load_pyobj() ? slot->has_unique_reference() : tensor->use_count() == 1;
            }
            if (available) {
                return outputs_[i];
            }
        }
        TORCH_CHECK(false, "preallocated output pool exhausted; release prior results or use a separate buffer");
    }

    AdvancedDeviceState bind_state(OutputSet& output, AdvancedLayout& plan, const Tensor& x,
                                    const Tensor& indices, const Tensor& weights, bool expand) const {
        auto state = state_;
        auto& base = state.base;
        base.route_plan = static_cast<TokenMetadata*>(plan.route_plan.data_ptr());
        base.rank_expert_counts = plan.rank_expert_counts.data_ptr<int>();
        base.source_offsets = output.source_offsets.data_ptr<int>();
        base.source_expert_prefix = output.source_expert_prefix.data_ptr<int>();
        base.expert_counts = output.expert_counts.data_ptr<int>();
        base.expert_offsets = output.expert_offsets.data_ptr<int>();
        base.unique_count = output.unique_count.data_ptr<int64_t>();
        base.expanded_count = output.expanded_count.data_ptr<int64_t>();
        base.expert_prefix = output.expert_prefix.data_ptr<int>();
        base.errors = output.errors.data_ptr<int64_t>();
        base.statistics = output.statistics.data_ptr<int64_t>();
        base.transport_timestamps_ns = output.transport_timestamps_ns.data_ptr<int64_t>();
        base.unique_payload = static_cast<std::uint16_t*>(output.unique_payload.data_ptr());
        base.unique_topk_indices = output.unique_topk_indices.data_ptr<int64_t>();
        base.unique_topk_weights = output.unique_topk_weights.data_ptr<float>();
        base.unique_source_metadata = output.unique_source_metadata.data_ptr<int64_t>();
        base.unique_expert_ordinals = output.unique_expert_ordinals.data_ptr<int>();
        state.input_payload = static_cast<std::uint16_t*>(x.data_ptr());
        state.topk_indices = indices.data_ptr<int64_t>();
        state.topk_weights = weights.defined() ? weights.data_ptr<float>() : nullptr;
        state.channel_remote_indices = plan.channel_remote_indices.data_ptr<int>();
        state.channel_num_remote_tokens = plan.channel_num_remote_tokens.data_ptr<int>();
        state.channel_rank_counts = plan.channel_rank_counts.data_ptr<int>();
        state.channel_endpoint_counts = plan.channel_endpoint_counts.data_ptr<int>();
        state.expected_receive_counts = output.expected_receive_counts.data_ptr<int>();
        state.expanded_payload = static_cast<std::uint16_t*>(output.expanded_payload.data_ptr());
        state.expanded_expert_indices = output.expanded_expert_indices.data_ptr<int64_t>();
        state.expanded_weights = output.expanded_weights.data_ptr<float>();
        state.expanded_source_metadata = output.expanded_source_metadata.data_ptr<int64_t>();
        state.expanded_capacity = output.expanded_payload.size(0);
        state.fuse_receive_expand = expand && config_.fuse_receive_expand;
        state.channel_statistics = output.channel_statistics.data_ptr<int64_t>();
        state.channel_timestamps_ns = output.channel_timestamps_ns.data_ptr<int64_t>();
        state.metadata_delay_ns = config_.metadata_delay_ns;
        state.tx_delay_ns = config_.tx_delay_ns;
        state.rx_delay_ns = config_.rx_delay_ns;
        state.scatter_delay_ns = config_.scatter_delay_ns;
        return state;
    }

    void launch_dispatch(const AdvancedDeviceState& state, OutputSet& output, int tokens,
                         int alignment, bool expand, bool zero_padding, bool reuse) const {
        const auto stream = at::cuda::getCurrentCUDAStream(device_);
        advanced_begin_epoch<<<1, 1, 0, stream>>>(state);
        if (!reuse) {
            if (tokens) {
                const int route_blocks = (tokens + kNumThreads - 1) / kNumThreads;
                advanced_route_flags<<<route_blocks, kNumThreads, 0, stream>>>(state, tokens);
            }
            advanced_stable_ordinals<<<shared_.base.num_ranks + shared_.base.num_experts, kNumThreads, 0, stream>>>(state, tokens);
        }
        advanced_prepare_channels_parallel<<<config_.channels, kNumThreads, 0, stream>>>(state, tokens, reuse);
        publish_counts<<<1, kNumThreads, 0, stream>>>(state.base);
        advanced_compute_receive_prefixes<<<1, 1, 0, stream>>>(state, alignment);
        if (config_.payload_dtype && tokens) {
            quantize_transport_payload<<<tokens, kNumThreads, 0, stream>>>(
                state.input_payload, static_cast<std::uint8_t*>(fp8_payload_.data_ptr()),
                fp8_scales_.data_ptr<float>(), tokens, shared_.base.num_hidden);
        }
        if (tokens) {
            advanced_copy_local_tokens<<<tokens, kNumThreads, 0, stream>>>(state, tokens);
        }
        const size_t dynamic_bytes = advanced_shared_memory_bytes(shared_.base.num_hidden, config_.stages, state.fuse_receive_expand);
        if (state.fuse_receive_expand &&
            config_.receive_expand_mode == AdvancedConfig::ReceiveExpandMode::Vectorized) {
            advanced_transport_vectorized<<<config_.channels, kNumThreads, dynamic_bytes, stream>>>(
                state, output.receive_expand_overlap.data_ptr<int64_t>());
        } else if (state.fuse_receive_expand &&
                   config_.receive_expand_mode == AdvancedConfig::ReceiveExpandMode::Pipelined) {
            constexpr int kNumPipelinedReceiveThreads = 352;
            advanced_transport_pipelined<<<config_.channels, kNumPipelinedReceiveThreads, dynamic_bytes, stream>>>(
                state, output.receive_expand_overlap.data_ptr<int64_t>());
        } else {
            // Unique mode has no receive tile, irrespective of the configured
            // expanded mode. Scalar retains the original fused implementation.
            advanced_transport<<<config_.channels, kNumThreads, dynamic_bytes, stream>>>(state);
        }
        if (expand && !state.fuse_receive_expand) {
            scatter_expert_inputs<<<std::max(shared_.base.num_ranks * shared_.base.num_max_tokens_per_rank,
                shared_.base.num_experts / shared_.base.num_ranks), kNumThreads, 0, stream>>>(
                state.base, static_cast<std::uint16_t*>(output.expanded_payload.data_ptr()), output.expanded_expert_indices.data_ptr<int64_t>(),
                output.expanded_weights.data_ptr<float>(), output.expanded_source_metadata.data_ptr<int64_t>(), int(output.expanded_payload.size(0)), zero_padding);
        } else if (expand && zero_padding) {
            advanced_zero_padding<<<shared_.base.num_experts / shared_.base.num_ranks, kNumThreads, 0, stream>>>(state);
        }
        advanced_reduce_statistics<<<1, 1, 0, stream>>>(state);
    }

    static py::dict make_result(OutputSet& output, bool expand, bool optimized_receive) {
        py::dict result;
        result["recv_x"] = expand ? output.expanded_payload : output.unique_payload;
        result["recv_topk_idx"] = expand ? output.expanded_expert_indices : output.unique_topk_indices;
        result["recv_topk_weights"] = expand ? output.expanded_weights : output.unique_topk_weights;
        result["source_meta"] = expand ? output.expanded_source_metadata : output.unique_source_metadata;
        result["receive_expand_overlap"] = optimized_receive ?
            py::cast(output.receive_expand_overlap) : py::none();
        result["unique_count"] = output.unique_count;
        result["expanded_count"] = output.expanded_count;
        result["expert_prefix"] = output.expert_prefix;
        result["expert_counts"] = output.expert_counts;
        result["expert_offsets"] = output.expert_offsets;
        result["source_offsets"] = output.source_offsets;
        result["error"] = output.errors;
        result["stats"] = output.statistics;
        result["transport_timestamps_ns"] = output.transport_timestamps_ns;
        result["channel_stats"] = output.channel_statistics;
        result["channel_timestamps_ns"] = output.channel_timestamps_ns;
        return result;
    }
};

}  // namespace pcie_ep::host_staged
