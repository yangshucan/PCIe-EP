#pragma once

#include "host_shared_memory.hpp"
#include "../kernels/host_staged/advanced_common.cuh"

#include <algorithm>
#include <pybind11/stl.h>
#include <atomic>
#include <map>
#include <new>
#include <set>
#include <stdexcept>

namespace pcie_ep::host_staged {

struct AdvancedConfig {
    // Host-only launch selection. AdvancedDeviceState and shared layouts keep
    // their existing ABI; unique receives always use the reference transport.
    enum class ReceiveExpandMode { Scalar, Vectorized, Pipelined };

    int stage = 6;
    int channels = 1;
    int stages = 2;
    int payload_slots = 256;
    int descriptor_slots = 256;
    int replica_slots = 256;
    int timeout_ms = 30000;
    int output_pool_size = 3;
    int relay_workers_per_numa = 1;
    int payload_dtype = 0;
    bool fuse_receive_expand = true;
    bool source_fusion = false;
    bool warp_specialization = true;
    bool enable_graph = false;
    bool relay = false;
    ReceiveExpandMode receive_expand_mode = ReceiveExpandMode::Scalar;
    std::uint64_t metadata_delay_ns = 0;
    std::uint64_t tx_delay_ns = 0;
    std::uint64_t rx_delay_ns = 0;
    std::uint64_t scatter_delay_ns = 0;
    std::uint64_t relay_delay_ns = 0;
    std::vector<int> rank_numa_nodes;
    std::vector<int> physical_numa_nodes;
    std::vector<int> rank_numa_domains;
    std::vector<int> relay_cpu_affinity;

    template <typename T>
    static T option(const py::dict& options, const char* key, T fallback) {
        return options.contains(key) ? py::cast<T>(options[key]) : fallback;
    }

    AdvancedConfig(int num_ranks, const py::dict& options) {
        stage = option(options, "stage", stage);
        TORCH_CHECK(stage >= 6 && stage <= 8, "retired development stage; use a current S6-S8 configuration");
        channels = option(options, "channels", channels);
        stages = option(options, "pipeline_stages", option(options, "stages", stages));
        payload_slots = option(options, "payload_slots", payload_slots);
        descriptor_slots = option(options, "descriptor_slots", descriptor_slots);
        replica_slots = option(options, "replica_slots", payload_slots);
        timeout_ms = option(options, "timeout_ms", timeout_ms);
        output_pool_size = option(options, "output_pool_size", output_pool_size);
        relay_workers_per_numa = option(options, "relay_workers_per_numa", relay_workers_per_numa);
        fuse_receive_expand = option(options, "fuse_receive_expand", fuse_receive_expand);
        source_fusion = option(options, "source_fusion", source_fusion);
        warp_specialization = option(options, "warp_specialization", warp_specialization);
        enable_graph = option(options, "enable_graph", enable_graph);
        const auto receive_mode = option(options, "receive_expand_mode", std::string("scalar"));
        if (receive_mode == "scalar") {
            receive_expand_mode = ReceiveExpandMode::Scalar;
        } else if (receive_mode == "vectorized") {
            receive_expand_mode = ReceiveExpandMode::Vectorized;
        } else if (receive_mode == "pipelined") {
            receive_expand_mode = ReceiveExpandMode::Pipelined;
        } else {
            TORCH_CHECK(false, "receive_expand_mode must be scalar, vectorized or pipelined");
        }
        TORCH_CHECK(receive_expand_mode == ReceiveExpandMode::Scalar ||
            (fuse_receive_expand && warp_specialization),
            "non-scalar receive expansion requires fuse_receive_expand and warp_specialization");
        const auto wire_dtype = option(options, "payload_dtype", option(options, "wire_dtype", std::string("bf16")));
        TORCH_CHECK(wire_dtype == "bf16" || wire_dtype == "fp8" || wire_dtype == "fp8_e4m3", "invalid wire dtype");
        payload_dtype = wire_dtype == "bf16" ? 0 : 1;
        const auto mode = option(options, "numa_mode", std::string("direct"));
        TORCH_CHECK(mode == "direct" || mode == "relay", "invalid NUMA mode");
        relay = mode == "relay";
        metadata_delay_ns = option<std::uint64_t>(options, "metadata_delay_ns", 0);
        tx_delay_ns = option<std::uint64_t>(options, "tx_delay_ns", 0);
        rx_delay_ns = option<std::uint64_t>(options, "rx_delay_ns", 0);
        scatter_delay_ns = option<std::uint64_t>(options, "scatter_delay_ns", 0);
        relay_delay_ns = option<std::uint64_t>(options, "relay_delay_ns", 0);
        rank_numa_nodes = option(options, "rank_numa_nodes", std::vector<int>{});
        relay_cpu_affinity = option(options, "relay_cpu_affinity", std::vector<int>{});
        if (rank_numa_nodes.empty()) rank_numa_nodes.assign(num_ranks, -1);
        TORCH_CHECK(int(rank_numa_nodes.size()) == num_ranks, "rank_numa_nodes length differs from EP size");
        std::set<int> nodes;
        for (int node : rank_numa_nodes) if (node >= 0) nodes.insert(node);
        if (nodes.empty()) nodes.insert(0);
        physical_numa_nodes.assign(nodes.begin(), nodes.end());
        for (int node : rank_numa_nodes) {
            const auto found = std::find(physical_numa_nodes.begin(), physical_numa_nodes.end(), node);
            rank_numa_domains.push_back(found == physical_numa_nodes.end() ? 0 : int(found - physical_numa_nodes.begin()));
        }
        TORCH_CHECK(channels >= 1 && channels <= 32, "channels must be 1..32");
        TORCH_CHECK(stages >= 2 && stages <= kNumMaxLocalStages, "pipeline_stages must be 2 or 3");
        TORCH_CHECK(payload_slots > 0 && descriptor_slots > 0 && replica_slots > 0 && timeout_ms > 0, "invalid advanced capacity or timeout");
        TORCH_CHECK(output_pool_size >= 2 && output_pool_size <= 8, "output_pool_size must be 2..8");
        TORCH_CHECK(relay_workers_per_numa >= 1 && relay_workers_per_numa <= channels, "invalid relay worker count");
        TORCH_CHECK(int(physical_numa_nodes.size()) <= kNumMaxNumaNodes, "too many NUMA domains");
    }

    AdvancedSharedMemoryLayout layout(int p, int t, int h, int k, int e) const {
        return make_advanced_shared_memory_layout(p, t, h, k, e, payload_slots,
            descriptor_slots, channels, stages, int(physical_numa_nodes.size()), replica_slots, payload_dtype);
    }
};

inline int get_device_numa_node() {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    char bus_id[32];
    C10_CUDA_CHECK(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device));
    std::string bus = bus_id;
    for (auto& character : bus) character = std::tolower(character);
    std::ifstream file("/sys/bus/pci/devices/" + bus + "/numa_node");
    int node = -1;
    file >> node;
    return node;
}

inline long bind_host_pages(void* address, size_t length, int numa_node, int& error) {
    if (!length || numa_node < 0) { error = 0; return -1; }
    const size_t bits = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask(size_t(numa_node) / bits + 1, 0);
    mask[size_t(numa_node) / bits] = 1ul << (size_t(numa_node) % bits);
    const long result = syscall(SYS_mbind, address, length, MPOL_BIND, mask.data(), mask.size() * bits, 0);
    error = result == 0 ? 0 : errno;
    return result;
}

inline void create_advanced_segments(std::string directory, int p, int t, int h, int k, int e, py::dict options) {
    const AdvancedConfig config(p, options);
    const auto layout = config.layout(p, t, h, k, e);
    for (int source = 0; source < p; ++source) {
        const auto path = source_segment_path(directory, source);
        const int fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        TORCH_CHECK(fd >= 0, "create advanced source segment failed: ", path);
        const int result = ftruncate(fd, layout.size_bytes);
        const int saved = errno;
        ::close(fd);
        TORCH_CHECK(result == 0, "ftruncate advanced source segment failed: ", saved);
    }
}

inline py::dict prepare_advanced_segment(std::string directory, int rank, int p, int t, int h, int k, int e, py::dict options) {
    const AdvancedConfig config(p, options);
    const auto layout = config.layout(p, t, h, k, e);
    const auto path = source_segment_path(directory, rank);
    const int fd = open(path.c_str(), O_RDWR);
    TORCH_CHECK(fd >= 0, "open advanced source segment failed");
    void* address = mmap(nullptr, layout.size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (address == MAP_FAILED) { ::close(fd); TORCH_CHECK(false, "mmap advanced source segment failed"); }
    py::dict result;
    py::list placements;
    try {
        const int source_node = get_device_numa_node();
        int error = 0;
        const long source_bind = bind_host_pages(address, layout.size_bytes, source_node, error);
        result["numa_node"] = source_node;
        result["mbind_return"] = source_bind;
        result["mbind_errno"] = error;
        // Source metadata/control stay on the source's node. Only payload
        // replica pages claim target-local placement; every payload slot is
        // page-isolated, including Q/R=1 and non-power-of-two capacities.
        if (config.relay) {
            for (int channel = 0; channel < config.channels; ++channel) {
                for (int domain = 0; domain < layout.num_numa_nodes; ++domain) {
                    const size_t slot = (size_t(channel) * layout.num_numa_nodes + domain) * layout.replica_slots;
                    const size_t offset = layout.replica_payload_offset + slot * layout.payload_stride_bytes;
                    const size_t length = size_t(layout.replica_slots) * layout.payload_stride_bytes;
                    TORCH_CHECK(offset % 4096 == 0 && length % 4096 == 0, "replica NUMA pages are not isolated");
                    int bind_error = 0;
                    const long bound = bind_host_pages(static_cast<char*>(address) + offset, length,
                        config.physical_numa_nodes[domain], bind_error);
                    py::dict span;
                    span["channel"] = channel;
                    span["domain"] = domain;
                    span["numa_node"] = config.physical_numa_nodes[domain];
                    span["offset"] = offset;
                    span["length"] = length;
                    span["mbind_return"] = bound;
                    span["mbind_errno"] = bind_error;
                    placements.append(span);
                }
            }
        }
        std::memset(address, 0, layout.size_bytes);
        result["replica_payload_spans"] = placements;
        result["segment_bytes"] = layout.size_bytes;
        result["payload_stride_bytes"] = layout.payload_stride_bytes;
        result["control_placement"] = "source NUMA; replica payload pages only are target-bound";
    } catch (...) {
        munmap(address, layout.size_bytes);
        ::close(fd);
        throw;
    }
    munmap(address, layout.size_bytes);
    ::close(fd);
    return result;
}

// Explicitly establish the trivial objects used by std::atomic_ref without
// reinitializing existing shared values. All ranks do this before dispatch.
inline void establish_advanced_objects(char* base, const AdvancedSharedMemoryLayout& z) {
    const auto signals = [&](size_t offset, size_t count) {
        for (size_t i = 0; i < count; ++i) ::new (base + offset + i * sizeof(Signal)) Signal;
    };
    signals(z.base.counts_ready_offset, 1);
    signals(z.base.counts_consumed_offset, z.base.num_ranks);
    signals(z.base.abort_offset, 1);
    signals(z.source_ack_offset, size_t(z.channels) * z.base.num_payload_slots * z.num_endpoints);
    signals(z.replica_ack_offset, size_t(z.channels) * z.num_numa_nodes * z.replica_slots * z.base.num_ranks);
    signals(z.relay_drained_offset, size_t(z.channels) * z.num_numa_nodes);
    signals(z.channel_abort_offset, z.channels);
    signals(z.relay_failed_offset, size_t(z.channels) * z.num_numa_nodes);
    const auto descriptors = [&](size_t offset, size_t count) {
        for (size_t i = 0; i < count; ++i) ::new (base + offset + i * sizeof(Descriptor)) Descriptor;
    };
    descriptors(z.source_descriptors_offset, size_t(z.channels) * z.num_endpoints * z.base.num_descriptor_slots);
    descriptors(z.replica_descriptors_offset, size_t(z.channels) * z.num_numa_nodes * z.base.num_ranks * z.base.num_descriptor_slots);
}

inline void release_advanced_segments(std::vector<MappedHostSegment>& segments, size_t bytes, bool checked) {
    int first_error = 0;
    for (auto& segment : segments) {
        if (!segment.device_address) {
            continue;
        }
        const auto status = cudaHostUnregister(segment.host_address);
        if (status == cudaSuccess) {
            segment.device_address = nullptr;
        } else if (!first_error) {
            first_error = int(status);
        }
    }
    // A failed unregister may leave an active GPU alias. Preserve every
    // mapping and file descriptor for a safe retry instead of unmapping a
    // partially registered segment. Constructor rollback retains them too.
    if (first_error) {
        if (checked) {
            TORCH_CHECK(false, "advanced cudaHostUnregister failed; mappings retained: ", first_error);
        }
        return;
    }
    for (auto& segment : segments) {
        munmap(segment.host_address, bytes);
        ::close(segment.file_descriptor);
    }
    segments.clear();
}

inline std::vector<MappedHostSegment> register_advanced_segments(const std::string& directory, const AdvancedSharedMemoryLayout& z) {
    std::vector<MappedHostSegment> segments;
    try {
        for (int source = 0; source < z.base.num_ranks; ++source) {
            const auto path = source_segment_path(directory, source);
            const int fd = open(path.c_str(), O_RDWR);
            TORCH_CHECK(fd >= 0, "open advanced source segment failed: ", path);
            void* address = mmap(nullptr, z.size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (address == MAP_FAILED) { ::close(fd); TORCH_CHECK(false, "mmap advanced source segment failed"); }
            segments.push_back({address, nullptr, fd});
            establish_advanced_objects(static_cast<char*>(address), z);
            C10_CUDA_CHECK(cudaHostRegister(address, z.size_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
            // Mark registration before fetching the alias so a later failure
            // still unregisters this segment during constructor rollback.
            segments.back().device_address = static_cast<char*>(address);
            void* device_address = nullptr;
            C10_CUDA_CHECK(cudaHostGetDevicePointer(&device_address, address, 0));
            segments.back().device_address = static_cast<char*>(device_address);
        }
    } catch (...) {
        release_advanced_segments(segments, z.size_bytes, false);
        throw;
    }
    return segments;
}

}  // namespace pcie_ep::host_staged
