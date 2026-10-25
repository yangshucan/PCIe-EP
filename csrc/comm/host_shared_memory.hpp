#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../kernels/host_staged/common.cuh"

namespace pcie_ep::host_staged {

namespace py = pybind11;

// Each process maps the same source-owned file, then registers that mapping
// with its local CUDA device. No device pointer is exchanged between ranks.
struct MappedHostSegment {
    void* host_address;
    char* device_address;
    int file_descriptor;
};

inline std::string source_segment_path(const std::string& directory, int source_rank) {
    return directory + "/source-" + std::to_string(source_rank);
}

inline void create_segments(std::string directory, int num_ranks,
                            int num_max_tokens_per_rank, int hidden, int num_topk,
                            int num_experts, int num_payload_slots,
                            int num_descriptor_slots, int stage) {
    const auto layout = make_shared_memory_layout(
        num_ranks, num_max_tokens_per_rank, hidden, num_topk, num_experts,
        num_payload_slots, num_descriptor_slots, stage);

    for (int source_rank = 0; source_rank < num_ranks; ++source_rank) {
        const auto path = source_segment_path(directory, source_rank);
        const int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        TORCH_CHECK(descriptor >= 0, "create segment failed: ", path);
        const int result = ftruncate(descriptor, layout.size_bytes);
        ::close(descriptor);
        TORCH_CHECK(result == 0, "ftruncate failed");
    }
}

// Python coordinates this owner-only first touch before any rank registers
// the pages. The reported mbind result remains best-effort diagnostics.
inline py::dict prepare_segment(std::string directory, int local_rank,
                                int num_ranks, int num_max_tokens_per_rank,
                                int hidden, int num_topk, int num_experts,
                                int num_payload_slots, int num_descriptor_slots,
                                int stage) {
    const auto layout = make_shared_memory_layout(
        num_ranks, num_max_tokens_per_rank, hidden, num_topk, num_experts,
        num_payload_slots, num_descriptor_slots, stage);
    const auto path = source_segment_path(directory, local_rank);
    const int descriptor = open(path.c_str(), O_RDWR);
    TORCH_CHECK(descriptor >= 0, "open owner segment failed");

    void* host_address = mmap(nullptr, layout.size_bytes, PROT_READ | PROT_WRITE,
                              MAP_SHARED, descriptor, 0);
    TORCH_CHECK(host_address != MAP_FAILED, "owner mmap failed");

    int device;
    cudaGetDevice(&device);
    char pci_bus_id[32];
    cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device);
    std::string pci_device_path = pci_bus_id;
    for (auto& character : pci_device_path) {
        character = std::tolower(character);
    }
    std::ifstream numa_node_file(
        "/sys/bus/pci/devices/" + pci_device_path + "/numa_node");
    int numa_node = -1;
    numa_node_file >> numa_node;

    long mbind_result = -1;
    int mbind_error = 0;
    if (numa_node >= 0 && numa_node < 64) {
        unsigned long node_mask = 1ul << numa_node;
        mbind_result = syscall(SYS_mbind, host_address, layout.size_bytes,
                               MPOL_BIND, &node_mask, 64, 0);
        mbind_error = errno;
    }
    std::memset(host_address, 0, layout.size_bytes);
    munmap(host_address, layout.size_bytes);
    ::close(descriptor);

    py::dict result;
    result["numa_node"] = numa_node;
    result["mbind_return"] = mbind_result;
    result["mbind_errno"] = mbind_result ? mbind_error : 0;
    result["segment_bytes"] = layout.size_bytes;
    return result;
}

inline std::vector<MappedHostSegment> register_source_segments(
        const std::string& directory, const SharedMemoryLayout& layout) {
    std::vector<MappedHostSegment> segments;
    for (int source_rank = 0; source_rank < layout.num_ranks; ++source_rank) {
        const auto path = source_segment_path(directory, source_rank);
        const int descriptor = open(path.c_str(), O_RDWR);
        TORCH_CHECK(descriptor >= 0, "open shared segment failed: ", path);
        void* host_address = mmap(nullptr, layout.size_bytes,
                                  PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
        TORCH_CHECK(host_address != MAP_FAILED, "mmap failed");
        segments.push_back({host_address, nullptr, descriptor});

        C10_CUDA_CHECK(cudaHostRegister(
            host_address, layout.size_bytes,
            cudaHostRegisterMapped | cudaHostRegisterPortable));
        void* device_address = nullptr;
        C10_CUDA_CHECK(cudaHostGetDevicePointer(&device_address, host_address, 0));
        segments.back().device_address = static_cast<char*>(device_address);
    }
    return segments;
}

// Explicit close is called only after Python has drained GPU work and reached
// its collective barrier. It does not add a new device synchronization here.
inline void unregister_source_segments(const std::vector<MappedHostSegment>& segments,
                                       size_t size_bytes) {
    for (const auto& segment : segments) {
        C10_CUDA_CHECK(cudaHostUnregister(segment.host_address));
        munmap(segment.host_address, size_bytes);
    }
    for (const auto& segment : segments) {
        ::close(segment.file_descriptor);
    }
}

// Preserve the existing destructor fallback: the owning Runtime first waits
// for its local device, then releases mappings without throwing CUDA errors.
inline void release_source_segments_on_destruction(
        const std::vector<MappedHostSegment>& segments, size_t size_bytes) {
    for (const auto& segment : segments) {
        cudaHostUnregister(segment.host_address);
        munmap(segment.host_address, size_bytes);
    }
    for (const auto& segment : segments) {
        ::close(segment.file_descriptor);
    }
}

}  // namespace pcie_ep::host_staged
