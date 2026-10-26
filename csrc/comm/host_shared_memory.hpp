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

}  // namespace pcie_ep::host_staged
