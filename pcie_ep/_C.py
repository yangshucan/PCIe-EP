"""Build and load the local CUDA extension without downloading dependencies.

The runtime remains one CUDA translation unit. Its implementation headers
live beside runtime.cu and are also available through the explicit include
directory below. This module is imported lazily when constructing a buffer.
"""

import os
from pathlib import Path

from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "csrc"
_CUDA_SOURCE = _SOURCE_DIRECTORY / "runtime.cu"
_CUDA_FLAGS = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++20"]
_HOST_FLAGS = ["-O2", "-std=c++20"]
# Use a separate TORCH_EXTENSIONS_DIR when comparing instrumented/release runs.
# No profiling allocation, clock read or counter exists in the default build.
if os.getenv("PCIE_EP_PROFILE", "0") == "1":
    _CUDA_FLAGS.append("-DPCIE_EP_ENABLE_PROFILING=1")
    _HOST_FLAGS.append("-DPCIE_EP_ENABLE_PROFILING=1")

_module = load(
    name="pcie_ep_cuda",
    sources=[str(_CUDA_SOURCE)],
    extra_include_paths=[str(_SOURCE_DIRECTORY)],
    extra_cuda_cflags=_CUDA_FLAGS,
    extra_cflags=_HOST_FLAGS,
    verbose=bool(int(os.getenv("PCIE_EP_BUILD_VERBOSE", "0"))),
)

AdvancedRuntime = _module.AdvancedRuntime
create_advanced_segments = _module.create_advanced_segments
prepare_advanced_segment = _module.prepare_advanced_segment
get_device_numa_node = _module.get_device_numa_node
