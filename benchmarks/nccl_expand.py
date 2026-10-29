"""GPU-only packing and stable route expansion for the NCCL SHM baseline.

NCCL transports source tokens together with their original routes and weights.
Packing and all three expansion kernels are included in full-dispatch timing.
This helper does not communicate, synchronize the CPU, read GPU scalar counts,
or change PCIe-EP's production implementation.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_EXTENSION = None


def _extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        _EXTENSION = load(
            name="pcie_ep_nccl_expand",
            sources=[str(Path(__file__).resolve().parent / "csrc" / "nccl_expand.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++20"],
            extra_cflags=["-O2", "-std=c++20"],
            verbose=bool(int(os.getenv("PCIE_EP_BUILD_VERBOSE", "0"))),
        )
    return _EXTENSION


class NCCLExpand:
    """Preallocate one fixed-shape NCCL all-gather and expert expansion case.

    ``packed_recv`` is source-rank-major. Expanded rows are ordered by local
    expert, source rank, source token, then original top-k branch. Only rows
    before ``expert_offsets[-1]`` contain output; unused capacity is undefined.
    No routing-dependent allocation or cached routing plan is used.
    """

    def __init__(self, world, tokens, hidden, topk, experts, rank, device):
        values = (world, tokens, hidden, topk, experts, rank)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("dimensions and rank must be integers")
        if min(world, tokens, hidden, topk, experts) <= 0:
            raise ValueError("dimensions must be positive")
        if experts % world or experts // world > 32:
            raise ValueError("experts must divide evenly across ranks, with at most 32 per rank")
        if rank < 0 or rank >= world:
            raise ValueError("rank is out of range")
        if hidden % 8 or topk % 4:
            raise ValueError("wire rows must preserve 16-byte vector alignment")
        if world * tokens * topk > 2**31 - 1 or hidden * 2 + topk * 12 > 2**31 - 1:
            raise ValueError("dimensions exceed int32 kernel capacity")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.world, self.tokens, self.hidden = world, tokens, hidden
        self.topk, self.experts, self.rank = topk, experts, rank
        self.device = device
        self.row_bytes = hidden * 2 + topk * 12
        self.capacity = world * tokens * topk
        local_experts = experts // world
        num_tokens = world * tokens
        num_tiles = (num_tokens + 255) // 256
        self._extension = _extension()
        self.packed_send = torch.empty((tokens, self.row_bytes), dtype=torch.uint8, device=device)
        self.packed_recv = torch.empty((num_tokens, self.row_bytes), dtype=torch.uint8, device=device)
        self.positions = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
        self.tile_counts = torch.empty((local_experts, num_tiles), dtype=torch.int32, device=device)
        self.tile_offsets = torch.empty_like(self.tile_counts)
        self.payload = torch.empty((self.capacity, hidden), dtype=torch.bfloat16, device=device)
        self.indices = torch.empty(self.capacity, dtype=torch.int64, device=device)
        self.weights = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.source_meta = torch.empty((self.capacity, 3), dtype=torch.int64, device=device)
        self.expert_counts = torch.empty(local_experts, dtype=torch.int32, device=device)
        self.expert_offsets = torch.empty(local_experts + 1, dtype=torch.int32, device=device)
        self._result = {
            "payload": self.payload,
            "indices": self.indices,
            "weights": self.weights,
            "source_meta": self.source_meta,
            "expert_counts": self.expert_counts,
            "expert_offsets": self.expert_offsets,
        }

    def pack(self, x, idx, weights):
        """Enqueue one vectorized pack on the calling CUDA stream."""
        if tuple(x.shape) != (self.tokens, self.hidden) or tuple(idx.shape) != (self.tokens, self.topk):
            raise ValueError("input dimensions do not match this case")
        self._extension.pack(x, idx, weights, self.packed_send)
        return self.packed_send

    def expand(self, packed_recv=None):
        """Enqueue count/ordinal, prefix and scatter kernels without CPU waits."""
        packed_recv = self.packed_recv if packed_recv is None else packed_recv
        self._extension.expand(
            packed_recv, self.positions, self.tile_counts, self.tile_offsets,
            self.expert_counts, self.expert_offsets, self.payload, self.indices,
            self.weights, self.source_meta, self.world, self.tokens, self.hidden,
            self.topk, self.experts, self.rank,
        )
        return self._result
