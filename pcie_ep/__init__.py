"""GPU-initiated PCIe host-staged expert-parallel dispatch."""

import torch

from .buffer import (
    EPBuffer,
    EPHandle,
    EventOverlap,
    HostStagedConfig,
    HostStagedEPBuffer,
    HostStagedEPHandle,
)
from .graph import GraphDispatch
from .buffers.handle import CachedLayout

__all__ = [
    "EPBuffer",
    "EPHandle",
    "EventOverlap",
    "HostStagedConfig",
    "HostStagedEPBuffer",
    "HostStagedEPHandle",
    "topk_idx_t",
    "GraphDispatch",
    "CachedLayout",
]

topk_idx_t = torch.int64
__version__ = "0.1.0"
