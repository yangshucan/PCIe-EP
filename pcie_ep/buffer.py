"""Compatibility imports for the original ``pcie_ep.buffer`` module.

The implementation now separates configuration, collective initialization,
CUDA overlap, and the expert-dispatch buffer. Existing import paths continue
referring to the same classes and helpers.
"""

# Keep shared modules accessible for existing unittest.mock targets. Patching
# torch.cuda or dist through this path also patches the implementation.
import torch
import torch.distributed as dist

from .buffers.ep import EPBuffer, HostStagedEPBuffer
from .buffers.handle import CachedLayout, EPHandle, HostStagedEPHandle
from .comm import _prepare_segments
from .config import HostStagedConfig
from .utils.event import EventOverlap, _record_tensors

__all__ = [
    "CachedLayout",
    "EPBuffer",
    "EPHandle",
    "EventOverlap",
    "HostStagedConfig",
    "HostStagedEPBuffer",
    "HostStagedEPHandle",
]
