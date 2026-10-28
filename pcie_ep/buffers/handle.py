"""Dispatch result metadata and routing lifetime checks."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ..utils.event import EventOverlap

if TYPE_CHECKING:
    from .ep import HostStagedEPBuffer


class HostStagedEPHandle:
    """GPU metadata describing a capacity-sized dispatch result.

    ``unique_count`` is a CUDA scalar.  Unique output uses its leading rows.
    Expanded output uses ``expert_offsets[e]`` and ``expert_counts[e]``;
    alignment gaps and unused capacity are not valid token rows.

    Attributes:
        source_meta: source group-local rank/token index/top-k slot for each
            unique or expanded row; unique rows use -1 for the top-k slot.
        source_offsets: exclusive per-source unique offsets, including zero.
        expert_counts: true counts per local expert, without alignment padding.
        expert_offsets: exclusive padded offsets for expert output segments.
        error, stats: GPU protocol status and deduplicated transport counters.
        receive_expand_overlap: optional CUDA int64 [channels, 2] for optimized
            expanded receives. Column zero counts conservative receive/scatter
            overlap; column one is an internal completed-ticket marker.
            Scalar/unique paths return None.
        completion: EventOverlap for stream ordering and deferred returns.

    This backend deliberately has no CPU count list. Eager result pools
    preserve outputs while external Tensor references exist; retaining every
    pool makes a new dispatch fail with pool exhaustion. Release an old
    result, or clone it and release the original tensors, to free its pool.
    S6 cached_layout detaches routing without retaining result-pool tensors.
    Graph results instead share one dedicated pool overwritten each replay.
    """

    def __init__(
        self,
        owner: "HostStagedEPBuffer",
        topk_idx: torch.Tensor,
        result: dict[str, torch.Tensor],
        do_expand: bool,
        expert_alignment: int,
        event: EventOverlap,
    ) -> None:
        self.buffer_id = owner.buffer_id
        self.generation = owner._generation
        self.group = owner.group
        self.global_ranks = owner.global_ranks
        self.do_expand = do_expand
        self.expert_alignment = expert_alignment
        self.num_experts = owner.num_experts
        self.num_max_tokens_per_rank = owner.num_max_tokens_per_rank
        self.num_sms = 0
        self._topk_idx = topk_idx
        self._topk_version = None if topk_idx.is_inference() else topk_idx._version
        self._route_signature = self._tensor_signature(topk_idx)
        self._layout = result.get("_layout")
        self.unique_count = result["unique_count"]
        self.expert_counts = result["expert_counts"]
        self.expert_offsets = result["expert_offsets"]
        self.source_meta = result["source_meta"]
        self.error = result["error"]
        self.stats = result["stats"]
        self.transport_timestamps_ns = result.get("transport_timestamps_ns")
        self.receive_expand_overlap = result.get("receive_expand_overlap")
        self.prefix_tx_timestamps_ns = result.get("prefix_tx_timestamps_ns")
        self.expanded_count = result["expanded_count"]
        self.num_recv_tokens = self.unique_count
        self.num_expanded_tokens = self.expanded_count
        self.num_recv_tokens_per_expert_list = None
        self.num_unaligned_recv_tokens_per_expert = self.expert_counts
        self.recv_src_metadata = self.source_meta
        self.source_offsets = result["source_offsets"]
        self.psum_num_recv_tokens_per_scaleup_rank = self.source_offsets[1:]
        self.psum_num_recv_tokens_per_expert = result.get("expert_prefix")
        self.dst_buffer_slot_idx = result.get("dst_buffer_slot_idx")
        self.completion = event
        # EventOverlap.release_handle=True releases its own reference only;
        # explicit error diagnostics still need this dispatch's completion.
        self._completion_event = event.event
        self._owner = owner

    @property
    def topk_idx(self) -> torch.Tensor:
        """Original routing tensor, guarded against ordinary in-place edits."""
        if self._topk_version is not None and self._topk_idx._version != self._topk_version:
            raise RuntimeError("topk_idx changed while its dispatch handle was in use")
        return self._topk_idx

    def check_errors(self) -> None:
        """Explicit diagnostic CPU synchronization; never called by dispatch."""
        self._owner.check_errors(self)

    def cache_layout(self) -> "CachedLayout":
        """Detach an immutable route cache without retaining result-pool tensors."""
        return CachedLayout(self)

    @property
    def cached_layout(self) -> "CachedLayout":
        return self.cache_layout()

    @staticmethod
    def _tensor_signature(tensor: torch.Tensor) -> tuple:
        return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)

    def validate_cached(self, owner: "HostStagedEPBuffer", x: torch.Tensor, alignment: int, do_expand: bool) -> Any:
        """Validate CPU routing metadata only; never read GPU counts."""
        if self.buffer_id != owner.buffer_id or self.generation != owner._generation:
            raise ValueError("Cached handle belongs to another buffer generation")
        if self._owner is not owner or self.group is not owner.group or self.global_ranks != owner.global_ranks:
            raise ValueError("Cached handle belongs to another EP group or owner map")
        if self.num_experts != owner.num_experts or self.num_max_tokens_per_rank != owner.num_max_tokens_per_rank:
            raise ValueError("Cached handle dimensions or expert ownership changed")
        route = self.topk_idx
        if self._topk_version is None:
            raise ValueError("Cached routing requires a tensor with a PyTorch version counter")
        if self._tensor_signature(route) != self._route_signature:
            raise ValueError("Cached routing tensor address, shape or strides changed")
        if x.shape[0] != route.shape[0] or x.shape[1] != owner.hidden:
            raise ValueError("Cached dispatch input dimensions changed")
        if (alignment, do_expand) != (self.expert_alignment, self.do_expand):
            raise ValueError("Cached dispatch alignment or expansion mode changed")
        if self._layout is None:
            raise RuntimeError("The native runtime did not provide an immutable cached layout")
        return self._layout


class CachedLayout:
    """Lightweight immutable routing plan without result-pool ownership.

    The route tensor, its original version and its buffer owner remain alive.
    No dispatch outputs, GPU counts, completion events or full result handles
    are retained. Native plan storage must be independent of result pools.
    """

    def __init__(self, handle: HostStagedEPHandle) -> None:
        if handle._layout is None:
            raise RuntimeError("This dispatch has no immutable native layout to cache")
        if handle._topk_version is None:
            raise ValueError("Cached routing requires a PyTorch version counter")
        self._topk_idx = handle.topk_idx
        for name in (
            "_topk_version",
            "_route_signature",
            "_layout",
            "_owner",
            "buffer_id",
            "generation",
            "group",
            "global_ranks",
            "num_experts",
            "num_max_tokens_per_rank",
            "expert_alignment",
            "do_expand",
        ):
            setattr(self, name, getattr(handle, name))

    topk_idx = HostStagedEPHandle.topk_idx
    _tensor_signature = staticmethod(HostStagedEPHandle._tensor_signature)
    validate_cached = HostStagedEPHandle.validate_cached


EPHandle = HostStagedEPHandle
