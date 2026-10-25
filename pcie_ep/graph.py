"""Explicit fixed-address CUDA Graph dispatch with GPU-only replay ordering."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist

from .buffers.handle import HostStagedEPHandle
from .utils.event import EventOverlap, _record_tensors


class GraphDispatch:
    """One captured dispatch bound to a buffer and fixed input/output addresses.

    Dynamic routing is recomputed on every replay. With cached_handle, routing
    must remain version-identical. Outputs are a dedicated graph pool and are
    overwritten by the next replay; clone data to retain replay history.
    """

    def __init__(
        self,
        owner: Any,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        weights: Optional[torch.Tensor],
        graph: Any,
        result: dict,
        alignment: int,
        expand: bool,
        cached_handle: Any,
    ) -> None:
        self._owner = owner
        self.x, self.topk_idx, self.topk_weights = x, topk_idx, weights
        self._graph, self._result = graph, result
        self.expert_alignment, self.do_expand = alignment, expand
        self._cached_handle = cached_handle
        self._bindings = tuple(self._signature(value) for value in (x, topk_idx, weights))
        self._generation = owner._generation
        self._closed = False
        self._last_event = None

    @staticmethod
    def _signature(tensor: Any) -> Any:
        if tensor is None:
            return None
        return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)

    def _validate_bindings(self) -> None:
        if self._closed or self._owner._generation != self._generation:
            raise RuntimeError("GraphDispatch is closed or its buffer generation changed")
        actual = tuple(self._signature(value) for value in (self.x, self.topk_idx, self.topk_weights))
        if actual != self._bindings:
            raise ValueError("GraphDispatch input address, shape, dtype or strides changed")
        if self._cached_handle is not None:
            self._cached_handle.validate_cached(
                self._owner,
                self.x,
                self.expert_alignment,
                self.do_expand,
            )

    def replay(self, *, previous_event: Any = None, async_with_compute_stream: bool = False) -> Any:
        """Replay using existing addresses; return the normal five-item result.

        In-place updates to x/weights and to dynamic routes are permitted when
        enqueued before this call. No CPU event wait or count readback occurs.
        """
        owner = self._owner
        with owner._lock:
            owner._check_live()
            self._validate_bindings()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("GraphDispatch.replay cannot be nested in another capture")
            compute = torch.cuda.current_stream(owner.device)
            with torch.cuda.stream(owner._comm_stream):
                owner._wait_for_dispatch_inputs(compute, previous_event)
                self._graph.replay()
                event = torch.cuda.Event()
                event.record(owner._comm_stream)
            refs = (owner, self, self.x, self.topk_idx, self.topk_weights, self._result)
            overlap = EventOverlap(event, refs, owner._register_consumer_stream)
            handle = HostStagedEPHandle(
                owner,
                self.topk_idx,
                self._result,
                self.do_expand,
                self.expert_alignment,
                overlap,
            )
            owner._last_event, owner._last_handle = event, handle
            self._last_event = event
            if not async_with_compute_stream:
                compute.wait_event(event)
                _record_tensors(self._result, compute)
                owner._register_consumer_stream(compute)
            return (
                self._result["recv_x"],
                self._result.get("recv_topk_idx"),
                self._result.get("recv_topk_weights") if self.topk_weights is not None else None,
                handle,
                overlap,
            )

    def close(self) -> None:
        """Explicit capture teardown may synchronize; replay never does."""
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.synchronize()
        self._graph.reset()
        self._closed = True


def capture_dispatch(
    owner: Any,
    x: torch.Tensor,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    *,
    handle: Any = None,
    do_expand: bool = False,
    expert_alignment: Optional[int] = None,
    do_zero_padding: bool = False,
) -> GraphDispatch:
    """Collectively prepare one graph per buffer; capture may synchronize.

    The graph has an independent preallocated result pool. Rebuild the buffer
    to capture different bindings. Eager dispatch and replay may interleave.
    """
    with owner._lock:
        owner._check_live()
        if not owner.config.enable_graph or owner.config.stage < 6:
            raise NotImplementedError("capture_dispatch requires S6+ with enable_graph=True")
        if owner._graph_capture is not None:
            raise RuntimeError("A buffer supports one fixed-address GraphDispatch; rebuild it to recapture")
        graph, cached, alignment, error = None, None, None, None
        try:
            if handle is not None:
                if topk_idx is not None:
                    raise ValueError("topk_idx must be None when a cached handle is supplied")
                topk_idx = handle.topk_idx
            if expert_alignment is None:
                alignment = handle.expert_alignment if handle is not None else 1
            else:
                alignment = expert_alignment
            if not isinstance(alignment, int) or not 1 <= alignment <= 256:
                raise ValueError("expert_alignment must be in [1,256]")
            if do_zero_padding and not do_expand:
                raise ValueError("do_zero_padding requires do_expand=True")
            owner._validate_input(x, "x", torch.bfloat16, (None, owner.hidden))
            owner._validate_input(topk_idx, "topk_idx", torch.int64, (x.shape[0], owner.num_topk))
            if x.shape[0] > owner.num_max_tokens_per_rank:
                raise ValueError("Graph input exceeds the prepared capacity")
            if topk_weights is not None:
                owner._validate_input(topk_weights, "topk_weights", torch.float32, topk_idx.shape)
            if handle is not None:
                cached = handle.validate_cached(owner, x, alignment, do_expand)
            # Explicit preparation may wait. Agree on validation/sync failures
            # before any peer enters capture, rather than stranding a peer at
            # a barrier when one rank rejects its local input.
            torch.cuda.synchronize(owner.device)
            graph = torch.cuda.CUDAGraph()
        except Exception as exc:
            error = f"rank {owner.rank_idx}: {type(exc).__name__}: {exc}"
        errors = [None] * owner.num_ranks
        dist.all_gather_object(errors, error, group=owner.control_group)
        if any(errors):
            if graph is not None:
                graph.reset()
            raise RuntimeError(f"Collective graph preparation failed: {errors}")

        result, error = None, None
        try:
            with torch.cuda.stream(owner._comm_stream):
                _record_tensors((x, topk_idx, topk_weights), owner._comm_stream)
            with torch.cuda.graph(graph, stream=owner._comm_stream):
                result = owner.runtime.dispatch(
                    x,
                    topk_idx,
                    topk_weights,
                    owner.config.stage,
                    alignment,
                    do_expand,
                    do_zero_padding,
                    cached,
                    True,
                )
        except Exception as exc:
            error = f"rank {owner.rank_idx}: {type(exc).__name__}: {exc}"
        errors = [None] * owner.num_ranks
        dist.all_gather_object(errors, error, group=owner.control_group)
        if any(errors):
            graph.reset()
            raise RuntimeError(f"Collective graph capture failed: {errors}")
        # A dynamic graph's routing scratch is updated at replay time and is
        # deliberately not exposed as an immutable cached-layout handle.
        result["_layout"] = cached
        captured = GraphDispatch(owner, x, topk_idx, topk_weights, graph, result, alignment, do_expand, handle)
        owner._graph_capture = captured
        return captured
