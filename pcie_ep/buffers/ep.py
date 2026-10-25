"""DeepEP-shaped buffer for GPU-initiated PCIe-only BF16 dispatch."""

from __future__ import annotations

import shutil
import threading
from typing import Any, Optional

import torch
import torch.distributed as dist

from .. import comm
from ..config import HostStagedConfig
from ..native import AdvancedExtensionAdapter
from ..utils.event import EventOverlap, _record_tensors
from .handle import HostStagedEPHandle


class HostStagedEPBuffer:
    """DeepEP-shaped API for GPU-initiated PCIe host-staged dispatch.

    Construction and ``close`` are collective and may synchronize the CPU.
    A supplied Gloo ``control_group`` must have the same ordered global ranks
    as ``group``.  If omitted, a Gloo group is used directly or a CPU-only
    auxiliary group is created.  No NCCL collective is issued by this class.

    The native runtime owns preallocated result pools: two in S1-S4 and
    ``config.output_pool_size`` in S5+. Python retains inputs and metadata,
    establishes producer/consumer stream dependencies, and exposes GPU valid
    counts; it never reads those counts during dispatch.
    """

    def __init__(
        self,
        group: Any = None,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        num_experts: int = 0,
        config: Optional[HostStagedConfig] = None,
        *,
        control_group: Any = None,
        host_staged_config: Optional[HostStagedConfig] = None,
        transport: str = "host_staged",
        deterministic: bool = True,
        require_reuse_nccl_comm: bool = False,
        explicitly_destroy: bool = False,
    ) -> None:
        """Prepare collective host mappings and fixed-capacity GPU storage.

        Arguments:
            group: existing EP process group; defaults to WORLD.
            num_max_tokens_per_rank: source token capacity shared by all ranks.
            hidden, num_topk, num_experts: fixed BF16 MoE dimensions.  Experts
                have uniform contiguous ownership across the EP ranks.
            config: host transport configuration; host_staged_config is its
                compatibility alias and cannot be supplied at the same time.
            control_group: optional Gloo group with the same ordered members.
            require_reuse_nccl_comm: reject missing private PyTorch reuse APIs
                instead of silently creating a new communicator.

        CPU submissions must be serialized.  Explicit close is collective;
        no Python destructor initiates implicit process-group communication.
        """
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before constructing a buffer")
        if not torch.cuda.is_available():
            raise RuntimeError("Host-staged EP requires NVIDIA CUDA")
        if transport != "host_staged":
            raise ValueError("This package implements only transport='host_staged'")
        if config is not None and host_staged_config is not None:
            raise ValueError("Supply config or host_staged_config, not both")
        self.config = config or host_staged_config or HostStagedConfig()
        for name, value in (
            ("num_max_tokens_per_rank", num_max_tokens_per_rank),
            ("hidden", hidden),
            ("num_topk", num_topk),
            ("num_experts", num_experts),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.group = group if group is not None else dist.group.WORLD
        self.rank_idx = dist.get_rank(self.group)
        self.num_ranks = dist.get_world_size(self.group)
        if self.rank_idx < 0:
            raise ValueError("This process is not a member of the supplied EP group")
        self.global_ranks = tuple(dist.get_process_group_ranks(self.group))
        if num_experts % self.num_ranks:
            raise ValueError("Experts must be uniformly divisible among EP ranks")
        if self.config.stage == 1 and self.num_ranks != 2:
            raise ValueError("S1 is the two-GPU transport baseline; use stage >= 2")
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden, self.num_topk, self.num_experts = hidden, num_topk, num_experts
        self.deterministic = deterministic
        self.explicitly_destroy = explicitly_destroy
        self.device = torch.cuda.current_device()
        self._comm_stream = torch.cuda.Stream(device=self.device)
        self._last_event = None
        self._last_handle: Optional[HostStagedEPHandle] = None
        self._generation = 0
        self._failed = False
        self._closed = False
        self._lock = threading.RLock()
        self._consumer_streams: dict[tuple[int, int], Any] = {}
        self._graph_capture = None
        self._owns_control_group = False
        self.runtime = None
        self.nccl_comm = None
        self.comm_reused = False
        self.comm_reuse_status = "not_an_nccl_group"

        # Borrow application-owned NCCL state; the transport itself uses no
        # NCCL data operations and never creates a managed communicator.
        borrowed = comm.borrow_nccl_comm(
            self.group,
            self.device,
            require_reuse=require_reuse_nccl_comm,
        )
        self.nccl_comm = borrowed.nccl_comm
        self.comm_reused = borrowed.reused
        self.comm_reuse_status = borrowed.status

        # CPU control is confined to construction/close.  Every peer must agree
        # on the host backing layout before any GPU registers its mappings.
        self.control_group, self._owns_control_group = comm.get_control_group(
            self.group,
            self.global_ranks,
            control_group,
        )
        comm.validate_group_configuration(
            self.control_group,
            self.num_ranks,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            num_experts,
            self.config,
        )

        # Loading/compiling CUDA is intentionally lazy until construction.
        from .. import _C

        extension = AdvancedExtensionAdapter(_C, self.config, self.control_group) if self.config.advanced else _C

        self.directory, self.buffer_id = comm.create_shared_segments(
            extension,
            self.rank_idx,
            self.num_ranks,
            self.global_ranks,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            num_experts,
            self.config,
            self.control_group,
        )
        try:
            comm._prepare_segments(
                extension,
                self.directory,
                self.rank_idx,
                self.num_ranks,
                num_max_tokens_per_rank,
                hidden,
                num_topk,
                num_experts,
                self.config.payload_slots,
                self.config.descriptor_slots,
                self.config.stage,
                self.control_group,
            )
        except Exception:
            if self.rank_idx == 0:
                shutil.rmtree(self.directory)
            raise
        self.runtime = comm.register_runtime(
            extension,
            self.directory,
            self.rank_idx,
            self.num_ranks,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            num_experts,
            self.config,
            self.control_group,
            self._comm_stream,
            self.device,
        )

    def get_comm_stream(self) -> Any:
        """Return this buffer's CUDA stream, independent of a borrowed comm."""
        return self._comm_stream

    @staticmethod
    def _stream_key(stream: Any) -> tuple[int, int]:
        return (stream.device.index, int(stream.cuda_stream))

    def _register_consumer_stream(self, stream: Any) -> None:
        """Remember GPU consumers even after their Python Tensor refs vanish."""
        with self._lock:
            key = self._stream_key(stream)
            if key != self._stream_key(self._comm_stream):
                self._consumer_streams[key] = stream

    def _wait_consumer_streams(self, compute: Any, compute_already_waited: bool) -> None:
        """Order manual pool reuse after every registered consumer's tail.

        record_stream protects allocator reclamation but cannot protect direct
        writes into a still-allocated pool.  These waits are GPU dependencies,
        not CPU completion waits.  CPU calls must be serialized by the caller.
        """
        skip = {self._stream_key(self._comm_stream)}
        if compute_already_waited:
            skip.add(self._stream_key(compute))
        # Keep the registry until close: a retained result may be consumed
        # again on B after another dispatch has already waited for B once.
        # Only streams are retained here, never historical output tensors.
        for key, stream in self._consumer_streams.items():
            if key not in skip:
                self._comm_stream.wait_stream(stream)
                skip.add(key)

    @staticmethod
    def capture() -> EventOverlap:
        """Record a producer dependency on the current CUDA stream."""
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        return EventOverlap(event)

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: Any = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[HostStagedEPHandle] = None,
        do_handle_copy: bool = False,
        do_cpu_sync: Optional[bool] = False,
        do_expand: bool = False,
        do_zero_padding: bool = False,
        use_tma_aligned_col_major_sf: bool = False,
        defer_epilogue: bool = False,
    ) -> Any:
        """Dispatch BF16 tokens with sender/receiver deduplication.

        Arguments:
            x: contiguous CUDA BF16 [T, hidden] on this buffer's device.
            topk_idx: contiguous CUDA int64 [T, K] of global expert IDs;
                -1 denotes an inactive slot.  Stored by reference in the handle.
            topk_weights: optional contiguous CUDA FP32 [T, K] weights.
            handle: S6 immutable routing cache, preferably detached with
                cache_layout; omit topk_idx and retain its original routing
                tensor unchanged. Token values and weights may change.
            previous_event: optional producer dependency.  Requires
                allocate_on_comm_stream=True; it never replaces dependencies
                registered by old output consumers.
            async_with_compute_stream: leave current-stream waiting to the
                caller.  False enqueues a GPU wait, not a CPU synchronize.
            do_expand: write per-expert rows rather than unique receive rows.
            expert_alignment: align expert segments; default is one.
            do_zero_padding: zero inter-expert alignment gaps in expand mode.
            defer_epilogue: return only EventOverlap; its first wait returns
                the four data/handle outputs.  Requires asynchronous mode.

        Returns:
            recv_x, recv_idx, recv_weights, handle, event, or just event when
            defer_epilogue=True.  Shapes describe capacity; actual valid ranges
            are GPU handle counts/offsets.  recv_weights is None without input
            weights. Retaining all native output pools makes later calls fail
            explicitly rather than waiting or allocating another output set.

        do_cpu_sync=None also means False. Wire compression is configured at
        construction; input/output stay BF16. QPs, SM overrides and combine
        are unsupported. Use capture_dispatch for explicit Graph preparation.
        """
        with self._lock:
            self._check_live()
            if do_cpu_sync:
                raise NotImplementedError("The main path supports do_cpu_sync=False only")
            cached_layout = None
            if handle is not None:
                if getattr(getattr(self, "config", None), "stage", 4) < 6:
                    raise NotImplementedError("Cached dispatch replay is an S6 optimization")
                if topk_idx is not None:
                    raise ValueError("topk_idx must be None when a cached handle is supplied")
                topk_idx = handle.topk_idx
            if do_handle_copy or num_qps or num_sms or use_tma_aligned_col_major_sf:
                raise NotImplementedError("Handle copy, QPs, SM overrides and FP8 scale layouts are unsupported")
            if previous_event is not None and not allocate_on_comm_stream:
                raise ValueError("previous_event requires allocate_on_comm_stream=True")
            if defer_epilogue and not async_with_compute_stream:
                raise ValueError("defer_epilogue requires async_with_compute_stream=True")
            if do_zero_padding and not do_expand:
                raise ValueError("do_zero_padding is valid only with do_expand=True")
            alignment = (
                (handle.expert_alignment if handle is not None else 1) if expert_alignment is None else expert_alignment
            )
            if not isinstance(alignment, int) or alignment < 1:
                raise ValueError("expert_alignment must be a positive integer")
            if num_experts not in (None, self.num_experts):
                raise ValueError("num_experts differs from the prepared buffer")
            if num_max_tokens_per_rank not in (None, self.num_max_tokens_per_rank):
                raise ValueError("num_max_tokens_per_rank differs from the prepared buffer")
            self._validate_input(x, "x", torch.bfloat16, (None, self.hidden))
            self._validate_input(topk_idx, "topk_idx", torch.int64, (x.shape[0], self.num_topk))
            if x.shape[0] > self.num_max_tokens_per_rank:
                raise ValueError("Input token count exceeds the prepared capacity")
            if topk_weights is not None:
                self._validate_input(topk_weights, "topk_weights", torch.float32, topk_idx.shape)
            if cumulative_local_expert_recv_stats is not None:
                self._validate_input(
                    cumulative_local_expert_recv_stats,
                    "cumulative_local_expert_recv_stats",
                    torch.int32,
                    (self.num_experts // self.num_ranks,),
                )
            if handle is not None:
                cached_layout = handle.validate_cached(self, x, alignment, do_expand)
            if torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("Use capture_dispatch for explicit S6 CUDA Graph preparation")
            compute = torch.cuda.current_stream(self.device)
            result, event = self._dispatch_on_comm_stream(
                x,
                topk_idx,
                topk_weights,
                cumulative_local_expert_recv_stats,
                alignment,
                do_expand,
                do_zero_padding,
                compute,
                previous_event,
                cached_layout,
            )
            refs = (self, x, topk_idx, topk_weights, result, cumulative_local_expert_recv_stats)
            if handle is not None:
                refs = (*refs, handle)
            overlap = EventOverlap(event, refs, self._register_consumer_stream)
            output_handle = HostStagedEPHandle(self, topk_idx, result, do_expand, alignment, overlap)
            self._last_event, self._last_handle = event, output_handle
            outputs = (
                result["recv_x"],
                result.get("recv_topk_idx"),
                result.get("recv_topk_weights") if topk_weights is not None else None,
                output_handle,
            )
            if not async_with_compute_stream:
                compute.wait_event(event)
                _record_tensors(result, compute)
                self._register_consumer_stream(compute)
            if defer_epilogue:
                overlap.register_hook_after_wait(lambda: outputs)
                return overlap
            return (*outputs, overlap)

    def _dispatch_on_comm_stream(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        cumulative_local_expert_recv_stats: Optional[torch.Tensor],
        alignment: int,
        do_expand: bool,
        do_zero_padding: bool,
        compute: Any,
        previous_event: Any,
        cached_layout: Any = None,
    ) -> Any:
        """Order GPU dependencies, launch dispatch, and record its completion.

        The native runtime owns the output pools. Stream registration must
        protect their manual reuse in addition to allocator record_stream.
        This method never waits for completion on the calling CPU thread.
        """
        with torch.cuda.stream(self._comm_stream):
            self._wait_for_dispatch_inputs(compute, previous_event)
            _record_tensors((x, topk_idx, topk_weights), self._comm_stream)
            arguments = (x, topk_idx, topk_weights, self.config.stage, alignment, do_expand, do_zero_padding)
            if self.config.advanced:
                result = self.runtime.dispatch(*arguments, cached_layout, False)
            else:
                result = self.runtime.dispatch(*arguments)
            if cumulative_local_expert_recv_stats is not None:
                cumulative_local_expert_recv_stats.add_(result["expert_counts"])
                cumulative_local_expert_recv_stats.record_stream(self._comm_stream)
            event = torch.cuda.Event()
            event.record(self._comm_stream)
        return result, event

    def _wait_for_dispatch_inputs(self, compute: Any, previous_event: Any) -> None:
        """Shared eager/graph producer ordering and output-consumer protection."""
        if previous_event is None:
            self._comm_stream.wait_stream(compute)
        elif hasattr(previous_event, "current_stream_wait"):
            previous_event.current_stream_wait()
        else:
            self._comm_stream.wait_event(previous_event)
        self._wait_consumer_streams(compute, compute_already_waited=previous_event is None)

    def capture_dispatch(self, *args: Any, **kwargs: Any) -> Any:
        """Prepare one fixed-address S6+ GraphDispatch collectively.

        Preparation may synchronize; returned replay() only enqueues GPU work.
        Dynamic routing may change in place. A supplied cached handle instead
        requires its original route identity and version on every replay.
        """
        from ..graph import capture_dispatch

        return capture_dispatch(self, *args, **kwargs)

    def cache_layout(self, handle: HostStagedEPHandle) -> Any:
        """Detach immutable S6 routing state without retaining an output pool.

        Pass the returned CachedLayout as dispatch(handle=...) and omit
        topk_idx. The original route tensor must remain unchanged; x and
        topk_weights may be updated with the usual producer dependencies.
        """
        self._check_live()
        if self.config.stage < 6:
            raise NotImplementedError("Cached layouts require S6")
        if handle.buffer_id != self.buffer_id or handle.generation != self._generation:
            raise ValueError("The handle belongs to another buffer generation")
        return handle.cache_layout()

    def _validate_input(self, tensor: Any, name: str, dtype: Any, shape: Any) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch Tensor")
        if not tensor.is_cuda or tensor.device.index != self.device:
            raise ValueError(f"{name} must be on this buffer's CUDA device")
        if tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous with dtype {dtype}")
        if tensor.ndim != len(shape) or any(
            expected is not None and actual != expected for actual, expected in zip(tensor.shape, shape)
        ):
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {tuple(shape)}")

    def _check_live(self) -> None:
        if self._closed or self.runtime is None:
            raise RuntimeError("This buffer is closed")
        if self._failed:
            raise RuntimeError("This buffer has failed; do not dispatch again")
        if torch.cuda.current_device() != self.device:
            raise RuntimeError("The current CUDA device differs from the buffer's device")

    def check_errors(self, handle: Optional[HostStagedEPHandle] = None) -> None:
        """Explicit test/diagnostic synchronization; excluded from timing."""
        selected = handle or self._last_handle
        if selected is None:
            return
        if selected.buffer_id != self.buffer_id or selected.generation != self._generation:
            raise ValueError("The handle belongs to a different buffer generation")
        selected._completion_event.synchronize()
        codes = selected.error.detach().cpu().reshape(-1).tolist()
        if any(codes):
            self._failed = True
            raise RuntimeError(f"PCIe dispatch GPU error on rank {self.rank_idx}: {codes}")

    def combine(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This backend provides dispatch only; combine/backward is not implemented")

    def close(self) -> None:
        """Collectively drain, unregister mappings, then unlink shared files.

        If a CUDA wait or the drain barrier fails, mappings are retained: a
        peer might still be accessing them.  This is not crash recovery.
        """
        with self._lock:
            if self._closed:
                return
            if self._graph_capture is not None:
                self._graph_capture.close()
                self._graph_capture = None
            if self._last_event is not None:
                self._last_event.synchronize()
            dist.barrier(group=self.control_group)
            self.runtime.close()
            self.runtime = None
            dist.barrier(group=self.control_group)
            if self.rank_idx == 0:
                shutil.rmtree(self.directory)
            self._closed = True
            self._generation += 1
            self._last_handle = None
            self._consumer_streams.clear()
            if self._owns_control_group:
                dist.destroy_process_group(self.control_group)

    destroy = close

    def __enter__(self) -> "HostStagedEPBuffer":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


EPBuffer = HostStagedEPBuffer
