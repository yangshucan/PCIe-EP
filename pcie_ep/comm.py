"""Communicator borrowing and collective host-memory initialization.

Only initialization and teardown use CPU collectives.  Dispatch never calls
these helpers, and no helper creates or destroys a managed NCCL communicator.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .config import HostStagedConfig


@dataclass(frozen=True)
class BorrowedNCCLComm:
    """Non-owning communicator pointer and the result of the reuse attempt."""

    nccl_comm: Optional[int] = None
    reused: bool = False
    status: str = "not_an_nccl_group"


def borrow_nccl_comm(group: Any, device: int, require_reuse: bool = False) -> BorrowedNCCLComm:
    """Borrow an existing PyTorch NCCL communicator without a fallback init.

    PyTorch versions without ``_comm_ptr`` remain usable in non-strict mode:
    the payload transport only needs the Gloo control group and host mappings.
    """
    result = BorrowedNCCLComm()
    if "nccl" in str(dist.get_backend(group)).lower():
        backend = group._get_backend(torch.device("cuda", device))
        comm_ptr = getattr(backend, "_comm_ptr", None)
        if comm_ptr is None:
            result = BorrowedNCCLComm(status="_comm_ptr unavailable")
        else:
            try:
                pointer = int(comm_ptr())
                reused = bool(pointer)
                result = BorrowedNCCLComm(
                    nccl_comm=pointer,
                    reused=reused,
                    status="borrowed" if reused else "uninitialized",
                )
            except (RuntimeError, TypeError) as exc:
                result = BorrowedNCCLComm(status=f"unavailable: {exc}")
    if require_reuse and not result.reused:
        raise RuntimeError(f"Cannot borrow an existing NCCL communicator: {result.status}")
    return result


def get_control_group(group: Any, global_ranks: Sequence[int], control_group: Any = None) -> Tuple[Any, bool]:
    """Select a matching CPU-only group and report whether the caller owns it."""
    owns_control_group = False
    if control_group is None:
        if str(dist.get_backend(group)).lower() == "gloo":
            control_group = group
        else:
            control_group = dist.new_group(
                ranks=list(global_ranks),
                backend="gloo",
                use_local_synchronization=True,
            )
            owns_control_group = True
    if str(dist.get_backend(control_group)).lower() != "gloo":
        raise ValueError("control_group must use Gloo; GPU collectives are not allowed")
    if tuple(dist.get_process_group_ranks(control_group)) != tuple(global_ranks):
        raise ValueError("control_group must have the same ordered EP ranks")
    return control_group, owns_control_group


def validate_group_configuration(
    control_group: Any, world: int, tmax: int, hidden: int, topk: int, experts: int, config: HostStagedConfig
) -> None:
    """Reject multi-host groups and mismatched shared-memory layouts."""
    state = (
        socket.gethostname(),
        tmax,
        hidden,
        topk,
        experts,
        config.payload_slots,
        config.descriptor_slots,
        config.stage,
        config.timeout_ms,
        tuple(config.native_options().items()),
    )
    states: list[Any] = [None] * world
    dist.all_gather_object(states, state, group=control_group)
    if any(other != state for other in states):
        raise ValueError("All EP ranks must be on one host with identical buffer configuration")


def create_shared_segments(
    extension: Any,
    rank: int,
    world: int,
    global_ranks: Sequence[int],
    tmax: int,
    hidden: int,
    topk: int,
    experts: int,
    config: HostStagedConfig,
    control_group: Any,
) -> Tuple[str, str]:
    """Create backing files once and broadcast offsets/identity, never pointers."""
    info: list[Any] = [None]
    if rank == 0:
        directory = None
        try:
            base = config.directory or ("/dev/shm" if Path("/dev/shm").is_dir() else None)
            if base is not None:
                Path(base).mkdir(parents=True, exist_ok=True)
            directory = tempfile.mkdtemp(prefix="pcie-ep-", dir=base)
            extension.create_segments(
                directory,
                world,
                tmax,
                hidden,
                topk,
                experts,
                config.payload_slots,
                config.descriptor_slots,
                config.stage,
            )
            info[0] = {"directory": directory, "id": uuid.uuid4().hex}
        except Exception as exc:
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
            info[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(info, src=global_ranks[0], group=control_group)
    if "error" in info[0]:
        raise RuntimeError(f"Shared staging initialization failed: {info[0]['error']}")
    return info[0]["directory"], info[0]["id"]


def _prepare_segments(
    extension: Any,
    directory: str,
    rank: int,
    world: int,
    tmax: int,
    hidden: int,
    topk: int,
    experts: int,
    slots: int,
    descriptor_slots: int,
    stage: int,
    control_group: Any,
) -> None:
    """First-touch each source on its owner before any peer CUDA-registers it."""
    error = None
    try:
        extension.prepare_segment(
            directory,
            rank,
            world,
            tmax,
            hidden,
            topk,
            experts,
            slots,
            descriptor_slots,
            stage,
        )
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors: list[Any] = [None] * world
    dist.all_gather_object(errors, error, group=control_group)
    # Even on failure, every first-touch worker must finish before an owner
    # removes the backing files or a peer attempts CUDA registration.
    dist.barrier(group=control_group)
    if any(errors):
        raise RuntimeError(f"Shared staging preparation failed: {errors}")


def register_runtime(
    extension: Any,
    directory: str,
    rank: int,
    world: int,
    tmax: int,
    hidden: int,
    topk: int,
    experts: int,
    config: HostStagedConfig,
    control_group: Any,
    comm_stream: Any,
    device: int,
) -> Any:
    """Register mappings, order CUDA initialization, and agree on success.

    A registration failure is reported collectively.  A successfully created
    local runtime is closed only after all peers have finished constructing.
    """
    runtime = None
    init_error = None
    try:
        runtime = extension.Runtime(
            directory,
            rank,
            world,
            tmax,
            hidden,
            topk,
            experts,
            config.payload_slots,
            config.descriptor_slots,
            config.timeout_ms,
            config.stage,
        )
        # The epoch initializer runs on the construction stream; the first
        # dispatch may come from another stream and must retain this ordering.
        comm_stream.wait_stream(torch.cuda.current_stream(device))
    except Exception as exc:
        init_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors: list[Any] = [None] * world
    dist.all_gather_object(errors, init_error, group=control_group)
    if any(errors):
        if runtime is not None:
            runtime.close()
        dist.barrier(group=control_group)
        if rank == 0:
            shutil.rmtree(directory)
        raise RuntimeError(f"GPU host-memory registration failed: {errors}")
    return runtime
