"""S4 stream, lifetime, error, and optional NCCL-borrowing integration checks.

Run with torchrun, for example:
  torchrun --standalone --nproc-per-node=2 tests/test_integration.py
  torchrun --standalone --nproc-per-node=8 tests/test_integration.py --nccl-reuse

All normal control collectives use Gloo.  --nccl-reuse additionally initializes
an existing NCCL communicator with a one-element Socket-only all-reduce outside
dispatch and timing.  Missing private _comm_ptr is reported as unavailable,
never as a successful communicator-reuse test.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

# The optional NCCL initialization must not use NVLink/P2P/NVLS/MNNVL/SHM/IB.
# Dispatch itself does not invoke NCCL or any GPU peer-memory operation.
for _name, _value in {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_MNNVL_ENABLE": "0",
    "NCCL_NET": "Socket",
}.items():
    os.environ[_name] = _value
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")

import torch
import torch.distributed as dist

from pcie_ep import EventOverlap, HostStagedConfig, HostStagedEPBuffer
from advanced_options import add_transport_arguments, transport_options
from test_correctness import gather_cpu, make_inputs, verify_result


def new_buffer(args, *, group=None, require_reuse_nccl_comm=False):
    return HostStagedEPBuffer(
        group=dist.group.WORLD if group is None else group,
        control_group=dist.group.WORLD,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        num_experts=args.experts,
        config=HostStagedConfig(
            payload_slots=2, descriptor_slots=3, timeout_ms=args.timeout_ms, **transport_options(args)
        ),
        require_reuse_nccl_comm=require_reuse_nccl_comm,
    )


def cpu_inputs(args, seed=7, increment=0):
    rank, world = dist.get_rank(), dist.get_world_size()
    values = make_inputs(rank, world, args.tokens, args.hidden, args.experts, args.topk, "multicast", seed)
    if increment:
        values = ((values[0] + increment).to(torch.bfloat16), values[1], values[2])
    return values, gather_cpu(values)


def gpu_inputs(values):
    return tuple(value.to(device="cuda", non_blocking=True) for value in values)


def verify(args, buffer, result, inputs, *, expand=False, alignment=1):
    buffer.check_errors(result[3])
    verify_result(result, inputs, dist.get_rank(), dist.get_world_size(), args.experts, args.topk, expand, alignment)


def snapshot_after_wait(result, *, wait=True):
    # Capture the payload on the consuming stream before any CPU diagnostic
    # sync, so missing stream dependencies cannot be hidden by a later sync.
    if wait:
        result[4].current_stream_wait()
    return (
        result[0].clone(),
        result[1].clone(),
        result[2].clone() if result[2] is not None else None,
        result[3],
        result[4],
    )


def stream_cases(args):
    values, inputs = cpu_inputs(args)
    construction = torch.cuda.Stream()
    with torch.cuda.stream(construction):
        buffer = new_buffer(args)
    try:
        # The first invocation is on a stream distinct from construction.
        compute = torch.cuda.Stream()
        with torch.cuda.stream(compute):
            x, idx, weights = gpu_inputs(values)
            result = buffer.dispatch(x, idx, weights)
            copied = snapshot_after_wait(result, wait=False)
        verify(args, buffer, copied, inputs)
        del copied, result

        # An explicit producer event is the only dependency from producer to
        # communication.  The consuming stream is independent as well.
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            x, idx, weights = gpu_inputs(values)
            torch.cuda._sleep(100_000)
            x.add_(1)
            previous = HostStagedEPBuffer.capture()
        expected = [(x_cpu.add(1).to(torch.bfloat16), i_cpu, w_cpu) for x_cpu, i_cpu, w_cpu in inputs]
        result = buffer.dispatch(
            x, idx, weights, previous_event=previous, allocate_on_comm_stream=True, async_with_compute_stream=True
        )
        with torch.cuda.stream(consumer):
            copied = snapshot_after_wait(result)
        verify(args, buffer, copied, expected)
        del copied, result, previous

        # Both default do_cpu_sync=False and compatibility None stay async.
        x, idx, weights = gpu_inputs(values)
        result = buffer.dispatch(x, idx, weights, do_cpu_sync=None)
        copied = snapshot_after_wait(result, wait=False)
        verify(args, buffer, copied, inputs)
        del copied, result

        # Deferred epilogue exposes four outputs once, and uses only a device
        # event wait.  The wrapper must not recompute layout on its second wait.
        event = buffer.dispatch(
            x,
            idx,
            weights,
            do_expand=True,
            expert_alignment=8,
            do_zero_padding=True,
            async_with_compute_stream=True,
            defer_epilogue=True,
        )
        if not isinstance(event, EventOverlap):
            raise AssertionError("defer_epilogue did not return EventOverlap")
        with torch.cuda.stream(consumer):
            outputs = event.wait()
            if not isinstance(outputs, tuple) or len(outputs) != 4:
                raise AssertionError("deferred wait did not return four outputs")
            copied = snapshot_after_wait((*outputs, event))
            if event.wait() is not None:
                raise AssertionError("deferred epilogue hook ran more than once")
        verify(args, buffer, copied, inputs, expand=True, alignment=8)
    finally:
        buffer.close()
    return {
        "status": "passed",
        "cases": [
            "construction_stream",
            "nondefault_compute_stream",
            "previous_event",
            "independent_consumer_stream",
            "cpu_sync_none",
            "defer_epilogue",
        ],
    }


def pool_case(args):
    values, inputs = cpu_inputs(args)
    next_values, next_inputs = cpu_inputs(args, increment=2)
    capacity = 3
    with new_buffer(args) as buffer:
        retained = []
        for index in range(capacity):
            current = buffer.dispatch(*gpu_inputs(values if index % 2 == 0 else next_values))
            verify(args, buffer, current, inputs if index % 2 == 0 else next_inputs)
            retained.append(current)
        del current
        count = int(retained[0][3].unique_count.cpu())
        snapshot = retained[0][0][:count].clone()

        def assert_exhausted():
            try:
                buffer.dispatch(*gpu_inputs(values))
            except RuntimeError as exc:
                if "pool exhausted" not in str(exc):
                    raise
            else:
                raise AssertionError("Retained outputs did not exhaust the prepared pool")

        assert_exhausted()
        if not torch.equal(retained[0][0][:count], snapshot):
            raise AssertionError("A later dispatch overwrote a retained result")
        del retained[0]
        gc.collect()
        reused = buffer.dispatch(*gpu_inputs(values))
        verify(args, buffer, reused, inputs)
        for index, result in enumerate(retained, start=1):
            verify(args, buffer, result, inputs if index % 2 == 0 else next_inputs)
        del result
        weights_only = reused[2]
        weights_snapshot = weights_only[:count].clone()
        del reused, retained
        gc.collect()
        retained = [buffer.dispatch(*gpu_inputs(next_values)) for _ in range(capacity - 1)]
        for result in retained:
            verify(args, buffer, result, next_inputs)
        del result
        assert_exhausted()
        if not torch.equal(weights_only[:count], weights_snapshot):
            raise AssertionError("Retained standalone weights were overwritten")
        del weights_only
        gc.collect()
        final = buffer.dispatch(*gpu_inputs(values))
        verify(args, buffer, final, inputs)
    return {
        "status": "passed",
        "retained_results": capacity,
        "overflow": "explicit error",
        "release_then_reuse": "passed",
        "retained_result_unchanged": True,
        "standalone_retained_weights": "passed",
    }


def error_case(args, kind):
    values, _ = cpu_inputs(args)
    with new_buffer(args) as buffer:
        x, idx, weights = gpu_inputs(values)
        # One bad producer must cause peer aborts without out-of-bounds copies.
        if dist.get_rank() == 0:
            if kind == "duplicate":
                idx[0, 1] = idx[0, 0]
            elif kind == "negative":
                idx[0, 0] = -2
            else:
                idx[0, 0] = args.experts
        result = buffer.dispatch(x, idx, weights, async_with_compute_stream=True)
        try:
            result[3].check_errors()
        except RuntimeError as exc:
            if "GPU error" not in str(exc):
                raise
        else:
            raise AssertionError("Invalid routing did not report a GPU error")
        codes = result[3].error.detach().cpu().tolist()
        try:
            buffer.dispatch(x, idx, weights)
        except RuntimeError as exc:
            if "failed" not in str(exc):
                raise
        else:
            raise AssertionError("A failed buffer accepted another dispatch")
        directory = Path(buffer.directory)
    dist.barrier()
    if directory.exists():
        raise AssertionError("Collective close left failed-buffer shared backing files")
    return {
        "status": "passed",
        "route_error": kind,
        "rank_error_codes": gather_cpu(codes),
        "sticky_abort": True,
        "collective_close": "passed",
    }


def nccl_case(args):
    group = dist.new_group(ranks=list(range(dist.get_world_size())), backend="nccl")
    try:
        # This Socket-only warmup initializes the application-owned NCCL comm;
        # it is outside transport dispatch, measurements, and reuse claims.
        warmup = torch.tensor([dist.get_rank() + 1.0], device="cuda")
        dist.all_reduce(warmup, group=group)
        if warmup.item() != dist.get_world_size() * (dist.get_world_size() + 1) / 2:
            raise AssertionError("NCCL Socket-only initialization warmup failed")
        values, inputs = cpu_inputs(args)
        with new_buffer(args, group=group) as buffer:
            available = buffer.comm_reused
            statuses = gather_cpu({"reused": available, "status": buffer.comm_reuse_status})
            if any(item["reused"] != available for item in statuses):
                raise AssertionError("NCCL borrowing availability differs among ranks")
            result = buffer.dispatch(*gpu_inputs(values))
            verify(args, buffer, result, inputs)
        if available:
            with new_buffer(args, group=group, require_reuse_nccl_comm=True) as strict:
                if strict.nccl_comm is None or not strict.comm_reused:
                    raise AssertionError("Strict borrowing did not retain the existing communicator")
            outcome = "passed"
        else:
            try:
                new_buffer(args, group=group, require_reuse_nccl_comm=True)
            except RuntimeError as exc:
                if "Cannot borrow" not in str(exc):
                    raise
            else:
                raise AssertionError("Strict communicator reuse silently accepted unavailable _comm_ptr")
            outcome = "unavailable"
        return {
            "status": outcome,
            "ranks": statuses,
            "host_transport_on_nccl_group": "passed",
            "initialization_collective": "one-element Socket-only NCCL all_reduce",
            "strict_reuse_check": "passed",
            "created_managed_nccl_comm": False,
        }
    finally:
        dist.destroy_process_group(group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_transport_arguments(parser)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--nccl-reuse", action="store_true")
    parser.add_argument("--only", choices=["all", "streams", "pools", "errors", "nccl"], default="all")
    parser.add_argument("--output", type=Path, default=Path("results/integration.json"))
    args = parser.parse_args()
    if args.tokens < 1 or args.topk < 2:
        parser.error("integration checks require tokens >= 1 and topk >= 2")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    records = {}
    cases = []
    if args.only in ("all", "streams"):
        cases.append(("streams", lambda: stream_cases(args)))
    if args.only in ("all", "pools"):
        cases.append(("output_pools", lambda: pool_case(args)))
    if args.only in ("all", "errors"):
        for kind in ("out_of_range", "negative", "duplicate"):
            cases.append((f"invalid_route_{kind}", lambda kind=kind: error_case(args, kind)))
    if args.nccl_reuse or args.only == "nccl":
        cases.append(("nccl_borrowing", lambda: nccl_case(args)))
    try:
        for name, function in cases:
            failure, result = None, None
            try:
                result = function()
            except Exception as exc:
                failure = f"rank {rank}: {type(exc).__name__}: {exc}"
            failures = [value for value in gather_cpu(failure) if value]
            records[name] = {"status": "failed", "errors": failures} if failures else result
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "world_size": world,
                            "records": records,
                            "nccl_safety_env": {
                                name: os.environ[name]
                                for name in (
                                    "NCCL_P2P_DISABLE",
                                    "NCCL_SHM_DISABLE",
                                    "NCCL_IB_DISABLE",
                                    "NCCL_NVLS_ENABLE",
                                    "NCCL_MNNVL_ENABLE",
                                    "NCCL_NET",
                                )
                            },
                        },
                        indent=2,
                    )
                )
                print(f"integration {name}: {records[name]['status']}", flush=True)
            if failures:
                raise AssertionError("; ".join(failures))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
