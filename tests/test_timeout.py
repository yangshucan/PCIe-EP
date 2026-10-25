"""S4 delayed-peer timeout and released-event diagnostic checks.

torchrun --standalone --nproc-per-node=2 tests/test_timeout.py
GPU sleep and CPU readbacks are explicit tests, outside the normal data path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from pcie_ep import EventOverlap, HostStagedConfig, HostStagedEPBuffer
from advanced_options import add_transport_arguments, transport_options
from test_correctness import gather_cpu, make_inputs, verify_result


def make_buffer(args, timeout_ms):
    return HostStagedEPBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=8,
        num_experts=256,
        config=HostStagedConfig(payload_slots=2, descriptor_slots=3, timeout_ms=timeout_ms, **transport_options(args)),
    )


def inputs(args):
    cpu = make_inputs(dist.get_rank(), dist.get_world_size(), args.tokens, args.hidden, 256, 8, "multicast")
    return tuple(item.to("cuda") for item in cpu), gather_cpu(cpu)


def release_case(args):
    gpu, reference = inputs(args)
    with make_buffer(args, 5000) as buffer:
        result = buffer.dispatch(*gpu, async_with_compute_stream=True)
        result[4].current_stream_wait(release_handle=True)
        if result[4].event is not None or result[4].extra_tensors is not None:
            raise AssertionError("EventOverlap did not release its owned references")
        buffer.check_errors(result[3])
        result[3].check_errors()
        replacement = EventOverlap(result[3]._completion_event)
        verify_result((*result[:4], replacement), reference, dist.get_rank(), dist.get_world_size(), 256, 8, False, 1)
    return {
        "status": "passed",
        "release_handle": True,
        "buffer_check_errors": "passed",
        "handle_check_errors": "passed",
    }


def timeout_case(args):
    gpu, _ = inputs(args)
    with make_buffer(args, args.timeout_ms) as buffer:
        if dist.get_rank() == dist.get_world_size() - 1:
            torch.cuda._sleep(args.delay_cycles)
        result = buffer.dispatch(*gpu, async_with_compute_stream=True)
        raised = False
        try:
            buffer.check_errors(result[3])
        except RuntimeError as exc:
            if "GPU error" not in str(exc):
                raise
            raised = True
        if not raised:
            raise AssertionError("Delayed peer did not trigger GPU timeout/abort")
        codes = result[3].error.detach().cpu().tolist()
        all_codes = gather_cpu(codes)
        # 12 is a prefix READY wait timeout.  Other peers may observe the
        # resulting abort (for example 19) before entering that wait.
        if not any(values[0] in (11, 12, 13, 14, 15) for values in all_codes):
            raise AssertionError(f"No timeout status was recorded: {all_codes}")
        try:
            buffer.dispatch(*gpu)
        except RuntimeError as exc:
            if "failed" not in str(exc):
                raise
        else:
            raise AssertionError("Timed-out buffer accepted a subsequent dispatch")
        directory = Path(buffer.directory)
    dist.barrier()
    if directory.exists():
        raise AssertionError("Timed-out shared backing survived collective close")
    return {
        "status": "passed",
        "timeout_ms": args.timeout_ms,
        "delayed_rank": dist.get_world_size() - 1,
        "delay_cycles": args.delay_cycles,
        "rank_error_codes": all_codes,
        "safe_collective_close": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_transport_arguments(parser)
    parser.add_argument("--mode", choices=["both", "release", "timeout"], default="both")
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--timeout-ms", type=int, default=50)
    parser.add_argument("--delay-cycles", type=int, default=500_000_000)
    parser.add_argument("--output", type=Path, default=Path("results/timeout.json"))
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if world < 2:
        raise ValueError("Timeout test requires at least two GPUs")
    records = {}
    cases = []
    if args.mode in ("both", "release"):
        cases.append(("released_event_diagnostics", release_case))
    if args.mode in ("both", "timeout"):
        cases.append(("delayed_peer_timeout", timeout_case))
    try:
        for name, function in cases:
            error, result = None, None
            try:
                result = function(args)
            except Exception as exc:
                error = f"rank {rank}: {type(exc).__name__}: {exc}"
            errors = [value for value in gather_cpu(error) if value]
            records[name] = {"status": "failed", "errors": errors} if errors else result
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"world_size": world, "records": records}, indent=2))
                print(f"{name}: {records[name]['status']}", flush=True)
            if errors:
                raise AssertionError("; ".join(errors))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
