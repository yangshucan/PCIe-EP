"""Pool reuse must wait for a GPU consumer after all host output refs vanish.

torchrun --standalone --nproc-per-node=2 tests/test_consumer_lifetime.py

Consumer B delays its payload read.  Python releases the first output and
submits enough later dispatches on compute A to overwrite both pool slots.
The captured B payload must still match the original dispatch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from advanced_options import add_transport_arguments, transport_options
from test_correctness import gather_cpu, make_inputs, reference


def run_case(args, use_previous_event=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    cpu = make_inputs(rank, world, args.tokens, args.hidden, 256, 8, "multicast")
    inputs = gather_cpu(cpu)
    expected_rows = reference(inputs, rank, world, 256, 8, 1)[0]
    expected = torch.stack([row[2] for row in expected_rows])
    x, idx, weights = (value.to("cuda") for value in cpu)
    changed_inputs = [x + (iteration + 1) * 4 for iteration in range(args.overwrites)]
    config = HostStagedConfig(payload_slots=2, descriptor_slots=3, **transport_options(args))
    with HostStagedEPBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=8,
        num_experts=256,
        config=config,
    ) as buffer:
        # This event precedes the old-output consumer.  It covers every new
        # input, so any additional compute-stream wait is specifically needed
        # to protect the old pool rather than prepare the next input.
        previous = HostStagedEPBuffer.capture() if use_previous_event else None
        first = buffer.dispatch(x, idx, weights, async_with_compute_stream=True)
        consumer = torch.cuda.current_stream() if use_previous_event else torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            first[4].current_stream_wait()
            torch.cuda._sleep(args.delay_cycles)
            captured = first[0][: len(expected_rows)].clone()
        # The buffer retains its last handle until the second call.  After
        # that call, the first pool has no external Tensor refs, although B's
        # queued GPU read can still be pending.
        del first
        gc.collect()
        for changed in changed_inputs:
            latest = buffer.dispatch(
                changed,
                idx,
                weights,
                previous_event=previous,
                allocate_on_comm_stream=use_previous_event,
                async_with_compute_stream=True,
            )
        buffer.check_errors(latest[3])
        torch.cuda.synchronize()
        if not torch.equal(captured.cpu(), expected):
            raise AssertionError("A later dispatch overwrote payload before independent consumer B read it")
    return {
        "status": "passed",
        "consumer_stream": "compute A" if use_previous_event else "independent B",
        "previous_event_predates_consumer": use_previous_event,
        "producer_stream": "A",
        "released_host_output_references": True,
        "later_dispatches": args.overwrites,
        "delay_cycles": args.delay_cycles,
        "original_payload_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_transport_arguments(parser)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--delay-cycles", type=int, default=100_000_000)
    parser.add_argument("--overwrites", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/consumer_lifetime.json"))
    args = parser.parse_args()
    if args.overwrites < 3:
        parser.error("At least three later dispatches are required to force pool reuse")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    try:
        records = {}
        for name, use_previous in [("independent_consumer", False), ("previous_event_and_compute_consumer", True)]:
            result, error = None, None
            try:
                result = run_case(args, use_previous)
            except Exception as exc:
                error = f"rank {rank}: {type(exc).__name__}: {exc}"
            errors = [value for value in gather_cpu(error) if value]
            records[name] = {"status": "failed", "errors": errors} if errors else result
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"world_size": world, "records": records}, indent=2))
                print(f"consumer lifetime {name}: {records[name]['status']}", flush=True)
            if errors:
                raise AssertionError("; ".join(errors))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
