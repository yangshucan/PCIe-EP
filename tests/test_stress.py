"""S3/S4 queue wrap, delayed-reader, and long-running epoch stress.

torchrun --standalone --nproc-per-node=8 tests/test_stress.py --iterations 100000

Dispatches are queued in bounded chunks. Synchronization and CPU reference
checks are explicit test diagnostics, outside any production dispatch path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from advanced_options import add_transport_arguments, transport_options
from test_correctness import gather_cpu, make_inputs, reference, verify_result


def stress_configuration(args, rank, world, device, payload_slots, descriptor_slots):
    config = HostStagedConfig(
        payload_slots=payload_slots,
        descriptor_slots=descriptor_slots,
        timeout_ms=args.timeout_ms,
        **transport_options(args),
    )
    buffers = [
        HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=args.tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            num_experts=args.experts,
            config=config,
        )
        for _ in range(args.buffers)
    ]
    test_inputs, references, expected_writes = [], [], []
    # Different payloads/routing distinguish adjacent epochs and two buffers.
    for seed, pattern in [(41, "random"), (99, "multicast"), (17, "hotspot"), (61, "partial"), (3, "none")]:
        count = args.tokens if pattern != "partial" else max(0, args.tokens - rank)
        cpu_input = make_inputs(rank, world, count, args.hidden, args.experts, args.topk, pattern, seed)
        references.append(gather_cpu(cpu_input))
        test_inputs.append(tuple(item.to(device) for item in cpu_input))
        expected_writes.append(reference(references[-1], rank, world, args.experts, args.topk, args.alignment)[4])
    stats_snapshots = []
    last_results = [None] * len(buffers)
    dispatch_count = source_token_reuses = 0
    started = time.perf_counter()
    try:
        for chunk_start in range(0, args.iterations, args.chunk):
            chunk_end = min(chunk_start + args.chunk, args.iterations)
            latest = {}
            for iteration in range(chunk_start, chunk_end):
                buffer_index = iteration % len(buffers)
                pattern = iteration % len(test_inputs)
                x, idx, weights = test_inputs[pattern]
                if args.delay_cycles and rank == world - 1 and iteration % args.delay_every == 0:
                    torch.cuda._sleep(args.delay_cycles)
                expand = bool(iteration & 1)
                result = buffers[buffer_index].dispatch(
                    x,
                    topk_idx=idx,
                    topk_weights=weights,
                    do_expand=expand,
                    expert_alignment=args.alignment,
                    do_zero_padding=expand,
                    do_cpu_sync=False,
                    async_with_compute_stream=True,
                )
                result[4].current_stream_wait()
                # Snapshot status for every dispatch before the runtime cycles
                # its output pool; validation cannot miss a transient failure.
                stats_snapshots.append(result[3].error.clone())
                last_results[buffer_index] = result
                latest[buffer_index] = (pattern, expand)
                source_token_reuses += expected_writes[pattern]
                dispatch_count += 1
            torch.cuda.synchronize()
            error = torch.stack(stats_snapshots).cpu()
            chunk_failure = None
            if torch.count_nonzero(error):
                chunk_failure = f"rank {rank} GPU error in chunk starting {chunk_start}: {error.tolist()}"
            chunk_failures = [item for item in gather_cpu(chunk_failure) if item]
            if chunk_failures:
                raise AssertionError("; ".join(chunk_failures))
            stats_snapshots.clear()
            failures = []
            for index, (pattern, expand) in latest.items():
                try:
                    buffers[index].check_errors(last_results[index][3])
                    verify_result(
                        last_results[index],
                        references[pattern],
                        rank,
                        world,
                        args.experts,
                        args.topk,
                        expand,
                        args.alignment,
                    )
                except Exception as exc:
                    failures.append(f"rank {rank}, buffer {index}, iteration {chunk_end}: {exc}")
            all_failures = [failure for items in gather_cpu(failures) for failure in items]
            if all_failures:
                raise AssertionError("; ".join(all_failures))
            if rank == 0 and (chunk_end == args.iterations or chunk_end % args.progress_every < args.chunk):
                print(f"stress Q={payload_slots} L={descriptor_slots}: {chunk_end}/{args.iterations}", flush=True)
        elapsed = time.perf_counter() - started
        all_reuses = gather_cpu(source_token_reuses)
        return {
            "payload_slots": payload_slots,
            "descriptor_slots": descriptor_slots,
            "dispatches_per_rank": dispatch_count,
            "buffers": args.buffers,
            "payload_slot_writes_per_rank": all_reuses,
            "payload_slot_generations_per_rank_lower_bound": [
                count // (payload_slots * args.channels * args.buffers) for count in all_reuses
            ],
            "minimum_slot_writes_per_rank": min(all_reuses),
            "delayed_rank": world - 1 if args.delay_cycles else None,
            "delay_cycles": args.delay_cycles,
            "elapsed_seconds": elapsed,
            "status": "passed",
        }
    finally:
        for buffer in reversed(buffers):
            buffer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_transport_arguments(parser)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--alignment", type=int, default=8)
    parser.add_argument("--payload-slots", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--descriptor-slots", type=int, nargs="+", default=[1, 3, 2])
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--delay-cycles", type=int, default=200000)
    parser.add_argument("--delay-every", type=int, default=97)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--output", type=Path, default=Path("results/stress.json"))
    args = parser.parse_args()
    if len(args.payload_slots) != len(args.descriptor_slots):
        parser.error("payload-slots and descriptor-slots lists must have equal lengths")
    if min(args.iterations, args.buffers, args.tokens, args.chunk, args.delay_every, args.progress_every) < 1:
        parser.error("iteration, buffer, token, chunk, and period counts must be positive")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    records = []
    try:
        for payload_slots, descriptor_slots in zip(args.payload_slots, args.descriptor_slots):
            record = stress_configuration(args, rank, world, device, payload_slots, descriptor_slots)
            records.append(record)
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "world_size": world,
                            "hidden": args.hidden,
                            "control_backend": "gloo",
                            "nvlink_payload_allowed": False,
                            "records": records,
                        },
                        indent=2,
                    )
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
