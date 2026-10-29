"""Focused profiling of the same workload as benchmark.py.

Use --profile-kind nsys under Nsight Systems with cudaProfilerApi capture.
Use --profile-kind torch for per-rank Chrome traces, or gpu for opt-in
device role counters. No diagnostic readback occurs inside dispatch.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import benchmark as bench
from pcie_ep import HostStagedConfig, HostStagedEPBuffer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-kind", choices=("torch", "nsys", "gpu"), default="gpu")
    profile, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = bench.parse_args()
    if args.execution_mode != "eager":
        raise ValueError("Focused driver currently measures eager dispatch only")
    if profile.profile_kind == "gpu" and (
        not args.expand or not args.warp_specialization or args.receive_expand_mode != "pipelined"
        or not args.fuse_receive_expand or args.wire_dtype != "bf16" or args.numa_mode != "direct"
    ):
        raise ValueError("GPU role profiling requires BF16, direct NUMA, pipelined expanded specialization")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    config = HostStagedConfig(
        payload_slots=args.payload_slots, descriptor_slots=args.descriptor_slots, **bench.transport_options(args)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        with HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=max(args.tokens),
            hidden=args.hidden,
            num_topk=args.topk,
            num_experts=args.experts,
            config=config,
        ) as buffer:
            for tokens in args.tokens:
                idx_cpu, weights_cpu = bench.routes(tokens, args.experts, args.topk, rank, args.seed)
                idx, weights = idx_cpu.cuda(), weights_cpu.cuda()
                generator = torch.Generator(device="cuda").manual_seed(args.seed + rank)
                x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device="cuda", generator=generator)
                for _ in range(args.warmup):
                    result = bench.dispatch(buffer, x, idx, weights, args)
                torch.cuda.synchronize()
                buffer.check_errors()
                dist.barrier()
                captured = []
                starts, ends = [], []
                profiler = (
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False,
                        with_stack=False,
                    )
                    if profile.profile_kind == "torch" else contextlib.nullcontext()
                )
                if profile.profile_kind == "nsys":
                    torch.cuda.cudart().cudaProfilerStart()
                with profiler as trace:
                    # CUPTI activation can be skewed across processes. Prime
                    # collection, then align ranks outside measured markers.
                    if profile.profile_kind in ("torch", "nsys"):
                        for _ in range(2):
                            result = bench.dispatch(buffer, x, idx, weights, args)
                        torch.cuda.synchronize()
                        dist.barrier()
                    for iteration in range(args.iterations):
                        marker = f"dispatch_T{tokens}_i{iteration}"
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        with torch.profiler.record_function(marker):
                            torch.cuda.nvtx.range_push(marker)
                            start.record()
                            result = bench.dispatch(buffer, x, idx, weights, args)
                            end.record()
                            torch.cuda.nvtx.range_pop()
                        starts.append(start)
                        ends.append(end)
                        handle = result[3]
                        data = {"prefix_tx": handle.prefix_tx_timestamps_ns.clone()}
                        role = getattr(handle, "profiling", None)
                        if role is not None:
                            data["roles"] = role.clone()
                        captured.append(data)
                    ends[-1].synchronize()
                    torch.cuda.synchronize()
                if profile.profile_kind == "nsys":
                    torch.cuda.cudart().cudaProfilerStop()
                buffer.check_errors()
                if trace is not None:
                    trace.export_chrome_trace(str(args.output / f"T{tokens}_rank{rank}.trace.json"))
                # GPU readback and CPU/Gloo coordination are outside capture.
                records.append({
                    "tokens": tokens,
                    "role_data_present": getattr(handle, "profiling", None) is not None,
                    "dispatch_us": [a.elapsed_time(b) * 1000 for a, b in zip(starts, ends)],
                    "samples": [{key: value.cpu().tolist() for key, value in item.items()} for item in captured],
                    "stats": bench.read_statistics(handle),
                })
                if rank == 0:
                    print(f"profile={profile.profile_kind} T={tokens} completed", flush=True)
        (args.output / f"rank{rank}.json").write_text(json.dumps({
            "rank": rank, "world": world, "kind": profile.profile_kind,
            "config": vars(args) | {"output": str(args.output)}, "records": records,
        }, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
