"""Audit a steady-state dispatch trace for CPU synchronization and count D2H.

torchrun --standalone --nproc-per-node=2 tests/test_profiler.py

Initialization, CUDA profiler start/stop, diagnostic synchronization and Gloo
collectives are outside the named steady-state region.  This is a behavior
audit of the sampled path, not a Python API latency benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from advanced_options import add_transport_arguments, transport_options
from test_correctness import gather_cpu, make_inputs

MARKER = "pcie_ep_steady_dispatch"
CPU_SYNCS = (
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
)
SCALAR_READS = ("aten::item", "aten::_local_scalar_dense", "aten::is_nonzero")


def audit_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    regions = [event for event in events if event.get("name") == MARKER and event.get("ph") == "X"]
    if not regions:
        raise AssertionError("Profiler trace contains no steady dispatch regions")

    def in_region(event):
        timestamp = event.get("ts")
        if timestamp is None:
            return False
        return any(
            region["ts"] <= timestamp <= region["ts"] + region.get("dur", 0) and region.get("pid") == event.get("pid")
            for region in regions
        )

    synchronization = [
        event.get("name")
        for event in events
        if any(name in event.get("name", "") for name in CPU_SYNCS) and in_region(event)
    ]
    scalar_reads = [event.get("name") for event in events if event.get("name") in SCALAR_READS and in_region(event)]
    # Nothing in the profile besides sampled dispatch is allowed to copy GPU
    # data back to the CPU.  Checking the full trace also catches async D2H
    # memcpy execution after its submitting CPU marker has ended.
    dtoh = [
        event.get("name")
        for event in events
        if any(token in event.get("name", "").lower() for token in ("dtoh", "device -> host", "device to host"))
    ]
    runtime_events = [
        event for event in events if event.get("cat") in ("cuda_runtime", "cuda_driver") and in_region(event)
    ]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    if not runtime_events or not kernels:
        raise AssertionError("Trace did not capture CUDA runtime calls and kernels; audit is inconclusive")
    report = {
        "status": "passed" if not (synchronization or scalar_reads or dtoh) else "failed",
        "steady_regions": len(regions),
        "captured_steady_cuda_calls": len(runtime_events),
        "captured_kernel_events": len(kernels),
        "cpu_synchronization_calls": synchronization,
        "gpu_scalar_readback_ops": scalar_reads,
        "device_to_host_memcopies": dtoh,
        "scope": "sampled steady dispatch only; excludes initialization, profiler start/stop and diagnostics",
        "trace": str(path),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_transport_arguments(parser)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--execution-mode", choices=["eager", "cached", "graph", "graph_cached"], default="eager")
    parser.add_argument("--output-dir", type=Path, default=Path("results/profiler"))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    cpu = make_inputs(rank, world, args.tokens, args.hidden, 256, 8, "multicast")
    gpu = tuple(value.to("cuda") for value in cpu)
    config = HostStagedConfig(
        payload_slots=2,
        descriptor_slots=3,
        enable_graph=args.execution_mode.startswith("graph"),
        **transport_options(args),
    )
    try:
        with HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=args.tokens,
            hidden=args.hidden,
            num_topk=8,
            num_experts=256,
            config=config,
        ) as buffer:
            result = buffer.dispatch(*gpu, do_expand=args.expand, expert_alignment=8, do_zero_padding=args.expand)
            buffer.check_errors(result[3])
            cache = buffer.cache_layout(result[3]) if args.execution_mode in ("cached", "graph_cached") else None
            graph = None
            if args.execution_mode.startswith("graph"):
                graph = buffer.capture_dispatch(
                    gpu[0],
                    gpu[1] if cache is None else None,
                    gpu[2],
                    handle=cache,
                    do_expand=args.expand,
                    expert_alignment=8,
                    do_zero_padding=args.expand,
                )
            # CPU diagnostic warmup and profiler setup are not sampled.
            dist.barrier()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                for _ in range(args.iterations):
                    with torch.profiler.record_function(MARKER):
                        if graph is not None:
                            result = graph.replay(async_with_compute_stream=True)
                        else:
                            result = buffer.dispatch(
                                gpu[0],
                                gpu[1] if cache is None else None,
                                gpu[2],
                                handle=cache,
                                do_expand=args.expand,
                                expert_alignment=8,
                                do_zero_padding=args.expand,
                                do_cpu_sync=False,
                                async_with_compute_stream=True,
                            )
                        result[4].current_stream_wait()
            buffer.check_errors(result[3])
            args.output_dir.mkdir(parents=True, exist_ok=True)
            trace = args.output_dir / f"rank-{rank}.trace.json"
            try:
                profiler.export_chrome_trace(str(trace))
                report = audit_trace(trace)
            except Exception as exc:
                report = {"status": "inconclusive", "trace": str(trace), "error": f"{type(exc).__name__}: {exc}"}
            reports = gather_cpu(report)
            if rank == 0:
                (args.output_dir / "audit.json").write_text(
                    json.dumps(
                        {
                            "world_size": world,
                            "expand": args.expand,
                            "execution_mode": args.execution_mode,
                            "transport_options": transport_options(args),
                            "reports": reports,
                        },
                        indent=2,
                    )
                )
                print(f"steady dispatch profiler audit: {[item['status'] for item in reports]}", flush=True)
            if any(item["status"] != "passed" for item in reports):
                raise AssertionError(
                    "Profiler found CPU synchronization, scalar readback or D2H during steady dispatch"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
