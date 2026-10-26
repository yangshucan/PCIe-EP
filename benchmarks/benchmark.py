"""Eight-rank host-staged dispatch benchmark (control traffic is Gloo/CPU only).

Run with ``torchrun --standalone --nproc-per-node=8 benchmarks/benchmark.py``.
Every reported latency is the maximum of per-rank CUDA-event durations. CPU
validation, topology discovery, reductions, and event synchronization occur
outside the timed intervals. Payload bandwidth excludes metadata and padding.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer


def routes(tokens: int, experts: int, topk: int, rank: int, seed: int):
    """Distinct expert IDs per token; reproducible without CUDA collectives."""
    generator = torch.Generator().manual_seed(seed + rank * 104729 + tokens)
    idx = torch.rand(tokens, experts, generator=generator).topk(topk, dim=1).indices
    weights = torch.rand(tokens, topk, generator=generator)
    weights /= weights.sum(dim=1, keepdim=True).clamp_min(1e-20)
    return idx.contiguous(), weights.contiguous()


def route_volume(idx: torch.Tensor, rank: int, world: int, experts: int):
    owners = idx // (experts // world)
    destinations = torch.zeros(idx.shape[0], world, dtype=torch.bool)
    if idx.numel():
        destinations.scatter_(1, owners, True)
    destinations[:, rank] = False
    return {
        "remote_source_tokens": int(destinations.any(dim=1).sum()),
        "remote_deliveries": int(destinations.sum()),
        "remote_expert_branches": int((owners != rank).sum()),
        "relay_token_copies": (
            sum(
                int(destinations[:, node * (world // 2) : (node + 1) * (world // 2)].any(dim=1).sum())
                for node in range(2)
                if node != rank // (world // 2)
            )
            if world == 8
            else 0
        ),
    }


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * p / 100
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def distribution(values):
    return {
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "p50_us": percentile(values, 50),
        "p95_us": percentile(values, 95),
        "p99_us": percentile(values, 99),
        "max_us": max(values),
    }


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def transport_events(handle):
    """Optional event pair supplied by the implementation, never inferred."""
    pair = getattr(handle, "transport_events", None)
    if pair is not None:
        return pair
    start = getattr(handle, "transport_start", None)
    end = getattr(handle, "transport_end", None)
    return (start, end) if start is not None and end is not None else None


def read_statistics(handle):
    value = getattr(handle, "stats", None)
    if isinstance(value, torch.Tensor):
        return {"raw": value.detach().cpu().tolist()}
    if isinstance(value, dict):
        return {
            key: item.detach().cpu().tolist() if isinstance(item, torch.Tensor) else item for key, item in value.items()
        }
    return value


def dispatch(buffer, x, idx, weights, args, cache=None, graph=None):
    if graph is not None:
        result = graph.replay(async_with_compute_stream=True)
        result[4].current_stream_wait()
        return result
    result = buffer.dispatch(
        x,
        topk_idx=idx if cache is None else None,
        topk_weights=weights,
        handle=cache,
        do_expand=args.expand,
        expert_alignment=args.alignment,
        do_cpu_sync=False,
        async_with_compute_stream=True,
    )
    result[4].current_stream_wait()
    return result


def benchmark_case(buffer, tokens, args, rank, world, device):
    idx_cpu, weights_cpu = routes(tokens, args.experts, args.topk, rank, args.seed)
    volumes = gather(route_volume(idx_cpu, rank, world, args.experts))
    global_volume = {key: sum(item[key] for item in volumes) for key in volumes[0]}
    idx = idx_cpu.to(device)
    weights = weights_cpu.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device=device, generator=generator)
    torch.cuda.synchronize()
    dist.barrier()
    result, cache, graph = None, None, None
    if args.execution_mode in ("cached", "graph_cached"):
        prepared = dispatch(buffer, x, idx, weights, args)
        buffer.check_errors(prepared[3])
        cache = buffer.cache_layout(prepared[3])
        del prepared
    if args.execution_mode.startswith("graph"):
        graph = buffer.capture_dispatch(
            x,
            idx if cache is None else None,
            weights,
            handle=cache,
            do_expand=args.expand,
            expert_alignment=args.alignment,
        )
    for _ in range(args.warmup):
        result = dispatch(buffer, x, idx, weights, args, cache, graph)
    torch.cuda.synchronize()
    buffer.check_errors()
    dist.barrier()

    starts, ends, transport, timestamps = [], [], [], []
    cpu_usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_start = cpu_usage.ru_utime + cpu_usage.ru_stime
    submit_start = time.perf_counter()
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = dispatch(buffer, x, idx, weights, args, cache, graph)
        end.record()
        starts.append(start)
        ends.append(end)
        pair = transport_events(result[3])
        if pair is not None:
            transport.append(pair)
        clock_ns = getattr(result[3], "transport_timestamps_ns", None)
        if clock_ns is not None:
            # Output pools are reused; preserve only the tiny per-round timer.
            # This copy is ordered after the completion wait and after end.
            timestamps.append(clock_ns.clone())
    submit_us = (time.perf_counter() - submit_start) * 1e6 / args.iterations
    ends[-1].synchronize()
    torch.cuda.current_stream().synchronize()
    cpu_usage = resource.getrusage(resource.RUSAGE_SELF)
    process_cpu_seconds = cpu_usage.ru_utime + cpu_usage.ru_stime - cpu_start
    elapsed_wall_seconds = time.perf_counter() - submit_start
    buffer.check_errors()
    local_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    local_transport_us = [start.elapsed_time(end) * 1000 for start, end in transport]
    if len(timestamps) == args.iterations:
        local_transport_us = [float((timer[1] - timer[0]).cpu()) / 1000 for timer in timestamps]
        if any(value <= 0 for value in local_transport_us):
            raise RuntimeError("invalid runtime transport timestamp interval")
    samples = gather(
        {
            "dispatch_us": local_us,
            "transport_us": local_transport_us,
            "submit_us": submit_us,
            "process_cpu_seconds": process_cpu_seconds,
            "measured_wall_seconds": elapsed_wall_seconds,
        }
    )
    elapsed_us = [max(sample["dispatch_us"][i] for sample in samples) for i in range(args.iterations)]
    transport_us = None
    if all(len(sample["transport_us"]) == args.iterations for sample in samples):
        transport_us = [max(sample["transport_us"][i] for sample in samples) for i in range(args.iterations)]

    bytes_per_token = args.hidden * (1 if args.wire_dtype == "fp8_e4m3" else 2)
    if args.wire_dtype == "fp8_e4m3":
        bytes_per_token += 4  # One FP32 token scale, published with payload.
    host_write_bytes = global_volume["remote_source_tokens"] * bytes_per_token
    wire_bytes = global_volume["remote_deliveries"] * bytes_per_token
    bus_bytes = host_write_bytes + wire_bytes
    mean_seconds = statistics.mean(elapsed_us) * 1e-6
    stats = gather(read_statistics(result[3]))
    if all(isinstance(item, dict) and "raw" in item for item in stats):
        actual_writes = sum(int(item["raw"][0]) for item in stats)
        actual_reads = sum(int(item["raw"][1]) for item in stats)
        if actual_writes != global_volume["remote_source_tokens"] or actual_reads != global_volume["remote_deliveries"]:
            raise AssertionError(
                f"runtime dedup counters differ from route volume: writes={actual_writes}, reads={actual_reads}, expected={global_volume}"
            )
    receive_expand_overlap = getattr(result[3], "receive_expand_overlap", None)
    # Diagnostic D2H is deliberately after the timed submission/completion
    # region. It describes only the final measured dispatch, not all samples.
    overlap_per_rank = gather(
        receive_expand_overlap.detach().cpu().tolist() if isinstance(receive_expand_overlap, torch.Tensor) else None
    )
    result_row = {
        "tokens_per_rank": tokens,
        "world_size": world,
        "hidden": args.hidden,
        "experts": args.experts,
        "topk": args.topk,
        "dtype": "bfloat16",
        "wire_dtype": args.wire_dtype,
        "transport_options": transport_options(args),
        "execution_mode": args.execution_mode,
        "receive_expand_mode": args.receive_expand_mode,
        "transport_threads_per_cta": (
            352 if args.expand and args.fuse_receive_expand and args.receive_expand_mode == "pipelined" else 256
        ),
        "do_expand": args.expand,
        "stage": args.stage,
        "payload_slots": args.payload_slots,
        "descriptor_slots": args.descriptor_slots,
        "expert_alignment": args.alignment,
        "num_max_tokens_per_rank": buffer.num_max_tokens_per_rank,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "seed": args.seed,
        **global_volume,
        "host_write_bytes": host_write_bytes,
        "unique_wire_bytes": wire_bytes,
        "host_bus_payload_bytes": bus_bytes,
        "useful_bf16_delivery_bytes": global_volume["remote_deliveries"] * args.hidden * 2,
        "relay_cpu_payload_read_write_bytes": (
            2 * global_volume["relay_token_copies"] * bytes_per_token if args.numa_mode == "relay" else 0
        ),
        "full_dispatch": distribution(elapsed_us),
        "full_dispatch_samples_us": elapsed_us,
        "per_rank_dispatch_samples_us": [sample["dispatch_us"] for sample in samples],
        "cpu_submission_us_per_rank": [sample["submit_us"] for sample in samples],
        "process_cpu_seconds_per_rank": [sample["process_cpu_seconds"] for sample in samples],
        "measured_wall_seconds_per_rank": [sample["measured_wall_seconds"] for sample in samples],
        "process_cpu_core_equivalents_per_rank": [
            sample["process_cpu_seconds"] / sample["measured_wall_seconds"] for sample in samples
        ],
        "unique_wire_GBps_aggregate": wire_bytes / mean_seconds / 1e9,
        "unique_wire_GBps_per_gpu_average": wire_bytes / mean_seconds / 1e9 / world,
        "host_bus_payload_GBps_aggregate": bus_bytes / mean_seconds / 1e9,
        "host_bus_payload_GBps_per_gpu_average": bus_bytes / mean_seconds / 1e9 / world,
        "runtime_stats_per_rank": stats,
        "receive_expand_overlap_per_rank": overlap_per_rank,
        "receive_expand_overlap_definition": (
            "Last measured dispatch only, shape [channels,2] per rank: column 0 counts RX producer "
            "warp 0 beginning ticket N before the preceding ticket's designated scatter warp 2 "
            "has published completion (a conservative overlap observation); column 1 is the "
            "monotonic scatter-warp-2 completed-ticket+1 watermark, not an event count. "
            "Only sum column 0. No duration or utilization implied. Pipelined mode counts these "
            "events; vectorized mode reports zeros; scalar/nof2/unique report null."
        ),
        "transport_only": distribution(transport_us) if transport_us else None,
    }
    if transport_us:
        result_row["transport_only_samples_us"] = transport_us
        seconds = statistics.mean(transport_us) * 1e-6
        result_row["transport_unique_wire_GBps_aggregate"] = wire_bytes / seconds / 1e9
        result_row["transport_host_bus_payload_GBps_aggregate"] = bus_bytes / seconds / 1e9
    if args.numa_mode == "relay":
        diagnostics = gather(buffer.runtime.relay_diagnostics())
        expected_copies = global_volume["relay_token_copies"]
        actual_copies = sum(item["copies"] for item in diagnostics)
        if actual_copies != expected_copies:
            raise AssertionError(f"NUMA relay copies={actual_copies}, expected={expected_copies}")
        result_row["relay_diagnostics_per_rank"] = diagnostics
    if graph is not None:
        graph.close()
    return result_row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--stage", type=int, choices=(6, 7, 8), default=6)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--warp-specialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pipeline-stages", type=int, choices=[2, 3], default=2)
    parser.add_argument("--fuse-receive-expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--receive-expand-mode", choices=["scalar", "vectorized", "pipelined"], default="pipelined")
    parser.add_argument("--source-fusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wire-dtype", choices=["bf16", "fp8_e4m3"], default="bf16")
    parser.add_argument("--numa-mode", choices=["direct", "relay"], default="direct")
    parser.add_argument("--replica-slots", type=int, default=256)
    parser.add_argument("--relay-workers", type=int, default=1)
    parser.add_argument("--relay-cpu-affinity", type=int, nargs="*", default=[])
    parser.add_argument("--execution-mode", choices=["eager", "cached", "graph", "graph_cached"], default="eager")
    parser.add_argument("--payload-slots", type=int, default=256)
    parser.add_argument("--descriptor-slots", type=int, default=256)
    parser.add_argument("--alignment", type=int, default=1)
    parser.add_argument("--expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", type=Path, default=Path("results/dispatch"))
    args = parser.parse_args()
    if args.execution_mode != "eager" and args.stage < 6:
        parser.error("cached/Graph execution requires S6+")
    if args.iterations < 1 or args.warmup < 0 or min(args.tokens) < 0:
        parser.error("invalid iteration or token count")
    return args


def transport_options(args):
    return {
        name: getattr(args, name)
        for name in (
            "stage",
            "channels",
            "warp_specialization",
            "pipeline_stages",
            "fuse_receive_expand",
            "receive_expand_mode",
            "source_fusion",
            "wire_dtype",
            "numa_mode",
            "replica_slots",
        )
    } | {
        "enable_graph": args.execution_mode.startswith("graph"),
        "relay_workers_per_numa": args.relay_workers,
        "relay_cpu_affinity": tuple(args.relay_cpu_affinity),
    }


def collect_environment(args):
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "control_backend": "gloo",
        "payload_path": "GPU SM -> mapped host RAM -> GPU SM",
        "nvlink_payload_allowed": False,
        "seed": args.seed,
        "latency_definition": "max per-rank CUDA-event full dispatch duration; mean of iteration maxima",
        "bandwidth_definition": "decimal GB/s; B=H*2 for BF16, H+4 for FP8+FP32 scale; unique_wire=F*B; GPU host payload=(U+F)*B; metadata/padding excluded; relay CPU read+write=2*R*B separately",
        "transport_timing_note": "GPU globaltimer ns around transport; includes fused receive/expand work when enabled, excludes separately launched layout/scatter; null if unavailable",
        "timing_note": (
            f"{args.execution_mode} submission; active GPU route/count publish/prefix are timed; "
            "CPU count readback, traffic accounting and CPU coordination are outside timing"
        ),
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total", "--format=csv"]
        ),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "numa": command_output(["numactl", "--hardware"]),
    }


def write_results(output_path, metadata, rows):
    """Persist completed cases after timing, so partial runs retain raw samples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2))
    columns = [
        "tokens_per_rank",
        "latency_mean_us",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "remote_source_tokens",
        "remote_deliveries",
        "host_write_bytes",
        "unique_wire_bytes",
        "host_bus_payload_bytes",
        "unique_wire_GBps_aggregate",
        "unique_wire_GBps_per_gpu_average",
        "host_bus_payload_GBps_aggregate",
        "host_bus_payload_GBps_per_gpu_average",
    ]
    with output_path.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, columns)
        writer.writeheader()
        for row in rows:
            values = {key: row[key] for key in columns if key in row}
            values.update(
                {f"latency_{name}_us": row["full_dispatch"][f"{name}_us"] for name in ("mean", "p50", "p95", "p99")}
            )
            writer.writerow(values)


def main():
    args = parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.experts % world or args.topk > args.experts:
        raise ValueError("expert count must be divisible by ranks, and top-k cannot exceed experts")
    config = HostStagedConfig(
        payload_slots=args.payload_slots, descriptor_slots=args.descriptor_slots, **transport_options(args)
    )
    metadata = collect_environment(args) if rank == 0 else None
    rows = []
    try:
        batches = [[tokens] for tokens in args.tokens] if args.execution_mode.startswith("graph") else [args.tokens]
        for batch in batches:
            with HostStagedEPBuffer(
                group=dist.group.WORLD,
                num_max_tokens_per_rank=max(args.tokens),
                hidden=args.hidden,
                num_topk=args.topk,
                num_experts=args.experts,
                config=config,
            ) as buffer:
                for tokens in batch:
                    row = benchmark_case(buffer, tokens, args, rank, world, device)
                    rows.append(row)
                    if rank == 0:
                        print(
                            f"T={tokens:5d}  mean={row['full_dispatch']['mean_us']:10.2f} us  "
                            f"wire={row['unique_wire_GBps_aggregate']:8.3f} GB/s  "
                            f"host-bus={row['host_bus_payload_GBps_aggregate']:8.3f} GB/s",
                            flush=True,
                        )
                        write_results(args.output, metadata, rows)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
