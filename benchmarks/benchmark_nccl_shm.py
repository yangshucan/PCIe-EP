"""Compare host-staged dispatch with NCCL SHM on identical inputs and routes.

Example: torchrun --standalone --nproc-per-node=8 \
    benchmarks/benchmark_nccl_shm.py --output results/nccl_shm/round1

``nccl_payload`` times a BF16 AllGather only. ``nccl_dispatch`` times packing
payload/routes/weights, one uint8 AllGather, and stable local expert expansion.
The Gloo world group owns all CPU coordination; the explicit NCCL group owns
only GPU AllGathers. Validation and CUDA completion waits are outside timing.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics

# The benchmark intentionally compares host-memory paths on an NVLink machine.
# Algorithm, protocol, channel count and host allocator retain NCCL defaults.
NCCL_PATH_ENV = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_MNNVL_ENABLE": "0",
    "NCCL_COLLNET_ENABLE": "0",
    "NCCL_IB_DISABLE": "1",
    # NCCL may otherwise prefer Socket for cross-NUMA peers. This baseline
    # explicitly measures SHM for every local GPU data connection.
    "NCCL_NET_DISABLE_INTRA": "1",
}
os.environ.update(NCCL_PATH_ENV)

import torch
import torch.distributed as dist

from benchmark import distribution, gather, route_volume, routes
from nccl_expand import NCCLExpand
from routing_fixture import load_fixture
from pcie_ep import HostStagedConfig, HostStagedEPBuffer


VARIANTS = ("pcie_ep", "nccl_payload", "nccl_dispatch")


def routing_tensors(args, tokens, rank):
    """Load saved CPU routes; a frozen run never calls the legacy RNG path."""
    if args.frozen_routing is not None:
        return args.frozen_routing.cpu_tensors(tokens, rank)
    return routes(tokens, args.experts, args.topk, rank, args.seed)


def tensor_sha256(value):
    """Hash exact contiguous tensor bytes; called only outside timed regions."""
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def global_sha256(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def input_fingerprints(idx_cpu, weights_cpu, x):
    local = {
        "rank": dist.get_rank(),
        "routes_sha256": tensor_sha256(idx_cpu),
        "weights_sha256": tensor_sha256(weights_cpu),
        "payload_sha256": tensor_sha256(x),
    }
    records = gather(local)
    routing = [{name: item[name] for name in ("rank", "routes_sha256", "weights_sha256")} for item in records]
    return {
        "per_rank": records,
        "global_routes_sha256": global_sha256(routing),
        "global_inputs_sha256": global_sha256(records),
        "hash_definition": "SHA256 of contiguous raw tensor bytes; route=int64, weight=float32, payload=BF16",
    }


def reference_layout(all_routes, rank, world, experts, topk, tokens):
    """Independent CPU expert/source/token/branch reference, without payloads."""
    local_experts = experts // world
    idx = torch.stack([item[0] for item in all_routes]).reshape(world * tokens, topk)
    weights = torch.stack([item[1] for item in all_routes]).reshape(world * tokens, topk)
    identities, expert_ids, branch_weights, counts = [], [], [], []
    for expert in range(local_experts):
        selected = (idx == rank * local_experts + expert).nonzero(as_tuple=False)
        count = selected.shape[0]
        counts.append(count)
        identities.append(torch.stack((selected[:, 0] // tokens, selected[:, 0] % tokens, selected[:, 1]), dim=1))
        expert_ids.append(torch.full((count,), expert, dtype=torch.int64))
        branch_weights.append(weights[selected[:, 0], selected[:, 1]])
    counts = torch.tensor(counts, dtype=torch.int64)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    metadata = torch.cat(identities)
    return {
        "expert_counts": counts,
        "expert_offsets": offsets,
        "source_meta": metadata,
        "indices": torch.cat(expert_ids),
        "weights": torch.cat(branch_weights),
        "source_rows": metadata[:, 0] * tokens + metadata[:, 1],
        "unique_count": int(((idx >= 0) & (idx // local_experts == rank)).any(dim=1).sum()),
    }


def allgather_gpu(output, value, group):
    """Order collective completion onto the current stream without CPU wait."""
    work = dist.all_gather_into_tensor(output, value, group=group, async_op=True)
    work.block_current_stream()
    return work


def pcie_dispatch(buffer, x, idx, weights):
    result = buffer.dispatch(
        x,
        topk_idx=idx,
        topk_weights=weights,
        do_expand=True,
        expert_alignment=1,
        do_cpu_sync=False,
        async_with_compute_stream=True,
    )
    result[4].current_stream_wait()
    return result


def timed_variant(call, variant, tokens, args, rank):
    """CUDA events span the full operation, including NCCL stream handshakes."""
    retained, result = [], None
    for _ in range(args.warmup):
        result = call()
        if variant != "pcie_ep":
            retained.append(result)
    torch.cuda.synchronize()
    dist.barrier()
    retained.clear()
    profiler = (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        if args.profile else contextlib.nullcontext()
    )
    starts, ends = [], []
    with profiler as trace:
        if args.profile:
            # CUPTI startup and rank skew are outside the steady markers.
            for _ in range(2):
                result = call()
                if variant != "pcie_ep":
                    retained.append(result)
            torch.cuda.synchronize()
            dist.barrier()
            retained.clear()
        for iteration in range(args.iterations):
            marker = f"{variant}_T{tokens}_i{iteration}"
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            scope = torch.profiler.record_function(marker) if args.profile else contextlib.nullcontext()
            with scope:
                start.record()
                result = call()
                # Keep asynchronous NCCL Work objects alive until completion.
                # PCIe result pools must be released as each result is replaced.
                if variant != "pcie_ep":
                    retained.append(result)
                end.record()
            starts.append(start)
            ends.append(end)
        ends[-1].synchronize()
    if trace is not None:
        args.profile.mkdir(parents=True, exist_ok=True)
        trace.export_chrome_trace(str(args.profile / f"{variant}_T{tokens}_rank{rank}.trace.json"))
    local_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    samples = gather(local_us)
    maxima = [max(sample[iteration] for sample in samples) for iteration in range(args.iterations)]
    return result, maxima, samples


def validate_transport(value, expected_hashes, tokens, chunk_rows):
    """Verify every byte of every source block, with bounded CPU copy size."""
    for source, expected in enumerate(expected_hashes):
        digest = hashlib.sha256()
        first = source * tokens
        for start in range(0, tokens, chunk_rows):
            block = value[first + start : first + min(start + chunk_rows, tokens)]
            raw = block.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
        if digest.hexdigest() != expected:
            raise AssertionError(f"AllGather bytes differ from input at source rank {source}")


def validate_expansion(result, reference, payload_recv, args):
    """Check counts, stable row identity, routes, weights and all BF16 elements."""
    count = reference["indices"].numel()
    for name in ("expert_counts", "expert_offsets", "source_meta", "indices", "weights"):
        actual = result[name]
        if name not in ("expert_counts", "expert_offsets"):
            actual = actual[:count]
        if name == "source_meta":
            actual = actual[:, :3]
        actual = actual.detach().cpu()
        expected = reference[name].to(actual.dtype)
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            raise AssertionError(f"expanded {name} differs from CPU routing reference")
    source_rows = reference["source_rows"].to(payload_recv.device)
    for start in range(0, count, args.validation_chunk_rows):
        end = min(start + args.validation_chunk_rows, count)
        expected = payload_recv.index_select(0, source_rows[start:end]).cpu().view(torch.int16)
        actual = result["payload"][start:end].detach().cpu().view(torch.int16)
        if not torch.equal(actual, expected):
            raise AssertionError(f"expanded BF16 payload differs at rows [{start}, {end})")
    return {"expanded_rows": count, "bf16_elements_checked": count * args.hidden, "stable_order": True}


def benchmark_case(buffer, tokens, args, rank, world, device, nccl_group):
    # Construct once per T. Every timed variant receives these exact tensors.
    idx_cpu, weights_cpu = routing_tensors(args, tokens, rank)
    idx, weights = idx_cpu.to(device), weights_cpu.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device=device, generator=generator)
    fingerprints = input_fingerprints(idx_cpu, weights_cpu, x)
    all_routes = gather((idx_cpu, weights_cpu))
    reference = reference_layout(all_routes, rank, world, args.experts, args.topk, tokens)
    volumes = gather(route_volume(idx_cpu, rank, world, args.experts))
    global_volume = {name: sum(item[name] for item in volumes) for name in volumes[0]}
    payload_recv = torch.empty((world * tokens, args.hidden), dtype=x.dtype, device=device)
    expand = NCCLExpand(world, tokens, args.hidden, args.topk, args.experts, rank, device) if "nccl_dispatch" in args.variants else None
    packed_hashes = None
    if expand is not None:
        expand.pack(x, idx, weights)
        packed_hashes = gather(tensor_sha256(expand.packed_send))
    payload_validated, records = False, []

    def run_pcie():
        return pcie_dispatch(buffer, x, idx, weights)

    def run_payload():
        return allgather_gpu(payload_recv, x, nccl_group)

    def run_nccl_dispatch():
        send = expand.pack(x, idx, weights)
        work = allgather_gpu(expand.packed_recv, send, nccl_group)
        return {"output": expand.expand(), "work": work}

    calls = {"pcie_ep": run_pcie, "nccl_payload": run_payload, "nccl_dispatch": run_nccl_dispatch}
    for variant in args.variants:
        result, maxima, per_rank = timed_variant(calls[variant], variant, tokens, args, rank)
        # A payload AllGather is an independent, bitwise-checked source oracle.
        # It is outside timing, and cached for all expansion checks at this T.
        if not payload_validated and variant != "nccl_payload":
            work = run_payload()
            torch.cuda.synchronize()
            del work
        validation = {"allgather_source_payload_bytes_checked": world * tokens * args.hidden * 2}
        failure = None
        try:
            if not payload_validated or variant == "nccl_payload":
                validate_transport(payload_recv, [item["payload_sha256"] for item in fingerprints["per_rank"]], tokens, args.validation_chunk_rows)
                payload_validated = True
            if variant == "pcie_ep":
                buffer.check_errors(result[3])
                output = dict(zip(("payload", "indices", "weights"), result[:3]))
                output.update({name: getattr(result[3], name) for name in ("source_meta", "expert_counts", "expert_offsets")})
                validation.update(validate_expansion(output, reference, payload_recv, args))
                if int(result[3].unique_count.cpu()) != reference["unique_count"]:
                    raise AssertionError("PCIe-EP unique token count differs from routing reference")
                expected = route_volume(idx_cpu, rank, world, args.experts)
                stats = result[3].stats.cpu().tolist()
                expected_reads = sum(
                    int(((item[0] // (args.experts // world)) == rank).any(dim=1).sum())
                    for source, item in enumerate(all_routes) if source != rank
                )
                if stats[0] != expected["remote_source_tokens"] or stats[1] != expected_reads:
                    raise AssertionError("PCIe-EP sender/receiver dedup counts differ from routes")
                validation["runtime_stats"] = stats
            elif variant == "nccl_dispatch":
                validate_transport(expand.packed_recv, packed_hashes, tokens, args.validation_chunk_rows)
                validation.update(validate_expansion(result["output"], reference, payload_recv, args))
                validation["packed_bytes_checked"] = expand.packed_recv.numel()
        except Exception as exc:
            failure = f"rank {rank}: {type(exc).__name__}: {exc}"
        failures = [item for item in gather(failure) if item is not None]
        if failures:
            raise AssertionError("; ".join(failures))
        validation_per_rank = gather(validation)
        seconds = statistics.mean(maxima) * 1e-6
        row = {
            "tokens_per_rank": tokens,
            "variant": variant,
            "variant_order": args.variants,
            "world_size": world,
            "experts": args.experts,
            "topk": args.topk,
            "hidden": args.hidden,
            "dtype": "bfloat16",
            "seed": args.seed,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "full_dispatch": distribution(maxima),
            "full_dispatch_samples_us": maxima,
            "per_rank_dispatch_samples_us": per_rank,
            "logical_bytes_per_gpu": tokens * args.topk * args.hidden * 2,
            "logical_GBps": tokens * args.topk * args.hidden * 2 / seconds / 1e9,
            "input_fingerprints": fingerprints,
            "route_volume": global_volume,
            "validation_per_rank": validation_per_rank,
        }
        if variant.startswith("nccl_"):
            send_bytes = tokens * args.hidden * 2 if variant == "nccl_payload" else expand.packed_send.numel()
            row.update({
                "allgather_send_bytes_per_gpu": send_bytes,
                "nccl_tests_total_receive_bytes_per_gpu": world * send_bytes,
                "nccl_tests_algbw_GBps": world * send_bytes / seconds / 1e9,
                "nccl_tests_busbw_GBps": (world - 1) * send_bytes / seconds / 1e9,
                "nccl_bandwidth_note": "AllGather size is total received bytes; algbw and busbw are collective conventions, not measured PCIe bytes. nccl_dispatch includes pack and expansion in t.",
            })
        records.append(row)
        if rank == 0:
            print(f"T={tokens:5d} {variant:13s} {row['full_dispatch']['mean_us']:11.3f} us  logical={row['logical_GBps']:8.3f} GB/s", flush=True)
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--routing-fixture", type=Path,
                        help="Frozen CPU routing JSON/NPZ; IDs and weights are loaded before timing, with no runtime route RNG")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--validation-chunk-rows", type=int, default=256)
    parser.add_argument("--profile", type=Path, help="Optional trace directory; profiler samples are diagnostic only")
    parser.add_argument("--output", type=Path, default=Path("results/nccl_shm/comparison"))
    args = parser.parse_args()
    if min(args.tokens) < 1 or args.warmup < 0 or args.iterations < 1 or args.validation_chunk_rows < 1:
        parser.error("token, iteration and validation chunk sizes must be positive (warmup may be zero)")
    if len(set(args.variants)) != len(args.variants) or len(set(args.tokens)) != len(args.tokens):
        parser.error("variants and tokens cannot contain duplicates")
    if args.hidden < 1 or args.experts < 1 or not 1 <= args.topk <= args.experts:
        parser.error("invalid hidden, expert or top-k dimensions")
    if args.hidden % 8 or args.topk % 4 or args.experts % 8 or args.experts // 8 > 32:
        parser.error("NCCL expansion requires H divisible by 8, K divisible by 4 and at most 32 experts/rank")
    args.frozen_routing = load_fixture(args.routing_fixture) if args.routing_fixture is not None else None
    if args.frozen_routing is not None:
        fixture = args.frozen_routing.manifest
        if any(getattr(args, key) != fixture[key] for key in ("experts", "topk", "hidden", "seed")):
            parser.error("Benchmark dimensions and seed must match the frozen routing fixture")
        available = {row["tokens_per_gpu"] for row in fixture["routes"]}
        if not set(args.tokens) <= available:
            parser.error("Every requested token count must exist in the frozen routing fixture")
    return args


def metadata(args, world):
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "control_backend": "gloo",
        "allgather_backend": "nccl",
        "world_size": world,
        "nvlink_payload_allowed": False,
        "nccl_environment": {
            name: os.environ.get(name, "unset (NCCL default)")
            for name in (*NCCL_PATH_ENV, "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_CUMEM_HOST_ENABLE", "NCCL_CUMEM_ENABLE", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS")
        },
        "profiling_enabled": bool(args.profile),
        "routing_fixture": args.frozen_routing.identity if args.frozen_routing is not None else None,
        "pcie_config": {"channels":32,"payload_slots":256,"descriptor_slots":256,
                        "pipeline_stages":2,"receive_expand_mode":"pipelined",
                        "source_fusion":True,"warp_specialization":True,
                        "fuse_receive_expand":True,"numa_mode":"direct",
                        "wire_dtype":"bf16","stage":6,"alignment":1,"do_expand":True},
        "nccl_completion_api": "Work.block_current_stream (GPU stream dependency; returns immediately)",
        "latency_definition": "Mean of iteration maxima across all ranks; full CUDA-event interval; eager, no per-iteration CPU synchronization",
        "bandwidth_definition": "logical_GBps=T*K*H*2/t/1e9 for every variant; no world-size factor or dedup adjustment",
        "variant_definition": {
            "pcie_ep": "Current C32/Q256/L256/two-stage BF16 direct mapped-host dispatch, pipelined receive expansion, source fusion; routing/counts/transport/expansion timed",
            "nccl_payload": "BF16 payload AllGather only; no routes/weights transport or expert expansion; lower-bound communication baseline",
            "nccl_dispatch": "GPU pack(payload BF16, routes int64, weights FP32), one uint8 AllGather, GPU stable expert/source/token/branch expansion; all timed. Expert-ready tensor baseline only; does not implement PCIe-EP unique layout, reusable handle or combine API.",
        },
        "path_evidence_required": "NCCL initialization logs must show SHM for every GPU data connection, and independent NVLink payload counters must remain unchanged; environment settings alone do not prove the path",
    }


def write_results(path, environment, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps({"metadata": environment, "results": rows}, indent=2))
    columns = ("tokens_per_rank", "variant", "latency_mean_us", "logical_GBps", "nccl_tests_algbw_GBps", "nccl_tests_busbw_GBps", "global_routes_sha256")
    with path.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in columns} | {
                "latency_mean_us": row["full_dispatch"]["mean_us"],
                "global_routes_sha256": row["input_fingerprints"]["global_routes_sha256"],
            })


def main():
    args = parse_args()
    if not hasattr(dist.Work, "block_current_stream"):
        raise RuntimeError("This benchmark requires Work.block_current_stream for asynchronous GPU completion ordering")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(device)
    if world != 8 or args.experts % world:
        raise ValueError("The comparison requires eight ranks and equally partitioned experts")
    nccl_group = dist.new_group(ranks=list(range(world)), backend="nccl")
    config = HostStagedConfig(
        channels=32, payload_slots=256, descriptor_slots=256,
        pipeline_stages=2, receive_expand_mode="pipelined",
        warp_specialization=True, fuse_receive_expand=True, source_fusion=True,
        wire_dtype="bf16", numa_mode="direct", stage=6,
    )
    environment = metadata(args, world) if rank == 0 else None
    rows = []
    try:
        context = (
            HostStagedEPBuffer(
                group=dist.group.WORLD, num_max_tokens_per_rank=max(args.tokens),
                hidden=args.hidden, num_topk=args.topk, num_experts=args.experts, config=config,
            )
            if "pcie_ep" in args.variants else contextlib.nullcontext()
        )
        with context as buffer:
            for tokens in args.tokens:
                rows.extend(benchmark_case(buffer, tokens, args, rank, world, device, nccl_group))
                if rank == 0:
                    write_results(args.output, environment, rows)
    finally:
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
