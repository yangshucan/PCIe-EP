"""Measure unique CPU backing, process mappings and setup cost on eight GPUs.

Run with torchrun. All /proc reads, CPU collectives, validation and explicit CUDA
waits are benchmark instrumentation outside production dispatch submission.
No transport code is patched. Raw samples are retained per process and phase.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer, comm
from pcie_ep.native import AdvancedExtensionAdapter

TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
ROOT = Path(__file__).resolve().parents[1]


def read_kib(path):
    fields = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"(\w+):\s+(\d+) kB", line)
        if match:
            fields[match[1]] = int(match[2]) * 1024
    return fields


def snapshot(directory):
    prefix = str(directory) + "/"
    fields = {key: 0 for key in (
        "Size", "Rss", "Pss", "Shared_Clean", "Shared_Dirty", "Private_Clean",
        "Private_Dirty", "Locked", "Swap", "Anonymous")}
    regions, active = [], None
    for line in Path("/proc/self/smaps").read_text().splitlines():
        if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
            active = None
            if prefix in line:
                active = {"header": line}
                regions.append(active)
        elif active is not None:
            match = re.fullmatch(r"(\w+):\s+(\d+) kB", line)
            if match:
                name, value = match.group(1), int(match.group(2)) * 1024
                active[name] = value
                if name in fields:
                    fields[name] += value
            elif line.startswith("VmFlags:"):
                active["VmFlags"] = line.split(":", 1)[1].strip()
    files = []
    if Path(directory).exists():
        for path in sorted(Path(directory).glob("source-*")):
            stat = path.stat()
            files.append(dict(name=path.name, inode=stat.st_ino, logical_bytes=stat.st_size,
                              allocated_bytes=stat.st_blocks * 512))
    status = read_kib("/proc/self/status")
    meminfo = read_kib("/proc/meminfo")
    return dict(pid=os.getpid(), mapped=fields, regions=regions, files=files,
                logical_backing_bytes=sum(item["logical_bytes"] for item in files),
                allocated_backing_bytes=sum(item["allocated_bytes"] for item in files),
                process={key: status.get(key) for key in ("VmRSS", "VmSize", "VmLck", "VmPin")},
                system={key: meminfo.get(key) for key in
                        ("Shmem", "Mlocked", "Unevictable", "MemAvailable")},
                numa_maps=[line for line in Path("/proc/self/numa_maps").read_text().splitlines()
                           if prefix in line])


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def layout_breakdown(world, channels, slots, descriptors, replicas, domains):
    """Independent byte arithmetic checked against actual fstat sizes."""
    def align(value, alignment=128):
        return (value + alignment - 1) // alignment * alignment

    cursor = sum(align(value) for value in (128, world * 128, 128, (world + 256) * 4))
    cursor = align(cursor, 4096)
    sizes = {"count_mailbox": cursor}

    def reserve(name, size):
        nonlocal cursor
        size = align(size)
        sizes[name] = size
        cursor += size

    source_slots = channels * slots
    replica_slots = channels * domains * replicas
    endpoints = world + domains
    stride = align(7168 * 2, 4096)
    reserve("channel_counts", channels * world * 4)
    reserve("endpoint_counts", channels * endpoints * 4)
    reserve("source_metadata", source_slots * 464)
    cursor = align(cursor, 4096)
    reserve("source_payload", source_slots * stride)
    reserve("source_scale", source_slots * 4)
    reserve("source_ack", source_slots * endpoints * 128)
    reserve("source_descriptor", channels * endpoints * descriptors * 128)
    reserve("replica_metadata", replica_slots * 464)
    cursor = align(cursor, 4096)
    reserve("replica_payload", replica_slots * stride)
    reserve("replica_scale", replica_slots * 4)
    reserve("replica_ack", replica_slots * world * 128)
    reserve("replica_descriptor", channels * domains * world * descriptors * 128)
    reserve("relay_drained", channels * domains * 128)
    reserve("channel_abort", channels * 128)
    reserve("relay_failed", channels * domains * 128)
    total = align(cursor, 4096)
    sizes["remaining_alignment"] = total - sum(sizes.values())
    return dict(per_source=sizes, source_segment_bytes=total,
                unique_backing_bytes=total * world, logical_replica_domains=domains)


def summarize_snapshot(values):
    first = values[0]
    assert all(value["files"] == first["files"] for value in values)
    return dict(logical_backing_bytes=first["logical_backing_bytes"],
                allocated_backing_bytes=first["allocated_backing_bytes"],
                sum_process_mapping_rss_bytes=sum(value["mapped"]["Rss"] for value in values),
                sum_process_mapping_pss_bytes=sum(value["mapped"]["Pss"] for value in values),
                max_mapping_size_per_process=max(value["mapped"]["Size"] for value in values),
                region_counts_per_rank=[len(value["regions"]) for value in values])


def make_inputs(rank, tokens):
    generator = torch.Generator().manual_seed(20261024 + rank * 7919)
    indices = torch.rand(tokens, 256, generator=generator).topk(8, dim=1).indices.contiguous()
    rows = torch.arange(tokens, device="cuda").view(-1, 1)
    columns = torch.arange(7168, device="cuda").view(1, -1)
    payload = ((rows % 23) * 2 + (columns % 47) / 8 + rank * 64).to(torch.bfloat16)
    return payload, indices.to("cuda"), indices


def verify_output(result, indices_by_rank, rank):
    handle = result[3]
    handle.check_errors()
    counts = handle.expert_counts.cpu()
    offsets = handle.expert_offsets.cpu()
    expected = torch.zeros(32, dtype=torch.int64)
    expected_unique = 0
    for indices in indices_by_rank:
        owned = (indices // 32) == rank
        expected_unique += int(owned.any(dim=1).sum())
        local = indices[owned] - rank * 32
        expected += torch.bincount(local, minlength=32)
    assert torch.equal(counts.to(torch.int64), expected)
    assert int(handle.unique_count.cpu()) == expected_unique
    assert int(handle.expanded_count.cpu()) == int(expected.sum())
    selected = []
    for expert, count in enumerate(counts.tolist()):
        if count:
            selected.extend((int(offsets[expert]), int(offsets[expert]) + count - 1))
    selected = sorted(set(selected))
    selection = torch.tensor(selected, dtype=torch.int64, device="cuda")
    metadata = handle.source_meta.index_select(0, selection).cpu()
    actual = result[0].index_select(0, selection).cpu()
    columns = torch.arange(7168).view(1, -1)
    reference = ((metadata[:, 1:2] % 23) * 2 + (columns % 47) / 8 + metadata[:, 0:1] * 64).to(torch.bfloat16)
    assert torch.equal(actual, reference)
    for source, token, branch in metadata.tolist():
        assert int(indices_by_rank[source][token, branch]) // 32 == rank
    return dict(status="passed", sampled_payload_rows=len(selected),
                expanded_count=int(expected.sum()), unique_count=expected_unique)


def run_case(name, output):
    rank, world = dist.get_rank(), dist.get_world_size()
    small = name == "direct_q64"
    config = HostStagedConfig(stage=8 if name == "relay_default" else 6,
                              channels=32, payload_slots=64 if small else 256,
                              descriptor_slots=64 if small else 256,
                              replica_slots=64 if small else 256,
                              numa_mode="relay" if name == "relay_default" else "direct",
                              receive_expand_mode="pipelined")
    phases = {}
    prepared = {}
    originals = {key: getattr(comm, key) for key in
                 ("create_shared_segments", "_prepare_segments", "register_runtime")}
    original_prepare = AdvancedExtensionAdapter.prepare_segment

    def measured_prepare(self, *args, **kwargs):
        value = original_prepare(self, *args, **kwargs)
        prepared.update(value)
        return value

    def timed(key):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            value = originals[key](*args, **kwargs)
            phases[key] = time.perf_counter() - start
            return value
        return wrapper

    for key in originals:
        setattr(comm, key, timed(key))
    AdvancedExtensionAdapter.prepare_segment = measured_prepare
    baseline = snapshot("/dev/shm/nonexistent-pcie-ep-host-buffer-probe")
    dist.barrier()
    start = time.perf_counter()
    try:
        buffer = HostStagedEPBuffer(num_max_tokens_per_rank=8192, hidden=7168,
                                    num_topk=8, num_experts=256, config=config)
    finally:
        for key, function in originals.items():
            setattr(comm, key, function)
        AdvancedExtensionAdapter.prepare_segment = original_prepare
    construction_seconds = time.perf_counter() - start
    torch.cuda.synchronize()
    directory = buffer.directory
    diagnostics = buffer.runtime.relay_diagnostics()
    domains = len(diagnostics["physical_numa_nodes"])
    layout = layout_breakdown(world, config.channels, config.payload_slots,
                              config.descriptor_slots, config.replica_slots, domains)
    phases["construction_total"] = construction_seconds
    snapshots = {"baseline": baseline, "constructed": snapshot(directory)}
    assert snapshots["constructed"]["logical_backing_bytes"] == layout["unique_backing_bytes"]
    assert diagnostics["segment_bytes"] == layout["source_segment_bytes"]
    assert snapshots["constructed"]["allocated_backing_bytes"] == layout["unique_backing_bytes"]
    tests = []
    grid = TOKENS if name == "direct_default" else (128,)
    for tokens in grid:
        payload, indices, indices_cpu = make_inputs(rank, tokens)
        all_indices = gather(indices_cpu)
        samples = []
        for iteration in range(4):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = buffer.dispatch(payload, indices, do_expand=True, do_cpu_sync=False)
            end.record()
            end.synchronize()  # Explicit benchmark wait, after production submission.
            if iteration:
                samples.append(begin.elapsed_time(end))
            if iteration < 3:
                del result
        validation = verify_output(result, all_indices, rank)
        current = snapshot(directory)
        assert current["allocated_backing_bytes"] == layout["unique_backing_bytes"]
        tests.append(dict(tokens_per_gpu=tokens, full_samples_ms=samples,
                          validation=validation, memory=current))
        del result, payload, indices, indices_cpu, all_indices
        gc.collect()
        if rank == 0:
            print(f"{name}: T={tokens} memory={current['allocated_backing_bytes']} bytes", flush=True)
    snapshots["after_dispatch"] = snapshot(directory)
    dist.barrier()
    start = time.perf_counter()
    buffer.close()
    phases["close_total"] = time.perf_counter() - start
    # close() has a drain barrier before rank 0 unlinks the files. Peers may
    # return before that unlink finishes; align the external OS observation.
    dist.barrier()
    del buffer
    gc.collect()
    snapshots["closed"] = snapshot(directory)
    assert not Path(directory).exists()
    assert snapshots["closed"]["mapped"]["Size"] == 0
    assert snapshots["closed"]["allocated_backing_bytes"] == 0
    local = dict(rank=rank, case=name, config=config.native_options(), setup_seconds=phases,
                 first_touch=prepared, native_diagnostics=diagnostics, layout=layout,
                 snapshots=snapshots, dispatch=tests)
    (output / f"{name}_rank{rank}.json").write_text(json.dumps(local, indent=2) + "\n")
    records = gather(local)
    if rank:
        return None
    summary = dict(case=name, status="passed", layout=layout,
                   physical_numa_nodes_per_rank=[record["first_touch"]["numa_node"] for record in records],
                   logical_replica_domains=domains,
                   setup_max_seconds={key: max(record["setup_seconds"][key] for record in records) for key in phases},
                   snapshots={phase: summarize_snapshot([record["snapshots"][phase] for record in records])
                              for phase in ("constructed", "after_dispatch", "closed")},
                   tokens=[dict(tokens_per_gpu=tokens,
                                max_rank_full_mean_ms=max(statistics.mean(record["dispatch"][i]["full_samples_ms"])
                                                          for record in records),
                                unique_backing_bytes=records[0]["dispatch"][i]["memory"]["allocated_backing_bytes"],
                                sampled_payload_rows_all_ranks=sum(record["dispatch"][i]["validation"]["sampled_payload_rows"]
                                                                  for record in records))
                           for i, tokens in enumerate(grid)])
    print(json.dumps(dict(case=name, status="passed", unique_GiB=layout["unique_backing_bytes"] / 2**30,
                         construction_max_s=summary["setup_max_seconds"]["construction_total"])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/host_buffer/20261027"))
    parser.add_argument("--cases", nargs="+", default=["direct_default", "direct_q64", "relay_default"],
                        choices=["direct_default", "direct_q64", "relay_default"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    assert dist.get_world_size() == 8
    from pcie_ep import _C
    warm = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    del warm
    report = dict(status="running", world_size=8, tmax=8192, hidden=7168, topk=8,
                  experts=256, dtype="bf16", torch_version=torch.__version__,
                  production_cpu_sync=False, source_commit="865cc495eafbd63e9745dd0a81c9d03d824afc89",
                  native_binary_sha256=hashlib.sha256(Path(_C._module.__file__).read_bytes()).hexdigest(),
                  notes=["smaps RSS summed across ranks double-counts shared pages; file blocks are unique physical backing",
                         "PSS is reported separately; CUDA host registration is not reliably reflected by VmLck/VmPin",
                         "setup phase times include their Gloo agreement; register_runtime includes GPU output workspace allocation",
                         "one construction per case; setup timings are observations, not repeated latency benchmarks",
                         "dispatch samples are memory-exercise diagnostics, not a replacement for the formal performance benchmark",
                         "all proc reads, CPU collectives, output readbacks and CUDA waits are external test instrumentation"], cases=[])
    for name in args.cases:
        value = run_case(name, args.output)
        if rank == 0:
            report["cases"].append(value)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
    if rank == 0:
        report["status"] = "passed"
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
