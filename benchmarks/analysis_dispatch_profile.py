"""Read focused dispatch traces and opt-in GPU-role counters without CUDA.

Example:
    python benchmarks/analysis_dispatch_profile.py --torch-dir results/profile/torch \
        --roles-dir results/profile/gpu --output results/profile/summary.json

GPU activities are assigned to CPU dispatch markers through CUDA runtime
correlation IDs, never by comparing asynchronous GPU and CPU time windows.
Only durations, rank/iteration numbers and an explicit configuration allowlist
are exported. Source paths, host metadata and absolute timestamps are omitted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics


MARKER = re.compile(r"^dispatch_T(\d+)_i(\d+)$")
TRACE_NAME = re.compile(r"^T(\d+)_rank(\d+)\.trace\.json$")
EXPECTED_KERNELS = (
    "advanced_begin_epoch", "advanced_route_flags", "advanced_stable_ordinals",
    "advanced_prepare_channels_parallel", "publish_counts", "advanced_transport_pipelined",
    "advanced_copy_local_tokens", "advanced_reduce_statistics",
)
CPU_SYNCS = (
    "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize",
    "cuCtxSynchronize", "cuStreamSynchronize", "cuEventSynchronize",
)
SCALAR_READS = {"aten::item", "aten::_local_scalar_dense", "aten::is_nonzero"}
CONFIG_FIELDS = (
    "hidden", "experts", "topk", "warmup", "iterations", "stage", "channels",
    "warp_specialization", "pipeline_stages", "fuse_receive_expand", "receive_expand_mode",
    "source_fusion", "wire_dtype", "numa_mode", "replica_slots", "execution_mode",
    "payload_slots", "descriptor_slots", "alignment", "expand", "seed",
)
# Keep this order synchronized with TransportProfileField in advanced_common.cuh.
ROLE_FIELDS = (
    "enter_ns", "exit_ns", "total_ns", "iterations", "work_samples", "work_ns", "fence_ns",
    "work_and_fence_ns", "first_publication_ns", "last_publication_ns", "failed_ns",
    "failed_calls", "failed_signal_loads", "drained_ns", "drained_calls", "drained_samples",
    "drained_ack_loads", "drained_relay_loads", "descriptor_probe_ns", "descriptor_probe_calls",
    "descriptor_probe_samples", "descriptor_not_ready", "local_stage_busy", "tx_finished_ns",
    "rx_finished_ns", "prefix_wait_ns", "available_ns", "available_calls", "items",
    "publications", "sample_stride", "available_samples",
)
ROLE_NAMES = ("tx_control", "rx_control", "tx_worker_0", "tx_worker_1", "tx_worker_2",
              "rx_producer_0", "rx_producer_1", "rx_producer_2",
              "scatter_consumer_0", "scatter_consumer_1", "scatter_consumer_2")
EXACT_COUNTERS = (
    "iterations", "failed_calls", "failed_signal_loads", "drained_calls", "drained_ack_loads",
    "drained_relay_loads", "descriptor_probe_calls", "descriptor_not_ready", "local_stage_busy",
    "available_calls", "items", "publications",
)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    if not values:
        return dict(count=0, mean=None, median=None, p95=None, minimum=None, maximum=None)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Nonfinite profiling measurement")
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=percentile(values, .95), minimum=min(values), maximum=max(values))


def load_rank_records(folder):
    ranks = {}
    configuration = None
    for path in sorted(folder.glob("rank*.json")):
        if not re.fullmatch(r"rank\d+\.json", path.name):
            continue
        raw = json.loads(path.read_text())
        rank = raw["rank"]
        current = {key: raw["config"][key] for key in CONFIG_FIELDS if key in raw["config"]}
        if configuration is None:
            configuration = current
        elif configuration != current:
            raise ValueError("Mismatched profiling configurations across ranks")
        if rank in ranks:
            raise ValueError("Duplicate rank record")
        records = {record["tokens"]: record for record in raw["records"]}
        if len(records) != len(raw["records"]):
            raise ValueError("Duplicate token record")
        ranks[rank] = dict(world=raw["world"], records=records)
    if not ranks:
        raise ValueError("No rank JSON records")
    worlds = {rank["world"] for rank in ranks.values()}
    if len(worlds) != 1 or set(ranks) != set(range(next(iter(worlds)))):
        raise ValueError("Incomplete rank coverage")
    token_sets = {tuple(sorted(rank["records"])) for rank in ranks.values()}
    if len(token_sets) != 1:
        raise ValueError("Mismatched batch coverage across ranks")
    return ranks, configuration


def marker_for(event, markers):
    timestamp = event.get("ts")
    if timestamp is None:
        return None
    matches = [marker for marker in markers
               if marker["pid"] == event.get("pid") and marker["tid"] == event.get("tid")
               and marker["ts"] <= timestamp <= marker["ts"] + marker["dur"]]
    if len(matches) > 1:
        raise ValueError("Overlapping CPU dispatch markers")
    return matches[0] if matches else None


def correlation(event):
    arguments = event.get("args", {})
    return arguments.get("correlation", arguments.get("correlationId"))


def is_dtoh(event):
    return any(text in event.get("name", "").lower()
               for text in ("dtoh", "device -> host", "device to host"))


def analyze_trace(path, rank, tokens, record):
    events = json.loads(path.read_text())["traceEvents"]
    markers = [event for event in events if event.get("cat") == "user_annotation"
               and event.get("ph") == "X" and MARKER.fullmatch(event.get("name", ""))]
    marker_by_iteration = {}
    for marker in markers:
        match = MARKER.fullmatch(marker["name"])
        if int(match[1]) != tokens:
            raise ValueError("Trace contains a different batch marker")
        iteration = int(match[2])
        if iteration in marker_by_iteration:
            raise ValueError("Duplicate dispatch marker")
        marker_by_iteration[iteration] = marker
    if set(marker_by_iteration) != set(range(len(record["dispatch_us"]))):
        raise ValueError("Incomplete iteration marker coverage")
    runtime = [event for event in events if event.get("cat") in ("cuda_runtime", "cuda_driver")]
    owned_runtime = {}
    runtime_by_iteration = defaultdict(list)
    for event in runtime:
        marker = marker_for(event, markers)
        if marker is not None:
            iteration = int(MARKER.fullmatch(marker["name"])[2])
            runtime_by_iteration[iteration].append(event)
            identifier = correlation(event)
            if identifier is not None:
                owned_runtime[identifier] = iteration
    gpu_by_iteration = defaultdict(list)
    outside_kernel_count = outside_dtoh_count = 0
    violations = []
    outside_sync_count = 0
    for event in events:
        name = event.get("name", "")
        cpu_sync = event.get("cat") in ("cuda_runtime", "cuda_driver") and any(item in name for item in CPU_SYNCS)
        scalar = name in SCALAR_READS
        if cpu_sync or scalar:
            marker = marker_for(event, markers)
            if marker is None:
                outside_sync_count += int(cpu_sync)
            else:
                violations.append(dict(rank=rank, tokens=tokens,
                    iteration=int(MARKER.fullmatch(marker["name"])[2]),
                    kind="cpu_sync" if cpu_sync else "scalar_readback", operation=name))
        if event.get("cat") == "kernel" or is_dtoh(event):
            iteration = owned_runtime.get(correlation(event))
            if iteration is None:
                outside_kernel_count += int(event.get("cat") == "kernel")
                outside_dtoh_count += int(is_dtoh(event))
            elif is_dtoh(event):
                violations.append(dict(rank=rank, tokens=tokens, iteration=iteration,
                                       kind="device_to_host", operation=name))
            else:
                gpu_by_iteration[iteration].append(event)
    dispatches, kernels = [], defaultdict(list)
    for iteration, marker in sorted(marker_by_iteration.items()):
        gpu = sorted(gpu_by_iteration[iteration], key=lambda event: event["ts"])
        names = [event["name"].split("(")[0].split("::")[-1] for event in gpu]
        if tuple(names) != EXPECTED_KERNELS:
            raise ValueError(f"Unexpected kernel sequence for rank {rank}, T{tokens}, iteration {iteration}: {names}")
        duration = sum(event["dur"] for event in gpu)
        span = gpu[-1]["ts"] + gpu[-1]["dur"] - gpu[0]["ts"]
        gaps = [gpu[index + 1]["ts"] - event["ts"] - event["dur"] for index, event in enumerate(gpu[:-1])]
        if min(gaps, default=0) < -1e-3:
            raise ValueError("Unexpected overlap between sequential dispatch kernels")
        dispatches.append(dict(rank=rank, iteration=iteration,
            cuda_event_us=record["dispatch_us"][iteration], sum_kernel_gpu_us=duration,
            kernel_span_gpu_us=span, kernel_gap_gpu_us=max(0, span - duration),
            cpu_marker_us=marker["dur"],
            launch_api_cpu_us=sum(event["dur"] for event in runtime_by_iteration[iteration]
                                  if "LaunchKernel" in event.get("name", ""))))
        for name, event in zip(names, gpu):
            arguments = event.get("args", {})
            kernels[name].append(dict(rank=rank, iteration=iteration, gpu_us=event["dur"],
                resources={key: arguments[key] for key in ("grid", "block", "registers per thread", "shared memory")
                           if key in arguments}))
    return dict(dispatches=dispatches, kernels=kernels, violations=violations,
                excluded_outside_marker_cpu_syncs=outside_sync_count,
                excluded_outside_marker_kernels=outside_kernel_count,
                excluded_outside_marker_dtoh=outside_dtoh_count)


def analyze_torch(folder):
    ranks, config = load_rank_records(folder)
    by_token = defaultdict(list)
    for path in sorted(folder.glob("*.trace.json")):
        match = TRACE_NAME.fullmatch(path.name)
        if match:
            tokens, rank = int(match[1]), int(match[2])
            if rank not in ranks or tokens not in ranks[rank]["records"]:
                raise ValueError("Trace lacks its rank record")
            by_token[tokens].append((rank, analyze_trace(path, rank, tokens, ranks[rank]["records"][tokens])))
    if set(by_token) != set(next(iter(ranks.values()))["records"]):
        raise ValueError("Missing batch trace")
    output = []
    all_violations = []
    for tokens, results in sorted(by_token.items()):
        if {rank for rank, _ in results} != set(ranks):
            raise ValueError("Incomplete rank trace coverage")
        dispatches = [item for _, result in results for item in result["dispatches"]]
        iterations = sorted({item["iteration"] for item in dispatches})
        if len(dispatches) != len(ranks) * len(iterations):
            raise ValueError("Incomplete rank/iteration trace coverage")
        maxima = [max(item["cuda_event_us"] for item in dispatches if item["iteration"] == iteration)
                  for iteration in iterations]
        violations = [item for _, result in results for item in result["violations"]]
        all_violations.extend(violations)
        kernel_rows = {}
        kernel_lookup = {}
        for name in EXPECTED_KERNELS:
            observations = [item for _, result in results for item in result["kernels"][name]]
            kernel_rows[name] = dict(gpu_us=distribution([item["gpu_us"] for item in observations]),
                                     per_rank_iteration=observations)
            kernel_lookup[name] = {(item["rank"], item["iteration"]): item["gpu_us"] for item in observations}
        critical = []
        for iteration in iterations:
            slowest = max((item for item in dispatches if item["iteration"] == iteration),
                          key=lambda item: item["cuda_event_us"])
            selected = (slowest["rank"], iteration)
            critical.append(slowest | dict(kernel_gpu_us={name: kernel_lookup[name][selected]
                                                         for name in EXPECTED_KERNELS}))
        output.append(dict(tokens=tokens, rank_count=len(ranks), iterations_per_rank=len(iterations),
            kernels_per_dispatch=len(EXPECTED_KERNELS), kernel_order=list(EXPECTED_KERNELS), kernels=kernel_rows,
            max_rank_cuda_event_us=distribution(maxima),
            critical_rank_kernel_gpu_us={name: distribution([item["kernel_gpu_us"][name] for item in critical])
                                         for name in EXPECTED_KERNELS},
            critical_rank_per_iteration=critical,
            dispatch_statistics={key: distribution([item[key] for item in dispatches]) for key in
                ("cuda_event_us", "sum_kernel_gpu_us", "kernel_span_gpu_us", "kernel_gap_gpu_us",
                 "cpu_marker_us", "launch_api_cpu_us")},
            per_rank_iteration=dispatches,
            audit=dict(status="passed" if not violations else "failed", violations=violations,
                excluded_outside_marker_cpu_syncs=sum(result["excluded_outside_marker_cpu_syncs"] for _, result in results),
                excluded_outside_marker_kernels=sum(result["excluded_outside_marker_kernels"] for _, result in results),
                excluded_outside_marker_dtoh=sum(result["excluded_outside_marker_dtoh"] for _, result in results))))
    return dict(configuration=config, world_size=len(ranks), batches=output,
                audit_status="passed" if not all_violations else "failed",
                note="Profiler observations, not formal performance samples. All-rank kernel means and max-rank event means use different aggregations; critical_rank_kernel_gpu_us selects the event-slowest rank separately per iteration.")


def timed_calls(observations, timer, samples, calls=None):
    nanoseconds = sum(row[timer] for row in observations)
    timed = sum(row[samples] for row in observations)
    result = dict(sum_timed_ns=nanoseconds, timed_call_count=timed,
                  pooled_ns_per_timed_call=nanoseconds / timed if timed else None,
                  per_observation_timed_ns=distribution([row[timer] for row in observations]))
    if calls:
        result["exact_call_count"] = sum(row[calls] for row in observations)
    complete = bool(calls) and all(row[samples] == row[calls] for row in observations)
    result["complete_call_timing"] = complete
    result["coverage"] = "all calls" if complete else "systematic stride-32 samples; no total-time extrapolation"
    if complete:
        lifetime = sum(row["total_ns"] for row in observations)
        result["fraction_of_pooled_role_lifetimes"] = nanoseconds / lifetime if lifetime else None
    return result


def role_summary(observations, worker):
    result = dict(
        observations=len(observations), total_role_ns=distribution([row["total_ns"] for row in observations]),
        prefix_wait_ns=distribution([row["prefix_wait_ns"] for row in observations]),
        exact_counters={key: dict(sum=sum(row[key] for row in observations),
            per_observation=distribution([row[key] for row in observations])) for key in EXACT_COUNTERS},
        abort_checks=timed_calls(observations, "failed_ns", "failed_calls", "failed_calls"),
        drain=timed_calls(observations, "drained_ns", "drained_samples", "drained_calls"),
        descriptor_probe=timed_calls(observations, "descriptor_probe_ns", "descriptor_probe_samples", "descriptor_probe_calls"),
        availability=timed_calls(observations, "available_ns", "available_samples", "available_calls"),
    )
    if worker:
        result.update(
            copy_work=timed_calls(observations, "work_ns", "work_samples", "iterations"),
            copy_fence=timed_calls(observations, "fence_ns", "work_samples", "iterations"),
            copy_work_and_fence=timed_calls(observations, "work_and_fence_ns", "work_samples", "iterations"),
        )
    return result


def analyze_roles(folder):
    ranks, config = load_rank_records(folder)
    by_token = defaultdict(list)
    for rank, raw in sorted(ranks.items()):
        for tokens, record in sorted(raw["records"].items()):
            if len(record["samples"]) != len(record["dispatch_us"]):
                raise ValueError("Incomplete role samples")
            for iteration, sample in enumerate(record["samples"]):
                channels = sample.get("roles")
                if not isinstance(channels, list) or len(channels) != config["channels"]:
                    raise ValueError("Missing profiling tensor or incorrect channel dimension")
                for channel, roles in enumerate(channels):
                    if len(roles) != len(ROLE_NAMES) or any(len(row) != len(ROLE_FIELDS) for row in roles):
                        raise ValueError("Role tensor must have shape [channels,11,32]")
                    decoded = [dict(zip(ROLE_FIELDS, values)) for values in roles]
                    for role in decoded:
                        if role["enter_ns"] <= 0 or role["exit_ns"] < role["enter_ns"]:
                            raise ValueError("Invalid role lifetime")
                        if role["total_ns"] != role["exit_ns"] - role["enter_ns"] or role["sample_stride"] != 32:
                            raise ValueError("Invalid role duration or instrumentation schema")
                    tx, rx = decoded[:2]
                    tails = dict(channel_role_span_ns=max(row["exit_ns"] for row in decoded)
                                  - min(row["enter_ns"] for row in decoded), prefix_wait_ns=rx["prefix_wait_ns"])
                    for name, begin, end in (
                        ("last_publish_to_tx_finished_ns", tx["last_publication_ns"], tx["tx_finished_ns"]),
                        ("tx_finished_to_tx_exit_ns", tx["tx_finished_ns"], tx["exit_ns"]),
                        ("rx_finished_to_rx_exit_ns", rx["rx_finished_ns"], rx["exit_ns"]),
                        ("last_publish_to_rx_finished_ns", tx["last_publication_ns"], rx["rx_finished_ns"]),
                        ("rx_finished_to_tx_exit_ns", rx["rx_finished_ns"], tx["exit_ns"]),
                        ("first_publish_after_enter_ns", tx["enter_ns"], tx["first_publication_ns"]),
                        ("last_publish_after_enter_ns", tx["enter_ns"], tx["last_publication_ns"]),
                    ):
                        if begin and end:
                            tails[name] = end - begin
                    if tx["tx_finished_ns"] and rx["rx_finished_ns"]:
                        tails["tx_finished_minus_rx_finished_ns"] = tx["tx_finished_ns"] - rx["rx_finished_ns"]
                    by_token[tokens].append(dict(rank=rank, iteration=iteration, channel=channel, roles=decoded, tails=tails))
    batches = []
    for tokens, observations in sorted(by_token.items()):
        tail_names = sorted({key for row in observations for key in row["tails"]})
        tails = {key: distribution([row["tails"][key] for row in observations if key in row["tails"]]) for key in tail_names}
        role_rows = {name: role_summary([row["roles"][index] for row in observations], index >= 2)
                     for index, name in enumerate(ROLE_NAMES)}
        batches.append(dict(tokens=tokens, channel_observations=len(observations), roles=role_rows, tails_ns=tails,
            per_rank_iteration_channel=[dict(rank=row["rank"], iteration=row["iteration"],
                                            channel=row["channel"], tails_ns=row["tails"]) for row in observations]))
    return dict(configuration=config, world_size=len(ranks), field_schema=list(ROLE_FIELDS), role_schema=list(ROLE_NAMES),
        batches=batches, notes=[
            "Durations use one lane-0 clock per role; roles and channels run concurrently and their times cannot be summed into kernel latency.",
            "Abort-check timers and call/load counters cover every instrumented call. Drain timing is complete only when every timed-sample count equals its call count; the older build sampled every 32nd drain call.",
            "Availability/descriptor and copy/fence timers use systematic stride-32 samples. complete_call_timing is inferred from actual sample/call counts, never from the shared sample_stride field alone.",
            "Call and signal-load counts are exact in the instrumented run; deterministic timing samples are not assumed unbiased and are not extrapolated.",
            "Worker iterations count completed tickets; control iterations count polling-loop iterations. Busy/not-ready counters count observations, not wait durations.",
            "Signed TX/RX tail differences describe local-GPU ordering. No absolute timestamps or cross-GPU clock comparison is exported.",
            "Instrumentation changes generated code and can perturb throughput; compare the instrumented CUDA-event latency against an uninstrumented run.",
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-dir", type=Path)
    parser.add_argument("--roles-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.torch_dir is None and args.roles_dir is None:
        parser.error("provide --torch-dir or --roles-dir")
    summary = dict(schema_version=1)
    if args.torch_dir is not None:
        summary["torch"] = analyze_torch(args.torch_dir)
    if args.roles_dir is not None:
        summary["roles"] = analyze_roles(args.roles_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="completed", sections=sorted(set(summary) - {"schema_version"}))))


if __name__ == "__main__":
    main()
