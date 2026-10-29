"""Compare two-round control-progress runs from anonymous CUDA-event samples.

Read two benchmark.py JSON files per variant, or one anonymous samples bundle.
This CPU-only program imports no Torch/native module. It validates the fixed
workload, recomputes iteration maxima across all eight ranks, and reports
per-GPU logical bandwidth T*K*H*2/t. Machine metadata and input paths are never
copied into its outputs. Runtime SHA-256 values are optional caller-supplied
provenance; this program does not infer a runtime identity from timing data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
ROUNDS = 2
ITERATIONS = 20
WORKLOAD = dict(world_size=8, experts=256, topk=8, hidden=7168, dtype="bfloat16", wire_dtype="bf16")
ROW_CONFIGURATION = dict(
    execution_mode="eager", receive_expand_mode="pipelined", transport_threads_per_cta=352,
    do_expand=True, stage=6, payload_slots=256, descriptor_slots=256, expert_alignment=1,
    num_max_tokens_per_rank=8192, warmup_iterations=5, measured_iterations=20, seed=20261024,
)
TRANSPORT_CONFIGURATION = dict(
    stage=6, channels=32, pipeline_stages=2, warp_specialization=True,
    fuse_receive_expand=True, receive_expand_mode="pipelined", source_fusion=True,
    wire_dtype="bf16", numa_mode="direct", replica_slots=256, enable_graph=False,
)


def valid_samples(values):
    if not isinstance(values, list) or len(values) != ITERATIONS:
        raise ValueError("Expected 20 timing samples per rank and round")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("CUDA-event durations must be finite positive numbers")
    return values


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def checked_runtime_sha(value):
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("Runtime SHA-256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def summarize_rounds(rounds):
    """Validate raw runs and retain only fixed configuration and timing arrays."""
    if len(rounds) != ROUNDS:
        raise ValueError("Expected two completed benchmark rounds per variant")
    samples = {tokens: [] for tokens in TOKENS}
    round_samples = {tokens: [] for tokens in TOKENS}
    anonymous_rounds = []
    full_configuration = None
    for document in rounds:
        rows = document.get("results", [])
        if not isinstance(rows, list) or sorted(row.get("tokens_per_rank", -1) for row in rows) != list(TOKENS):
            raise ValueError("Expected all ten distinct token counts in each round")
        anonymous_rows = []
        for row in sorted(rows, key=lambda item: item["tokens_per_rank"]):
            for key, expected in (WORKLOAD | ROW_CONFIGURATION).items():
                if row.get(key) != expected or type(row.get(key)) is not type(expected):
                    raise ValueError(f"Unexpected or missing benchmark field: {key}")
            options = row.get("transport_options")
            if not isinstance(options, dict):
                raise ValueError("Missing transport_options")
            for key, expected in TRANSPORT_CONFIGURATION.items():
                if options.get(key) != expected or type(options.get(key)) is not type(expected):
                    raise ValueError(f"Unexpected or missing transport option: {key}")
            # Also compare auxiliary flags across raw files. They are not
            # exported: only the fixed, explicitly allowed fields are public.
            current = {key: row[key] for key in ROW_CONFIGURATION} | {"transport_options": options}
            if full_configuration is None:
                full_configuration = current
            elif current != full_configuration:
                raise ValueError("Benchmark configuration changes across batches or rounds")
            recorded = valid_samples(row.get("full_dispatch_samples_us"))
            ranks = row.get("per_rank_dispatch_samples_us")
            if not isinstance(ranks, list) or len(ranks) != WORKLOAD["world_size"]:
                raise ValueError("Expected eight ranks of CUDA-event samples")
            for rank in ranks:
                valid_samples(rank)
            maxima = [max(rank[index] for rank in ranks) for index in range(ITERATIONS)]
            if any(not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)
                   for actual, expected in zip(recorded, maxima)):
                raise ValueError("Recorded full-dispatch sample does not match the maximum rank duration")
            distribution = row.get("full_dispatch")
            if distribution is not None and not math.isclose(
                distribution.get("mean_us", float("nan")), statistics.mean(maxima), rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ValueError("Recorded full-dispatch mean disagrees with raw rank samples")
            tokens = row["tokens_per_rank"]
            logical_bytes = tokens * WORKLOAD["topk"] * WORKLOAD["hidden"] * 2
            if "logical_bytes_per_gpu" in row and row["logical_bytes_per_gpu"] != logical_bytes:
                raise ValueError("Recorded logical bytes disagree with T*K*H*2")
            expected_bandwidth = logical_bytes / statistics.mean(maxima) / 1000
            if "logical_GBps" in row and not math.isclose(
                row["logical_GBps"], expected_bandwidth, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("Recorded logical bandwidth disagrees with raw full-dispatch samples")
            samples[tokens].extend(maxima)
            round_samples[tokens].append(maxima)
            anonymous_rows.append(dict(
                tokens_per_rank=tokens, **WORKLOAD, **ROW_CONFIGURATION,
                transport_options=TRANSPORT_CONFIGURATION.copy(),
                full_dispatch_samples_us=maxima, per_rank_dispatch_samples_us=ranks,
            ))
        anonymous_rounds.append({"results": anonymous_rows})
    return samples, round_samples, full_configuration, anonymous_rounds


def compare(baseline_rounds, optimized_rounds):
    baseline, baseline_by_round, baseline_config, baseline_anonymous = summarize_rounds(baseline_rounds)
    optimized, optimized_by_round, optimized_config, optimized_anonymous = summarize_rounds(optimized_rounds)
    if baseline_config != optimized_config:
        changed = [key for key in baseline_config if baseline_config[key] != optimized_config.get(key)]
        raise ValueError(f"Mismatched baseline/optimized configuration: {', '.join(changed)}")
    rows = []
    for tokens in TOKENS:
        baseline_us = statistics.mean(baseline[tokens])
        optimized_us = statistics.mean(optimized[tokens])
        baseline_means = [statistics.mean(values) for values in baseline_by_round[tokens]]
        optimized_means = [statistics.mean(values) for values in optimized_by_round[tokens]]
        logical_bytes = tokens * WORKLOAD["topk"] * WORKLOAD["hidden"] * 2
        rows.append(dict(
            tokens_per_gpu=tokens, logical_bytes_per_gpu=logical_bytes,
            baseline_full_mean_ms=baseline_us / 1000, optimized_full_mean_ms=optimized_us / 1000,
            baseline_logical_GBps=logical_bytes / baseline_us / 1000,
            optimized_logical_GBps=logical_bytes / optimized_us / 1000,
            speedup=baseline_us / optimized_us,
            latency_reduction_percent=(1 - optimized_us / baseline_us) * 100,
            baseline_full_p50_ms=percentile(baseline[tokens], 50) / 1000,
            baseline_full_p95_ms=percentile(baseline[tokens], 95) / 1000,
            optimized_full_p50_ms=percentile(optimized[tokens], 50) / 1000,
            optimized_full_p95_ms=percentile(optimized[tokens], 95) / 1000,
            baseline_round1_mean_ms=baseline_means[0] / 1000,
            baseline_round2_mean_ms=baseline_means[1] / 1000,
            optimized_round1_mean_ms=optimized_means[0] / 1000,
            optimized_round2_mean_ms=optimized_means[1] / 1000,
            baseline_round_mean_max_over_min=max(baseline_means) / min(baseline_means),
            optimized_round_mean_max_over_min=max(optimized_means) / min(optimized_means),
            baseline_sample_count=len(baseline[tokens]), optimized_sample_count=len(optimized[tokens]),
        ))
    report = dict(
        formula="T*K*H*2/t", bandwidth_unit="decimal GB/s",
        T_scope="input tokens per GPU; no extra world-size factor or deduplication",
        latency_definition="mean of 40 iteration maximum-of-8-rank full dispatch CUDA-event durations",
        rounds_per_variant=ROUNDS, measured_iterations_per_round=ITERATIONS,
        samples_per_batch_per_variant=ROUNDS * ITERATIONS,
        workload=WORKLOAD | dict(element_bytes=2),
        configuration=ROW_CONFIGURATION | {"transport_options": TRANSPORT_CONFIGURATION},
        all_batches_faster=all(row["speedup"] > 1 for row in rows),
        rows=rows,
    )
    anonymous = dict(
        timing_unit="microseconds", workload=WORKLOAD | dict(element_bytes=2),
        variant_definition="baseline and optimized runtime implementations with identical benchmark configuration",
        variants=dict(baseline=baseline_anonymous, optimized=optimized_anonymous),
    )
    return report, anonymous


def markdown(report):
    lines = [
        "| Tokens/GPU | Baseline ms | Optimized ms | Baseline GB/s | Optimized GB/s | Speedup | Latency reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['tokens_per_gpu']} | {row['baseline_full_mean_ms']:.3f} | "
            f"{row['optimized_full_mean_ms']:.3f} | {row['baseline_logical_GBps']:.3f} | "
            f"{row['optimized_logical_GBps']:.3f} | {row['speedup']:.3f}x | "
            f"{row['latency_reduction_percent']:.2f}% |"
        )
    lines += [
        "", "Bandwidth = T*K*H*2/t: T per GPU, K=8, H=7168, BF16; decimal GB/s, no extra GPU-count factor.",
        "Each variant: two rounds x 20 samples/batch. Each sample is recomputed as the maximum across eight ranks.",
        "Full latency is the mean of all 40 samples/batch, including layout, transport, expansion and GPU stream waits. "
        "CPU setup and diagnostic readback are outside the benchmark's timed interval.",
        "The exported JSON/CSV also retain pooled p50/p95 and the two round means. These are observations, not confidence intervals.",
        "Pipelined transport uses 352 threads (11 warps)/CTA in both variants. Bandwidth is logical expert volume, not measured PCIe/UPI bytes.",
    ]
    runtime = report.get("runtime_sha256")
    if runtime:
        lines += ["", f"Caller-supplied baseline runtime SHA-256: `{runtime['baseline']}`.",
                  f"Caller-supplied optimized runtime SHA-256: `{runtime['optimized']}`."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path, help="Anonymous samples bundle for both variants")
    source.add_argument("--baseline", type=Path, nargs=2, metavar=("ROUND1", "ROUND2"))
    parser.add_argument("--optimized", type=Path, nargs=2, metavar=("ROUND1", "ROUND2"))
    parser.add_argument("--baseline-runtime-sha256")
    parser.add_argument("--optimized-runtime-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline_sha = checked_runtime_sha(args.baseline_runtime_sha256)
    optimized_sha = checked_runtime_sha(args.optimized_runtime_sha256)
    if (baseline_sha is None) != (optimized_sha is None):
        parser.error("Supply both runtime SHA-256 values or neither")
    runtime = dict(baseline=baseline_sha, optimized=optimized_sha) if baseline_sha else None
    if args.samples:
        if args.optimized is not None:
            parser.error("--samples cannot be combined with --optimized")
        payload = args.samples.read_bytes()
        document = json.loads(payload)
        variants = document.get("variants", {})
        if set(variants) != {"baseline", "optimized"}:
            raise ValueError("Samples bundle must contain baseline and optimized variants")
        report, anonymous = compare(variants["baseline"], variants["optimized"])
        recorded_runtime = document.get("runtime_sha256")
        if recorded_runtime is not None:
            if not isinstance(recorded_runtime, dict) or set(recorded_runtime) != {"baseline", "optimized"}:
                raise ValueError("Malformed runtime SHA-256 provenance")
            recorded_runtime = {key: checked_runtime_sha(value) for key, value in recorded_runtime.items()}
            if any(value is None for value in recorded_runtime.values()):
                raise ValueError("Samples runtime SHA-256 values cannot be null")
            if runtime is not None and runtime != recorded_runtime:
                raise ValueError("Runtime SHA-256 parameters disagree with samples provenance")
            runtime = recorded_runtime
        report["anonymous_samples_sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        if args.optimized is None:
            parser.error("--baseline requires two --optimized files")
        contents = {name: [path.read_bytes() for path in paths]
                    for name, paths in (("baseline", args.baseline), ("optimized", args.optimized))}
        report, anonymous = compare(
            [json.loads(data) for data in contents["baseline"]],
            [json.loads(data) for data in contents["optimized"]],
        )
        report["input_sha256"] = {name: [hashlib.sha256(data).hexdigest() for data in payloads]
                                  for name, payloads in contents.items()}
        report["input_hash_scope"] = "Original input byte digests; machine metadata and paths are not exported"
    if runtime is not None:
        report["runtime_sha256"] = runtime
        anonymous["runtime_sha256"] = runtime
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_payload = (json.dumps(anonymous, indent=2) + "\n").encode()
    (args.output_dir / "control_progress_samples.json").write_bytes(samples_payload)
    report["exported_samples_sha256"] = hashlib.sha256(samples_payload).hexdigest()
    prefix = args.output_dir / "control_progress_comparison"
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(report["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["rows"])
    table = markdown(report)
    prefix.with_suffix(".md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
