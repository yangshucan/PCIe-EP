"""Compare baseline and prefix/TX-overlap runs using anonymous timing data.

Supply one anonymous samples file, or two benchmark JSON files per variant,
with all ten batches and 20 CUDA-event samples per batch per round. This
CPU-only report recomputes the per-GPU logical bandwidth from raw samples,
including for older benchmark JSON files without logical bandwidth fields.
Machine metadata and source paths are never copied into the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
ROUNDS = 2
ITERATIONS = 20
WORKLOAD = dict(world_size=8, experts=256, topk=8, hidden=7168, dtype="bfloat16", wire_dtype="bf16")
CONFIG_FIELDS = (
    "transport_options", "execution_mode", "receive_expand_mode", "transport_threads_per_cta",
    "do_expand", "stage", "payload_slots", "descriptor_slots", "expert_alignment",
    "num_max_tokens_per_rank", "warmup_iterations", "seed",
)


def valid_samples(values):
    if not isinstance(values, list) or len(values) != ITERATIONS:
        raise ValueError("Expected 20 timing samples per round")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Invalid CUDA-event duration")
    return values


def summarize_rounds(rounds):
    """Validate complete raw benchmark rounds and pool iteration maxima."""
    if len(rounds) != ROUNDS:
        raise ValueError("Expected two completed benchmark rounds per variant")
    samples = {tokens: [] for tokens in TOKENS}
    configuration = None
    for group in rounds:
        rows = group.get("results", [])
        if sorted(row.get("tokens_per_rank", -1) for row in rows) != list(TOKENS):
            raise ValueError("Expected all ten distinct batch sizes per round")
        for row in rows:
            for key, expected in WORKLOAD.items():
                if row.get(key) != expected:
                    raise ValueError(f"Unexpected workload field: {key}")
            if row.get("measured_iterations") != ITERATIONS:
                raise ValueError("Expected measured_iterations=20")
            missing = [key for key in CONFIG_FIELDS if key not in row]
            if missing:
                raise ValueError(f"Missing benchmark configuration fields: {', '.join(missing)}")
            current = {key: row[key] for key in CONFIG_FIELDS}
            if configuration is None:
                configuration = current
            elif current != configuration:
                raise ValueError("Mismatched benchmark configuration across batches or rounds")
            maxima = valid_samples(row.get("full_dispatch_samples_us"))
            ranks = row.get("per_rank_dispatch_samples_us")
            if not isinstance(ranks, list) or len(ranks) != WORKLOAD["world_size"]:
                raise ValueError("Expected eight ranks of timing samples")
            for rank in ranks:
                valid_samples(rank)
            if any(not math.isclose(value, max(rank[index] for rank in ranks), rel_tol=1e-9, abs_tol=1e-6)
                   for index, value in enumerate(maxima)):
                raise ValueError("Sample does not match the maximum rank duration")
            samples[row["tokens_per_rank"]].extend(maxima)
    return samples, configuration


def compare(baseline_rounds, optimized_rounds):
    baseline, baseline_config = summarize_rounds(baseline_rounds)
    optimized, optimized_config = summarize_rounds(optimized_rounds)
    if baseline_config != optimized_config:
        changed = [key for key in CONFIG_FIELDS if baseline_config[key] != optimized_config[key]]
        raise ValueError(f"Mismatched baseline/optimized configuration: {', '.join(changed)}")
    rows = []
    for tokens in TOKENS:
        baseline_us = statistics.mean(baseline[tokens])
        optimized_us = statistics.mean(optimized[tokens])
        logical_bytes = tokens * WORKLOAD["topk"] * WORKLOAD["hidden"] * 2
        rows.append(dict(
            tokens_per_gpu=tokens,
            logical_bytes_per_gpu=logical_bytes,
            baseline_full_mean_ms=baseline_us / 1000,
            optimized_full_mean_ms=optimized_us / 1000,
            baseline_logical_GBps=logical_bytes / baseline_us / 1000,
            optimized_logical_GBps=logical_bytes / optimized_us / 1000,
            speedup=baseline_us / optimized_us,
            latency_reduction_percent=(1 - optimized_us / baseline_us) * 100,
            baseline_sample_count=len(baseline[tokens]),
            optimized_sample_count=len(optimized[tokens]),
        ))
    return dict(
        formula="T*K*H*2/t",
        bandwidth_unit="decimal GB/s",
        T_scope="input tokens per GPU; no extra world-size factor or deduplication",
        latency_definition="mean of all iteration maximum-rank full dispatch CUDA-event samples",
        rounds_per_variant=ROUNDS,
        measured_iterations_per_round=ITERATIONS,
        samples_per_batch_per_variant=ROUNDS * ITERATIONS,
        world_size=WORKLOAD["world_size"],
        workload=WORKLOAD | dict(element_bytes=2),
        configuration=baseline_config,
        rows=rows,
    )


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
    lines.extend([
        "", "Bandwidth = T*K*H*2/t, T per GPU, K=8, H=7168, BF16; decimal GB/s.",
        "Each variant: 2 rounds x 20 samples per batch; each sample is the maximum across 8 ranks.",
        "Latency is the mean of all 40 samples per batch. Speedup = baseline latency / optimized latency.",
    ])
    return "\n".join(lines) + "\n"


def report_from_samples(path):
    """Reproduce the comparison using only the publishable samples bundle."""
    raw = json.loads(path.read_text())
    variants = raw.get("variants", {})
    if set(variants) != {"baseline", "optimized"}:
        raise ValueError("Expected baseline and optimized variants in samples file")
    report = compare(variants["baseline"], variants["optimized"])
    report["anonymous_samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path, help="One anonymous samples bundle containing both variants")
    source.add_argument("--baseline", type=Path, nargs="+", help="Two raw baseline benchmark JSON files")
    parser.add_argument("--optimized", type=Path, nargs="+", help="Two raw optimized benchmark JSON files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples is not None:
        if args.optimized is not None:
            parser.error("--samples cannot be combined with --optimized")
        report = report_from_samples(args.samples)
    else:
        if args.optimized is None:
            parser.error("--baseline requires --optimized")
        baseline = [json.loads(path.read_text()) for path in args.baseline]
        optimized = [json.loads(path.read_text()) for path in args.optimized]
        report = compare(baseline, optimized)
        report["anonymous_input_sha256"] = {
            name: [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            for name, paths in (("baseline", args.baseline), ("optimized", args.optimized))
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "prefix_tx_comparison"
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
