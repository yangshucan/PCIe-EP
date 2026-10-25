"""Reproduce the latest pipelined F2 table using anonymous timing samples.

This CPU-only report uses T*K*H*element_bytes/t, with T per GPU. It accepts
either the included anonymous data or a newly completed two-round run folder.
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
ROOT = Path(__file__).resolve().parents[1]


def load_measurements(args):
    if args.run_dir is None:
        return json.loads(args.input.read_text())["rounds"]
    rounds = []
    for name in ("round1", "round2"):
        path = args.run_dir / name / "dispatch_pipelined.json"
        raw = json.loads(path.read_text())
        rows = []
        for row in raw["results"]:
            for key, value in dict(world_size=8, topk=8, hidden=7168, experts=256,
                                   dtype="bfloat16", receive_expand_mode="pipelined").items():
                if row.get(key) != value:
                    raise ValueError(f"Unexpected workload field: {key}")
            rows.append(dict(tokens_per_gpu=row["tokens_per_rank"],
                             full_dispatch_samples_us=row["full_dispatch_samples_us"],
                             per_rank_dispatch_samples_us=row["per_rank_dispatch_samples_us"]))
        rounds.append(dict(round=name, measurements=rows))
    return rounds


def summarize(rounds):
    if len(rounds) != 2:
        raise ValueError("Expected two completed measurement rounds")
    samples = {tokens: [] for tokens in TOKENS}
    for group in rounds:
        rows = group["measurements"]
        if sorted(row["tokens_per_gpu"] for row in rows) != list(TOKENS):
            raise ValueError("Expected all ten distinct batch sizes per round")
        for row in rows:
            values = row["full_dispatch_samples_us"]
            ranks = row["per_rank_dispatch_samples_us"]
            if len(values) != 20 or len(ranks) != 8 or any(len(rank) != 20 for rank in ranks):
                raise ValueError("Incomplete 8-rank timing samples")
            if any(not math.isfinite(value) or value <= 0 for rank in ranks for value in rank):
                raise ValueError("Invalid CUDA-event duration")
            if any(not math.isclose(value, max(rank[i] for rank in ranks), rel_tol=1e-9)
                   for i, value in enumerate(values)):
                raise ValueError("Sample does not match the maximum rank duration")
            samples[row["tokens_per_gpu"]].extend(values)
    output = []
    for tokens, values in samples.items():
        mean_us = statistics.mean(values)
        payload = tokens * 8 * 7168 * 2
        output.append(dict(tokens_per_gpu=tokens, topk=8, hidden=7168, element_bytes=2,
                           logical_bytes_per_gpu=payload, full_mean_ms=mean_us / 1000,
                           bandwidth_GBps=payload / mean_us / 1000, sample_count=len(values)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "benchmarks/data/f2_pipelined_samples.json")
    parser.add_argument("--run-dir", type=Path, help="A local run directory containing round1 and round2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/data")
    args = parser.parse_args()
    rows = summarize(load_measurements(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(variant="pipelined", formula="T*K*H*b/t", bandwidth_unit="decimal GB/s",
                  T_scope="input tokens per GPU; no extra world-size factor", rows=rows)
    if args.run_dir is None:
        report["anonymous_samples_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    (args.output_dir / "f2_pipelined_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "f2_pipelined_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("| Tokens/GPU | Full dispatch ms | Bandwidth GB/s |")
    print("|---:|---:|---:|")
    for row in rows:
        print(f"| {row['tokens_per_gpu']} | {row['full_mean_ms']:.3f} | {row['bandwidth_GBps']:.3f} |")


if __name__ == "__main__":
    main()
