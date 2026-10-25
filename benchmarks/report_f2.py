"""CPU-only F2 comparison from preserved, counterbalanced BF16 benchmark JSON.

python benchmarks/report_f2.py --results-dir results/f2 --no-plots
python benchmarks/report_f2.py --results-dir results/f2

No PyTorch/CUDA import or network access. Matplotlib is optional for --no-plots.
Different run IDs remain separate; missing rows or audits are never filled in.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

from report_advanced import TOKENS, validate_row

VARIANTS = {
    "scalar_f2": "Scalar F2",
    "nof2": "No F2",
    "vectorized": "Vectorized F2",
    "pipelined": "Pipelined F2 (11 warps)",
}
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, percent):
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def distribution(values):
    return dict(mean_us=statistics.mean(values), p50_us=percentile(values, 50), p95_us=percentile(values, 95))


def validate_options(row, variant, source):
    options = row["transport_options"]
    expected_mode = variant if variant in ("vectorized", "pipelined") else "scalar"
    expected = dict(
        stage=6,
        channels=32,
        pipeline_stages=2,
        warp_specialization=True,
        source_fusion=True,
        numa_mode="direct",
        wire_dtype="bf16",
        fuse_receive_expand=variant != "nof2",
        receive_expand_mode=expected_mode,
    )
    for key, value in expected.items():
        if options.get(key) != value:
            raise ValueError(f"{source}: {key}={options.get(key)!r}, expected {value!r}")
    for key, value in dict(payload_slots=256, descriptor_slots=256, expert_alignment=1, execution_mode="eager").items():
        if row.get(key) != value:
            raise ValueError(f"{source}: {key} must be {value!r}")
    for key, value in dict(
        seed=20261024, warmup_iterations=5, measured_iterations=20, num_max_tokens_per_rank=8192
    ).items():
        if key in row and row[key] != value:
            raise ValueError(f"{source}: {key} must be {value!r}")
    if row.get("transport_threads_per_cta") != (352 if variant == "pipelined" else 256):
        raise ValueError(f"{source}: wrong CTA size for {variant}")


def load_sources(results_dir):
    datasets = []
    for variant in VARIANTS:
        for source in sorted((results_dir / "runs").glob(f"*/round*/dispatch_{variant}.json")):
            record = json.loads(source.read_text())
            run_id, round_id = source.parent.parent.name, source.parent.name
            if round_id not in ("round1", "round2"):
                raise ValueError(f"{source}: unknown counterbalanced round")
            rows = []
            for raw in record["results"]:
                normalized = validate_row(raw, source)
                validate_options(raw, variant, source)
                samples = [float(item) for item in raw["full_dispatch_samples_us"]]
                if len(samples) != 20:
                    raise ValueError(f"{source}: expected 20 completed measured samples per T")
                transport = [float(item) for item in raw.get("transport_only_samples_us", [])]
                if transport and len(transport) != len(samples):
                    raise ValueError(f"{source}: incomplete transport samples")
                if transport:
                    for key, value in distribution(transport).items():
                        if not math.isclose(value, raw["transport_only"][key], rel_tol=1e-6):
                            raise ValueError(f"{source}: transport {key} does not match raw samples")
                for key in ("p50_us", "p95_us"):
                    if not math.isclose(distribution(samples)[key], raw["full_dispatch"][key], rel_tol=1e-6):
                        raise ValueError(f"{source}: {key} does not match raw samples")
                stats = raw.get("runtime_stats_per_rank", [])
                if (
                    len(stats) != 8
                    or sum(int(item["raw"][0]) for item in stats) != raw["remote_source_tokens"]
                    or sum(int(item["raw"][1]) for item in stats) != raw["remote_deliveries"]
                ):
                    raise ValueError(f"{source}: runtime U/F counters disagree with routes")
                overlap = raw.get("receive_expand_overlap_per_rank", [None] * 8)
                if len(overlap) != 8:
                    raise ValueError(f"{source}: incomplete rank overlap diagnostics")
                overlap_counts = []
                for matrix in overlap:
                    if variant in ("scalar_f2", "nof2"):
                        if matrix is not None:
                            raise ValueError(f"{source}: scalar path should not expose new diagnostics")
                        overlap_counts.append(None)
                    else:
                        if not isinstance(matrix, list) or len(matrix) != 32 or any(len(row) != 2 for row in matrix):
                            raise ValueError(f"{source}: expected overlap shape [32,2]")
                        if variant == "vectorized" and any(any(row) for row in matrix):
                            raise ValueError(f"{source}: vectorized diagnostic must remain zero")
                        overlap_counts.append(sum(int(row[0]) for row in matrix))
                rows.append(
                    dict(
                        normalized,
                        run_id=run_id,
                        round_id=round_id,
                        variant=variant,
                        full_samples_us=samples,
                        transport_samples_us=transport,
                        transport_p50_us=distribution(transport)["p50_us"] if transport else None,
                        transport_p95_us=distribution(transport)["p95_us"] if transport else None,
                        # Diagnostic arithmetic only: maxima may belong to different ranks.
                        full_minus_transport_mean_us=(
                            (normalized["full_mean_us"] - normalized["transport_mean_us"])
                            if normalized["transport_mean_us"]
                            else None
                        ),
                        overlap_count_last_dispatch_per_rank=overlap_counts,
                        receive_expand_diagnostic_raw_last_dispatch=overlap,
                        source=str(source.resolve()),
                        source_sha256=sha256(source),
                    )
                )
            tokens = [row["tokens_per_rank"] for row in rows]
            if len(tokens) != len(set(tokens)) or any(token not in TOKENS for token in tokens):
                raise ValueError(f"{source}: duplicate or unexpected token count")
            datasets.append(
                dict(
                    run_id=run_id,
                    round_id=round_id,
                    variant=variant,
                    complete_10_token_grid=sorted(tokens) == list(TOKENS),
                    source=str(source.resolve()),
                    source_sha256=sha256(source),
                    rows=rows,
                )
            )
    if not datasets:
        raise ValueError(f"No completed runs/*/round*/dispatch_*.json in {results_dir}; no measurements invented")
    # Paired comparisons require identical routing and byte accounting at each T.
    volumes = {}
    for dataset in datasets:
        for row in dataset["rows"]:
            key = row["run_id"], row["tokens_per_rank"]
            volume = tuple(row[name] for name in ("remote_source_tokens", "remote_deliveries", "unique_wire_bytes"))
            if key in volumes and volumes[key] != volume:
                raise ValueError(f"{row['source']}: routing differs between paired measurements")
            volumes[key] = volume
    return datasets


def audit_sources(results_dir, datasets):
    evidence = []
    for run_id, round_id in sorted({(item["run_id"], item["round_id"]) for item in datasets}):
        directory = results_dir / "runs" / run_id / round_id
        delta_path = directory / "nvlink_counter_delta.json"
        delta = json.loads(delta_path.read_text()) if delta_path.exists() else None
        counter_passed = bool(
            delta
            and delta.get("status") == "passed"
            and delta.get("counter_count") == 288
            and delta.get("complete")
            and delta.get("monotonic")
            and delta.get("all_payload_deltas_zero")
        )
        exit_path = directory / "round.exit"
        exit_code = exit_path.read_text().strip() if exit_path.exists() else None
        manifest_path = directory / "executed_source_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        run_directory = directory.parent
        final_path = run_directory / "final_source_manifest.json"
        final = json.loads(final_path.read_text()) if final_path.exists() else None
        evidence.append(
            dict(
                run_id=run_id,
                round_id=round_id,
                counter_audit_passed=counter_passed,
                round_exit=exit_code,
                manifest_path=str(manifest_path.resolve()),
                manifest_sha256=sha256(manifest_path) if manifest else None,
                source_unchanged_at_end=bool(manifest and final and manifest == final),
                counter_path=str(delta_path.resolve()),
                counter_sha256=sha256(delta_path) if delta else None,
                status=(
                    "passed"
                    if counter_passed and exit_code == "0" and manifest and final == manifest
                    else "incomplete_or_failed"
                ),
            )
        )
    return evidence


def aggregate(datasets):
    groups = {}
    for dataset in datasets:
        for row in dataset["rows"]:
            groups.setdefault((row["run_id"], row["variant"], row["tokens_per_rank"]), []).append(row)
    summaries = []
    for (run_id, variant, token), rows in sorted(groups.items()):
        samples = [sample for row in rows for sample in row["full_samples_us"]]
        transports = [sample for row in rows for sample in row["transport_samples_us"]]
        means = {row["round_id"]: row["full_mean_us"] for row in rows}
        full = distribution(samples)
        summaries.append(
            dict(
                run_id=run_id,
                variant=variant,
                tokens_per_rank=token,
                round_count=len(rows),
                round_full_mean_us=means,
                round_mean_min_us=min(means.values()),
                round_mean_max_us=max(means.values()),
                round_mean_max_over_min=max(means.values()) / min(means.values()),
                sample_count=len(samples),
                full_mean_us=full["mean_us"],
                full_p50_us=full["p50_us"],
                full_p95_us=full["p95_us"],
                transport_mean_us=statistics.mean(transports) if transports else None,
                transport_p50_us=percentile(transports, 50) if transports else None,
                transport_p95_us=percentile(transports, 95) if transports else None,
                unique_wire_bytes=rows[0]["unique_wire_bytes"],
                unique_wire_GBps_aggregate=rows[0]["unique_wire_bytes"] / full["mean_us"] / 1000,
                unique_wire_GBps_per_gpu_average=rows[0]["unique_wire_bytes"] / full["mean_us"] / 1000 / 8,
                overlap_count_last_dispatch_per_round={
                    row["round_id"]: row["overlap_count_last_dispatch_per_rank"] for row in rows
                },
            )
        )
    return summaries


def write_csv(path, rows):
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, keys)
        writer.writeheader()
        writer.writerows(
            {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            for row in rows
        )


def plot_run(run_id, summaries, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = dict(zip(VARIANTS, ("#6b7280", "#e48b20", "#1976b9", "#218c56")))
    for variant, label in VARIANTS.items():
        rows = sorted(
            (row for row in summaries if row["run_id"] == run_id and row["variant"] == variant),
            key=lambda row: row["tokens_per_rank"],
        )
        if not rows:
            continue
        tokens = [row["tokens_per_rank"] for row in rows]
        axes[0].plot(tokens, [row["full_mean_us"] / 1000 for row in rows], "o-", label=label, color=colors[variant])
        axes[0].fill_between(
            tokens,
            [row["round_mean_min_us"] / 1000 for row in rows],
            [row["round_mean_max_us"] / 1000 for row in rows],
            color=colors[variant],
            alpha=0.14,
        )
        axes[1].plot(tokens, [row["unique_wire_GBps_aggregate"] for row in rows], "o-", color=colors[variant])
        axes[1].fill_between(
            tokens,
            [row["unique_wire_bytes"] / row["round_mean_max_us"] / 1000 for row in rows],
            [row["unique_wire_bytes"] / row["round_mean_min_us"] / 1000 for row in rows],
            color=colors[variant],
            alpha=0.14,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Full dispatch latency (ms)")
    axes[1].set_ylabel("Remote BF16 delivery bandwidth (aggregate GB/s)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(TOKENS, labels=[str(token) for token in TOKENS], rotation=40)
        ax.set_xlabel("Tokens per GPU")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
    fig.suptitle("BF16 direct dispatch · 8 GPUs · C32 · H7168 · top-k 8")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.29, wspace=0.30)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{run_id}_f2_comparison.{suffix}", dpi=180)
    plt.close(fig)


def fmt(value, scale=1):
    return "—" if value is None else f"{value / scale:.3f}"


def markdown(datasets, summaries, evidence, plots):
    lines = [
        "# F2 receive/expand comparison",
        "",
        "BF16 direct, DP8/EP8, 256 experts, top-k 8, H7168, C32, Q=L256/channel, pipeline stages 2, "
        "source fusion enabled, expanded output, alignment 1, seed 20261024. Five warmups and twenty measured iterations per point per round.",
        "",
        "Full dispatch is the primary comparison: each sample is the maximum CUDA-event duration across eight ranks. "
        "The two rounds run in opposite order. Their raw results and means remain separate; pooled p50/p95 combine equal-sized rounds "
        "and are not confidence intervals. Shading, when present, spans the two round means, not per-iteration variance.",
        "",
        "Remote BF16 delivery bandwidth = F×7168×2 / full latency, decimal aggregate GB/s across eight GPUs. "
        "F counts each remote source-token/destination-GPU pair once, independently of local expert multiplicity. "
        "This is logical useful payload throughput; it does not measure physical PCIe/UPI traffic and excludes metadata/control/padding.",
        "",
        "Transport intervals include fused receive/expand work for F2 paths and exclude the separately launched scatter for no-F2. "
        "Their subtraction from full latency is diagnostic arithmetic, not an isolated scatter timer: rank maxima may select different GPUs. "
        "Pipelined F2 uses 352 threads / 11 warps per CTA; the other paths use 256 / 8. "
        "The comparison therefore includes the resource and scheduling effect of three extra warps.",
        "",
        "Overlap diagnostics describe only the last measured dispatch. Sum column 0 only: it conservatively observes RX producer "
        "starting a token before the preceding token's designated scatter warp publishes completion. Column 1 is a completed-ticket+1 "
        "watermark. These counts do not establish overlap duration, utilization, or performance causality.",
        "",
        "## Coverage and execution evidence",
        "",
        "| Run | Round | Variant | Complete 10T | Raw source SHA256 |",
        "|---|---|---|---|---|",
    ]
    for dataset in datasets:
        lines.append(
            f"| {dataset['run_id']} | {dataset['round_id']} | {VARIANTS[dataset['variant']]} | "
            f"{dataset['complete_10_token_grid']} | `{dataset['source_sha256']}` |"
        )
    lines += ["", "| Run | Round | 288 counters unchanged | Exit | Source unchanged |", "|---|---|---|---:|---|"]
    for item in evidence:
        lines.append(
            f"| {item['run_id']} | {item['round_id']} | {item['counter_audit_passed']} | {item['round_exit']} | {item['source_unchanged_at_end']} |"
        )
    for run_id in sorted({row["run_id"] for row in summaries}):
        lines += ["", f"## {run_id}", ""]
        if plots:
            lines += [f"![F2 comparison]({run_id}_f2_comparison.png)", ""]
        lines += [
            "| T/GPU | Path | Rounds | Full mean ms | Round means min–max ms | Pooled p50 ms | Pooled p95 ms | Transport mean ms | F×B/full GB/s |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        rows = sorted(
            (row for row in summaries if row["run_id"] == run_id),
            key=lambda row: (row["tokens_per_rank"], list(VARIANTS).index(row["variant"])),
        )
        for row in rows:
            lines.append(
                f"| {row['tokens_per_rank']} | {VARIANTS[row['variant']]} | {row['round_count']} | "
                f"{fmt(row['full_mean_us'], 1000)} | {fmt(row['round_mean_min_us'], 1000)}–{fmt(row['round_mean_max_us'], 1000)} | "
                f"{fmt(row['full_p50_us'], 1000)} | {fmt(row['full_p95_us'], 1000)} | "
                f"{fmt(row['transport_mean_us'], 1000)} | {fmt(row['unique_wire_GBps_aggregate'])} |"
            )
        paired = {(row["variant"], row["tokens_per_rank"]): row for row in rows}
        lines += [
            "",
            "At T8192, completed paired full-latency ratios (greater than 1 means the named new path is faster):",
            "",
        ]
        found = False
        for variant in ("vectorized", "pipelined"):
            candidate = paired.get((variant, 8192))
            if candidate:
                for baseline in ("scalar_f2", "nof2"):
                    reference = paired.get((baseline, 8192))
                    if reference:
                        found = True
                        lines.append(
                            f"- {VARIANTS[baseline]} / {VARIANTS[variant]}: "
                            f"{reference['full_mean_us'] / candidate['full_mean_us']:.3f}×; "
                            f"new-path round-mean max/min {candidate['round_mean_max_over_min']:.3f}×."
                        )
        if not found:
            lines.append("No complete T8192 comparison is available yet.")
    lines += [
        "",
        "The JSON/CSV report includes per-round full/transport p50 and p95, raw-source hashes, overlap diagnostics and audit paths. "
        "Incomplete audits or a missing second round remain visible above; no absent result is estimated.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/f2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output = args.output_dir or results_dir / "report"
    datasets = load_sources(results_dir)
    evidence = audit_sources(results_dir, datasets)
    summaries = aggregate(datasets)
    output.mkdir(parents=True, exist_ok=True)
    raw_rows = [row for dataset in datasets for row in dataset["rows"]]
    report = dict(
        generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        datasets=datasets,
        summaries=summaries,
        execution_evidence=evidence,
        aggregation="Within each run ID, pool equal-sized 20-sample rounds; retain individual round means/ranges; no run IDs pooled",
        plots_requested=not args.no_plots,
    )
    (output / "f2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(output / "f2_per_round.csv", raw_rows)
    write_csv(output / "f2_summary.csv", summaries)
    if not args.no_plots:
        for run_id in sorted({row["run_id"] for row in summaries}):
            plot_run(run_id, summaries, output)
    (output / "f2_report.md").write_text(markdown(datasets, summaries, evidence, not args.no_plots))
    print(f"CPU report: {len(datasets)} raw files, {len(raw_rows)} rows; {output}")


if __name__ == "__main__":
    main()
