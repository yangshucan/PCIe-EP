"""Render S5-S8 tables and static figures from completed benchmark JSON rows.

This program runs on CPU and imports no CUDA/PyTorch code. Missing or interrupted
configurations remain visible as incomplete coverage; no measurements are filled
in. Examples::

    python benchmarks/report_advanced.py --no-plots
    python benchmarks/report_advanced.py --baseline results/refactor/dispatch.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
EXPECTED = {
    "dispatch_s5": "S5 warp + F2",
    "dispatch_s6": "S6 source fusion",
    "dispatch_s8_direct": "S8 direct",
    "dispatch_s8_relay": "S8 NUMA relay",
    "dispatch_cached": "S6 cached layout",
    "dispatch_graph": "S6 Graph",
    "dispatch_graph_cached": "S6 cached Graph",
    "dispatch_s7_fp8": "S7 FP8 wire",
}


def positive(value: Any, description: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{description} must be positive and finite: {value}")
    return number


def validate_row(row: dict, source: Path) -> dict:
    """Check units and deduplicated traffic before comparing any records."""
    label = f"{source}: T={row.get('tokens_per_rank')}"
    world = int(row["world_size"])
    hidden = int(row["hidden"])
    if world != 8 or (int(row["experts"]), int(row["topk"]), hidden) != (256, 8, 7168):
        raise ValueError(f"{label}: expected DP8/EP8/E256/K8/H7168")
    if row["dtype"] != "bfloat16" or not row.get("do_expand", False):
        raise ValueError(f"{label}: expected BF16 input/output and expanded expert rows")
    wire_dtype = row.get("wire_dtype", "bf16")
    if wire_dtype not in ("bf16", "fp8_e4m3"):
        raise ValueError(f"{label}: unsupported wire dtype {wire_dtype}")
    wire_bytes_per_token = hidden * 2 if wire_dtype == "bf16" else hidden + 4
    source_tokens, deliveries = int(row["remote_source_tokens"]), int(row["remote_deliveries"])
    expected_bytes = {
        "host_write_bytes": source_tokens * wire_bytes_per_token,
        "unique_wire_bytes": deliveries * wire_bytes_per_token,
        "host_bus_payload_bytes": (source_tokens + deliveries) * wire_bytes_per_token,
    }
    for key, expected in expected_bytes.items():
        if int(row[key]) != expected:
            raise ValueError(f"{label}: {key}={row[key]}, expected {expected}")
    full_us = positive(row["full_dispatch"]["mean_us"], f"{label}: full latency")
    transport = row.get("transport_only")
    transport_us = positive(transport["mean_us"], f"{label}: transport latency") if transport else None
    checks = {
        "unique_wire_GBps_aggregate": expected_bytes["unique_wire_bytes"] / full_us / 1000,
        "host_bus_payload_GBps_aggregate": expected_bytes["host_bus_payload_bytes"] / full_us / 1000,
        "unique_wire_GBps_per_gpu_average": expected_bytes["unique_wire_bytes"] / full_us / 1000 / world,
        "host_bus_payload_GBps_per_gpu_average": expected_bytes["host_bus_payload_bytes"] / full_us / 1000 / world,
    }
    if transport_us:
        checks.update(
            {
                "transport_unique_wire_GBps_aggregate": expected_bytes["unique_wire_bytes"] / transport_us / 1000,
                "transport_host_bus_payload_GBps_aggregate": expected_bytes["host_bus_payload_bytes"]
                / transport_us
                / 1000,
            }
        )
    for key, expected in checks.items():
        if key in row and not math.isclose(float(row[key]), expected, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{label}: recorded {key} disagrees with bytes/mean latency")
    samples = row.get("full_dispatch_samples_us", [])
    if samples:
        if any(not math.isfinite(float(item)) or item <= 0 for item in samples):
            raise ValueError(f"{label}: invalid full-dispatch sample")
        if not math.isclose(sum(samples) / len(samples), full_us, rel_tol=1e-6):
            raise ValueError(f"{label}: mean does not match recorded samples")
    return {
        "tokens_per_rank": int(row["tokens_per_rank"]),
        "stage": int(row.get("stage", 4)),
        "wire_dtype": wire_dtype,
        "execution_mode": row.get("execution_mode", "eager"),
        "channels": int(row.get("transport_options", {}).get("channels", 1)),
        "payload_slots_per_channel": int(row.get("payload_slots", 256)),
        "descriptor_slots_per_endpoint_channel": int(row.get("descriptor_slots", 256)),
        "full_mean_us": full_us,
        "full_p50_us": row["full_dispatch"].get("p50_us"),
        "full_p95_us": row["full_dispatch"].get("p95_us"),
        "transport_mean_us": transport_us,
        "transport_fraction": transport_us / full_us if transport_us else None,
        "unique_wire_GBps_aggregate": checks["unique_wire_GBps_aggregate"],
        "host_bus_payload_GBps_aggregate": checks["host_bus_payload_GBps_aggregate"],
        "unique_wire_GBps_per_gpu_average": checks["unique_wire_GBps_per_gpu_average"],
        "host_bus_payload_GBps_per_gpu_average": checks["host_bus_payload_GBps_per_gpu_average"],
        "transport_unique_wire_GBps_aggregate": checks.get("transport_unique_wire_GBps_aggregate"),
        "transport_host_bus_payload_GBps_aggregate": checks.get("transport_host_bus_payload_GBps_aggregate"),
        "remote_source_tokens": source_tokens,
        "remote_deliveries": deliveries,
        "host_write_bytes": expected_bytes["host_write_bytes"],
        "unique_wire_bytes": expected_bytes["unique_wire_bytes"],
        "host_bus_payload_bytes": expected_bytes["host_bus_payload_bytes"],
        "relay_cpu_read_write_bytes": int(row.get("relay_cpu_payload_read_write_bytes", 0)),
        "relay_cpu_read_write_GBps": int(row.get("relay_cpu_payload_read_write_bytes", 0)) / full_us / 1000,
        "sample_count": len(samples),
    }


def load_dataset(source: Path, results_dir: Path, baseline: bool = False) -> dict:
    record = json.loads(source.read_text())
    if not isinstance(record.get("results"), list) or not record["results"]:
        raise ValueError(f"{source}: no completed benchmark rows")
    rows = sorted(record["results"], key=lambda row: row["tokens_per_rank"])
    token_counts = [row["tokens_per_rank"] for row in rows]
    if len(set(token_counts)) != len(token_counts):
        raise ValueError(f"{source}: duplicate token counts")
    options = rows[0].get("transport_options", {})
    for row in rows:
        if row.get("transport_options", {}) != options:
            raise ValueError(f"{source}: transport options change within one file")
    relative = "baseline_s4" if baseline else str(source.relative_to(results_dir).with_suffix(""))
    name = "S4 baseline" if baseline else EXPECTED.get(source.stem, relative)
    normalized = [validate_row(row, source) for row in rows]
    return {
        "name": name,
        "key": relative,
        "source": str(source.resolve()),
        "metadata": record.get("metadata", {}),
        "options": options,
        "rows": normalized,
        "is_sweep": "sweep" in source.relative_to(results_dir).parts if not baseline else False,
        "baseline": baseline,
        "complete_10_token_grid": token_counts == list(TOKENS),
    }


def discover(results_dir: Path, baseline: Path | None) -> list[dict]:
    sources = [
        path
        for path in sorted(results_dir.rglob("dispatch*.json"))
        if not {"logs", "report", "previous_outputs"}.intersection(path.relative_to(results_dir).parts)
    ]
    datasets = [load_dataset(path, results_dir) for path in sources]
    if baseline is not None:
        datasets.insert(0, load_dataset(baseline, results_dir, baseline=True))
    if not datasets:
        raise ValueError(f"No completed dispatch*.json inputs in {results_dir}; no report generated")
    return datasets


def fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_report(datasets: list[dict], output_dir: Path) -> str:
    lines = [
        "# PCIe EP S5–S8 measured results",
        "",
        f"Generated from {len(datasets)} input files at {dt.datetime.now(dt.timezone.utc).isoformat()}.",
        "",
        "Only completed JSON rows are included. Missing points are not estimated. "
        "Each latency is the mean of per-iteration maxima across eight GPUs; bandwidth uses decimal GB/s.",
        "",
        "- BF16 wire bytes/token = H×2. FP8 wire bytes/token = H+4, including its FP32 scale.",
        "- Remote delivery bandwidth = F×bytes/token÷latency. GPU host payload read+write bandwidth = (U+F)×bytes/token÷latency.",
        "- U counts each remotely sent source token once; F counts one delivery per destination GPU. Expert expansion does not multiply F.",
        "- Per-GPU values are aggregate/8 averages, not measured symmetric per-rank physical link rates.",
        "- CPU relay read+write traffic is reported separately. Metadata/control/padding traffic is excluded.",
        "- Transport timing includes work inside the transport kernel, including fused receive/expand when enabled. "
        "Separately launched layout or scatter kernels are excluded; this interval is not an isolated physical-link transfer time.",
        "",
        "## Input coverage",
        "",
        "| Dataset | Completed token counts | Full 10-point grid | Input |",
        "|---|---|---|---|",
    ]
    present = {Path(item["source"]).stem for item in datasets}
    for item in datasets:
        link = Path(os.path.relpath(item["source"], output_dir)).as_posix()
        tokens = ", ".join(str(row["tokens_per_rank"]) for row in item["rows"])
        lines.append(
            f"| {item['name']} | {tokens} | {'yes' if item['complete_10_token_grid'] else 'no'} | [JSON]({link}) |"
        )
    missing = [name for key, name in EXPECTED.items() if key not in present]
    if missing:
        lines += ["", "Not present: " + ", ".join(missing) + "."]
    for item in datasets:
        if item["is_sweep"]:
            continue
        lines += [
            "",
            f"## {item['name']}",
            "",
            "Configuration: `" + json.dumps(item["options"], sort_keys=True) + "`.",
            "",
            "| T/GPU | Full mean µs | p50 µs | p95 µs | Transport mean µs | Wire GB/s, aggregate | GPU host R+W GB/s, aggregate | Wire GB/s, avg/GPU | GPU host R+W GB/s, avg/GPU |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in item["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["tokens_per_rank"]),
                        fmt(row["full_mean_us"]),
                        fmt(row["full_p50_us"]),
                        fmt(row["full_p95_us"]),
                        fmt(row["transport_mean_us"]),
                        fmt(row["unique_wire_GBps_aggregate"]),
                        fmt(row["host_bus_payload_GBps_aggregate"]),
                        fmt(row["unique_wire_GBps_per_gpu_average"]),
                        fmt(row["host_bus_payload_GBps_per_gpu_average"]),
                    ]
                )
                + " |"
            )
        if item["options"].get("numa_mode") == "relay":
            lines += [
                "",
                "| T/GPU | CPU relay read+write MiB | CPU relay read+write GB/s, aggregate |",
                "|---:|---:|---:|",
            ]
            for row in item["rows"]:
                lines.append(
                    f"| {row['tokens_per_rank']} | {row['relay_cpu_read_write_bytes'] / 2**20:.3f} | {row['relay_cpu_read_write_GBps']:.3f} |"
                )
    sweep_rows = [(item, row) for item in datasets if item["is_sweep"] for row in item["rows"]]
    if sweep_rows:
        lines += [
            "",
            "## Channel and capacity sweep",
            "",
            "Per-channel Q=L=256 increases total outstanding capacity with C. The fixed-total series instead uses Q=L=256/C; "
            "compare both before attributing a change to additional CTAs.",
            "",
            "| Dataset | C | Q/channel | Total Q | T/GPU | Full mean µs | Transport mean µs | Wire GB/s, aggregate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item, row in sweep_rows:
            channels, slots = row["channels"], row["payload_slots_per_channel"]
            lines.append(
                f"| {item['key']} | {channels} | {slots} | {channels * slots} | {row['tokens_per_rank']} | "
                f"{row['full_mean_us']:.3f} | {fmt(row['transport_mean_us'])} | {row['unique_wire_GBps_aggregate']:.3f} |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "These files establish measured latency and byte-accounted effective bandwidth only. They do not establish PCIe saturation, "
        "SM occupancy, NUMA placement, absence of NVLink traffic, numerical accuracy, or protocol correctness by themselves. "
        "Use the corresponding profiler, placement, NVLink-counter and correctness records for those checks.",
        "",
    ]
    return "\n".join(lines)


def write_tables(datasets: list[dict], output_dir: Path) -> None:
    flat = [{"dataset": item["name"], "source": item["source"], **row} for item in datasets for row in item["rows"]]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "datasets": datasets,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "REPORT.md").write_text(markdown_report(datasets, output_dir))


def plot_performance(datasets: list[dict], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filenames = []
    ordinary = [item for item in datasets if not item["is_sweep"]]
    for wire_dtype in ("bf16", "fp8_e4m3"):
        group = [item for item in ordinary if item["rows"][0]["wire_dtype"] == wire_dtype]
        if not group:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        for item in group:
            rows = item["rows"]
            tokens = [row["tokens_per_rank"] for row in rows]
            axes[0].plot(tokens, [row["full_mean_us"] for row in rows], "-o", label=item["name"], markersize=4)
            axes[1].plot(
                tokens, [row["unique_wire_GBps_aggregate"] for row in rows], "-o", label=item["name"], markersize=4
            )
        axes[0].set(xscale="log", yscale="log", xlabel="Tokens per GPU", ylabel="Full-dispatch mean latency (µs)")
        axes[1].set(xscale="log", xlabel="Tokens per GPU", ylabel="Remote delivery bandwidth, aggregate (GB/s)")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
        figure.suptitle(f"PCIe host-staged EP · 8 GPUs · E256 K8 H7168 · {wire_dtype} wire")
        for suffix in ("png", "svg"):
            name = f"performance_{wire_dtype}.{suffix}"
            figure.savefig(output_dir / name, dpi=180)
            filenames.append(name)
        plt.close(figure)

    sweeps = [item for item in datasets if item["is_sweep"]]
    if sweeps:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
        for axis, tokens in zip(axes, (16, 1024, 8192)):
            series = {}
            for item in sweeps:
                row = next((row for row in item["rows"] if row["tokens_per_rank"] == tokens), None)
                if row is None:
                    continue
                options = item["options"]
                fixed = "fixed_total" in item["key"]
                variant = (
                    "warp + F2"
                    if options.get("fuse_receive_expand")
                    else "warp" if options.get("warp_specialization") else "homogeneous"
                )
                label = f"{variant}, {'total Q=256' if fixed else 'Q/channel=256'}"
                series.setdefault(label, []).append((row["channels"], row["full_mean_us"]))
            for label, values in sorted(series.items()):
                values.sort()
                axis.plot(
                    [value[0] for value in values], [value[1] for value in values], "-o", label=label, markersize=4
                )
            axis.set(
                xscale="log",
                yscale="log",
                xlabel="Channels / transport CTAs",
                ylabel="Full-dispatch mean latency (µs)",
                title=f"T={tokens} per GPU",
            )
            axis.set_xticks([1, 2, 4, 8, 16, 32], labels=[1, 2, 4, 8, 16, 32])
            axis.grid(True, alpha=0.25)
        axes[-1].legend(fontsize=7)
        figure.suptitle("S5 channel sweep: concurrency and outstanding capacity are separate variables")
        for suffix in ("png", "svg"):
            name = f"channel_sweep.{suffix}"
            figure.savefig(output_dir / name, dpi=180)
            filenames.append(name)
        plt.close(figure)
    return filenames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/s5_s8")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline", type=Path, help="Optional recorded S4 dispatch.json for comparison")
    parser.add_argument(
        "--no-plots", action="store_true", help="Generate only Markdown, CSV and JSON; requires no plotting package"
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "report").resolve()
    try:
        datasets = discover(results_dir, args.baseline.resolve() if args.baseline else None)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(datasets, output_dir)
    figures = [] if args.no_plots else plot_performance(datasets, output_dir)
    if figures:
        with (output_dir / "REPORT.md").open("a") as handle:
            handle.write("\n## Figures\n\n")
            for filename in figures:
                if filename.endswith(".png"):
                    handle.write(f"![{Path(filename).stem}]({filename})\n\n")
    print(
        json.dumps(
            {
                "datasets": len(datasets),
                "completed_rows": sum(len(item["rows"]) for item in datasets),
                "output_dir": str(output_dir),
                "figures": figures,
            }
        )
    )


if __name__ == "__main__":
    main()
