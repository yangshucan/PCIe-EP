"""Render the original S4 benchmark report from its recorded measurements.

The readability refactor has its own records in ``results/refactor``. This
report preserves the original benchmark and the source snapshot to which its
PCIe path audit refers; it does not reinterpret those as new measurements.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TEMPLATE_PATH = Path(__file__).parent / "templates/baseline_results.md"
TOKENS_PER_RANK = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def read_record(relative_path):
    return json.loads((RESULTS_DIR / relative_path).read_text())


def validate_records():
    """Require the actual baseline validation records before rendering tables."""
    assert (RESULTS_DIR / "pipeline_final.exit").read_text().strip() == "0"
    rows = read_record("dispatch.json")["results"]
    assert tuple(row["tokens_per_rank"] for row in rows) == TOKENS_PER_RANK
    for row in rows:
        dimensions = tuple(row[key] for key in ("world_size", "experts", "topk", "hidden", "dtype"))
        assert dimensions == (8, 256, 8, 7168, "bfloat16")

    stress = read_record("s4_stress.json")["records"]
    assert len(stress) == 3
    assert all(record["status"] == "passed" and record["dispatches_per_rank"] == 100000 for record in stress)
    large = read_record("large.json")
    assert large["status"] == "passed"
    audit = read_record("pcie_path_audit.json")
    assert not audit.get("superseded_by_lifecycle_fix", False)
    links = audit["nvlink_s4_benchmark_measurement"]
    assert links["all_payload_deltas_zero"] and links["complete_gpu_link_direction_sets"]
    profiler = read_record("profiler_expanded/audit.json")
    assert profiler["expand"] and all(record["status"] == "passed" for record in profiler["reports"])
    for filename in ("consumer_lifetime.json", "integration.json"):
        assert all(record["status"] == "passed" for record in read_record(filename)["records"].values())
    return rows, large, links


def format_dispatch_table(rows):
    lines = [
        "| 每卡 tokens | Dispatch 平均 ms | Dispatch p50 ms | Dispatch p95 ms | Transport 平均 ms | 远端交付 GB/s | Host 读写 GB/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        dispatch = row["full_dispatch"]
        transport = row["transport_only"]
        lines.append(
            f"| {row['tokens_per_rank']} | {dispatch['mean_us'] / 1000:.3f} "
            f"| {dispatch['p50_us'] / 1000:.3f} | {dispatch['p95_us'] / 1000:.3f} "
            f"| {transport['mean_us'] / 1000:.3f} | {row['unique_wire_GBps_aggregate']:.3f} "
            f"| {row['host_bus_payload_GBps_aggregate']:.3f} |"
        )
    return "\n".join(lines)


def format_transport_table(rows):
    lines = [
        "| 每卡 tokens | 远端交付量 MiB | Host 写量 MiB | Transport 平均 ms | Transport 远端 GB/s | Transport Host 读写 GB/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tokens_per_rank']} | {row['unique_wire_bytes'] / 2**20:.3f} "
            f"| {row['host_write_bytes'] / 2**20:.3f} | {row['transport_only']['mean_us'] / 1000:.3f} "
            f"| {row['transport_unique_wire_GBps_aggregate']:.3f} "
            f"| {row['transport_host_bus_payload_GBps_aggregate']:.3f} |"
        )
    return "\n".join(lines)


def write_report(rows, large, links):
    replacements = {
        "DISPATCH_TABLE": format_dispatch_table(rows),
        "TRANSPORT_TABLE": format_transport_table(rows),
        "PAYLOAD_ELEMENTS": f"{large['payload_elements_checked_total']:,}",
        "PAYLOAD_GB": f"{large['payload_bytes_checked_total'] / 1e9:.3f}",
        "TX_DELTA": str(links["Tx_total_delta_KiB"]),
        "RX_DELTA": str(links["Rx_total_delta_KiB"]),
    }
    document = TEMPLATE_PATH.read_text()
    for name, value in replacements.items():
        document = document.replace("{{" + name + "}}", value)
    (ROOT / "RESULTS.md").write_text(document)


def plot_performance(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = [row["tokens_per_rank"] for row in rows]
    figure, (latency_axes, bandwidth_axes) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    latency_axes.plot(tokens, [row["full_dispatch"]["mean_us"] / 1000 for row in rows], "-o", label="Full dispatch")
    latency_axes.plot(
        tokens, [row["transport_only"]["mean_us"] / 1000 for row in rows], "--s", label="Transport kernel"
    )
    latency_axes.set(xscale="log", yscale="log", xlabel="Tokens per GPU", ylabel="Mean latency (ms)")
    latency_axes.legend()
    latency_axes.grid(True, alpha=0.25)
    bandwidth_axes.plot(
        tokens, [row["unique_wire_GBps_aggregate"] for row in rows], "-o", label="Remote payload / dispatch time"
    )
    bandwidth_axes.plot(
        tokens,
        [row["host_bus_payload_GBps_aggregate"] for row in rows],
        "-s",
        label="Host write + read / dispatch time",
    )
    bandwidth_axes.set(
        xscale="log", ylim=(0, 3.5), xlabel="Tokens per GPU", ylabel="Aggregate effective bandwidth (GB/s)"
    )
    bandwidth_axes.legend(fontsize=8)
    bandwidth_axes.grid(True, alpha=0.25)
    figure.suptitle("PCIe host-staged EP: S4 baseline, 8 H20D GPUs, BF16 H=7168, top-k=8")
    figure.savefig(RESULTS_DIR / "performance.png", dpi=180)
    figure.savefig(RESULTS_DIR / "performance.svg")
    plt.close(figure)


def main():
    rows, large, links = validate_records()
    write_report(rows, large, links)
    plot_performance(rows)
    print(ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
