"""Summarize Nsight Systems GPU metrics during concurrent transport.

Reads an exported SQLite file without changing it. Metrics are pooled across
all eight GPUs only in the intersection of their transport intervals, so no
private GPU identifiers or profiler metadata enter the output. Sampling is
100 us in the recorded experiment; values are diagnostic percentages.
"""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict


def summarize(database, iterations):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    kernels = defaultdict(list)
    for device, start, end in connection.execute(
        "SELECT k.deviceId,k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.demangledName "
        "WHERE s.value LIKE '%advanced_transport_pipelined%' ORDER BY k.start"
    ):
        kernels[device].append((start, end))
    if len(kernels) != 8 or any(len(v) < iterations for v in kernels.values()):
        raise ValueError(f"Incomplete eight-GPU capture: {[(k,len(v)) for k,v in kernels.items()]}")
    windows = []
    for index in range(iterations):
        start = max(v[-iterations + index][0] for v in kernels.values())
        end = min(v[-iterations + index][1] for v in kernels.values())
        if end > start:
            windows.append((start, end))
    names = {
        (type_id, metric): name
        for type_id, metric, name in connection.execute(
            "SELECT DISTINCT typeId,metricId,metricName FROM TARGET_INFO_GPU_METRICS"
        )
        if name.endswith("[Throughput %]")
    }
    values = defaultdict(list)
    for timestamp, type_id, metric, value in connection.execute(
        "SELECT timestamp,typeId,metricId,value FROM GPU_METRICS"
    ):
        name = names.get((type_id, metric))
        if name is not None and any(start <= timestamp < end for start, end in windows):
            values[name].append(value)
    if not values:
        raise ValueError("No GPU metric samples in concurrent transport windows")
    nvlink = {key: max(v) for key, v in values.items() if key.startswith("NVLink")}
    result = {
        "iterations": iterations,
        "transport_kernel_counts_per_device": {str(k): len(v) for k,v in kernels.items()},
        "concurrent_window_count": len(windows),
        "concurrent_window_total_us": sum(b-a for a,b in windows)/1000,
        "metric_scope": "All eight GPU metric sources, samples within intersection of each iteration's eight transport kernels",
        "sampling_frequency_hz": 10000,
        "units": "Throughput % only; raw frequency/count fields with exporter-specific scaling are omitted",
        "nvlink_sample_maxima": nvlink,
        "metrics": {name: {"mean": statistics.mean(v), "median": statistics.median(v), "max": max(v), "samples": len(v)}
                    for name, v in values.items()},
    }
    connection.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.output, "w") as output:
        json.dump(summarize(args.database, args.iterations), output, indent=2)
