"""Require complete, monotonic and unchanged NVLink payload counters."""

import argparse
import json
from pathlib import Path

EXPECTED_KEYS = {(gpu, link, direction) for gpu in range(8) for link in range(18) for direction in ("Tx", "Rx")}


def read_counters(path):
    record = json.loads(path.read_text())
    if record["commands"]["data"]["returncode"] != 0:
        raise RuntimeError(f"Counter collection failed: {path}")
    entries = record["data_counters"]
    values = {(entry["gpu"], entry["link"], entry["direction"]): entry["KiB"] for entry in entries}
    if len(entries) != len(EXPECTED_KEYS) or set(values) != EXPECTED_KEYS:
        raise RuntimeError(f"Expected all 288 GPU/link/direction counters: {path}")
    return values


def compare_counters(before_path, after_path):
    before, after = read_counters(before_path), read_counters(after_path)
    deltas = []
    for key in sorted(EXPECTED_KEYS):
        gpu, link, direction = key
        deltas.append(
            {
                "gpu": gpu,
                "link": link,
                "direction": direction,
                "delta_KiB": after[key] - before[key],
            }
        )
    monotonic = all(entry["delta_KiB"] >= 0 for entry in deltas)
    unchanged = all(entry["delta_KiB"] == 0 for entry in deltas)
    return {
        "before": str(before_path),
        "after": str(after_path),
        "counter_count": len(deltas),
        "complete": True,
        "monotonic": monotonic,
        "all_payload_deltas_zero": unchanged,
        "status": "passed" if monotonic and unchanged else "failed",
        "deltas": deltas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_counters(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"NVLink payload counters: {report['status']} ({report['counter_count']} values)")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
