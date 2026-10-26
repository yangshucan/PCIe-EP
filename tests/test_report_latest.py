"""CPU-only integrity checks for the public latest-version timing dataset."""

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_latest", ROOT / "benchmarks/report_latest.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class LatestReportTests(unittest.TestCase):
    def test_public_table_is_reproducible_from_anonymous_rank_timings(self):
        raw = json.loads((ROOT / "benchmarks/data/f2_pipelined_samples.json").read_text())
        saved = json.loads((ROOT / "benchmarks/data/f2_pipelined_summary.json").read_text())
        self.assertEqual(report.summarize(raw["rounds"]), saved["rows"])
        self.assertEqual(len(saved["rows"]), 10)

    def test_bandwidth_uses_per_gpu_topk_volume_without_world_size_factor(self):
        raw = json.loads((ROOT / "benchmarks/data/f2_pipelined_samples.json").read_text())
        row = report.summarize(raw["rounds"])[-1]
        self.assertEqual(row["logical_bytes_per_gpu"], 8192 * 8 * 7168 * 2)
        self.assertAlmostEqual(row["bandwidth_GBps"], 19.699677803144098)

    def test_inconsistent_rank_duration_is_rejected(self):
        raw = json.loads((ROOT / "benchmarks/data/f2_pipelined_samples.json").read_text())
        raw["rounds"][0]["measurements"][0]["full_dispatch_samples_us"][0] *= 2
        with self.assertRaisesRegex(ValueError, "maximum rank"):
            report.summarize(raw["rounds"])


if __name__ == "__main__":
    unittest.main()
