"""CPU-only checks for the baseline/prefix-TX dispatch comparison report."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_prefix_tx", ROOT / "benchmarks/report_prefix_tx.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rounds(duration):
    rows = []
    for tokens in report.TOKENS:
        row = report.WORKLOAD | dict(
            tokens_per_rank=tokens,
            measured_iterations=20,
            transport_options=dict(channels=32, warp_specialization=True),
            execution_mode="eager",
            receive_expand_mode="pipelined",
            transport_threads_per_cta=352,
            do_expand=True,
            stage=6,
            payload_slots=256,
            descriptor_slots=256,
            expert_alignment=1,
            num_max_tokens_per_rank=8192,
            warmup_iterations=5,
            seed=20261024,
            full_dispatch_samples_us=[duration] * 20,
            per_rank_dispatch_samples_us=[[duration] * 20 for _ in range(8)],
        )
        rows.append(row)
    return [dict(metadata=dict(host="private-host", source_commit="baseline"), results=copy.deepcopy(rows))
            for _ in range(2)]


class PrefixTxReportTests(unittest.TestCase):
    def test_public_comparison_reproduces_from_anonymous_samples(self):
        saved = json.loads((ROOT / "benchmarks/data/prefix_tx_comparison.json").read_text())
        reproduced = report.report_from_samples(ROOT / "benchmarks/data/prefix_tx_samples.json")
        self.assertEqual(reproduced, saved)
        self.assertNotIn("anonymous_input_sha256", saved)

    def test_anonymous_bundle_keeps_only_configuration_timings_and_source_hashes(self):
        bundle = json.loads((ROOT / "benchmarks/data/prefix_tx_samples.json").read_text())
        self.assertEqual(set(bundle), {"schema_version", "provenance", "variants"})
        fields = set(report.WORKLOAD) | set(report.CONFIG_FIELDS) | {
            "tokens_per_rank", "measured_iterations", "full_dispatch_samples_us", "per_rank_dispatch_samples_us",
        }
        self.assertEqual(set(bundle["variants"]), {"baseline", "optimized"})
        for variant in bundle["variants"]:
            self.assertEqual(len(bundle["variants"][variant]), 2)
            hashes = bundle["provenance"]["raw_input_sha256"][variant]
            self.assertEqual(len(hashes), 2)
            for value in hashes:
                self.assertRegex(value, r"^[0-9a-f]{64}$")
            for group in bundle["variants"][variant]:
                self.assertEqual(set(group), {"results"})
                for row in group["results"]:
                    self.assertEqual(set(row), fields)

    def test_bandwidth_and_comparison_use_per_gpu_topk_bytes(self):
        output = report.compare(rounds(2000), rounds(1000))
        row = output["rows"][-1]
        expected = 8192 * 8 * 7168 * 2
        self.assertEqual(row["logical_bytes_per_gpu"], expected)
        self.assertEqual(row["baseline_logical_GBps"], expected / 2e6)
        self.assertEqual(row["optimized_logical_GBps"], expected / 1e6)
        self.assertEqual(row["speedup"], 2)
        self.assertEqual(row["latency_reduction_percent"], 50)
        self.assertEqual(row["baseline_sample_count"], 40)
        self.assertNotIn("private-host", repr(output))
        self.assertNotIn("source_commit", repr(output))

    def test_mean_uses_every_iteration_maximum_in_both_rounds(self):
        baseline = rounds(1000)
        for row in baseline[1]["results"]:
            row["full_dispatch_samples_us"] = [3000] * 20
            row["per_rank_dispatch_samples_us"] = [[3000] * 20 for _ in range(8)]
        self.assertEqual(report.compare(baseline, rounds(1000))["rows"][0]["baseline_full_mean_ms"], 2)

    def test_legacy_json_does_not_need_precomputed_logical_fields(self):
        output = report.compare(rounds(2000), rounds(1000))
        self.assertEqual(len(output["rows"]), 10)
        self.assertIn("40 samples", report.markdown(output))

    def test_workload_mismatch_is_rejected(self):
        optimized = rounds(1000)
        optimized[0]["results"][0]["hidden"] = 4096
        with self.assertRaisesRegex(ValueError, "hidden"):
            report.compare(rounds(1000), optimized)

    def test_transport_configuration_mismatch_is_rejected(self):
        optimized = rounds(1000)
        for group in optimized:
            for row in group["results"]:
                row["transport_options"]["channels"] = 16
        with self.assertRaisesRegex(ValueError, "baseline/optimized configuration"):
            report.compare(rounds(1000), optimized)

    def test_configuration_mismatch_within_round_is_rejected(self):
        baseline = rounds(1000)
        baseline[0]["results"][1]["payload_slots"] = 64
        with self.assertRaisesRegex(ValueError, "across batches or rounds"):
            report.compare(baseline, rounds(1000))

    def test_inconsistent_maximum_rank_sample_is_rejected(self):
        baseline = rounds(1000)
        baseline[0]["results"][0]["per_rank_dispatch_samples_us"][3][4] = 1001
        with self.assertRaisesRegex(ValueError, "maximum rank"):
            report.compare(baseline, rounds(1000))

    def test_missing_round_batch_or_sample_is_rejected(self):
        cases = [rounds(1000)[:1], rounds(1000), rounds(1000)]
        cases[1][0]["results"].pop()
        cases[2][0]["results"][0]["full_dispatch_samples_us"].pop()
        for baseline in cases:
            with self.subTest(baseline=cases.index(baseline)), self.assertRaises(ValueError):
                report.compare(baseline, rounds(1000))

    def test_invalid_timing_is_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1, True):
            baseline = rounds(1000)
            baseline[0]["results"][0]["full_dispatch_samples_us"][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid CUDA-event"):
                report.compare(baseline, rounds(1000))


if __name__ == "__main__":
    unittest.main()
