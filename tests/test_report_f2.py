"""CPU-only checks for F2 reporting units, pooling and evidence handling."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import report_f2


def synthetic_record(variant, shift=0):
    """Synthetic fixture only; never saved under experimental results/."""
    samples = [100.0 + step + shift for step in range(20)]
    distribution = report_f2.distribution(samples)
    mode = variant if variant in ("vectorized", "pipelined") else "scalar"
    row = dict(
        tokens_per_rank=16,
        world_size=8,
        hidden=7168,
        experts=256,
        topk=8,
        dtype="bfloat16",
        wire_dtype="bf16",
        do_expand=True,
        stage=6,
        execution_mode="eager",
        payload_slots=256,
        descriptor_slots=256,
        expert_alignment=1,
        transport_threads_per_cta=352 if variant == "pipelined" else 256,
        transport_options=dict(
            stage=6,
            channels=32,
            pipeline_stages=2,
            warp_specialization=True,
            source_fusion=True,
            numa_mode="direct",
            wire_dtype="bf16",
            fuse_receive_expand=variant != "nof2",
            receive_expand_mode=mode,
        ),
        remote_source_tokens=128,
        remote_deliveries=896,
        host_write_bytes=128 * 7168 * 2,
        unique_wire_bytes=896 * 7168 * 2,
        host_bus_payload_bytes=1024 * 7168 * 2,
        full_dispatch=distribution,
        full_dispatch_samples_us=samples,
        transport_only=None,
        runtime_stats_per_rank=[dict(raw=[16, 112, 112, 0, 0, 0, 0, 0]) for _ in range(8)],
        receive_expand_overlap_per_rank=(
            [None] * 8 if mode == "scalar" else [[[0, 0] for _ in range(32)] for _ in range(8)]
        ),
    )
    return dict(metadata=dict(synthetic_fixture=True), results=[row])


class F2ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pcie-ep-f2-report-unit-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, variant="pipelined", round_id="round1", record=None):
        path = self.root / "runs" / "synthetic" / round_id / f"dispatch_{variant}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record or synthetic_record(variant)))
        return path

    def test_no_inputs_are_not_estimated(self):
        with self.assertRaisesRegex(ValueError, "No completed"):
            report_f2.load_sources(self.root)

    def test_partial_grid_remains_partial_and_second_round_missing(self):
        self.save()
        datasets = report_f2.load_sources(self.root)
        self.assertFalse(datasets[0]["complete_10_token_grid"])
        summary = report_f2.aggregate(datasets)[0]
        self.assertEqual(summary["round_count"], 1)
        self.assertEqual(summary["sample_count"], 20)
        evidence = report_f2.audit_sources(self.root, datasets)[0]
        self.assertEqual(evidence["status"], "incomplete_or_failed")

    def test_equal_sized_rounds_are_pooled_with_range_preserved(self):
        self.save()
        self.save(round_id="round2", record=synthetic_record("pipelined", shift=100))
        summary = report_f2.aggregate(report_f2.load_sources(self.root))[0]
        self.assertEqual(summary["full_mean_us"], 159.5)
        self.assertEqual(summary["round_mean_min_us"], 109.5)
        self.assertEqual(summary["round_mean_max_us"], 209.5)
        self.assertEqual(summary["sample_count"], 40)
        self.assertAlmostEqual(summary["unique_wire_GBps_aggregate"], 896 * 7168 * 2 / 159500)

    def test_completion_watermark_is_not_counted_as_overlap(self):
        record = synthetic_record("pipelined")
        record["results"][0]["receive_expand_overlap_per_rank"][0][0] = [3, 197]
        self.save(record=record)
        row = report_f2.load_sources(self.root)[0]["rows"][0]
        self.assertEqual(row["overlap_count_last_dispatch_per_rank"][0], 3)

    def test_wrong_traffic_cta_and_modes_are_rejected(self):
        for mutate in (
            lambda row: row.update(unique_wire_bytes=1),
            lambda row: row.update(transport_threads_per_cta=256),
            lambda row: row["transport_options"].update(receive_expand_mode="scalar"),
            lambda row: row["runtime_stats_per_rank"][0]["raw"].__setitem__(0, 999),
        ):
            record = synthetic_record("pipelined")
            mutate(record["results"][0])
            self.save(record=record)
            with self.assertRaises(ValueError):
                report_f2.load_sources(self.root)

    def test_different_routes_between_paths_are_rejected(self):
        self.save()
        record = synthetic_record("vectorized")
        row = record["results"][0]
        row.update(remote_deliveries=904, unique_wire_bytes=904 * 7168 * 2, host_bus_payload_bytes=1032 * 7168 * 2)
        row["runtime_stats_per_rank"][0]["raw"][1] += 8
        self.save("vectorized", record=record)
        with self.assertRaisesRegex(ValueError, "routing differs"):
            report_f2.load_sources(self.root)

    def test_complete_audit_requires_288_counters_exit_and_unchanged_source(self):
        path = self.save()
        directory = path.parent
        manifest = {"sha256": {"synthetic.py": "123"}}
        (directory / "executed_source_manifest.json").write_text(json.dumps(manifest))
        (directory.parent / "final_source_manifest.json").write_text(json.dumps(manifest))
        (directory / "round.exit").write_text("0\n")
        delta = dict(status="passed", counter_count=288, complete=True, monotonic=True, all_payload_deltas_zero=True)
        (directory / "nvlink_counter_delta.json").write_text(json.dumps(delta))
        datasets = report_f2.load_sources(self.root)
        self.assertEqual(report_f2.audit_sources(self.root, datasets)[0]["status"], "passed")
        delta["counter_count"] = 287
        (directory / "nvlink_counter_delta.json").write_text(json.dumps(delta))
        self.assertEqual(report_f2.audit_sources(self.root, datasets)[0]["status"], "incomplete_or_failed")


if __name__ == "__main__":
    unittest.main()
