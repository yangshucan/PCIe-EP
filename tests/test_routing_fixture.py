"""CPU checks for immutable, identical-input skewed top-4 comparisons."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("routing_fixture", _ROOT / "benchmarks/routing_fixture.py")
routing = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = routing
_SPEC.loader.exec_module(routing)


class SamplingTest(unittest.TestCase):
    def test_marginal_moments_match_published_targets(self):
        probabilities, exponent = routing.calibrated_marginals()
        statistics = routing.moments(probabilities)
        self.assertAlmostEqual(probabilities.sum(), 4, places=13)
        self.assertAlmostEqual(statistics["cv"], 164.1 / 128, places=12)
        self.assertAlmostEqual(statistics["max_over_mean"], 11.948, places=12)
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))
        self.assertAlmostEqual(exponent, 0.601004685695461, places=12)

    def test_fixed_size_sampler_preserves_marginals(self):
        probabilities = np.array([0.8, 0.6, 0.4, 0.15, 0.05])
        indices = routing.sample_routes(probabilities, 30000, 2, np.random.default_rng(7),
                                        "random_permutation_systematic_pps")
        self.assertFalse(np.any(indices[:, 0] == indices[:, 1]))
        actual = np.bincount(indices.reshape(-1), minlength=5) / len(indices)
        np.testing.assert_allclose(actual, probabilities, rtol=0, atol=0.012)
        # Top-k branch order is independently shuffled, not probability sorted.
        first_slot = np.bincount(indices[:, 0], minlength=5) / len(indices)
        np.testing.assert_allclose(first_slot, probabilities / 2, rtol=0, atol=0.012)

    def test_hotspot_mass_and_distinct_experts(self):
        probabilities, model = routing.probability_model("hotspot_20_80", 20261024)
        hot = model["design_hot_expert_ids"]
        self.assertEqual(len(hot), 51)
        self.assertAlmostEqual(probabilities[hot].sum(), 0.8)
        self.assertAlmostEqual(probabilities.sum(), 1)
        indices = routing.sample_routes(probabilities, 4096, 4, np.random.default_rng(3), model["sampler"])
        self.assertFalse(np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0))
        share = np.isin(indices, hot).mean()
        self.assertGreater(share, 0.75)
        self.assertLess(share, 0.83)

    def test_calibrated_fixture_empirical_moments(self):
        probabilities, model = routing.probability_model("nvidia_calibrated", 20261024)
        indices = routing.sample_routes(probabilities, 32768, 4, np.random.default_rng(17), model["sampler"])
        statistics = routing.moments(np.bincount(indices.reshape(-1), minlength=256))
        self.assertAlmostEqual(statistics["cv"], 164.1 / 128, delta=0.035)
        self.assertAlmostEqual(statistics["max_over_mean"], 11.948, delta=0.6)

    def test_profile_rejects_other_dimensions(self):
        with self.assertRaises(ValueError):
            routing.calibrated_marginals(128, 4)


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "routes.json"
        routing.create_fixture(self.path, "nvidia_calibrated", tokens=(4, 16))

    def manifest(self):
        return json.loads(self.path.read_text())

    def save_manifest(self, value):
        self.path.write_text(json.dumps(value, indent=2) + "\n")

    def mutate_arrays(self, mutate):
        manifest = self.manifest()
        path = self.path.parent / manifest["data_file"]
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        mutate(arrays)
        np.savez_compressed(path, **arrays)
        manifest["tensors_sha256"] = routing.file_sha256(path)
        self.save_manifest(manifest)

    def test_roundtrip_hashes_prefix_statistics_and_immutability(self):
        fixture = routing.load_fixture(self.path)
        self.assertEqual(fixture.identity["manifest_sha256"], routing.file_sha256(self.path))
        self.assertEqual(fixture.manifest["routes"][-1]["statistics"]["selected_expert_branches"], 8 * 16 * 4)
        row = fixture.manifest["routes"][0]
        self.assertEqual(row["per_rank"][0]["routes_sha256"], routing.array_sha256(fixture.indices[0, :4]))
        self.assertEqual(row["route_volume"]["remote_deliveries"],
                         np.array(row["statistics"]["source_destination_unique_tokens"]).sum() -
                         np.trace(row["statistics"]["source_destination_unique_tokens"]))
        with self.assertRaises(ValueError):
            fixture.indices[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            fixture.weights[0, 0, 0] = 0

    def test_generation_is_deterministic(self):
        second_path = self.path.parent / "other.json"
        routing.create_fixture(second_path, "nvidia_calibrated", tokens=(4, 16))
        first, second = routing.load_fixture(self.path), routing.load_fixture(second_path)
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_array_equal(first.weights, second.weights)
        self.assertEqual(first.identity["fixture_id"], second.identity["fixture_id"])

    def test_does_not_overwrite_inputs(self):
        with self.assertRaises(ValueError):
            routing.create_fixture(self.path, "hotspot_20_80", tokens=(4, 16))

    def test_rejects_corrupted_tensor_bytes(self):
        manifest = self.manifest()
        data_path = self.path.parent / manifest["data_file"]
        with data_path.open("ab") as output:
            output.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            routing.load_fixture(self.path)

    def test_rejects_valid_archive_with_duplicate_expert(self):
        self.mutate_arrays(lambda arrays: arrays["indices"][0, 0].fill(1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            routing.load_fixture(self.path)

    def test_rejects_valid_archive_with_out_of_range_expert(self):
        self.mutate_arrays(lambda arrays: arrays["indices"].__setitem__((0, 0, 0), 256))
        with self.assertRaisesRegex(ValueError, "invalid"):
            routing.load_fixture(self.path)

    def test_rejects_invalid_weights(self):
        self.mutate_arrays(lambda arrays: arrays["weights"].__setitem__((0, 0, 0), np.nan))
        with self.assertRaisesRegex(ValueError, "weights"):
            routing.load_fixture(self.path)

    def test_rejects_wrong_tensor_dtype(self):
        self.mutate_arrays(lambda arrays: arrays.__setitem__("indices", arrays["indices"].astype(np.int32)))
        with self.assertRaisesRegex(ValueError, "dtype"):
            routing.load_fixture(self.path)

    def test_rejects_forged_load_statistics(self):
        manifest = self.manifest()
        manifest["routes"][0]["statistics"]["expert_counts"][0] += 1
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "statistics"):
            routing.load_fixture(self.path)

    def test_rejects_forged_route_fingerprint(self):
        manifest = self.manifest()
        manifest["routes"][0]["per_rank"][0]["routes_sha256"] = "0" * 64
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "hashes"):
            routing.load_fixture(self.path)

    def test_rejects_changed_probability_model(self):
        manifest = self.manifest()
        manifest["probability_model"]["source"]["published_mean"] = 129
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "probability model"):
            routing.load_fixture(self.path)

    def test_rejects_npz_path_outside_fixture_directory(self):
        manifest = self.manifest()
        manifest["data_file"] = "../routes.npz"
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "basename"):
            routing.load_fixture(self.path)

    def test_cpu_tensors_are_exact_independent_copies(self):
        fixture = routing.load_fixture(self.path)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(from_numpy=lambda value: value)}):
            indices, weights = fixture.cpu_tensors(4, 0)
        np.testing.assert_array_equal(indices, fixture.indices[0, :4])
        np.testing.assert_array_equal(weights, fixture.weights[0, :4])
        self.assertFalse(np.shares_memory(indices, fixture.indices))
        self.assertFalse(np.shares_memory(weights, fixture.weights))

    def test_frozen_benchmark_branch_cannot_call_rng(self):
        source = ast.parse((_ROOT / "benchmarks/benchmark_nccl_shm.py").read_text())
        helper = next(item for item in source.body if isinstance(item, ast.FunctionDef) and item.name == "routing_tensors")
        def forbidden(*args):
            raise AssertionError("Runtime routing RNG was called")
        expected = (object(), object())
        fixture = SimpleNamespace(cpu_tensors=lambda tokens, rank: expected)
        namespace = {"routes": forbidden}
        exec(compile(ast.Module(body=[copy.deepcopy(helper)], type_ignores=[]), "routing_tensors", "exec"), namespace)
        self.assertIs(namespace["routing_tensors"](SimpleNamespace(frozen_routing=fixture), 16, 0), expected)


if __name__ == "__main__":
    unittest.main()
