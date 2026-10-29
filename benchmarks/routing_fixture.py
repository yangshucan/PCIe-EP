"""Create and validate frozen, CPU-generated top-4 routing fixtures.

The NPZ stores concrete expert IDs and FP32 weights, not RNG seeds alone.
Benchmarks load these arrays before timing; generation never runs on a GPU.
The NVIDIA-calibrated case matches two published aggregate moments and is a
synthetic workload. It does not reproduce a model's routing trace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
PROFILES = ("nvidia_calibrated", "hotspot_20_80")
GENERATOR = "pcie_ep_frozen_routes_v1"
NVIDIA_SOURCE = "https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html#offline-ep-load-balancer"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def array_sha256(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    return {"mean": mean, "stddev": float(values.std()),
            "cv": float(values.std() / mean), "max_over_mean": float(values.max() / mean),
            "min_over_mean": float(values.min() / mean)}


def calibrated_marginals(experts=256, topk=4):
    """Return a positive inclusion vector with sum K and the target moments.

    One entry fixes max/mean. The remaining entries form a smooth power-law
    tail; a scalar bisection fixes CV. These are inclusion probabilities,
    rather than multinomial weights, so their first-order moments survive
    fixed-size, distinct-expert sampling.
    """
    if (experts, topk) != (256, 4):
        raise ValueError("The published-moment synthetic profile requires E256/K4")
    target_cv, maximum = 164.1 / 128.0, 11.948
    remaining = experts - maximum
    target_square_sum = experts * (1 + target_cv**2) - maximum**2
    ranks = np.arange(1, experts, dtype=np.float64)
    low, high = 0.0, 2.0
    for _ in range(100):
        exponent = (low + high) / 2
        tail = ranks ** -exponent
        tail *= remaining / tail.sum()
        if np.square(tail).sum() < target_square_sum:
            low = exponent
        else:
            high = exponent
    relative = np.concatenate(([maximum], tail))
    probabilities = relative * (topk / experts)
    if probabilities.max() > 1 or tail.max() > maximum:
        raise ValueError("Invalid fixed-size inclusion probabilities")
    return probabilities, exponent


def probability_model(profile, seed, experts=256, topk=4):
    """Place the probability spectrum with one seed-derived ID permutation."""
    if profile not in PROFILES or (experts, topk) != (256, 4):
        raise ValueError("Supported fixtures are the two E256/K4 profiles")
    placement = np.random.default_rng(np.random.SeedSequence([seed, 982451653])).permutation(experts)
    if profile == "nvidia_calibrated":
        spectrum, exponent = calibrated_marginals(experts, topk)
        source = {
            "kind": "synthetic_moment_match",
            "url": NVIDIA_SOURCE,
            "table": "Table 1, No EPLB, By Expert Slot",
            "published_mean": 128.0, "published_stddev": 164.1,
            "published_imbalance_ratio": 10.948,
            "imbalance_definition": "(maximum - mean) / mean",
            "target_cv": 164.1 / 128.0, "target_max_over_mean": 11.948,
            "original_workload": "DeepSeek-R1 E256/top8/EP32 on 32 GB200 GPUs",
            "limitation": "Published metrics are averages over layers and iterations, not one complete probability vector. This E256/top4 fixture matches only first-order marginal moments; it does not replay pair coactivation, temporal, domain, or source correlations.",
        }
        model = {"semantics": "first_order_inclusion_probability", "sampler": "random_permutation_systematic_pps",
                 "tail_exponent": exponent, "source": source}
    else:
        hot = int(experts * 0.2)
        spectrum = np.full(experts, 0.2 / (experts - hot), dtype=np.float64)
        spectrum[:hot] = 0.8 / hot
        model = {"semantics": "sequential_sampling_weight", "sampler": "gumbel_topk_without_replacement",
                 "hot_expert_count": hot, "hot_sampling_mass": 0.8,
                 "source": {"kind": "user_selected_sensitivity_case",
                            "limitation": "20% means floor(0.2*256)=51 experts. 80% is pre-sampling weight mass, not a promise of 80% selected branches after top4 without replacement."}}
    probabilities = np.empty_like(spectrum)
    probabilities[placement] = spectrum
    model.update({"expert_id_placement": "one seed-derived random permutation; no manual GPU placement",
                  "placement_seed": seed, "spectrum_to_expert_id": placement.tolist(),
                  "probabilities": probabilities.tolist(), "theoretical_moments": moments(probabilities),
                  "design_hot_expert_ids": placement[:int(experts * 0.2)].tolist()})
    return probabilities, model


def sample_routes(probabilities, tokens, topk, rng, sampler, chunk_size=1024):
    """Offline fixed-size sampling; return distinct IDs in random slot order."""
    experts = len(probabilities)
    result = np.empty((tokens, topk), dtype="<i8")
    for start in range(0, tokens, chunk_size):
        count = min(chunk_size, tokens - start)
        if sampler == "random_permutation_systematic_pps":
            # Random order avoids assigning persistent coactivation pairs to
            # neighboring expert IDs. Any order preserves inclusion marginals.
            order = np.argsort(rng.random((count, experts)), axis=1)
            cumulative = np.cumsum(probabilities[order], axis=1)
            cumulative[:, -1] = topk
            thresholds = rng.random((count, 1)) + np.arange(topk)
            positions = np.empty((count, topk), dtype=np.int64)
            for branch in range(topk):
                positions[:, branch] = (cumulative <= thresholds[:, branch, None]).sum(axis=1)
            chosen = np.take_along_axis(order, positions, axis=1)
        elif sampler == "gumbel_topk_without_replacement":
            uniform = np.maximum(rng.random((count, experts)), np.finfo(np.float64).tiny)
            keys = np.log(probabilities)[None, :] - np.log(-np.log(uniform))
            chosen = np.argpartition(keys, -topk, axis=1)[:, -topk:]
        else:
            raise ValueError("Unsupported fixed routing sampler")
        # Branch order is independent of probability rank and shared by both
        # measured implementations, including exact FP32 branch weights.
        slots = np.argsort(rng.random((count, topk)), axis=1)
        result[start:start + count] = np.take_along_axis(chosen, slots, axis=1)
    return result


def routing_statistics(indices, experts, hot_experts):
    world, tokens, topk = indices.shape
    owners = indices // (experts // world)
    counts = np.bincount(indices.reshape(-1), minlength=experts)
    deliveries = np.zeros((world, world), dtype=np.int64)
    remote_source_tokens = remote_expert_branches = relay_token_copies = 0
    for source in range(world):
        destination = np.any(owners[source, :, :, None] == np.arange(world), axis=1)
        deliveries[source] = destination.sum(axis=0)
        remote = destination.copy()
        remote[:, source] = False
        remote_source_tokens += int(remote.any(axis=1).sum())
        remote_expert_branches += int((owners[source] != source).sum())
        if world == 8:
            for node in range(2):
                if node != source // 4:
                    relay_token_copies += int(remote[:, node * 4:(node + 1) * 4].any(axis=1).sum())
    rank_counts = counts.reshape(world, experts // world).sum(axis=1)
    unique = deliveries.sum(axis=0)
    hot_count = int(experts * 0.2)
    total = int(counts.sum())
    volume = {"remote_source_tokens": remote_source_tokens,
              "remote_deliveries": int(deliveries.sum() - np.trace(deliveries)),
              "remote_expert_branches": remote_expert_branches,
              "relay_token_copies": relay_token_copies}
    return volume, {
        "selected_expert_branches": total, "expert_counts": counts.tolist(),
        "expert_load": moments(counts),
        "top_20pct_actual_expert_branch_share": float(np.sort(counts)[-hot_count:].sum() / total),
        "design_hot_expert_branch_share": float(counts[hot_experts].sum() / total),
        "rank_expanded_branches": rank_counts.tolist(), "rank_expanded_load": moments(rank_counts),
        "rank_unique_received_tokens": unique.tolist(), "rank_unique_load": moments(unique),
        "source_destination_unique_tokens": deliveries.tolist(),
        "remote_delivery_fraction_of_allgather": volume["remote_deliveries"] / (world * (world - 1) * tokens),
    }


def case_record(indices, weights, tokens, model):
    world = indices.shape[0]
    records = [{"rank": rank, "routes_sha256": array_sha256(indices[rank, :tokens]),
                "weights_sha256": array_sha256(weights[rank, :tokens])} for rank in range(world)]
    volume, statistics = routing_statistics(indices[:, :tokens], len(model["probabilities"]), model["design_hot_expert_ids"])
    return {"tokens_per_gpu": tokens, "per_rank": records,
            "global_routes_sha256": digest(records), "route_volume": volume,
            "statistics": statistics}


def create_fixture(output, profile, seed=20261024, tokens=TOKENS, world=8, experts=256, topk=4):
    """Write concrete immutable test inputs and their anonymous manifest."""
    output = Path(output)
    if output.suffix != ".json" or output.exists() or output.with_suffix(".npz").exists():
        raise ValueError("Output must be a new JSON manifest with no existing NPZ sibling")
    if world != 8 or not tokens or any(type(t) is not int or t < 1 for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Fixture requires eight ranks and distinct positive token counts")
    probabilities, model = probability_model(profile, seed, experts, topk)
    maximum = max(tokens)
    indices = np.empty((world, maximum, topk), dtype="<i8")
    weights = np.empty((world, maximum, topk), dtype="<f4")
    for rank in range(world):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rank, 49979687, PROFILES.index(profile)]))
        indices[rank] = sample_routes(probabilities, maximum, topk, rng, model["sampler"])
        rank_weights = rng.random((maximum, topk), dtype=np.float32)
        weights[rank] = rank_weights / rank_weights.sum(axis=1, keepdims=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_file = output.with_suffix(".npz")
    np.savez_compressed(data_file, indices=indices, weights=weights,
                        probabilities=np.asarray(probabilities, dtype="<f8"))
    manifest = {"schema_version": 2, "generator": GENERATOR, "numpy_version": np.__version__,
                "profile": profile, "world_size": world, "experts": experts, "topk": topk,
                "hidden": 7168, "seed": seed, "max_tokens_per_rank": maximum,
                "data_file": data_file.name, "tensors_sha256": file_sha256(data_file),
                "array_dtypes": {"indices": "int64", "weights": "float32", "probabilities": "float64"},
                "prefix_rule": "Every rank and T uses the first T rows of its stored max-T routing; no runtime RNG",
                "probability_model": model,
                "routes": [case_record(indices, weights, t, model) for t in sorted(tokens)]}
    manifest["fixture_id"] = digest({"profile": profile, "indices_sha256": array_sha256(indices),
                                      "weights_sha256": array_sha256(weights),
                                      "probability_model": model})
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass(frozen=True)
class FrozenRouting:
    manifest: dict
    indices: np.ndarray
    weights: np.ndarray
    identity: dict

    def cpu_tensors(self, tokens, rank):
        """Return fresh CPU tensors without importing Torch in the generator."""
        import torch

        if tokens not in {row["tokens_per_gpu"] for row in self.manifest["routes"]}:
            raise ValueError("Token count is absent from the routing fixture")
        if not 0 <= rank < self.manifest["world_size"]:
            raise ValueError("Rank is outside the routing fixture")
        return (torch.from_numpy(self.indices[rank, :tokens].copy()),
                torch.from_numpy(self.weights[rank, :tokens].copy()))


def load_fixture(path):
    """Verify every stored case on CPU before any CUDA setup or timing."""
    path = Path(path)
    manifest_hash = file_sha256(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2 or manifest.get("generator") != GENERATOR:
        raise ValueError("Unsupported frozen routing fixture schema")
    if (manifest.get("world_size"), manifest.get("experts"), manifest.get("topk"), manifest.get("hidden")) != (8, 256, 4, 7168):
        raise ValueError("Frozen routing fixture dimensions differ from E256/K4/DP8/H7168")
    name = manifest.get("data_file")
    if not isinstance(name, str) or Path(name).name != name or name in (".", "..") or not name.endswith(".npz"):
        raise ValueError("Fixture data_file must be an NPZ basename beside the manifest")
    data_path = path.parent / name
    if file_sha256(data_path) != manifest.get("tensors_sha256"):
        raise ValueError("Frozen routing NPZ SHA256 mismatch")
    with np.load(data_path, allow_pickle=False) as data:
        if set(data.files) != {"indices", "weights", "probabilities"}:
            raise ValueError("Unexpected routing NPZ arrays")
        indices, weights, probabilities = (data[key] for key in ("indices", "weights", "probabilities"))
    maximum = manifest.get("max_tokens_per_rank")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Invalid maximum token count")
    shape = (8, maximum, 4)
    if indices.shape != shape or weights.shape != shape or indices.dtype != np.dtype("<i8") or weights.dtype != np.dtype("<f4"):
        raise ValueError("Frozen route/weight shape or dtype mismatch")
    if probabilities.shape != (256,) or probabilities.dtype != np.dtype("<f8"):
        raise ValueError("Frozen probability shape or dtype mismatch")
    if indices.min() < 0 or indices.max() >= 256 or np.any(np.diff(np.sort(indices, axis=2), axis=2) == 0):
        raise ValueError("Frozen routes contain invalid or duplicate experts")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(axis=2), 1, rtol=1e-6, atol=1e-6):
        raise ValueError("Frozen weights must be finite nonnegative normalized FP32 values")
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("Frozen sampling probabilities must be positive and finite")
    expected_probabilities, model = probability_model(manifest.get("profile"), manifest.get("seed"))
    if model != manifest.get("probability_model") or not np.array_equal(probabilities, expected_probabilities):
        raise ValueError("Frozen probability model does not match its profile definition")
    cases = manifest.get("routes", [])
    tokens = [row.get("tokens_per_gpu") for row in cases]
    if not tokens or any(type(t) is not int or not 1 <= t <= maximum for t in tokens) or tokens != sorted(set(tokens)) or max(tokens) != maximum:
        raise ValueError("Invalid frozen routing token cases")
    for row in cases:
        if case_record(indices, weights, row["tokens_per_gpu"], model) != row:
            raise ValueError("Frozen route hashes, traffic or load statistics mismatch")
    fixture_id = digest({"profile": manifest["profile"], "indices_sha256": array_sha256(indices),
                         "weights_sha256": array_sha256(weights), "probability_model": model})
    if fixture_id != manifest.get("fixture_id"):
        raise ValueError("Frozen routing fixture identity mismatch")
    identity = {key: manifest[key] for key in ("schema_version", "generator", "profile", "tensors_sha256", "fixture_id")}
    identity["manifest_sha256"] = manifest_hash
    indices.flags.writeable = weights.flags.writeable = False
    return FrozenRouting(manifest, indices, weights, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    create_fixture(args.output, args.profile, args.seed)
    fixture = load_fixture(args.output)
    print(json.dumps({"identity": fixture.identity, "largest_case_statistics": fixture.manifest["routes"][-1]["statistics"]}, indent=2))


if __name__ == "__main__":
    main()
