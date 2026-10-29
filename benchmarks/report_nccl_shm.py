"""Recompute paired NCCL SHM/PCIe-EP results from exact-route timing samples.

This CPU-only tool rejects mismatched routing, weights, payloads, workloads,
and profiler runs. Every latency is recomputed as the mean of 40 iteration
maxima across eight ranks. Only anonymous, explicitly selected data is exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
VARIANTS = ("pcie_ep", "nccl_payload", "nccl_dispatch")
EXPECTED = dict(world_size=8, experts=256, topk=8, hidden=7168,
                dtype="bfloat16", seed=20261024, warmup_iterations=5,
                measured_iterations=20)
REQUIRED_ENV = dict(NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="0",
                    NCCL_NVLS_ENABLE="0", NCCL_MNNVL_ENABLE="0",
                    NCCL_COLLNET_ENABLE="0", NCCL_IB_DISABLE="1", NCCL_NET_DISABLE_INTRA="1")
PCIE_CONFIG = dict(channels=32, payload_slots=256, descriptor_slots=256,
                   pipeline_stages=2, receive_expand_mode="pipelined", source_fusion=True,
                   warp_specialization=True, fuse_receive_expand=True, numa_mode="direct",
                   wire_dtype="bf16", stage=6, alignment=1, do_expand=True)


def digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA256 fingerprint")
    return value


def fingerprints(value):
    records = value.get("per_rank", [])
    if len(records) != 8 or [item.get("rank") for item in records] != list(range(8)):
        raise ValueError("Expected rank-ordered input fingerprints for eight ranks")
    clean = [{"rank": item["rank"], **{key: sha256(item.get(key)) for key in
              ("routes_sha256", "weights_sha256", "payload_sha256")}} for item in records]
    routing = [{key: item[key] for key in ("rank", "routes_sha256", "weights_sha256")} for item in clean]
    if digest(routing) != value.get("global_routes_sha256") or digest(clean) != value.get("global_inputs_sha256"):
        raise ValueError("Global input fingerprint does not match rank fingerprints")
    return dict(per_rank=clean, global_routes_sha256=digest(routing), global_inputs_sha256=digest(clean))


def samples(row):
    ranks = row.get("per_rank_dispatch_samples_us", [])
    if len(ranks) != 8 or any(len(values) != 20 for values in ranks):
        raise ValueError("Expected eight ranks of 20 measured samples")
    for values in ranks:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Latency must be finite and positive")
    maxima = [max(values[i] for values in ranks) for i in range(20)]
    recorded = row.get("full_dispatch_samples_us", [])
    if len(recorded) != 20 or any(not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6)
                                   for x, y in zip(recorded, maxima)):
        raise ValueError("Stored iteration maxima differ from rank samples")
    return ranks, maxima


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(documents, route_fixture=None, route_fixture_digest=None):
    if len(documents) != 2:
        raise ValueError("Expected two complete paired rounds")
    timings = {(tokens, variant): [] for tokens in TOKENS for variant in VARIANTS}
    by_round = {key: [] for key in timings}
    identities, volumes, rounds, versions = {}, {}, [], None
    nccl_configuration, workload, fixture_identity = None, None, None
    for document in documents:
        metadata = document.get("metadata", {})
        if metadata.get("profiling_enabled") is not False:
            raise ValueError("Formal comparison requires an unprofiled run")
        if metadata.get('pcie_config') != PCIE_CONFIG:
            raise ValueError("PCIe-EP configuration differs from the current default")
        environment = metadata.get("nccl_environment", {})
        if any(environment.get(key) != expected for key, expected in REQUIRED_ENV.items()):
            raise ValueError("NCCL host-memory path configuration mismatch")
        configuration = dict(REQUIRED_ENV)
        for key in ('NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS'):
            value = environment.get(key, 'unset (NCCL default)')
            if value != 'unset (NCCL default)' and not (isinstance(value, str) and value.isdigit()):
                raise ValueError('Invalid channel configuration')
            configuration[key] = value
        for key, allowed in (
            ('NCCL_PROTO', {'Simple', 'LL', 'LL128'}),
            ('NCCL_ALGO', {'Ring', 'Tree', 'CollnetDirect', 'CollnetChain', 'NVLS', 'NVLSTree', 'PAT'}),
        ):
            value = environment.get(key, 'unset (NCCL default)')
            if value != 'unset (NCCL default)' and value not in allowed:
                raise ValueError('Unsupported algorithm or protocol configuration')
            configuration[key] = value
        if nccl_configuration is not None and configuration != nccl_configuration:
            raise ValueError('NCCL tuning configuration changes between paired rounds')
        nccl_configuration = configuration
        current_versions = {key: metadata.get(key) for key in ("pytorch", "cuda", "nccl")}
        if any(value is None for value in current_versions.values()):
            raise ValueError("Missing installed software versions")
        if versions is not None and current_versions != versions:
            raise ValueError("Software versions change between paired rounds")
        versions = current_versions
        rows = document.get("results", [])
        if len(rows) != 30 or {(row.get("tokens_per_rank"), row.get("variant")) for row in rows} != set(timings):
            raise ValueError("Each round must contain ten batches and all three variants exactly once")
        current_fixture = metadata.get("routing_fixture")
        if rounds and current_fixture != fixture_identity:
            raise ValueError("Fixed routing fixture changes between paired rounds")
        fixture_identity = current_fixture
        clean_rows = []
        for row in rows:
            current_workload = {key: row.get(key) for key in EXPECTED}
            expected = dict(EXPECTED, topk=current_workload.get("topk"))
            if (type(expected["topk"]) is not int or expected["topk"] not in (4, 8) or
                    any(type(current_workload[key]) is not type(value) or current_workload[key] != value
                        for key, value in expected.items())):
                raise ValueError("Workload or measurement settings differ from the fixed experiment")
            if workload is not None and current_workload != workload:
                raise ValueError("Workload changes between variants or paired rounds")
            workload = current_workload
            tokens, variant = row["tokens_per_rank"], row["variant"]
            identity = fingerprints(row.get("input_fingerprints", {}))
            if tokens in identities and identity != identities[tokens]:
                raise ValueError("Routes, top-k slots, weights or payloads differ between variants/rounds")
            identities[tokens] = identity
            volume = row.get("route_volume", {})
            keys = ("remote_source_tokens", "remote_deliveries", "remote_expert_branches", "relay_token_copies")
            if any(type(volume.get(key)) is not int or volume[key] < 0 for key in keys):
                raise ValueError("Missing route traffic volume")
            volume = {key: volume[key] for key in keys}
            if tokens in volumes and volume != volumes[tokens]:
                raise ValueError("Route volumes differ between variants/rounds")
            volumes[tokens] = volume
            ranks, maxima = samples(row)
            timings[tokens, variant].extend(maxima)
            by_round[tokens, variant].append(statistics.mean(maxima))
            validation = row.get("validation_per_rank", [])
            if len(validation) != 8 or any(item.get("allgather_source_payload_bytes_checked") != 8*tokens*7168*2 for item in validation):
                raise ValueError("Missing complete source payload validation")
            if variant != "nccl_payload" and any(item.get("stable_order") is not True or
                    item.get("bf16_elements_checked", -1) != item.get("expanded_rows", -2)*7168 for item in validation):
                raise ValueError("Missing complete expert expansion validation")
            validation_keys = ("allgather_source_payload_bytes_checked", "expanded_rows", "bf16_elements_checked", "stable_order", "packed_bytes_checked")
            clean_rows.append({"tokens_per_rank": tokens, "variant": variant,
                "per_rank_dispatch_samples_us": ranks, "full_dispatch_samples_us": maxima,
                "input_fingerprints": identity, "route_volume": volume,
                "validation_per_rank": [{key: item[key] for key in validation_keys if key in item} for item in validation]})
        rounds.append(clean_rows)

    comparison = []
    if workload["topk"] == 4 and (route_fixture is None or fixture_identity is None):
        raise ValueError("Top-4 comparisons require an explicit persisted routing fixture")
    if workload["topk"] == 4:
        identity_keys = ("schema_version", "generator", "profile", "tensors_sha256", "fixture_id")
        if (not isinstance(fixture_identity, dict) or route_fixture.get("schema_version") != 2 or
                any(fixture_identity.get(key) != route_fixture.get(key) for key in identity_keys) or
                any(route_fixture.get(key) != workload[key]
                    for key in ("world_size", "experts", "topk", "hidden", "seed"))):
            raise ValueError("Measured fixture identity or configuration differs from the supplied manifest")
        for key in ("manifest_sha256", "tensors_sha256", "fixture_id"):
            sha256(fixture_identity.get(key))
        if route_fixture_digest is not None and fixture_identity["manifest_sha256"] != route_fixture_digest:
            raise ValueError("Measured fixture manifest SHA256 differs from the supplied file")
    if route_fixture is not None:
        expected_routes = {row['tokens_per_gpu']: row for row in route_fixture['routes']}
        if set(expected_routes) != set(TOKENS):
            raise ValueError("Route fixture must contain all ten batches")
        for tokens in TOKENS:
            expected = expected_routes[tokens]
            if identities[tokens]['global_routes_sha256'] != expected['global_routes_sha256'] or volumes[tokens] != expected['route_volume']:
                raise ValueError("Measured routing differs from the original experiment route fixture")
    for tokens in TOKENS:
        row = {"tokens_per_gpu": tokens, "routes_sha256": identities[tokens]["global_routes_sha256"],
               "inputs_sha256": identities[tokens]["global_inputs_sha256"], "route_volume": volumes[tokens]}
        if route_fixture is not None and "statistics" in expected_routes[tokens]:
            row["route_statistics"] = expected_routes[tokens]["statistics"]
        for variant in VARIANTS:
            values = timings[tokens, variant]
            mean = statistics.mean(values)
            row[variant] = dict(mean_ms=mean/1000, p50_ms=percentile(values, 50)/1000,
                p95_ms=percentile(values, 95)/1000, round_mean_ms=[v/1000 for v in by_round[tokens, variant]],
                logical_GBps=tokens*workload["topk"]*7168*2/(mean*1e-6)/1e9)
        row["pcie_speedup_vs_payload"] = row["nccl_payload"]["mean_ms"]/row["pcie_ep"]["mean_ms"]
        row["pcie_speedup_vs_dispatch"] = row["nccl_dispatch"]["mean_ms"]/row["pcie_ep"]["mean_ms"]
        row["nccl_payload"]["nccl_tests_busbw_GBps"] = row["nccl_payload"]["logical_GBps"]*7/workload["topk"]
        row["remote_delivery_fraction_of_allgather"] = volumes[tokens]["remote_deliveries"]/(8*7*tokens)
        comparison.append(row)
    return dict(schema_version=1, workload=workload, software=versions,
        **({"routing_fixture": fixture_identity} if fixture_identity is not None else {}),
        nccl_configuration=nccl_configuration, rounds=rounds, comparison=comparison,
        bandwidth_definition=f"Per-GPU logical expert volume T*K*H*2/t/1e9; K={workload['topk']}, H=7168, BF16. No world-size multiplier.",
        latency_definition="Two rounds, each 5 warmup and 20 measured iterations; mean of 40 maxima across eight ranks.",
        comparison_definition="nccl_payload is communication only; nccl_dispatch includes GPU pack, AllGather of payload/routes/weights and stable expert expansion. PCIe-EP includes its full eager expanded dispatch.")


def anonymous_rounds(bundle):
    """Restore only the fields needed to recheck exported rank samples."""
    workload = bundle.get('workload', {})
    if bundle.get('schema_version') != 1 or workload not in (EXPECTED, dict(EXPECTED, topk=4)):
        raise ValueError('Unsupported anonymous sample schema or workload')
    metadata = dict(bundle['software'], profiling_enabled=False,
                    pcie_config=PCIE_CONFIG, nccl_environment=bundle['nccl_configuration'])
    if 'routing_fixture' in bundle:
        metadata['routing_fixture'] = bundle['routing_fixture']
    return [{'metadata':metadata, 'results':[dict(workload, **row) for row in rows]}
            for rows in bundle['rounds']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rounds", type=Path, nargs='*')
    parser.add_argument('--samples', type=Path, help='Recompute an anonymous exported comparison JSON')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-fixture", type=Path,
                        default=Path(__file__).resolve().parent/'data/nccl_shm_routes.json')
    args = parser.parse_args()
    if args.samples is not None:
        if args.rounds:
            parser.error('Use either two raw rounds or --samples')
        documents = anonymous_rounds(json.loads(args.samples.read_text()))
    else:
        if len(args.rounds) != 2:
            parser.error('Expected two raw rounds or --samples')
        documents = [json.loads(path.read_text()) for path in args.rounds]
    fixture_bytes = args.route_fixture.read_bytes()
    result = compare(documents, json.loads(fixture_bytes), hashlib.sha256(fixture_bytes).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"nccl_shm_comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["| Tokens/GPU | PCIe-EP ms | NCCL SHM payload ms | NCCL SHM dispatch ms | EP GB/s | NCCL payload GB/s | NCCL dispatch GB/s | EP speedup vs full |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    columns = ("tokens_per_gpu", "pcie_ep_ms", "nccl_payload_ms", "nccl_dispatch_ms", "pcie_ep_GBps", "nccl_payload_GBps", "nccl_dispatch_GBps", "pcie_speedup_vs_dispatch")
    with (args.output_dir/"nccl_shm_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, columns); writer.writeheader()
        for row in result["comparison"]:
            flat = {"tokens_per_gpu":row["tokens_per_gpu"], "pcie_speedup_vs_dispatch":row["pcie_speedup_vs_dispatch"]}
            for variant in VARIANTS:
                flat[variant+"_ms"] = row[variant]["mean_ms"]
                flat[variant+"_GBps"] = row[variant]["logical_GBps"]
            writer.writerow(flat)
            lines.append(f"| {row['tokens_per_gpu']} | " + " | ".join(f"{flat[key]:.3f}" for key in columns[1:-1])+f" | {row['pcie_speedup_vs_dispatch']:.3f}x |")
    (args.output_dir/"nccl_shm_comparison.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
