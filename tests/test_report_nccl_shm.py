"""CPU-only checks for exact-input paired comparison and sample accounting."""

import copy
import importlib.util
from pathlib import Path
import unittest


_PATH = Path(__file__).resolve().parents[1] / "benchmarks/report_nccl_shm.py"
_SPEC = importlib.util.spec_from_file_location("report_nccl_shm", _PATH)
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def fixture():
    records = [{"rank": rank, "routes_sha256": "a"*64,
                "weights_sha256": "b"*64, "payload_sha256": "c"*64} for rank in range(8)]
    routing = [{key: item[key] for key in ("rank", "routes_sha256", "weights_sha256")} for item in records]
    fingerprints = dict(per_rank=records, global_routes_sha256=report.digest(routing),
                        global_inputs_sha256=report.digest(records))
    rows = []
    for tokens in report.TOKENS:
        for index, variant in enumerate(report.VARIANTS):
            ranks = [[float((index+1)*100 + rank + iteration) for iteration in range(20)] for rank in range(8)]
            maxima = [max(values[i] for values in ranks) for i in range(20)]
            validation = [dict(allgather_source_payload_bytes_checked=8*tokens*7168*2,
                               expanded_rows=tokens, bf16_elements_checked=tokens*7168,
                               stable_order=True) for _ in range(8)]
            rows.append(dict(report.EXPECTED, tokens_per_rank=tokens, variant=variant,
                input_fingerprints=copy.deepcopy(fingerprints), per_rank_dispatch_samples_us=ranks,
                full_dispatch_samples_us=maxima, route_volume=dict(remote_source_tokens=tokens*8,
                remote_deliveries=tokens*32, remote_expert_branches=tokens*56, relay_token_copies=tokens*8),
                validation_per_rank=validation))
    document = dict(metadata=dict(profiling_enabled=False, nccl_environment=report.REQUIRED_ENV, pcie_config=report.PCIE_CONFIG,
                                 pytorch="test", cuda="test", nccl=[2,29,7]), results=rows)
    return [copy.deepcopy(document), copy.deepcopy(document)]


class CompareTest(unittest.TestCase):
    def test_recompute_max_rank_and_bandwidth(self):
        result = report.compare(fixture())
        row = result["comparison"][0]
        self.assertAlmostEqual(row["pcie_ep"]["mean_ms"], 0.1165)
        self.assertAlmostEqual(row["pcie_ep"]["logical_GBps"], 16*8*7168*2/(116.5e-6)/1e9)
        self.assertAlmostEqual(row["pcie_speedup_vs_dispatch"], 316.5/116.5)
        self.assertEqual(len(result["rounds"]), 2)

    def test_top4_bandwidth_uses_four_branches_and_seven_remote_copies(self):
        documents = fixture()
        for document in documents:
            document['metadata']['routing_fixture'] = dict(schema_version=2, generator='test', profile='test', tensors_sha256='d'*64, fixture_id='f'*64, manifest_sha256='e'*64)
            for row in document['results']:
                row['topk'] = 4
        route_fixture = {'routes': [
            {'tokens_per_gpu': tokens,
             'global_routes_sha256': documents[0]['results'][0]['input_fingerprints']['global_routes_sha256'],
             'route_volume': documents[0]['results'][0]['route_volume'] | {
                 key: value*tokens//16 for key, value in documents[0]['results'][0]['route_volume'].items()}}
            for tokens in report.TOKENS]}
        route_fixture.update(dict(schema_version=2, generator='test', profile='test', tensors_sha256='d'*64, fixture_id='f'*64, **{key: documents[0]['results'][0][key] for key in ('world_size','experts','topk','hidden','seed')}))
        result = report.compare(documents, route_fixture, 'e'*64)
        with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
            report.compare(documents, route_fixture, '0'*64)
        changed = copy.deepcopy(route_fixture); changed['fixture_id'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'fixture identity'):
            report.compare(documents, changed, 'e'*64)
        row = result['comparison'][0]
        self.assertAlmostEqual(row['pcie_ep']['logical_GBps'], 16*4*7168*2/(116.5e-6)/1e9)
        self.assertAlmostEqual(row['nccl_payload']['nccl_tests_busbw_GBps'],
                               16*7*7168*2/(216.5e-6)/1e9)
        self.assertEqual(result, report.compare(report.anonymous_rounds(result), route_fixture))

    def test_top4_requires_persisted_fixture(self):
        documents = fixture()
        for document in documents:
            for row in document['results']: row['topk'] = 4
        with self.assertRaisesRegex(ValueError, 'persisted routing fixture'):
            report.compare(documents)

    def test_reject_changed_topk_or_fixture(self):
        documents = fixture()
        documents[1]['results'][0]['topk'] = 4
        with self.assertRaisesRegex(ValueError, 'Workload changes'):
            report.compare(documents)
        documents = fixture()
        documents[0]['metadata']['routing_fixture'] = {'manifest_sha256':'a'*64}
        with self.assertRaisesRegex(ValueError, 'fixture changes'):
            report.compare(documents)

    def test_reject_wrong_global_route_digest(self):
        documents = fixture(); documents[0]["results"][0]["input_fingerprints"]["global_routes_sha256"] = "0"*64
        with self.assertRaises(ValueError): report.compare(documents)

    def test_reject_changed_route_with_valid_digest(self):
        documents = fixture(); value = documents[1]["results"][0]["input_fingerprints"]
        value["per_rank"][0]["routes_sha256"] = "d"*64
        routing = [{key:item[key] for key in ("rank", "routes_sha256", "weights_sha256")} for item in value["per_rank"]]
        value["global_routes_sha256"] = report.digest(routing)
        value["global_inputs_sha256"] = report.digest(value["per_rank"])
        with self.assertRaisesRegex(ValueError, "differ between"): report.compare(documents)

    def test_reject_changed_payload(self):
        documents = fixture(); value = documents[0]["results"][1]["input_fingerprints"]
        value["per_rank"][0]["payload_sha256"] = "d"*64
        value["global_inputs_sha256"] = report.digest(value["per_rank"])
        with self.assertRaisesRegex(ValueError, "differ between"): report.compare(documents)

    def test_reject_wrong_max_rank(self):
        documents = fixture(); documents[0]["results"][0]["full_dispatch_samples_us"][0] += 1
        with self.assertRaises(ValueError): report.compare(documents)

    def test_reject_profiled_or_wrong_path(self):
        for key in ("profile", "p2p"):
            with self.subTest(key=key):
                documents = fixture()
                if key == "profile": documents[0]["metadata"]["profiling_enabled"] = True
                else: documents[0]["metadata"]["nccl_environment"]["NCCL_P2P_DISABLE"] = "0"
                with self.assertRaises(ValueError): report.compare(documents)

    def test_reject_missing_validation_or_samples(self):
        for key in ("validation_per_rank", "per_rank_dispatch_samples_us"):
            with self.subTest(key=key):
                documents = fixture(); documents[0]["results"][0][key].pop()
                with self.assertRaises(ValueError): report.compare(documents)

    def test_reject_incomplete_round(self):
        documents = fixture(); documents[1]["results"].pop()
        with self.assertRaises(ValueError): report.compare(documents)

    def test_reject_pair_that_differs_from_original_routes(self):
        documents = fixture()
        reference = {'routes':[{'tokens_per_gpu':tokens, 'global_routes_sha256':'0'*64,
                               'route_volume':documents[0]['results'][0]['route_volume']}
                              for tokens in report.TOKENS]}
        with self.assertRaisesRegex(ValueError, 'original experiment'):
            report.compare(documents, reference)

    def test_reject_changed_channel_configuration(self):
        documents = fixture()
        documents[1]['metadata']['nccl_environment']['NCCL_MIN_NCHANNELS']='32'
        with self.assertRaisesRegex(ValueError, 'configuration changes'):
            report.compare(documents)

    def test_reject_changed_protocol_configuration(self):
        documents = fixture()
        documents[1]['metadata']['nccl_environment']['NCCL_PROTO']='Simple'
        with self.assertRaisesRegex(ValueError, 'configuration changes'):
            report.compare(documents)

    def test_anonymous_samples_roundtrip_recomputes(self):
        result = report.compare(fixture())
        restored = report.compare(report.anonymous_rounds(result))
        self.assertEqual(result, restored)


if __name__ == "__main__":
    unittest.main()
