"""S6 cache/Graph CPU contracts and optional distributed BF16 validation.

python -m unittest discover -s tests -p test_s6.py -v
torchrun --standalone --nproc-per-node=2 tests/test_s6.py --distributed
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from pcie_ep import EventOverlap, GraphDispatch, HostStagedConfig, HostStagedEPBuffer
from pcie_ep.buffers.handle import CachedLayout, HostStagedEPHandle
from pcie_ep.native import AdvancedExtensionAdapter
from pcie_ep.graph import capture_dispatch


def cpu_handle():
    owner = SimpleNamespace(
        buffer_id="test",
        _generation=0,
        group=object(),
        global_ranks=(0, 1),
        num_experts=256,
        num_max_tokens_per_rank=16,
        hidden=8,
    )
    idx = torch.arange(8).repeat(2, 1)
    result = {
        "unique_count": torch.tensor([2]),
        "expert_counts": torch.tensor([2]),
        "expert_offsets": torch.tensor([0, 2]),
        "expanded_count": torch.tensor([2]),
        "source_meta": torch.zeros((2, 3), dtype=torch.int64),
        "source_offsets": torch.tensor([0, 1, 2]),
        "error": torch.zeros(8, dtype=torch.int64),
        "stats": torch.zeros(8, dtype=torch.int64),
        "_layout": object(),
    }
    handle = HostStagedEPHandle(owner, idx, result, False, 1, EventOverlap(Mock()))
    return owner, idx, handle


class CachedLayoutTests(unittest.TestCase):
    def test_lightweight_cache_has_no_output_pool_references(self):
        owner, idx, handle = cpu_handle()
        cache = handle.cache_layout()
        self.assertIsInstance(cache, CachedLayout)
        self.assertIs(cache.topk_idx, idx)
        self.assertIs(cache._layout, handle._layout)
        for field in ("expert_counts", "source_offsets", "error", "completion", "source_meta"):
            self.assertFalse(hasattr(cache, field), field)
        self.assertIs(cache.validate_cached(owner, torch.zeros(2, 8), 1, False), handle._layout)

    def test_lightweight_cache_does_not_keep_original_handle_alive(self):
        owner, idx, handle = cpu_handle()
        original = weakref.ref(handle)
        cache = handle.cached_layout
        del handle
        gc.collect()
        self.assertIsNone(original())
        self.assertIs(cache.topk_idx, idx)
        self.assertIsNotNone(cache.validate_cached(owner, torch.zeros(2, 8), 1, False))

    def test_inplace_route_change_invalidates_cache(self):
        owner, idx, handle = cpu_handle()
        cache = handle.cache_layout()
        idx.add_(1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            cache.validate_cached(owner, torch.zeros(2, 8), 1, False)

    def test_generation_group_shape_and_alignment_are_checked(self):
        for change in ("generation", "group", "shape", "alignment"):
            owner, idx, handle = cpu_handle()
            cache = handle.cache_layout()
            x, alignment = torch.zeros(2, 8), 1
            if change == "generation":
                owner._generation += 1
            elif change == "group":
                owner.group = object()
            elif change == "shape":
                x = torch.zeros(3, 8)
            else:
                alignment = 8
            with self.subTest(change=change), self.assertRaises(ValueError):
                cache.validate_cached(owner, x, alignment, False)

    def test_dynamic_graph_allows_route_content_updates_but_not_resize(self):
        owner, idx, handle = cpu_handle()
        x = torch.zeros(2, 8)
        graph = GraphDispatch(owner, x, idx, None, Mock(), {}, 1, False, None)
        idx.add_(1)
        graph._validate_bindings()
        idx.resize_(3, 8)
        with self.assertRaisesRegex(ValueError, "address, shape"):
            graph._validate_bindings()

    def test_static_graph_rejects_cached_route_updates(self):
        owner, idx, handle = cpu_handle()
        graph = GraphDispatch(owner, torch.zeros(2, 8), idx, None, Mock(), {}, 1, False, handle.cache_layout())
        idx.add_(1)
        with self.assertRaises(RuntimeError):
            graph._validate_bindings()


class AdvancedConfigurationTests(unittest.TestCase):
    def test_features_have_explicit_stage_gates(self):
        for kwargs in (
            {"stage": 6, "wire_dtype": "fp8_e4m3"},
            {"stage": 7, "numa_mode": "relay"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                HostStagedConfig(**kwargs)
        HostStagedConfig(
            stage=8,
            channels=4,
            pipeline_stages=3,
            warp_specialization=True,
            source_fusion=True,
            fuse_receive_expand=True,
            wire_dtype="fp8_e4m3",
            numa_mode="relay",
            enable_graph=True,
        )

    def test_native_adapter_uses_current_collective_setup(self):
        extension = Mock()
        config = HostStagedConfig(stage=6, channels=4, fuse_receive_expand=True)
        adapter = AdvancedExtensionAdapter(extension, config)
        adapter.create_segments("dir", 8, 16, 7168, 8, 256, 256, 256, 6)
        extension.create_advanced_segments.assert_called_once_with(
            "dir",
            8,
            16,
            7168,
            8,
            256,
            adapter.options,
        )
        self.assertEqual(adapter.options["channels"], 4)


class GraphPreparationTests(unittest.TestCase):
    def owner(self):
        return SimpleNamespace(
            _lock=threading.RLock(),
            _check_live=Mock(),
            _graph_capture=None,
            config=HostStagedConfig(stage=6, enable_graph=True),
            _validate_input=Mock(),
            num_ranks=2,
            rank_idx=0,
            control_group=object(),
            device=0,
            hidden=8,
            num_topk=8,
            num_max_tokens_per_rank=16,
            runtime=Mock(),
        )

    def test_local_validation_error_is_exchanged_before_capture(self):
        owner = self.owner()
        owner._validate_input.side_effect = ValueError("bad local input")

        def gather(errors, error, **kwargs):
            errors[:] = [error, None]

        with patch("pcie_ep.graph.dist.all_gather_object", side_effect=gather) as collective, patch(
            "pcie_ep.graph.torch.cuda.synchronize"
        ) as synchronize:
            with self.assertRaisesRegex(RuntimeError, "bad local input"):
                capture_dispatch(owner, torch.zeros(2, 8), torch.zeros(2, 8, dtype=torch.int64))
        collective.assert_called_once()
        synchronize.assert_not_called()
        owner.runtime.dispatch.assert_not_called()

    def test_remote_preparation_error_releases_unused_graph(self):
        owner, graph = self.owner(), Mock()

        def gather(errors, error, **kwargs):
            errors[:] = [error, "rank 1: preparation failed"]

        with patch("pcie_ep.graph.dist.all_gather_object", side_effect=gather), patch(
            "pcie_ep.graph.torch.cuda.synchronize"
        ), patch("pcie_ep.graph.torch.cuda.CUDAGraph", return_value=graph):
            with self.assertRaisesRegex(RuntimeError, "rank 1"):
                capture_dispatch(owner, torch.zeros(2, 8), torch.zeros(2, 8, dtype=torch.int64))
        graph.reset.assert_called_once()
        owner.runtime.dispatch.assert_not_called()


def distributed_main():
    import torch.distributed as dist
    from test_correctness import gather_cpu, make_inputs, verify_result

    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--stage", type=int, choices=(6, 7, 8), default=6)
    parser.add_argument("--numa-mode", choices=("direct", "relay"), default="direct")
    parser.add_argument("--output", type=Path, default=Path("results/s6.json"))
    args = parser.parse_args()
    if args.numa_mode == "relay" and args.stage < 8:
        parser.error("NUMA relay requires --stage 8")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    cpu = make_inputs(rank, world, 17, 256, 256, 8, "multicast")
    inputs = gather_cpu(cpu)
    x, idx, weights = (value.to("cuda") for value in cpu)
    config = HostStagedConfig(
        stage=args.stage,
        channels=2,
        source_fusion=True,
        enable_graph=True,
        fuse_receive_expand=True,
        payload_slots=2,
        descriptor_slots=3,
        numa_mode=args.numa_mode,
    )

    def update_values(base, step):
        updated = ((base[0].float() + step * 0.25).to(torch.bfloat16), base[1], base[2] + step * 0.125)
        x.copy_(updated[0].to(x.device))
        weights.copy_(updated[2].to(weights.device))
        return updated

    def expect_changed_route_rejection(call):
        try:
            call()
        except RuntimeError as error:
            if "changed" not in str(error):
                raise
        else:
            raise AssertionError("An in-place route change was accepted by a static cache")

    try:
        with HostStagedEPBuffer(
            group=dist.group.WORLD, num_max_tokens_per_rank=17, hidden=256, num_topk=8, num_experts=256, config=config
        ) as buffer:
            first = buffer.dispatch(x, idx, weights, do_expand=True)
            buffer.check_errors(first[3])
            cache = buffer.cache_layout(first[3])
            del first
            gc.collect()
            current = cpu
            for iteration in range(25):
                current = update_values(cpu, iteration + 1)
                result = buffer.dispatch(x, topk_weights=weights, handle=cache, do_expand=True)
                result[4].wait()
                if iteration in (0, 12, 24):
                    buffer.check_errors(result[3])
                    inputs = gather_cpu(current)
                    verify_result(result, inputs, rank, world, 256, 8, True, 1)
            buffer.check_errors(result[3])
            verify_result(result, inputs, rank, world, 256, 8, True, 1)
            del result
            graph = buffer.capture_dispatch(x, idx, weights, do_expand=True)
            for _ in range(5):
                result = graph.replay()
                buffer.check_errors(result[3])
                verify_result(result, inputs, rank, world, 256, 8, True, 1)
                eager = buffer.dispatch(x, topk_weights=weights, handle=cache, do_expand=True)
                buffer.check_errors(eager[3])
                del eager
            # Dynamic graph re-reads routes. Rotate all expert IDs in place.
            idx.copy_((idx + 32) % 256)
            changed = (current[0], (current[1] + 32) % 256, current[2])
            changed_inputs = gather_cpu(changed)
            result = graph.replay()
            buffer.check_errors(result[3])
            verify_result(result, changed_inputs, rank, world, 256, 8, True, 1)

        # Static cache graph uses independent fixed-address inputs and keeps
        # routing unchanged while token values and weights vary each replay.
        del result
        with HostStagedEPBuffer(
            group=dist.group.WORLD, num_max_tokens_per_rank=17, hidden=256, num_topk=8, num_experts=256, config=config
        ) as buffer:
            prepared = buffer.dispatch(x, idx, weights, do_expand=True, expert_alignment=8)
            buffer.check_errors(prepared[3])
            cache = buffer.cache_layout(prepared[3])
            del prepared
            static_graph = buffer.capture_dispatch(
                x,
                topk_weights=weights,
                handle=cache,
                do_expand=True,
                expert_alignment=8,
            )
            for iteration in range(2):
                current = update_values(changed, iteration + 1)
                result = static_graph.replay()
                buffer.check_errors(result[3])
                verify_result(result, gather_cpu(current), rank, world, 256, 8, True, 8)
            idx.add_(1).remainder_(256)
            expect_changed_route_rejection(static_graph.replay)
            expect_changed_route_rejection(
                lambda: buffer.dispatch(
                    x,
                    topk_weights=weights,
                    handle=cache,
                    do_expand=True,
                    expert_alignment=8,
                )
            )
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "cached_repeats": 25,
                        "dynamic_graph_replays": 6,
                        "static_graph_replays": 2,
                        "dynamic_routes": True,
                        "cached_mutation_rejected": True,
                        "updated_values_and_weights": True,
                        "eager_graph_interleave": True,
                        "stage": args.stage,
                        "numa_mode": args.numa_mode,
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--distributed" in os.sys.argv:
        distributed_main()
    else:
        unittest.main()
