"""CPU-only API contract checks; transport integration lives in correctness.py."""

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from pcie_ep import EventOverlap, HostStagedConfig, HostStagedEPBuffer, HostStagedEPHandle
from pcie_ep.buffer import _prepare_segments
from test_profiler import audit_trace


class ModuleContractTests(unittest.TestCase):
    def test_original_exports_refer_to_the_split_implementation(self):
        import pcie_ep
        import pcie_ep.buffer as compatibility
        from pcie_ep.buffers.ep import HostStagedEPBuffer as buffer_class
        from pcie_ep.buffers.handle import HostStagedEPHandle as handle_class
        from pcie_ep.comm import _prepare_segments as prepare_segments
        from pcie_ep.config import HostStagedConfig as config_class
        from pcie_ep.utils.event import EventOverlap as event_class

        self.assertIs(pcie_ep.HostStagedEPBuffer, buffer_class)
        self.assertIs(compatibility.EPBuffer, buffer_class)
        self.assertIs(compatibility.EPHandle, handle_class)
        self.assertIs(compatibility.HostStagedConfig, config_class)
        self.assertIs(compatibility.EventOverlap, event_class)
        self.assertIs(compatibility._prepare_segments, prepare_segments)

    def test_gloo_group_does_not_initialize_nccl(self):
        from pcie_ep.comm import borrow_nccl_comm

        group = Mock()
        with (
            patch("pcie_ep.buffer.dist.get_backend", return_value="gloo"),
            patch("pcie_ep.buffer.dist.new_group") as new_group,
            patch("pcie_ep.buffer.dist.all_gather_object") as gather,
        ):
            borrowed = borrow_nccl_comm(group, device=0)
        self.assertFalse(borrowed.reused)
        self.assertIsNone(borrowed.nccl_comm)
        self.assertEqual(borrowed.status, "not_an_nccl_group")
        group._get_backend.assert_not_called()
        new_group.assert_not_called()
        gather.assert_not_called()

    def test_nccl_pointer_is_borrowed_without_creating_a_group(self):
        from pcie_ep.comm import borrow_nccl_comm

        group = Mock()
        group._get_backend.return_value._comm_ptr.return_value = 123
        with (
            patch("pcie_ep.buffer.dist.get_backend", return_value="nccl"),
            patch("pcie_ep.buffer.dist.new_group") as new_group,
        ):
            borrowed = borrow_nccl_comm(group, device=0, require_reuse=True)
        self.assertEqual(borrowed.nccl_comm, 123)
        self.assertTrue(borrowed.reused)
        self.assertEqual(borrowed.status, "borrowed")
        group._get_backend.assert_called_once_with(torch.device("cuda", 0))
        group._get_backend.return_value._comm_ptr.assert_called_once_with()
        new_group.assert_not_called()

    def test_unavailable_nccl_pointer_has_no_fallback_initialization(self):
        from pcie_ep.comm import borrow_nccl_comm

        group = Mock()
        group._get_backend.return_value = SimpleNamespace()
        with (
            patch("pcie_ep.buffer.dist.get_backend", return_value="nccl"),
            patch("pcie_ep.buffer.dist.new_group") as new_group,
            patch("pcie_ep.buffer.dist.all_gather_object") as gather,
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot borrow"):
                borrow_nccl_comm(group, device=0, require_reuse=True)
        new_group.assert_not_called()
        gather.assert_not_called()

    def test_runtime_initialization_orders_construction_stream_without_cpu_wait(self):
        from pcie_ep.comm import register_runtime

        calls = []
        extension, runtime, comm_stream, construction_stream = Mock(), Mock(), Mock(), Mock()
        extension.Runtime.side_effect = lambda *args: calls.append("runtime") or runtime
        comm_stream.wait_stream.side_effect = lambda stream: calls.append("wait-construction")

        def gather(output, error, group):
            calls.append("gather")
            output[:] = [error, None]

        with (
            patch("pcie_ep.buffer.torch.cuda.current_stream", return_value=construction_stream),
            patch("pcie_ep.buffer.dist.all_gather_object", side_effect=gather),
        ):
            actual = register_runtime(
                extension,
                "/dev/shm/test",
                0,
                2,
                16,
                7168,
                8,
                256,
                HostStagedConfig(),
                "control",
                comm_stream,
                0,
            )
        self.assertIs(actual, runtime)
        self.assertEqual(calls, ["runtime", "wait-construction", "gather"])
        comm_stream.wait_stream.assert_called_once_with(construction_stream)
        comm_stream.synchronize.assert_not_called()
        runtime.close.assert_not_called()


class ConfigTests(unittest.TestCase):
    def test_bounded_ring_defaults(self):
        config = HostStagedConfig()
        self.assertEqual((config.payload_slots, config.descriptor_slots, config.stage), (256, 256, 6))

    def test_invalid_capacity_rejected(self):
        for kwargs in ({"payload_slots": 0}, {"descriptor_slots": -1}, {"timeout_ms": 0}, {"stage": 9}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HostStagedConfig(**kwargs)

    def test_retired_runtime_stages_are_rejected(self):
        for stage in range(1, 6):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "retired"):
                HostStagedConfig(stage=stage)
        config = HostStagedConfig()
        self.assertEqual(config.channels, 32)
        self.assertTrue(config.warp_specialization and config.fuse_receive_expand and config.source_fusion)

    def test_receive_expand_mode_default_and_native_options(self):
        self.assertEqual(HostStagedConfig().receive_expand_mode, "pipelined")
        for mode in ("scalar", "vectorized", "pipelined"):
            config = HostStagedConfig(
                stage=6, fuse_receive_expand=True, warp_specialization=True, receive_expand_mode=mode
            )
            self.assertEqual(config.native_options()["receive_expand_mode"], mode)

    def test_receive_expand_mode_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "receive_expand_mode"):
            HostStagedConfig(receive_expand_mode="unknown")
        for mode in ("vectorized", "pipelined"):
            with self.subTest(mode=mode):
                for flags in ({"fuse_receive_expand": False}, {"warp_specialization": False}):
                    with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "requires"):
                        HostStagedConfig(stage=6, receive_expand_mode=mode, **flags)

    def test_receive_expand_mode_test_argument_bridge(self):
        import argparse
        from advanced_options import add_transport_arguments, transport_options

        parser = argparse.ArgumentParser()
        add_transport_arguments(parser, stage=6)
        args = parser.parse_args(
            ["--receive-expand-mode", "pipelined", "--fuse-receive-expand", "--warp-specialization"]
        )
        config = HostStagedConfig(**transport_options(args))
        self.assertEqual(config.receive_expand_mode, "pipelined")

    def test_receive_expand_mode_is_checked_collectively(self):
        from pcie_ep.comm import validate_group_configuration

        config = HostStagedConfig(stage=6, fuse_receive_expand=True, warp_specialization=True)

        def gather(states, state, **kwargs):
            different = dict(state[-1])
            different["receive_expand_mode"] = "vectorized"
            states[:] = [state, (*state[:-1], tuple(different.items()))]

        with patch("pcie_ep.comm.dist.all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(ValueError, "identical buffer configuration"):
                validate_group_configuration("control", 2, 16, 7168, 8, 256, config)

    def test_relay_worker_defaults_and_native_options(self):
        default = HostStagedConfig()
        self.assertEqual(default.relay_workers_per_numa, 1)
        self.assertEqual(default.relay_cpu_affinity, ())
        config = HostStagedConfig(
            stage=8, channels=4, numa_mode="relay", relay_workers_per_numa=2, relay_cpu_affinity=(0, 2, 64, 66)
        )
        options = config.native_options()
        self.assertEqual(options["relay_workers_per_numa"], 2)
        self.assertEqual(options["relay_cpu_affinity"], [0, 2, 64, 66])

    def test_relay_worker_configuration_requires_s8(self):
        for kwargs in ({"relay_workers_per_numa": 2}, {"relay_cpu_affinity": (0,)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(NotImplementedError, "S8"):
                HostStagedConfig(stage=7, channels=4, **kwargs)

    def test_relay_worker_bounds_and_cpu_affinity(self):
        for kwargs in (
            {"relay_workers_per_numa": 0},
            {"relay_workers_per_numa": 5},
            {"relay_workers_per_numa": 1.5},
            {"relay_cpu_affinity": (-1,)},
            {"relay_cpu_affinity": (0, 0)},
            {"relay_cpu_affinity": (0.5,)},
            {"relay_cpu_affinity": [0, 1]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HostStagedConfig(stage=8, channels=4, **kwargs)


class EventTests(unittest.TestCase):
    def test_wait_registers_consumer_before_epilogue_without_cpu_sync(self):
        order = []
        stream, native_event = Mock(), Mock()
        stream.wait_event.side_effect = lambda event: order.append("wait")
        callback = Mock(side_effect=lambda value: order.append("consumer"))
        event = EventOverlap(native_event, on_wait=callback)
        event.register_hook_after_wait(lambda: order.append("epilogue"))
        with patch("pcie_ep.buffer.torch.cuda.current_stream", return_value=stream):
            event.current_stream_wait(release_handle=True)
        self.assertEqual(order, ["wait", "consumer", "epilogue"])
        callback.assert_called_once_with(stream)
        native_event.synchronize.assert_not_called()
        self.assertIsNone(event._on_wait)

    def test_wait_enqueues_stream_dependency_without_cpu_wait(self):
        native_event = Mock()
        stream = Mock()
        event = EventOverlap(native_event)
        hook = Mock(return_value=("x", "idx", "weights", "handle"))
        event.register_hook_after_wait(hook)
        with patch("pcie_ep.buffer.torch.cuda.current_stream", return_value=stream):
            self.assertEqual(event.wait(), ("x", "idx", "weights", "handle"))
            self.assertIsNone(event.wait())
        self.assertEqual(stream.wait_event.call_count, 2)
        native_event.synchronize.assert_not_called()
        hook.assert_called_once()

    def test_context_exit_wait_and_release(self):
        stream = Mock()
        event = EventOverlap(Mock(), extra_tensors=())
        with patch("pcie_ep.buffer.torch.cuda.current_stream", return_value=stream):
            with event(release_handle=True):
                stream.wait_event.assert_not_called()
        stream.wait_event.assert_called_once()
        self.assertIsNone(event.event)
        self.assertIsNone(event.extra_tensors)
        with self.assertRaises(RuntimeError):
            event.wait()

    def test_hook_cannot_be_silently_replaced(self):
        event = EventOverlap(Mock())
        event.register_hook_after_wait(lambda: None)
        with self.assertRaises(RuntimeError):
            event.register_hook_after_wait(lambda: None)


class PreparationTests(unittest.TestCase):
    def test_first_touch_finishes_before_register_phase(self):
        calls = []
        extension = Mock()
        extension.prepare_segment.side_effect = lambda *args: calls.append("prepare")

        def gather(output, error, group):
            calls.append("gather")
            output[:] = [error, None]

        with (
            patch("pcie_ep.buffer.dist.all_gather_object", side_effect=gather),
            patch("pcie_ep.buffer.dist.barrier", side_effect=lambda **kwargs: calls.append("barrier")),
        ):
            _prepare_segments(extension, "/dev/shm/test", 0, 2, 16, 7168, 8, 256, 2, 2, 4, "control")
        self.assertEqual(calls, ["prepare", "gather", "barrier"])
        extension.prepare_segment.assert_called_once_with("/dev/shm/test", 0, 2, 16, 7168, 8, 256, 2, 2, 4)

    def test_prepare_error_is_collectively_reported_before_exit(self):
        extension = Mock()
        extension.prepare_segment.side_effect = OSError("first-touch failed")
        observed = []

        def gather(output, error, group):
            observed.append(error)
            output[:] = [error, None]

        with (
            patch("pcie_ep.buffer.dist.all_gather_object", side_effect=gather),
            patch("pcie_ep.buffer.dist.barrier") as barrier,
        ):
            with self.assertRaisesRegex(RuntimeError, "first-touch failed"):
                _prepare_segments(extension, "/dev/shm/test", 0, 2, 16, 7168, 8, 256, 2, 2, 4, "control")
        self.assertIn("rank 0", observed[0])
        barrier.assert_called_once_with(group="control")

    def test_remote_prepare_error_stops_local_registration(self):
        extension = Mock()

        def gather(output, error, group):
            output[:] = [None, "rank 1: first-touch failed"]

        with (
            patch("pcie_ep.buffer.dist.all_gather_object", side_effect=gather),
            patch("pcie_ep.buffer.dist.barrier") as barrier,
        ):
            with self.assertRaisesRegex(RuntimeError, "rank 1"):
                _prepare_segments(extension, "/dev/shm/test", 0, 2, 16, 7168, 8, 256, 2, 2, 4, "control")
        barrier.assert_called_once_with(group="control")


class HandleMetadataTests(unittest.TestCase):
    def test_native_metadata_is_aliased_and_source_prefix_is_inclusive(self):
        owner = SimpleNamespace(
            buffer_id="test",
            _generation=0,
            group="group",
            global_ranks=(2, 7),
            num_experts=256,
            num_max_tokens_per_rank=16,
        )
        result = {
            "unique_count": torch.tensor([3]),
            "expert_counts": torch.tensor([1, 2]),
            "expert_offsets": torch.tensor([0, 4, 8]),
            "source_meta": torch.zeros((3, 3), dtype=torch.int64),
            "error": torch.zeros(8, dtype=torch.int64),
            "stats": torch.zeros(8, dtype=torch.int64),
            "transport_timestamps_ns": torch.zeros(2, dtype=torch.int64),
            "expanded_count": torch.tensor([3]),
            "expert_prefix": torch.tensor([1, 6]),
            "source_offsets": torch.tensor([0, 2, 3]),
        }
        handle = HostStagedEPHandle(
            owner, torch.zeros((2, 8), dtype=torch.int64), result, True, 4, EventOverlap(Mock())
        )
        self.assertIs(handle.expanded_count, result["expanded_count"])
        self.assertIs(handle.expert_counts, result["expert_counts"])
        self.assertIs(handle.source_offsets, result["source_offsets"])
        self.assertTrue(torch.equal(handle.psum_num_recv_tokens_per_scaleup_rank, torch.tensor([2, 3])))
        self.assertIs(handle.psum_num_recv_tokens_per_expert, result["expert_prefix"])
        self.assertIsNone(handle.num_recv_tokens_per_expert_list)
        self.assertIsNone(handle.receive_expand_overlap)
        result["receive_expand_overlap"] = torch.tensor([[0, 1], [2, 4]], dtype=torch.int64)
        optimized = HostStagedEPHandle(
            owner, torch.zeros((2, 8), dtype=torch.int64), result, True, 4, EventOverlap(Mock())
        )
        self.assertIs(optimized.receive_expand_overlap, result["receive_expand_overlap"])

    def test_error_diagnostics_survive_overlap_release(self):
        buffer = object.__new__(HostStagedEPBuffer)
        buffer.buffer_id, buffer._generation, buffer.rank_idx = "test", 0, 0
        buffer._failed = False
        native_event = Mock()
        overlap = EventOverlap(native_event)
        selected = SimpleNamespace(
            buffer_id="test",
            generation=0,
            completion=overlap,
            _completion_event=native_event,
            error=torch.zeros(8, dtype=torch.int64),
        )
        with patch("pcie_ep.buffer.torch.cuda.current_stream", return_value=Mock()):
            overlap.current_stream_wait(release_handle=True)
        self.assertIsNone(overlap.event)
        buffer.check_errors(selected)
        native_event.synchronize.assert_called_once()
        self.assertFalse(buffer._failed)


class ConsumerStreamTests(unittest.TestCase):
    def setUp(self):
        self.buffer = object.__new__(HostStagedEPBuffer)
        self.buffer._lock = threading.RLock()
        self.buffer._consumer_streams = {}
        self.compute = SimpleNamespace(device=torch.device("cuda", 0), cuda_stream=1)
        self.consumer = SimpleNamespace(device=torch.device("cuda", 0), cuda_stream=2)
        self.buffer._comm_stream = SimpleNamespace(device=torch.device("cuda", 0), cuda_stream=99, wait_stream=Mock())

    def test_registered_consumer_is_waited_and_compute_wait_is_deduplicated(self):
        self.buffer._register_consumer_stream(self.compute)
        self.buffer._register_consumer_stream(self.consumer)
        self.buffer._register_consumer_stream(self.consumer)
        self.assertEqual(len(self.buffer._consumer_streams), 2)
        self.buffer._wait_consumer_streams(self.compute, compute_already_waited=True)
        self.buffer._comm_stream.wait_stream.assert_called_once_with(self.consumer)
        self.assertEqual(len(self.buffer._consumer_streams), 2)

    def test_consumer_stream_tail_is_waited_again_on_later_dispatch(self):
        self.buffer._register_consumer_stream(self.consumer)
        self.buffer._wait_consumer_streams(self.compute, compute_already_waited=True)
        self.buffer._wait_consumer_streams(self.compute, compute_already_waited=True)
        self.assertEqual(self.buffer._comm_stream.wait_stream.call_count, 2)

    def test_previous_event_does_not_drop_registered_compute_dependency(self):
        self.buffer._register_consumer_stream(self.compute)
        self.buffer._wait_consumer_streams(self.compute, compute_already_waited=False)
        self.buffer._comm_stream.wait_stream.assert_called_once_with(self.compute)

    def test_communication_stream_is_not_registered_as_a_consumer(self):
        self.buffer._register_consumer_stream(self.buffer._comm_stream)
        self.assertFalse(self.buffer._consumer_streams)


class ProfilerAuditTests(unittest.TestCase):
    def audit(self, additions=()):
        events = [
            {"name": "pcie_ep_steady_dispatch", "ph": "X", "pid": 1, "ts": 100, "dur": 100},
            {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "pid": 1, "ts": 120},
            {"name": "route", "cat": "kernel", "pid": 2, "ts": 400},
            {"name": "cudaDeviceSynchronize", "cat": "cuda_runtime", "pid": 1, "ts": 300},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": [*events, *additions]}))
            return audit_trace(path)

    def test_profiler_teardown_sync_is_outside_steady_scope(self):
        self.assertEqual(self.audit()["status"], "passed")

    def test_cpu_sync_inside_dispatch_is_detected(self):
        report = self.audit([{"name": "cudaStreamSynchronize", "pid": 1, "ts": 150}])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cpu_synchronization_calls"], ["cudaStreamSynchronize"])

    def test_async_dtoh_after_cpu_marker_is_still_detected(self):
        report = self.audit([{"name": "Memcpy DtoH (Device -> Host)", "pid": 2, "ts": 500}])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["device_to_host_memcopies"]), 1)

    def test_gpu_scalar_readback_inside_dispatch_is_detected(self):
        report = self.audit([{"name": "aten::_local_scalar_dense", "pid": 1, "ts": 150}])
        self.assertEqual(report["status"], "failed")


class RejectionTests(unittest.TestCase):
    def setUp(self):
        # Rejection happens before tensor use or CUDA submission.  No process
        # group, CUDA device, or compiled extension is needed for these checks.
        self.buffer = object.__new__(HostStagedEPBuffer)
        self.buffer._lock = threading.RLock()
        self.buffer._check_live = lambda: None
        self.buffer.num_experts = 256
        self.buffer.num_max_tokens_per_rank = 8192
        self.buffer.hidden, self.buffer.num_topk = 7168, 8

    def test_cpu_sync_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(NotImplementedError, "do_cpu_sync=False"):
            self.buffer.dispatch(None, do_cpu_sync=True)

    def test_cached_replay_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(NotImplementedError, "S6"):
            self.buffer.dispatch(None, handle=object())

    def test_unsupported_qps_and_handle_copy(self):
        for kwargs in ({"num_qps": 1}, {"num_sms": 4}, {"do_handle_copy": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                self.buffer.dispatch(None, **kwargs)

    def test_event_and_deferred_contract(self):
        with self.assertRaisesRegex(ValueError, "allocate_on_comm_stream"):
            self.buffer.dispatch(None, previous_event=Mock())
        with self.assertRaisesRegex(ValueError, "async_with_compute_stream"):
            self.buffer.dispatch(None, defer_epilogue=True)

    def test_expansion_and_capacity_contract(self):
        for kwargs in (
            {"expert_alignment": 0},
            {"do_zero_padding": True},
            {"num_experts": 128},
            {"num_max_tokens_per_rank": 4096},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.buffer.dispatch(None, **kwargs)

    def test_cpu_tensor_rejected_before_submission(self):
        self.buffer.device = 0
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            self.buffer.dispatch(torch.empty(1, 7168, dtype=torch.bfloat16))

    def test_combine_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(NotImplementedError, "dispatch only"):
            self.buffer.combine(None)


if __name__ == "__main__":
    unittest.main()
