"""Bounded control-plane abort, prefix, drain and asynchronous-submit checks.

Examples (run each torchrun under an external process timeout as well):
  torchrun --standalone --nproc-per-node=2 tests/test_control_progress.py --quick
  torchrun --standalone --nproc-per-node=8 tests/test_control_progress.py \
      --modes scalar vectorized pipelined --output results/current/control_progress.json

Completion queries, CPU references, Gloo and error readbacks below are test
diagnostics. Every production dispatch explicitly uses do_cpu_sync=False.
The watchdog exits the process on a stuck event: unwinding a buffer context
would otherwise call collective close and could hang on that same kernel.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from test_correctness import gather_cpu, make_inputs, verify_result


EXPERTS, TOPK = 256, 8
PROFILE_EXIT_NS = 1
PROFILE_TX_FINISHED_NS = 23
PROFILE_TX_ABORT_OBSERVED_NS = 24  # TX-only alias; RX field 24 is RX finished.
REMOTE_ABORT_AFTER_NS = 20_000_000
SLOW_READER_NS = 100_000_000


class Verification:
    def __init__(self, args):
        self.args = args
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.records = []

    def save(self, status="running"):
        if self.rank == 0:
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            self.args.output.write_text(json.dumps(dict(
                status=status, world_size=self.world, hidden=self.args.hidden,
                watchdog_seconds=self.args.watchdog_seconds, records=self.records,
            ), indent=2) + "\n")

    def wait(self, result, label, started):
        event = result[3]._completion_event
        deadline = started + self.args.watchdog_seconds
        while True:
            completed = event.query()
            if time.perf_counter() >= deadline:
                diagnostic = dict(
                    status="watchdog_expired", rank=self.rank, case=label,
                    elapsed_seconds=time.perf_counter() - started,
                    limit_seconds=self.args.watchdog_seconds,
                )
                path = self.args.output.with_name(
                    f"{self.args.output.stem}.rank{self.rank}.watchdog.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(diagnostic, indent=2) + "\n")
                print(json.dumps(diagnostic), file=sys.stderr, flush=True)
                if completed:
                    raise AssertionError("GPU completion exceeded the control-progress deadline")
                os._exit(3)
            if completed:
                return time.perf_counter() - started
            time.sleep(0.001)

    def buffer(self, mode, channels, q=2, length=3, stages=2, timeout_ms=5000):
        config = HostStagedConfig(
            stage=6, channels=channels, payload_slots=q, descriptor_slots=length,
            pipeline_stages=stages, timeout_ms=timeout_ms,
            warp_specialization=True, fuse_receive_expand=True,
            receive_expand_mode=mode, source_fusion=True,
        )
        return HostStagedEPBuffer(
            group=dist.group.WORLD, num_max_tokens_per_rank=self.args.tokens,
            hidden=self.args.hidden, num_topk=TOPK, num_experts=EXPERTS, config=config,
        )

    def cpu_inputs(self, pattern, epoch=0, *, unequal=False):
        tokens = 0 if pattern == "empty" else self.args.tokens
        if unequal and tokens:
            tokens = max(1, tokens - self.rank)
        source_pattern = "none" if pattern == "empty" else "local" if pattern == "remote" else pattern
        x, indices, weights = make_inputs(
            self.rank, self.world, tokens, self.args.hidden,
            EXPERTS, TOPK, source_pattern, seed=700 + epoch,
        )
        if pattern == "remote":
            destination = (self.rank + 1) % self.world
            indices.copy_((torch.arange(TOPK) + destination * (EXPERTS // self.world)).repeat(tokens, 1))
        # Adjacent epochs must not pass by accidentally reusing old payloads.
        x = (x.float() + epoch * 4).to(torch.bfloat16)
        return x.contiguous(), indices, (weights + epoch * 0.03125).contiguous()

    def to_gpu(self, cpu):
        gpu = tuple(value.to("cuda") for value in cpu)
        torch.cuda.synchronize()  # Input/setup completion, before dispatch.
        return gpu

    def dispatch(self, buffer, gpu, *, expand=True):
        return buffer.dispatch(
            *gpu, do_expand=expand, expert_alignment=8 if expand else 1,
            do_zero_padding=expand, do_cpu_sync=False,
            async_with_compute_stream=True,
        )

    def diagnostic_error(self, buffer, result):
        # wait() has already established completion; this explicit API also
        # marks a failed Python buffer terminal before checking reuse below.
        raised = False
        try:
            buffer.check_errors(result[3])
        except RuntimeError as exc:
            if "GPU error" not in str(exc):
                raise
            raised = True
        code = int(result[3].error.detach().cpu().reshape(-1)[0])
        if raised != bool(code):
            raise AssertionError("GPU error code and check_errors disagree")
        return code

    def reject_failed_reuse(self, buffer, gpu, code):
        if not code:
            return
        try:
            self.dispatch(buffer, gpu)
        except RuntimeError as exc:
            if "failed" not in str(exc):
                raise
        else:
            raise AssertionError("Failed buffer accepted another dispatch")

    def guardian_profile(self, result):
        profile = result[3].profiling
        if profile is None:
            if self.args.require_profiling:
                raise AssertionError("PCIE_EP_PROFILE=1 build is required")
            return dict(available=False)
        if profile.ndim != 3 or profile.shape[0] < 1 or profile.shape[1] < 1 or profile.shape[2] <= 24:
            raise AssertionError(f"Unexpected profiling shape: {tuple(profile.shape)}")
        # All three values come from channel 0 / TX role 0 on this GPU.
        # Never subtract timestamps obtained from different GPUs.
        row = profile[0, 0].detach().cpu().tolist()
        finished, aborted, exited = (int(row[index]) for index in (
            PROFILE_TX_FINISHED_NS, PROFILE_TX_ABORT_OBSERVED_NS, PROFILE_EXIT_NS,
        ))
        if not 0 < finished < aborted < exited:
            raise AssertionError(f"TX guardian did not observe abort after finishing: {finished, aborted, exited}")
        return dict(available=True, first_tx_finished_ns=finished,
                    tx_abort_observed_ns=aborted, exit_ns=exited)

    def remote_abort(self, mode, channels):
        cpu = self.cpu_inputs("local")
        if self.rank == 1:
            # Only rank 1 sends remotely, and every branch goes to observer 0.
            cpu[1].copy_(torch.arange(TOPK).repeat(cpu[0].shape[0], 1))
        gpu = self.to_gpu(cpu)
        with self.buffer(mode, channels) as buffer:
            if self.rank == 0:
                buffer.runtime.set_test_delays(0, 0, SLOW_READER_NS, 0)
            buffer.runtime.set_test_control_abort_after_ns(
                REMOTE_ABORT_AFTER_NS if self.rank == 1 else 0
            )
            dist.barrier()
            started = time.perf_counter()
            result = self.dispatch(buffer, gpu)
            submitted_ms = (time.perf_counter() - started) * 1000
            elapsed = self.wait(result, "remote_abort", started)
            code = self.diagnostic_error(buffer, result)
            if code not in (0, 12, 15, 19):
                raise AssertionError(f"Unexpected remote-abort error code: {code}")
            if self.rank in (0, 1) and code != 15:
                raise AssertionError(f"Observer/source must report error 15, got {code}")
            profile = self.guardian_profile(result) if self.rank == 0 else None
            self.reject_failed_reuse(buffer, gpu, code)
            return dict(error_code=code, elapsed_seconds=elapsed,
                        cpu_submit_ms=submitted_ms, observer_profile=profile)

    def prefix_timeout(self, channels):
        cpu = self.cpu_inputs("multicast")
        gpu = self.to_gpu(cpu)
        with self.buffer("pipelined", channels, timeout_ms=self.args.prefix_timeout_ms) as buffer:
            dist.barrier()
            # Withhold one source's counts READY by delaying its comm-stream
            # dependency. Healthy peers must terminate their prefix wait;
            # the delayed source later observes the terminal mapped abort.
            if self.rank == self.world - 1:
                torch.cuda._sleep(self.args.prefix_delay_cycles)
            started = time.perf_counter()
            result = self.dispatch(buffer, gpu)
            elapsed = self.wait(result, "prefix_timeout", started)
            code = self.diagnostic_error(buffer, result)
            if self.rank != self.world - 1 and code not in (12, 15):
                raise AssertionError(f"Prefix waiter did not report timeout/abort: {code}")
            if code not in (12, 15, 19):
                raise AssertionError(f"Unexpected prefix failure code: {code}")
            self.reject_failed_reuse(buffer, gpu, code)
            return dict(error_code=code, elapsed_seconds=elapsed,
                        delayed_rank=self.world - 1, timeout_ms=self.args.prefix_timeout_ms)

    def mailbox_timeout(self, channels):
        cpu = self.cpu_inputs("local")
        inputs = gather_cpu(cpu)
        gpu = self.to_gpu(cpu)
        timeout_ms = 50
        with self.buffer("pipelined", channels, timeout_ms=timeout_ms) as buffer:
            # Rank 1 is the sole writer of source 0's counts-consumed ACK
            # for receiver 1. Its prefix warp captures all counts normally,
            # but the private hook suppresses just this one publication.
            # No CPU thread writes or repairs a mapped protocol flag.
            buffer.runtime.set_test_skip_counts_ack_source(0 if self.rank == 1 else -1)
            dist.barrier()
            started = time.perf_counter()
            first = self.dispatch(buffer, gpu)
            first_elapsed = self.wait(first, "mailbox_first_epoch", started)
            failure = None
            try:
                buffer.check_errors(first[3])
                verify_result(first, inputs, self.rank, self.world, EXPERTS, TOPK, True, 8)
            except Exception as exc:
                failure = f"rank {self.rank}, mailbox first epoch: {type(exc).__name__}: {exc}"
            failures = [item for item in gather_cpu(failure) if item]
            if failures:
                raise AssertionError("; ".join(failures))
            del first

            # Re-enabling future ACKs cannot retroactively acknowledge the
            # previous epoch. Source 0 must therefore fail in begin before
            # it may overwrite that still-owned counts mailbox.
            buffer.runtime.set_test_skip_counts_ack_source(-1)
            next_gpu = self.to_gpu(self.cpu_inputs("local", epoch=1))
            dist.barrier()
            started = time.perf_counter()
            second = self.dispatch(buffer, next_gpu)
            elapsed = self.wait(second, "mailbox_previous_ack_timeout", started)
            code = self.diagnostic_error(buffer, second)
            if self.rank == 0:
                if code != 11:
                    raise AssertionError(f"Source 0 must report prior-counts ACK timeout 11, got {code}")
            elif code not in (12, 15, 19):
                raise AssertionError(f"Peer did not terminate after mailbox abort: {code}")
            self.reject_failed_reuse(buffer, next_gpu, code)
            return dict(
                first_epoch_status="passed", first_epoch_elapsed_seconds=first_elapsed,
                withheld_ack_source=0, withheld_ack_receiver=1,
                second_epoch_error_code=code, second_epoch_elapsed_seconds=elapsed,
                timeout_ms=timeout_ms, terminal_reuse_rejected=True,
            )

    def success(self, mode, channels, q, length, stages):
        diagnostics = []
        with self.buffer(mode, channels, q, length, stages) as buffer:
            if self.rank == self.world - 1:
                buffer.runtime.set_test_delays(0, 0, 50_000, 30_000)
            patterns = ("remote", "multicast", "none", "empty", "local", "partial", "remote")
            for epoch, pattern in enumerate(patterns):
                cpu = self.cpu_inputs(pattern, epoch, unequal=pattern in ("multicast", "partial"))
                inputs = gather_cpu(cpu)
                gpu = self.to_gpu(cpu)
                expand = epoch % 2 == 0
                started = time.perf_counter()
                result = self.dispatch(buffer, gpu, expand=expand)
                elapsed = self.wait(result, "bounded_ring_success", started)
                failure = None
                try:
                    buffer.check_errors(result[3])
                    verify_result(result, inputs, self.rank, self.world,
                                  EXPERTS, TOPK, expand, 8 if expand else 1)
                except Exception as exc:
                    failure = f"rank {self.rank}, epoch {epoch}: {type(exc).__name__}: {exc}"
                failures = [item for item in gather_cpu(failure) if item]
                if failures:
                    # All ranks leave this buffer together instead of one
                    # rank entering collective close during peers' next epoch.
                    raise AssertionError("; ".join(failures))
                diagnostics.append(dict(epoch=epoch, pattern=pattern, expand=expand,
                                        elapsed_seconds=elapsed, status="passed"))
                del result
        return dict(epochs=diagnostics)

    def asynchronous_submit(self, mode):
        # Three bounded-ring tokens leave ample pending GPU work without
        # multiplying the artificial 100-ms delay by the full stress shape.
        cpu = tuple(value[:3].contiguous() for value in self.cpu_inputs("remote"))
        inputs = gather_cpu(cpu)
        gpu = self.to_gpu(cpu)
        with self.buffer(mode, 1, 1, 2) as buffer:
            warmup = self.dispatch(buffer, gpu)
            self.wait(warmup, "async_warmup", time.perf_counter())
            failure = None
            try:
                buffer.check_errors(warmup[3])
            except Exception as exc:
                failure = f"rank {self.rank}, warmup: {type(exc).__name__}: {exc}"
            failures = [item for item in gather_cpu(failure) if item]
            if failures:
                raise AssertionError("; ".join(failures))
            del warmup
            buffer.runtime.set_test_delays(0, 0, SLOW_READER_NS, 0)
            started = time.perf_counter()
            result = self.dispatch(buffer, gpu)
            submit_ms = (time.perf_counter() - started) * 1000
            pending_on_return = not result[3]._completion_event.query()
            elapsed = self.wait(result, "asynchronous_submit", started)
            if not pending_on_return:
                raise AssertionError("Dispatch did not return while delayed GPU work was pending")
            buffer.check_errors(result[3])
            verify_result(result, inputs, self.rank, self.world, EXPERTS, TOPK, True, 8)
            return dict(pending_on_return=pending_on_return, cpu_submit_ms=submit_ms,
                        elapsed_seconds=elapsed, do_cpu_sync=False)

    def case(self, label, function):
        failure, diagnostic = None, None
        try:
            diagnostic = function()
        except Exception as exc:
            failure = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        failures = [item for item in gather_cpu(failure) if item]
        per_rank = gather_cpu(diagnostic)
        self.records.append(dict(**label, status="failed" if failures else "passed",
                                 failures=failures, per_rank=per_rank))
        self.save("failed" if failures else "running")
        if self.rank == 0:
            print(json.dumps(dict(**label, status=self.records[-1]["status"])), flush=True)
        if failures:
            raise AssertionError("; ".join(failures))

    def run(self):
        if self.args.case in ("all", "remote_abort"):
            for mode in self.args.modes:
                for channels in (1, 32):
                    self.case(dict(case="remote_abort", mode=mode, channels=channels),
                              lambda m=mode, c=channels: self.remote_abort(m, c))
        if self.args.case in ("all", "prefix_timeout"):
            for channels in (1, 32):
                self.case(dict(case="prefix_timeout", channels=channels),
                          lambda c=channels: self.prefix_timeout(c))
        if self.args.case in ("all", "mailbox_timeout"):
            for channels in (1, 32):
                self.case(dict(case="mailbox_timeout", channels=channels),
                          lambda c=channels: self.mailbox_timeout(c))
        if self.args.case in ("all", "success"):
            matrix = ([(1, 1, 2, 2), (32, 2, 1, 3)] if self.args.quick else
                      [(c, q, length, stages) for c in (1, 2, 32)
                       for q, length in ((1, 2), (2, 1)) for stages in (2, 3)])
            for mode in self.args.modes:
                for channels, q, length, stages in matrix:
                    self.case(dict(case="bounded_ring_success", mode=mode, channels=channels,
                                   payload_slots=q, descriptor_slots=length, stages=stages),
                              lambda m=mode, c=channels, p=q, l=length, s=stages: self.success(m, c, p, l, s))
        if self.args.case in ("all", "async"):
            for mode in self.args.modes:
                self.case(dict(case="asynchronous_submit", mode=mode),
                          lambda m=mode: self.asynchronous_submit(m))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "remote_abort", "prefix_timeout", "mailbox_timeout", "success", "async"), default="all")
    parser.add_argument("--modes", nargs="+", choices=("scalar", "vectorized", "pipelined"), default=["pipelined"])
    parser.add_argument("--quick", action="store_true", help="Use two bounded-ring configurations instead of twelve")
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--watchdog-seconds", type=float, default=2.0)
    parser.add_argument("--prefix-timeout-ms", type=int, default=50)
    parser.add_argument("--prefix-delay-cycles", type=int, default=500_000_000)
    parser.add_argument("--require-profiling", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/current/control_progress.json"))
    args = parser.parse_args()
    if (args.tokens < 3 or args.hidden < 1 or args.watchdog_seconds <= 0 or
            args.prefix_timeout_ms < 1 or args.prefix_delay_cycles < 1):
        parser.error("requires tokens>=3, hidden>=1, positive timeout, delay and watchdog")
    dist.init_process_group("gloo", timeout=timedelta(seconds=max(30, args.watchdog_seconds * 5)))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if world < 2 or EXPERTS % world or EXPERTS // world < TOPK:
        parser.error("requires >=2 ranks with >=8 consecutive local experts each")
    verification = Verification(args)
    verification.save()
    try:
        verification.run()
        verification.save("passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
