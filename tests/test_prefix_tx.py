"""GPU prefix/TX overlap, stable dispatch order, and cache/Graph regression.

torchrun --standalone --nproc-per-node=8 tests/test_prefix_tx.py --quick
CPU Gloo collectives and completion waits below are test diagnostics only.
Every eager dispatch uses do_cpu_sync=False; Graph replay only enqueues work.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from test_correctness import gather_cpu, make_inputs, verify_result


EXPERTS, TOPK = 256, 8
PREFIX_DELAY_NS = 20_000_000


class Verification:
    def __init__(self, args):
        self.args = args
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.records = []
        self.previous_prefix_end = {}

    def save(self, status="running"):
        if self.rank == 0:
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            self.args.output.write_text(json.dumps(dict(
                status=status, world_size=self.world, hidden=self.args.hidden,
                experts=EXPERTS, topk=TOPK, prefix_delay_ns=PREFIX_DELAY_NS,
                records=self.records,
            ), indent=2) + "\n")

    def inputs(self, pattern, *, unequal=False):
        tokens = 17 if self.args.quick else 65
        if pattern == "empty":
            tokens, pattern = 0, "none"
        if unequal:
            tokens = 0 if self.rank == 0 else max(1, tokens - self.rank * 2)
        cpu = make_inputs(self.rank, self.world, tokens, self.args.hidden, EXPERTS, TOPK, pattern)
        if pattern == "remote":
            # Several local experts on one remote GPU exercise both dedup stages.
            cpu[1].copy_((torch.arange(TOPK) + ((self.rank + 1) % self.world) * (EXPERTS // self.world))
                         .repeat(tokens, 1))
        return cpu

    def verify(self, buffer, result, cpu, label, expand, alignment, delay_ns):
        inputs = gather_cpu(cpu)
        failure, diagnostic = None, None
        try:
            # This reference checks every row in expert/source/token order,
            # including payload, branch, weights, counts and alignment padding.
            verify_result(result, inputs, self.rank, self.world, EXPERTS, TOPK, expand, alignment)
            times = result[3].prefix_tx_timestamps_ns
            assert isinstance(times, torch.Tensor) and times.is_cuda, "missing CUDA prefix/TX timestamps"
            assert times.dtype == torch.int64 and tuple(times.shape) == (buffer.config.channels + 1, 2)
            times = times.detach().cpu()
            prefix_start, prefix_end = times[0].tolist()
            assert prefix_start > self.previous_prefix_end.get(buffer.buffer_id, 0), "stale prefix state on replay"
            assert prefix_end >= prefix_start + delay_ns, "prefix owner delay or READY timestamp missing"
            self.previous_prefix_end[buffer.buffer_id] = prefix_end
            active = [False] * buffer.config.channels
            for token, route in enumerate(cpu[1]):
                destinations = route[route >= 0] // (EXPERTS // self.world)
                active[token % buffer.config.channels] |= bool((destinations != self.rank).any())
            overlap = []
            for channel, remote in enumerate(active):
                start, done = times[channel + 1].tolist()
                if remote:
                    assert 0 < start <= done, f"missing first payload completion: channel {channel}"
                    if prefix_start < done < prefix_end:
                        overlap.append(channel)
                else:
                    assert start == done == 0, f"TX timestamp on zero-work channel {channel}"
            if delay_ns and any(active):
                assert overlap, "no complete payload (all TX warps and metadata) inside the prefix interval"
            diagnostic = dict(timestamps_ns=times.tolist(), early_payload_channels=overlap)
        except Exception as exc:
            failure = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        failures = [value for value in gather_cpu(failure) if value]
        per_rank = gather_cpu(diagnostic)
        self.records.append(dict(**label, expand=expand, alignment=alignment,
                                 status="failed" if failures else "passed",
                                 failures=failures, diagnostics_per_rank=per_rank))
        self.save("failed" if failures else "running")
        if failures:
            raise AssertionError("; ".join(failures))
        if self.rank == 0:
            print(json.dumps(dict(**label, expand=expand, status="passed")), flush=True)

    def eager(self, buffer, cpu, label, *, expand, delay_ns=0):
        alignment = 8 if expand else 1
        buffer.runtime.set_test_prefix_delay_ns(delay_ns)
        x, idx, weights = (value.to("cuda") for value in cpu)
        result = buffer.dispatch(x, idx, weights, do_expand=expand, expert_alignment=alignment,
                                 do_zero_padding=expand, do_cpu_sync=False)
        self.verify(buffer, result, cpu, label, expand, alignment, delay_ns)

    def cached_and_graph(self, buffer, label, *, expand, graph_mode=None):
        # Unequal shapes include a rank with zero tokens on every cache replay.
        base = self.inputs("multicast", unequal=True)
        x, idx, weights = (value.to("cuda") for value in base)
        alignment = 8 if expand else 1
        buffer.runtime.set_test_prefix_delay_ns(PREFIX_DELAY_NS)
        prepared = buffer.dispatch(x, idx, weights, do_expand=expand, expert_alignment=alignment,
                                   do_zero_padding=expand, do_cpu_sync=False)
        self.verify(buffer, prepared, base, dict(**label, execution="cache_prepare"),
                    expand, alignment, PREFIX_DELAY_NS)
        cache = prepared[3].cache_layout()
        del prepared
        repeats = 2 if self.args.quick else 3
        for replay in range(repeats):
            cpu = ((base[0].float() + (replay + 1) * 4).to(torch.bfloat16), base[1],
                   base[2] + (replay + 1) * 0.125)
            x.copy_(cpu[0].to("cuda"))
            weights.copy_(cpu[2].to("cuda"))
            result = buffer.dispatch(x, handle=cache, topk_weights=weights, do_expand=expand,
                                     expert_alignment=alignment, do_zero_padding=expand, do_cpu_sync=False)
            self.verify(buffer, result, cpu, dict(**label, execution="cached", replay=replay),
                        expand, alignment, PREFIX_DELAY_NS)
            del result
        if graph_mode is None:
            return
        graph = buffer.capture_dispatch(
            x, topk_idx=idx if graph_mode == "graph" else None, topk_weights=weights,
            handle=cache if graph_mode == "graph_cached" else None, do_expand=expand,
            expert_alignment=alignment, do_zero_padding=expand,
        )
        try:
            for replay in range(repeats):
                cpu = ((base[0].float() + (replay + 5) * 4).to(torch.bfloat16), base[1],
                       base[2] + (replay + 5) * 0.125)
                x.copy_(cpu[0].to("cuda"))
                weights.copy_(cpu[2].to("cuda"))
                result = graph.replay(async_with_compute_stream=True)
                self.verify(buffer, result, cpu, dict(**label, execution=graph_mode, replay=replay),
                            expand, alignment, PREFIX_DELAY_NS)
                del result
        finally:
            graph.close()

    def run(self):
        configurations = [(channels, q, length) for channels in (1, 2, 32) for q, length in ((1, 2), (2, 1))]
        for index, (channels, q, length) in enumerate(configurations):
            graph_mode = "graph_cached" if index == 0 else "graph" if index == 5 else None
            config = HostStagedConfig(channels=channels, payload_slots=q, descriptor_slots=length,
                                      enable_graph=graph_mode is not None)
            with HostStagedEPBuffer(group=dist.group.WORLD, num_max_tokens_per_rank=65,
                                    hidden=self.args.hidden, num_topk=TOPK, num_experts=EXPERTS,
                                    config=config) as buffer:
                label = dict(channels=channels, q=q, l=length)
                cases = [("multicast", True, False), ("remote", False, False), ("local", True, False),
                         ("multicast", False, True), ("empty", True, False), ("empty", False, False)]
                if not self.args.quick:
                    cases += [("random", expand, False) for expand in (False, True)]
                    cases += [("partial", expand, False) for expand in (False, True)]
                    cases += [("local", False, False), ("remote", True, False)]
                for pattern, expand, unequal in cases:
                    delay = PREFIX_DELAY_NS if pattern in ("multicast", "remote") else 0
                    self.eager(buffer, self.inputs(pattern, unequal=unequal),
                               dict(**label, execution="eager", pattern=pattern, unequal=unequal),
                               expand=expand, delay_ns=delay)
                self.cached_and_graph(buffer, label, expand=index % 2 == 0, graph_mode=graph_mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/current/prefix_tx.json"))
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if world < 2 or EXPERTS % world or TOPK > EXPERTS // world:
        parser.error("requires >= 2 ranks and >= 8 contiguous experts per rank")
    verification = Verification(args)
    verification.save()
    try:
        verification.run()
        verification.save("passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
