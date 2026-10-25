"""F2 receive/expand verification against the independent CPU routing reference.

All collectives are CPU Gloo. Synchronization and diagnostic D2H copies below
belong to validation, never to the production dispatch submission path.
Example: torchrun --standalone --nproc-per-node=8 tests/test_f2.py --suite overlap
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from pcie_ep.quantization import fp8_roundtrip_reference
from test_correctness import gather_cpu, make_inputs, verify_result

EXPERTS, TOPK = 256, 8


def config_for(mode, stages, payload_slots, descriptor_slots, **overrides):
    options = dict(
        stage=6,
        channels=2,
        pipeline_stages=stages,
        payload_slots=payload_slots,
        descriptor_slots=descriptor_slots,
        replica_slots=payload_slots,
        warp_specialization=True,
        fuse_receive_expand=True,
        receive_expand_mode=mode,
        source_fusion=True,
        timeout_ms=30000,
    )
    options.update(overrides)
    return HostStagedConfig(**options)


class Verification:
    def __init__(self, args):
        self.args = args
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.records = []
        self.started = time.monotonic()

    def save(self, status="running"):
        if self.rank == 0:
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            self.args.output.write_text(
                json.dumps(
                    dict(
                        status=status,
                        world_size=self.world,
                        suite=self.args.suite,
                        modes=self.args.modes,
                        elapsed_seconds=time.monotonic() - self.started,
                        reference="CPU source/token/topk enumeration; bitwise BF16 payload, identity, weights, counts and padding",
                        overlap_definition="[channel,0] conservative producer-before-scatter-warp-2 completion count; column 1 is a completed-ticket+1 watermark",
                        records=self.records,
                    ),
                    indent=2,
                )
                + "\n"
            )

    def buffer(self, config, hidden, capacity=65):
        return HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=capacity,
            hidden=hidden,
            num_topk=TOPK,
            num_experts=EXPERTS,
            config=config,
        )

    def inputs(self, tokens, hidden, pattern, seed=20261024, unequal=False):
        if unequal:
            tokens = 0 if self.rank == 0 else max(1, tokens - self.rank)
        return make_inputs(self.rank, self.world, tokens, hidden, EXPERTS, TOPK, pattern, seed)

    def verify(self, buffer, result, cpu, label, expand=True, alignment=8, require_overlap=False):
        # The reference is constructed independently from source CPU tensors.
        inputs = gather_cpu(cpu)
        if buffer.config.wire_dtype == "fp8_e4m3":
            inputs = [
                (payload if source == self.rank else fp8_roundtrip_reference(payload), idx, weights)
                for source, (payload, idx, weights) in enumerate(inputs)
            ]
        failure, diagnostics = None, None
        try:
            buffer.check_errors(result[3])
            verify_result(result, inputs, self.rank, self.world, EXPERTS, TOPK, expand, alignment)
            diagnostic = getattr(result[3], "receive_expand_overlap", None)
            if not expand:
                if diagnostic is not None:
                    raise AssertionError("unique fallback must not expose the new transport diagnostic")
            else:
                if not isinstance(diagnostic, torch.Tensor):
                    raise AssertionError("optimized expanded path did not return receive_expand_overlap")
                diagnostics = diagnostic.detach().cpu()
                if tuple(diagnostics.shape) != (buffer.config.channels, 2) or diagnostics.dtype != torch.int64:
                    raise AssertionError(f"unexpected diagnostic shape/dtype: {diagnostics.shape}/{diagnostics.dtype}")
                if buffer.config.receive_expand_mode == "vectorized" and torch.count_nonzero(diagnostics):
                    raise AssertionError("vectorized diagnostic must be zero")
                if require_overlap and int(diagnostics[:, 0].sum()) <= 0:
                    raise AssertionError("pipelined producer/scatter overlap was not observed on this rank")
                diagnostics = diagnostics.tolist()
        except Exception as exc:
            failure = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        failures = [value for value in gather_cpu(failure) if value]
        diagnostic_per_rank = gather_cpu(diagnostics)
        if failures:
            self.records.append(dict(**label, status="failed", failures=failures))
            self.save("failed")
            raise AssertionError("; ".join(failures))
        self.records.append(
            dict(
                **label,
                status="passed",
                expand=expand,
                alignment=alignment,
                receive_expand_overlap_per_rank=diagnostic_per_rank,
                overlap_required_on_every_rank=require_overlap,
            )
        )
        self.save()
        if self.rank == 0:
            print(json.dumps(dict(**label, status="passed")), flush=True)

    def eager(self, buffer, cpu, label, expand=True, alignment=8, require_overlap=False):
        x, idx, weights = (value.to("cuda") for value in cpu)
        result = buffer.dispatch(
            x,
            idx,
            weights,
            do_expand=expand,
            expert_alignment=alignment,
            do_zero_padding=expand,
            do_cpu_sync=False,
        )
        self.verify(buffer, result, cpu, label, expand, alignment, require_overlap)

    def correctness(self):
        windows = [(1, 2), (2, 1)]
        patterns = (
            ["random", "multicast"]
            if self.args.quick
            else ["random", "multicast", "hotspot", "partial", "local", "none"]
        )
        for mode in self.args.modes:
            for stages in (2, 3):
                for q, l in windows:
                    config = config_for(mode, stages, q, l)
                    with self.buffer(config, 7168) as buffer:
                        for pattern in patterns:
                            label = dict(
                                suite="correctness", mode=mode, stages=stages, q=q, l=l, pattern=pattern, hidden=7168
                            )
                            self.eager(buffer, self.inputs(17, 7168, pattern), label)
                        for name, cpu in [
                            ("unequal_with_zero_rank", self.inputs(17, 7168, "partial", unequal=True)),
                            ("all_empty", self.inputs(0, 7168, "none")),
                        ]:
                            self.eager(
                                buffer, cpu, dict(suite="correctness", mode=mode, stages=stages, q=q, l=l, pattern=name)
                            )
                        self.eager(
                            buffer,
                            self.inputs(17, 7168, "multicast"),
                            dict(suite="unique_fallback", mode=mode, stages=stages, q=q, l=l),
                            expand=False,
                        )
                        if (q, l) == (1, 2):
                            self.eager(
                                buffer,
                                self.inputs(17, 7168, "multicast"),
                                dict(suite="alignment_one", mode=mode, stages=stages, q=q, l=l, hidden=7168),
                                alignment=1,
                            )
                # Hidden size not divisible by a uint4 BF16 vector exercises the
                # scalar tail/fallback while preserving every payload element.
                with self.buffer(config_for(mode, stages, 2, 1), 129) as buffer:
                    self.eager(
                        buffer,
                        self.inputs(19, 129, "multicast"),
                        dict(suite="odd_hidden", mode=mode, stages=stages, q=2, l=1, hidden=129),
                    )

    def delays(self):
        for mode in self.args.modes:
            for stages in (2, 3):
                for delay in ("metadata_delay_ns", "tx_delay_ns", "rx_delay_ns", "scatter_delay_ns"):
                    q, l = (1, 2) if delay in ("metadata_delay_ns", "rx_delay_ns") else (2, 1)
                    config = config_for(mode, stages, q, l, **{delay: 100000})
                    with self.buffer(config, 7168) as buffer:
                        for iteration in range(2 if self.args.quick else 4):
                            self.eager(
                                buffer,
                                self.inputs(23, 7168, "multicast", seed=17 + iteration),
                                dict(
                                    suite="delays",
                                    mode=mode,
                                    stages=stages,
                                    q=q,
                                    l=l,
                                    delay=delay,
                                    delay_ns=100000,
                                    iteration=iteration,
                                ),
                            )

    def overlap(self):
        if "pipelined" not in self.args.modes:
            return
        # Ample credits remove queue starvation from the deliberate scatter
        # delay. All ranks receive many remote tokens, even with one channel.
        for stages in (2, 3):
            config = config_for(
                "pipelined",
                stages,
                128,
                128,
                channels=1,
                replica_slots=128,
                scatter_delay_ns=100000,
            )
            with self.buffer(config, 7168) as buffer:
                for iteration in range(2):
                    self.eager(
                        buffer,
                        self.inputs(65, 7168, "multicast", seed=101 + iteration),
                        dict(
                            suite="overlap",
                            mode="pipelined",
                            channels=1,
                            stages=stages,
                            q=128,
                            l=128,
                            scatter_delay_ns=100000,
                            iteration=iteration,
                        ),
                        require_overlap=True,
                    )

    def combinations(self):
        settings = [(6, "bf16", "direct"), (7, "fp8_e4m3", "direct"), (8, "bf16", "relay"), (8, "fp8_e4m3", "relay")]
        for mode in self.args.modes:
            for stages in (2, 3):
                for stage, wire_dtype, numa_mode in settings:
                    config = config_for(
                        mode,
                        stages,
                        2,
                        1,
                        stage=stage,
                        enable_graph=True,
                        wire_dtype=wire_dtype,
                        numa_mode=numa_mode,
                    )
                    base = self.inputs(9, 256, "multicast")
                    # Each buffer owns exactly one fixed-address graph.
                    for execution in ("graph", "graph_cached"):
                        with self.buffer(config, 256, capacity=9) as buffer:
                            self.eager(
                                buffer,
                                base,
                                dict(
                                    suite="combination_eager",
                                    mode=mode,
                                    stages=stages,
                                    stage=stage,
                                    wire_dtype=wire_dtype,
                                    numa_mode=numa_mode,
                                    following_execution=execution,
                                ),
                            )
                            x, idx, weights = (value.to("cuda") for value in base)
                            cache = None
                            if execution == "graph_cached":
                                prepared = buffer.dispatch(x, idx, weights, do_expand=True, expert_alignment=8)
                                buffer.check_errors(prepared[3])
                                cache = buffer.cache_layout(prepared[3])
                                del prepared
                            graph = buffer.capture_dispatch(
                                x,
                                topk_idx=idx if cache is None else None,
                                topk_weights=weights,
                                handle=cache,
                                do_expand=True,
                                expert_alignment=8,
                                do_zero_padding=True,
                            )
                            try:
                                for replay in range(3):
                                    # Fixed addresses, new payload/weights each
                                    # replay. Dynamic Graph also re-reads routes.
                                    routes = (base[1] + 32 * replay) % EXPERTS if cache is None else base[1]
                                    # Four exceeds the BF16 ULP at every rank's
                                    # generated magnitude, so payload changes
                                    # cannot disappear through BF16 rounding.
                                    cpu = (
                                        (base[0].float() + replay * 4.0).to(torch.bfloat16),
                                        routes,
                                        base[2] + replay * 0.125,
                                    )
                                    x.copy_(cpu[0].to("cuda"))
                                    weights.copy_(cpu[2].to("cuda"))
                                    if cache is None:
                                        idx.copy_(cpu[1].to("cuda"))
                                    result = graph.replay(async_with_compute_stream=True)
                                    self.verify(
                                        buffer,
                                        result,
                                        cpu,
                                        dict(
                                            suite="combinations",
                                            mode=mode,
                                            stages=stages,
                                            stage=stage,
                                            wire_dtype=wire_dtype,
                                            numa_mode=numa_mode,
                                            execution=execution,
                                            replay=replay,
                                        ),
                                    )
                                    del result
                            finally:
                                graph.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "correctness", "delays", "overlap", "combinations"], default="all")
    parser.add_argument("--modes", nargs="+", choices=["vectorized", "pipelined"], default=["vectorized", "pipelined"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/f2/correctness.json"))
    args = parser.parse_args()
    args.modes = list(dict.fromkeys(args.modes))
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if world < 2 or EXPERTS % world or TOPK > EXPERTS // world:
        parser.error("requires at least two ranks and >= 8 contiguous experts per rank")
    verification = Verification(args)
    verification.save()
    try:
        for suite in ("correctness", "delays", "overlap", "combinations"):
            if args.suite in ("all", suite):
                getattr(verification, suite)()
        verification.save("passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
