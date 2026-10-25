"""Eight-rank regression for shared kernel attributes across live buffers.

torchrun --standalone --nproc-per-node=8 tests/test_f2_buffers.py \
    --output results/f2/regression/mixed_buffers.json
Use --allow-failure only when collecting the original-binary diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
sys.path.insert(0, str(PROJECT_ROOT))

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from test_correctness import gather_cpu, make_inputs, verify_result

WORLD, TOKENS, EXPERTS, TOPK = 8, 17, 256, 8
USER_HIDDEN, OPTIN_HIDDEN, SMALL_HIDDEN = 7168, 8192, 128


def configuration(mode, stages):
    return HostStagedConfig(
        stage=6,
        channels=2,
        payload_slots=2,
        descriptor_slots=2,
        pipeline_stages=stages,
        warp_specialization=True,
        fuse_receive_expand=True,
        source_fusion=True,
        receive_expand_mode=mode,
        timeout_ms=30_000,
    )


def new_buffer(mode, hidden, stages):
    return HostStagedEPBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=TOKENS,
        hidden=hidden,
        num_topk=TOPK,
        num_experts=EXPERTS,
        config=configuration(mode, stages),
    )


def attempt(buffer, rank, phase, step):
    cpu = make_inputs(rank, WORLD, TOKENS, buffer.hidden, EXPERTS, TOPK, "multicast")
    cpu = ((cpu[0].float() + step * 4).to(torch.bfloat16), cpu[1], cpu[2] + step * 0.125)
    expected = gather_cpu(cpu)
    x, indices, weights = (value.to("cuda") for value in cpu)
    failure = None
    try:
        result = buffer.dispatch(
            x,
            indices,
            weights,
            do_expand=True,
            expert_alignment=8,
            do_zero_padding=True,
            do_cpu_sync=False,
        )
        buffer.check_errors(result[3])
        verify_result(result, expected, rank, WORLD, EXPERTS, TOPK, True, 8)
    except Exception as error:
        failure = f"rank {rank}: {type(error).__name__}: {error}"
    finally:
        # If a launch fails, earlier count/prefix kernels can still be queued
        # without a new wrapper completion event. Drain before unmapping.
        try:
            torch.cuda.synchronize()
        except Exception as error:
            failure = f"{failure or ''}; drain rank {rank}: {type(error).__name__}: {error}"
    failures = [value for value in gather_cpu(failure) if value is not None]
    return {"phase": phase, "status": "failed" if failures else "passed", "failures": failures}


def run_case(large_mode, small_mode, rank):
    phases = []
    # Same-kernel cases exceed the default 48 KiB cap and exercise opt-in;
    # mixed-mode cases retain the user's H=7168 shape.
    large_hidden = USER_HIDDEN if large_mode == "scalar" else OPTIN_HIDDEN
    with new_buffer(large_mode, large_hidden, 3) as large:
        phases.append(attempt(large, rank, "large_before_small", 0))
        if phases[-1]["status"] == "passed":
            # Both buffers are live here. A small-buffer constructor must not
            # lower any shared function cap required by the large buffer.
            with new_buffer(small_mode, SMALL_HIDDEN, 2) as small:
                phases.append(attempt(small, rank, "small_while_large_alive", 1))
                phases.append(attempt(large, rank, "large_after_small", 2))
    return {
        "large_mode": large_mode,
        "small_mode": small_mode,
        "large_hidden": large_hidden,
        "small_hidden": SMALL_HIDDEN,
        "large_pipeline_stages": 3,
        "small_pipeline_stages": 2,
        "status": "passed" if all(item["status"] == "passed" for item in phases) else "failed",
        "phases": phases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/f2/regression/mixed_buffers.json")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    records = []
    try:
        if dist.get_world_size() != WORLD:
            raise ValueError("This regression requires exactly eight GPU ranks")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        for large_mode, small_mode in (
            ("scalar", "vectorized"),
            ("scalar", "pipelined"),
            ("vectorized", "vectorized"),
            ("pipelined", "pipelined"),
        ):
            records.append(run_case(large_mode, small_mode, rank))
        status = "passed" if all(item["status"] == "passed" for item in records) else "failed"
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "status": status,
                        "world": WORLD,
                        "tokens_per_rank": TOKENS,
                        "experts": EXPERTS,
                        "topk": TOPK,
                        "channels": 2,
                        "payload_slots": 2,
                        "descriptor_slots": 2,
                        "expanded": True,
                        "expert_alignment": 8,
                        "reference": "independent CPU source/token/topk enumeration via verify_result",
                        "records": records,
                    },
                    indent=2,
                )
                + "\n"
            )
        if status != "passed" and not args.allow_failure:
            raise AssertionError("An existing live buffer failed after another buffer was constructed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
