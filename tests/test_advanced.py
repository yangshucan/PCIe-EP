"""Reference checks for independently selectable S5--S8 BF16 optimizations.

This test deliberately uses small, unequal source/descriptor windows. Every
case validates the complete payload, stable expert order, and both dedup
counters against the existing independent CPU reference.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from advanced_options import add_transport_arguments, transport_options
from test_correctness import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_transport_arguments(parser, stage=5)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/s5_s8/correctness.json"))
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(device)
    started, records = time.perf_counter(), []
    windows = [(2, 3)] if args.quick else [(1, 1), (2, 3), (3, 2), (256, 256)]
    patterns = (
        ["random", "multicast", "none"]
        if args.quick
        else ["random", "local", "multicast", "hotspot", "partial", "none"]
    )
    try:
        for q, length in windows:
            config = HostStagedConfig(payload_slots=q, descriptor_slots=length, **transport_options(args))
            with HostStagedEPBuffer(
                group=dist.group.WORLD,
                num_max_tokens_per_rank=129,
                hidden=args.hidden,
                num_topk=8,
                num_experts=256,
                config=config,
            ) as buffer:
                cases = [
                    dict(pattern=pattern, tokens=17, expand=expand, alignment=1)
                    for pattern in patterns
                    for expand in [False, True]
                ]
                cases.extend(
                    dict(pattern="random", tokens=tokens, unequal=unequal, expand=True, alignment=alignment)
                    for tokens, unequal, alignment in [(0, False, 1), (129, True, 1), (33, False, 16)]
                )
                cases.extend(
                    dict(pattern="random", tokens=3 + i % 5, expand=bool(i % 2), alignment=8, seed=100 + i)
                    for i in range(args.repeat)
                )
                cases.append(dict(pattern="multicast", tokens=5, expand=True, alignment=1, weights=False))
                for case in cases:
                    case["hidden"] = args.hidden
                    run_case(buffer, case, rank, world, device, 256, 8)
                    records.append({**case, "payload_slots": q, "descriptor_slots": length, "status": "passed"})
                if rank == 0:
                    print(f"S{args.stage} C={args.channels} Q={q} L={length}: {len(cases)} cases passed", flush=True)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "world_size": world,
                        "options": transport_options(args),
                        "records": records,
                        "elapsed_seconds": time.perf_counter() - started,
                        "nvlink_payload_allowed": False,
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
