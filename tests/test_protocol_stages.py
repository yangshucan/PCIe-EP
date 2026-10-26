"""Delay the final metadata/payload/host/shared reader of each local stage.

The CPU reference checks every output element after many ring wraps. A
separate run with ample credit and slow TX proves that the next metadata
task is prepared while the preceding payload remains in flight.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from test_correctness import gather_cpu, make_inputs, verify_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(6, 7, 8), default=6)
    parser.add_argument("--numa-mode", choices=["direct", "relay"], default="direct")
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--delay-ns", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path("results/s5_s8/protocol_stages.json"))
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    records = []
    try:
        for pipeline in [2, 3]:
            config = HostStagedConfig(
                stage=args.stage,
                channels=4,
                payload_slots=2,
                descriptor_slots=1,
                replica_slots=3,
                numa_mode=args.numa_mode,
                warp_specialization=True,
                fuse_receive_expand=True,
                pipeline_stages=pipeline,
            )
            with HostStagedEPBuffer(
                group=dist.group.WORLD,
                num_max_tokens_per_rank=65,
                hidden=args.hidden,
                num_topk=8,
                num_experts=256,
                config=config,
            ) as buffer:
                for delayed in range(4):
                    delays = [0] * 4
                    delays[delayed] = args.delay_ns
                    buffer.runtime.set_test_delays(*delays)
                    for repeat in range(3):
                        cpu = make_inputs(
                            rank, world, 65 - rank, args.hidden, 256, 8, "random", seed=200 + delayed * 17 + repeat
                        )
                        inputs = gather_cpu(cpu)
                        gpu = tuple(value.to("cuda") for value in cpu)
                        result = buffer.dispatch(
                            *gpu, do_expand=True, expert_alignment=8, do_zero_padding=True, do_cpu_sync=False
                        )
                        failure = None
                        try:
                            buffer.check_errors(result[3])
                            verify_result(result, inputs, rank, world, 256, 8, True, 8)
                        except Exception as exc:
                            failure = f"rank {rank}: {type(exc).__name__}: {exc}"
                        failures = [value for value in gather_cpu(failure) if value]
                        if failures:
                            raise AssertionError("; ".join(failures))
                    records.append(
                        {
                            "pipeline_stages": pipeline,
                            "delay": ["metadata_writer", "tx_writer", "host_reader", "scatter_reader"][delayed],
                            "delay_ns": args.delay_ns,
                            "epochs": 3,
                            "status": "passed",
                        }
                    )
        # Ample ring capacity isolates actual metadata/payload overlap from
        # descriptor or source-slot starvation in the safety runs above.
        config = HostStagedConfig(
            stage=6, channels=1, payload_slots=128, descriptor_slots=128, warp_specialization=True, pipeline_stages=3
        )
        with HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=64,
            hidden=args.hidden,
            num_topk=8,
            num_experts=256,
            config=config,
        ) as buffer:
            buffer.runtime.set_test_delays(0, args.delay_ns, 0, 0)
            gpu = tuple(value.to("cuda") for value in make_inputs(rank, world, 64, args.hidden, 256, 8, "multicast"))
            result = buffer.dispatch(*gpu, do_cpu_sync=False)
            buffer.check_errors(result[3])
            overlap = gather_cpu(int(result[3].stats[3].cpu()))
            if not all(value > 0 for value in overlap):
                raise AssertionError(f"No cross-token metadata/TX overlap on some ranks: {overlap}")
            records.append({"metadata_prepared_while_previous_payload_inflight_per_rank": overlap, "status": "passed"})
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {"world_size": world, "stage": args.stage, "numa_mode": args.numa_mode, "records": records},
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
