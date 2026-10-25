"""S8 CPU/GPU interoperation, zero epochs, two-level ACK and NUMA placement."""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer
from test_correctness import gather_cpu, make_inputs, verify_result


def relay_copies(indices, source, rank_nodes, experts):
    experts_per_rank = experts // len(rank_nodes)
    copies = 0
    for token in indices:
        domains = {rank_nodes[int(expert) // experts_per_rank] for expert in token if int(expert) >= 0}
        copies += len(domains - {rank_nodes[source]})
    return copies


def verify_placement(diagnostics):
    source_node = diagnostics["rank_numa_nodes"][diagnostics["rank"]]
    spans = [(diagnostics["source_payload_placement"], source_node)]
    spans.extend((span, span["target_numa_node"]) for span in diagnostics["replica_payload_placement"])
    for span, expected in spans:
        if span["query_return"] != 0:
            raise AssertionError(f"Actual NUMA page query failed: {span}")
        if any(node != expected for node in span["sample_status"]):
            raise AssertionError(f"Payload pages are not on NUMA {expected}: {span}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-placement", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/s5_s8/relay.json"))
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    config = HostStagedConfig(
        stage=8,
        channels=4,
        payload_slots=2,
        descriptor_slots=1,
        replica_slots=3,
        numa_mode="relay",
        warp_specialization=True,
        pipeline_stages=3,
        fuse_receive_expand=True,
        source_fusion=True,
        relay_workers_per_numa=args.workers,
    )
    records = []
    try:
        with HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=33,
            hidden=args.hidden,
            num_topk=8,
            num_experts=256,
            config=config,
        ) as buffer:
            # An idle worker must wait for epoch 1 instead of timing out
            # against its initial started_ns=0. This is test setup only.
            time.sleep(0.1)
            idle = buffer.runtime.relay_diagnostics()
            if any(flow["epoch"] or flow["error"] for worker in idle["workers"] for flow in worker["flows"]):
                raise AssertionError(f"Relay worker failed before its first epoch: {idle}")
            # Adjacent zero/nonzero epochs stress CPU mailbox consumption.
            for iteration, pattern in enumerate(["none", "random", "local", "multicast", "none", "hotspot"] * 3):
                tokens = 0 if iteration % 7 == 0 else 33 - rank
                cpu = make_inputs(rank, world, tokens, args.hidden, 256, 8, pattern, seed=600 + iteration)
                inputs = gather_cpu(cpu)
                gpu = tuple(value.to("cuda") for value in cpu)
                buffer.runtime.set_relay_delay_ns(50000 if iteration % 3 == 0 else 0)
                expand = bool(iteration % 2)
                result = buffer.dispatch(
                    *gpu, do_expand=expand, expert_alignment=8, do_zero_padding=expand, do_cpu_sync=False
                )
                failure = None
                diagnostics = None
                try:
                    buffer.check_errors(result[3])
                    verify_result(result, inputs, rank, world, 256, 8, expand, 8)
                    diagnostics = buffer.runtime.relay_diagnostics()
                    expected = relay_copies(cpu[1], rank, diagnostics["rank_numa_nodes"], 256)
                    if diagnostics["copies"] != expected or diagnostics["payload_bytes"] != expected * args.hidden * 2:
                        raise AssertionError(f"NUMA multicast copies differ: expected={expected}, actual={diagnostics}")
                    for worker in diagnostics["workers"]:
                        if worker["runtime_error"]:
                            raise AssertionError(worker["runtime_error"])
                        for flow in worker["flows"]:
                            if (
                                flow["epoch"] != diagnostics["epoch"]
                                or flow["drained_epoch"] != diagnostics["epoch"]
                                or flow["copied_tokens"] != flow["expected_tokens"]
                                or flow["error"]
                            ):
                                raise AssertionError(f"Relay completion preceded complete consumption: {flow}")
                    if args.require_placement:
                        verify_placement(diagnostics)
                except Exception as exc:
                    failure = f"rank {rank}: {type(exc).__name__}: {exc}"
                failures = [item for item in gather_cpu(failure) if item]
                if failures:
                    raise AssertionError("; ".join(failures))
                records.append(
                    {
                        "iteration": iteration,
                        "pattern": pattern,
                        "per_rank": gather_cpu(diagnostics),
                        "status": "passed",
                    }
                )
            buffer.runtime.set_relay_delay_ns(0)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "world_size": world,
                        "records": records,
                        "require_placement": args.require_placement,
                        "status": "passed",
                    },
                    indent=2,
                )
            )
            print(
                f"S8 relay: {len(records)} epochs, copies/ACK/drained passed; placement_required={args.require_placement}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
