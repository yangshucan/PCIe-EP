"""Distributed bitwise dispatch/reference checks; all test collectives use CPU Gloo.

Example: torchrun --standalone --nproc-per-node=8 tests/test_correctness.py
The CPU copies below are validation only; the dispatch API stays asynchronous.
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


def gather_cpu(value, group=None):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def make_inputs(rank, world, tokens, hidden, experts, topk, pattern, seed=7):
    generator = torch.Generator().manual_seed(seed + rank * 7919)
    local_experts = experts // world
    if pattern == "local":
        idx = (torch.arange(topk) + rank * local_experts).repeat(tokens, 1)
    elif pattern == "multicast":
        # Two or more local experts on one destination plus other destinations.
        ids = []
        for local_expert in range(local_experts):
            for dest in range(world):
                ids.append(dest * local_experts + local_expert)
        idx = torch.tensor(ids[:topk], dtype=torch.int64).repeat(tokens, 1)
    elif pattern == "hotspot":
        idx = torch.arange(topk).repeat(tokens, 1)
    else:
        idx = torch.rand(tokens, experts, generator=generator).topk(topk, dim=1).indices
    if pattern == "none":
        idx.fill_(-1)
    elif pattern == "partial" and tokens:
        idx[:, ::3] = -1
    weights = torch.rand(tokens, topk, generator=generator).to(torch.float32)
    # Derive exact BF16 source rows independently on CPU; duplicate payload
    # contents are allowed and must never merge different token identities.
    x = ((torch.arange(tokens).view(-1, 1) % 23) * 2 + (torch.arange(hidden).view(1, -1) % 47) / 8 + rank * 64).to(
        torch.bfloat16
    )
    return x.contiguous(), idx.contiguous(), weights.contiguous()


def reference(inputs, rank, world, experts, topk, alignment):
    local_experts = experts // world
    unique = []
    expanded = [[] for _ in range(local_experts)]
    host_writes = host_reads = 0
    for source, (x, idx, weights) in enumerate(inputs):
        for token in range(x.shape[0]):
            route = idx[token]
            valid = route >= 0
            owners = route[valid] // local_experts
            destinations = set(owners.tolist())
            if source == rank and any(dest != source for dest in destinations):
                host_writes += 1
            slots = [
                slot for slot in range(topk) if int(route[slot]) >= 0 and int(route[slot]) // local_experts == rank
            ]
            if not slots:
                continue
            if source != rank:
                host_reads += 1
            out_idx = torch.full((topk,), -1, dtype=torch.int64)
            out_weights = torch.zeros(topk, dtype=torch.float32)
            for slot in slots:
                local = int(route[slot]) % local_experts
                out_idx[slot] = local
                out_weights[slot] = weights[token, slot]
                expanded[local].append((source, token, slot, x[token], weights[token, slot]))
            unique.append((source, token, x[token], out_idx, out_weights))
    counts = [len(rows) for rows in expanded]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + (count + alignment - 1) // alignment * alignment)
    return unique, expanded, counts, offsets, host_writes, host_reads


def assert_tensor_equal(actual, expected, label):
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise AssertionError(f"{label}: actual shape={tuple(actual.shape)}, expected shape={tuple(expected.shape)}")


def verify_result(result, inputs, rank, world, experts, topk, expand, alignment):
    recv_x, recv_idx, recv_weights, handle, event = result
    event.current_stream_wait()
    torch.cuda.synchronize()
    unique, expanded, counts, offsets, host_writes, host_reads = reference(
        inputs, rank, world, experts, topk, alignment
    )
    x = recv_x.detach().cpu()
    idx = recv_idx.detach().cpu()
    weights = recv_weights.detach().cpu() if recv_weights is not None else None
    meta = handle.source_meta.detach().cpu()
    assert int(handle.unique_count.detach().cpu()) == len(unique), "unique count mismatch"
    assert_tensor_equal(
        handle.expert_counts.detach().cpu(), torch.tensor(counts, dtype=handle.expert_counts.dtype), "expert counts"
    )
    assert_tensor_equal(
        handle.expert_offsets.detach().cpu(),
        torch.tensor(offsets, dtype=handle.expert_offsets.dtype),
        "expert padded offsets",
    )
    if not expand:
        for row, (source, token, payload, route_idx, route_weights) in enumerate(unique):
            assert_tensor_equal(x[row], payload, f"unique payload row={row}")
            assert_tensor_equal(idx[row], route_idx.to(idx.dtype), f"unique routes row={row}")
            if weights is not None:
                assert_tensor_equal(weights[row], route_weights, f"unique weights row={row}")
            assert_tensor_equal(
                meta[row, :2], torch.tensor([source, token], dtype=meta.dtype), f"unique identity row={row}"
            )
    else:
        for expert, rows in enumerate(expanded):
            for ordinal, (source, token, slot, payload, weight) in enumerate(rows):
                row = offsets[expert] + ordinal
                assert_tensor_equal(x[row], payload, f"expanded payload expert={expert} row={row}")
                assert int(idx[row]) == expert, f"expanded expert ID row={row}"
                if weights is not None:
                    assert weights[row] == weight, f"expanded weight row={row}"
                assert_tensor_equal(
                    meta[row, :3], torch.tensor([source, token, slot], dtype=meta.dtype), f"expanded identity row={row}"
                )
            padding = slice(offsets[expert] + len(rows), offsets[expert + 1])
            assert not torch.count_nonzero(x[padding]), f"payload padding expert={expert}"
            if weights is not None:
                assert not torch.count_nonzero(weights[padding]), f"weight padding expert={expert}"
            assert (idx[padding] == -1).all(), f"index padding expert={expert}"
    stats = handle.stats.detach().cpu() if isinstance(handle.stats, torch.Tensor) else None
    if stats is not None:
        assert int(stats[0]) == host_writes, f"sender dedup bytes: writes={int(stats[0])} expected={host_writes}"
        assert int(stats[1]) == host_reads, f"receiver dedup bytes: reads={int(stats[1])} expected={host_reads}"
        local_experts = experts // world
        send_descriptors = sum(
            len(set((route[route >= 0] // local_experts).tolist()) - {rank}) for route in inputs[rank][1]
        )
        # S1/S2 use batch headers and have no descriptor queue.
        assert int(stats[2]) in (0, send_descriptors), "descriptor publication mismatch"
    error = handle.error.detach().cpu()
    assert not torch.count_nonzero(error), f"GPU error: {error.tolist()}"


def run_case(buffer, case, rank, world, device, experts, topk, group=None):
    tokens = case["tokens"]
    if case.get("unequal"):
        tokens = 0 if rank == 0 else max(1, tokens - rank * 3)
    inputs_cpu = make_inputs(
        rank, world, tokens, case["hidden"], experts, topk, case["pattern"], seed=case.get("seed", 7)
    )
    inputs = gather_cpu(inputs_cpu, group)
    x, idx, weights = (item.to(device) for item in inputs_cpu)
    result = buffer.dispatch(
        x,
        topk_idx=idx,
        topk_weights=weights if case.get("weights", True) else None,
        do_expand=case["expand"],
        expert_alignment=case["alignment"],
        do_cpu_sync=False,
        do_zero_padding=case["expand"],
    )
    failure = None
    try:
        buffer.check_errors()
        verify_result(result, inputs, rank, world, experts, topk, case["expand"], case["alignment"])
    except Exception as exc:
        failure = f"rank {rank}: {type(exc).__name__}: {exc}"
    failures = [item for item in gather_cpu(failure, group) if item is not None]
    if failures:
        raise AssertionError("; ".join(failures))


def assert_rejected(call, label):
    try:
        call()
    except (TypeError, ValueError, RuntimeError, NotImplementedError):
        return
    raise AssertionError(f"unsupported or invalid input was accepted: {label}")


def check_api_rejections(buffer, rank, world, device, experts, topk, hidden):
    inputs = make_inputs(rank, world, 2, hidden, experts, topk, "local")
    x, idx, weights = (item.to(device) for item in inputs)
    for label, overrides in [
        ("CPU sync", {"do_cpu_sync": True}),
        ("FP32 payload", {"x": x.float()}),
        ("wrong top-k shape", {"topk_idx": idx[:, :-1]}),
        ("wrong weight shape", {"topk_weights": weights[:, :-1]}),
        ("CPU input", {"x": inputs[0]}),
        ("noncontiguous payload", {"x": x.t().contiguous().t()}),
        ("wrong expert count", {"num_experts": experts + 1}),
        ("handle copy", {"do_handle_copy": True}),
        ("QP override", {"num_qps": 1}),
    ]:
        kwargs = {"x": x, "topk_idx": idx, "topk_weights": weights, "do_cpu_sync": False}
        kwargs.update(overrides)
        assert_rejected(lambda kwargs=kwargs: buffer.dispatch(**kwargs), label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--output", type=Path, default=Path("results/correctness.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.experts % world or args.experts // world < args.topk:
        parser.error("test requires uniform contiguous ownership and >= top-k experts per rank")
    if args.stage == 1 and world != 2:
        parser.error("S1 requires exactly two ranks")
    started = time.perf_counter()
    records = []
    slots = [(256, 256)] if args.stage < 3 or args.quick else [(1, 1), (2, 3), (3, 2), (256, 256)]
    try:
        for payload_slots, descriptor_slots in slots:
            config = HostStagedConfig(stage=args.stage, payload_slots=payload_slots, descriptor_slots=descriptor_slots)
            with HostStagedEPBuffer(
                group=dist.group.WORLD,
                num_max_tokens_per_rank=129,
                hidden=args.hidden,
                num_topk=args.topk,
                num_experts=args.experts,
                config=config,
            ) as buffer:
                check_api_rejections(buffer, rank, world, device, args.experts, args.topk, args.hidden)
                patterns = ["random", "local", "multicast", "hotspot", "partial", "none"]
                if args.quick:
                    patterns = ["random", "multicast", "none"]
                for pattern in patterns:
                    for expand in ([False] if args.stage == 1 else [False, True]):
                        case = {
                            "pattern": pattern,
                            "tokens": 17,
                            "hidden": args.hidden,
                            "expand": expand,
                            "alignment": 1,
                        }
                        run_case(buffer, case, rank, world, device, args.experts, args.topk)
                        records.append(
                            {
                                **case,
                                "payload_slots": payload_slots,
                                "descriptor_slots": descriptor_slots,
                                "status": "passed",
                            }
                        )
                for tokens, unequal, alignment in [(0, False, 1), (129, True, 1), (33, False, 16)]:
                    case = {
                        "pattern": "random",
                        "tokens": tokens,
                        "hidden": args.hidden,
                        "expand": args.stage != 1,
                        "alignment": alignment,
                        "unequal": unequal,
                    }
                    run_case(buffer, case, rank, world, device, args.experts, args.topk)
                    records.append(
                        {
                            **case,
                            "payload_slots": payload_slots,
                            "descriptor_slots": descriptor_slots,
                            "status": "passed",
                        }
                    )
                case = {
                    "pattern": "multicast",
                    "tokens": 5,
                    "hidden": args.hidden,
                    "expand": args.stage != 1,
                    "alignment": 1,
                    "weights": False,
                }
                run_case(buffer, case, rank, world, device, args.experts, args.topk)
                records.append(
                    {**case, "payload_slots": payload_slots, "descriptor_slots": descriptor_slots, "status": "passed"}
                )
                # Repeated epochs vary inputs and verify slot/sequence reuse.
                for repeat in range(5 if args.quick else 25):
                    case = {
                        "pattern": "random",
                        "tokens": 3 + repeat % 5,
                        "hidden": args.hidden,
                        "expand": args.stage != 1,
                        "alignment": 1,
                        "seed": repeat + 100,
                    }
                    run_case(buffer, case, rank, world, device, args.experts, args.topk)
                if rank == 0:
                    print(f"S{args.stage} Q={payload_slots} L={descriptor_slots}: passed", flush=True)
        if args.stage == 4 and world >= 4 and not args.quick:
            # Disjoint noncontiguous rank groups exercise rank-local ownership;
            # a second group with identical members exercises group isolation.
            groups = []
            for parity in [0, 1]:
                members = list(range(parity, world, 2))
                for _ in range(2):
                    selected = dist.new_group(members, backend="gloo")
                    if rank in members:
                        groups.append((selected, members.index(rank), len(members)))
            for group, group_rank, group_world in groups:
                if args.experts % group_world:
                    continue
                with HostStagedEPBuffer(
                    group=group,
                    num_max_tokens_per_rank=19,
                    hidden=args.hidden,
                    num_topk=args.topk,
                    num_experts=args.experts,
                    config=HostStagedConfig(stage=4, payload_slots=2, descriptor_slots=3),
                ) as buffer:
                    case = {
                        "pattern": "multicast",
                        "tokens": 19,
                        "hidden": args.hidden,
                        "expand": True,
                        "alignment": 8,
                        "unequal": True,
                    }
                    run_case(buffer, case, group_rank, group_world, device, args.experts, args.topk, group)
                records.append({"noncontiguous_group": True, "world_size": group_world, "status": "passed"})
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "stage": args.stage,
                        "world_size": world,
                        "records": records,
                        "elapsed_seconds": time.perf_counter() - started,
                        "control_backend": "gloo",
                        "nvlink_payload_allowed": False,
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
