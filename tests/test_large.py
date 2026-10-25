"""Full-payload, GPU-side S4 validation at the requested largest workload.

torchrun --standalone --nproc-per-node=8 tests/test_large.py

All valid BF16 payload elements are checked, in bounded GPU chunks. Gloo
exchanges only CPU routes/weights and diagnostics, never token payloads.
This is correctness validation; its CPU synchronization is outside benchmarks.
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
from advanced_options import add_transport_arguments, transport_options


def gather_cpu(value):
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def payload_pattern(source: torch.Tensor, token: torch.Tensor, columns: torch.Tensor):
    """Finite BF16 raw bits derived only from (source rank, token, column)."""
    key = source.to(torch.int64).view(-1, 1) * 1000003
    key = key + token.to(torch.int64).view(-1, 1) * 9176 + columns.view(1, -1) * 131
    key = (key ^ (key >> 16)) * 0x45D9F3B
    key = key ^ (key >> 16)
    # Exponents 16..235 exclude zero/subnormal, infinity, and NaN. Both signs
    # and every mantissa bit vary. No floating-point arithmetic generates x.
    bits = (key & 127) | ((((key >> 7) % 220) + 16) << 7) | (((key >> 20) & 1) << 15)
    return bits.to(torch.int16).contiguous().view(torch.bfloat16)


def make_routes(tokens, experts, topk, rank, seed):
    generator = torch.Generator().manual_seed(seed + rank * 104729)
    idx = torch.rand(tokens, experts, generator=generator).topk(topk, dim=1).indices.contiguous()
    weights = torch.rand(tokens, topk, generator=generator)
    weights /= weights.sum(dim=1, keepdim=True)
    return idx, weights.contiguous()


def cpu_reference(all_routes, rank, world, experts, alignment):
    idx = torch.stack([item[0] for item in all_routes])
    weights = torch.stack([item[1] for item in all_routes])
    local_experts = experts // world
    mask = (idx >= rank * local_experts) & (idx < (rank + 1) * local_experts)
    unique_meta = mask.any(dim=2).nonzero(as_tuple=False)
    topk_idx = torch.where(mask, idx - rank * local_experts, -1)
    topk_weights = torch.where(mask, weights, 0)
    expanded_meta = mask.nonzero(as_tuple=False)
    expanded_experts = idx[mask] - rank * local_experts
    order = torch.argsort(expanded_experts, stable=True)
    expanded_meta = expanded_meta[order]
    expanded_experts = expanded_experts[order]
    counts = torch.bincount(expanded_experts, minlength=local_experts)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), ((counts + alignment - 1) // alignment * alignment).cumsum(0)]
    )
    raw_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    destinations = torch.zeros(idx.shape[1], world, dtype=torch.bool)
    destinations.scatter_(1, idx[rank] // local_experts, True)
    destinations[:, rank] = False
    return {
        "unique_meta": unique_meta,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "expanded_meta": expanded_meta,
        "counts": counts,
        "offsets": offsets,
        "raw_offsets": raw_offsets,
        "weights": weights,
        "writes": int(destinations.any(dim=1).sum()),
        "reads": int((unique_meta[:, 0] != rank).sum()),
        "descriptors": int(destinations.sum()),
    }


def check_equal(actual, expected, label):
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise AssertionError(f"{label}: expected shape={tuple(expected.shape)}, actual shape={tuple(actual.shape)}")


def check_result(result, ref, args, device, rank, world, expand, columns):
    recv_x, recv_idx, recv_weights, handle, event = result
    event.current_stream_wait()
    torch.cuda.synchronize()
    unique_count = int(handle.unique_count.detach().cpu())
    check_equal(handle.expert_counts.detach().cpu().to(torch.int64), ref["counts"], "expert counts")
    check_equal(handle.expert_offsets.detach().cpu().to(torch.int64), ref["offsets"], "padded expert offsets")
    if unique_count != ref["unique_meta"].shape[0]:
        raise AssertionError(f"unique count {unique_count} != {ref['unique_meta'].shape[0]}")
    stats = handle.stats.detach().cpu().tolist()
    if stats[:3] != [ref["writes"], ref["reads"], ref["descriptors"]]:
        raise AssertionError(f"dedup counters {stats[:3]} != {[ref['writes'], ref['reads'], ref['descriptors']]}")
    chunks = valid_rows = padding_rows = 0

    def check_chunk(begin, end, expected_meta, expert=None):
        nonlocal chunks, valid_rows
        meta = handle.source_meta[begin:end]
        expected_meta_gpu = expected_meta.to(device)
        width = 3 if expand else 2
        check_equal(meta[:, :width], expected_meta_gpu[:, :width], f"metadata rows {begin}:{end}")
        expected_payload = payload_pattern(meta[:, 0], meta[:, 1], columns)
        check_equal(recv_x[begin:end], expected_payload, f"full BF16 payload rows {begin}:{end}")
        source, token = expected_meta[:, 0], expected_meta[:, 1]
        if expand:
            expected_idx = torch.full((end - begin,), expert, dtype=recv_idx.dtype, device=device)
            expected_weights = ref["weights"][source, token, expected_meta[:, 2]].to(device)
        else:
            expected_idx = ref["topk_idx"][source, token].to(device)
            expected_weights = ref["topk_weights"][source, token].to(device)
        check_equal(recv_idx[begin:end], expected_idx, f"expert routes rows {begin}:{end}")
        check_equal(recv_weights[begin:end], expected_weights, f"weights rows {begin}:{end}")
        chunks += 1
        valid_rows += end - begin

    if not expand:
        for begin in range(0, unique_count, args.chunk_rows):
            end = min(begin + args.chunk_rows, unique_count)
            check_chunk(begin, end, ref["unique_meta"][begin:end])
    else:
        for expert in range(args.experts // world):
            begin = int(ref["offsets"][expert])
            count = int(ref["counts"][expert])
            raw = int(ref["raw_offsets"][expert])
            for ordinal in range(0, count, args.chunk_rows):
                size = min(args.chunk_rows, count - ordinal)
                check_chunk(
                    begin + ordinal,
                    begin + ordinal + size,
                    ref["expanded_meta"][raw + ordinal : raw + ordinal + size],
                    expert,
                )
            end = int(ref["offsets"][expert + 1])
            if end > begin + count:
                padding = slice(begin + count, end)
                if (
                    torch.count_nonzero(recv_x[padding])
                    or torch.count_nonzero(recv_weights[padding])
                    or not torch.all(recv_idx[padding] == -1)
                    or not torch.all(handle.source_meta[padding] == -1)
                ):
                    raise AssertionError(f"padding expert {expert}")
                padding_rows += end - begin - count
    return {
        "rank": rank,
        "mode": "expanded" if expand else "unique",
        "unique_rows": unique_count,
        "valid_rows_checked": valid_rows,
        "padding_rows_checked": padding_rows,
        "payload_elements_checked": valid_rows * args.hidden,
        "payload_bytes_checked": valid_rows * args.hidden * 2,
        "padding_bytes_checked": padding_rows * args.hidden * 2,
        "chunk_rows": args.chunk_rows,
        "chunks_checked": chunks,
        "runtime_stats": stats,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_transport_arguments(parser)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--payload-slots", type=int, default=256)
    parser.add_argument("--descriptor-slots", type=int, default=256)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--mode", choices=["both", "unique", "expanded"], default="both")
    parser.add_argument("--output", type=Path, default=Path("results/large.json"))
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.experts, args.topk, args.chunk_rows) < 1:
        parser.error("dimensions and chunk size must be positive")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.experts % world or args.topk > args.experts:
        parser.error("expert count must divide world size and top-k must not exceed experts")
    started = time.perf_counter()
    local_routes = make_routes(args.tokens, args.experts, args.topk, rank, args.seed)
    all_routes = gather_cpu(local_routes)
    ref = cpu_reference(all_routes, rank, world, args.experts, args.alignment)
    idx, weights = (item.to(device) for item in local_routes)
    columns = torch.arange(args.hidden, dtype=torch.int64, device=device)
    x = torch.empty(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)
    for begin in range(0, args.tokens, args.chunk_rows):
        end = min(begin + args.chunk_rows, args.tokens)
        token = torch.arange(begin, end, dtype=torch.int64, device=device)
        source = torch.full((end - begin,), rank, dtype=torch.int64, device=device)
        x[begin:end].copy_(payload_pattern(source, token, columns))
    records = []
    try:
        with HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=args.tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            num_experts=args.experts,
            config=HostStagedConfig(
                payload_slots=args.payload_slots, descriptor_slots=args.descriptor_slots, **transport_options(args)
            ),
        ) as buffer:
            for expand in ([False, True] if args.mode == "both" else [args.mode == "expanded"]):
                result = buffer.dispatch(
                    x,
                    topk_idx=idx,
                    topk_weights=weights,
                    do_expand=expand,
                    expert_alignment=args.alignment,
                    do_zero_padding=expand,
                    do_cpu_sync=False,
                )
                failure = None
                record = None
                try:
                    buffer.check_errors(result[3])
                    record = check_result(result, ref, args, device, rank, world, expand, columns)
                except Exception as exc:
                    failure = f"rank {rank}: {type(exc).__name__}: {exc}"
                failures = [item for item in gather_cpu(failure) if item is not None]
                if failures:
                    raise AssertionError("; ".join(failures))
                mode_records = gather_cpu(record)
                records.extend(mode_records)
                if rank == 0:
                    print(
                        f"large {'expanded' if expand else 'unique'}: "
                        f"{sum(item['payload_elements_checked'] for item in mode_records):,} BF16 elements checked",
                        flush=True,
                    )
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "world_size": world,
                        "tokens_per_rank": args.tokens,
                        "hidden": args.hidden,
                        "experts": args.experts,
                        "topk": args.topk,
                        "dtype": "bfloat16",
                        "transport_options": transport_options(args),
                        "payload_slots": args.payload_slots,
                        "descriptor_slots": args.descriptor_slots,
                        "expert_alignment": args.alignment,
                        "seed": args.seed,
                        "control_backend": "gloo",
                        "payload_reference": "finite BF16 raw-bit source/token/column hash",
                        "nvlink_payload_allowed": False,
                        "payloads_allgathered": False,
                        "payload_elements_checked_total": sum(item["payload_elements_checked"] for item in records),
                        "payload_bytes_checked_total": sum(item["payload_bytes_checked"] for item in records),
                        "elapsed_seconds": time.perf_counter() - started,
                        "records": records,
                        "status": "passed",
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
