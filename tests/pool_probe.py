"""Tiny, diagnostic-only output pool ownership probe (torchrun with 2 ranks)."""

import gc
import json
import os
import sys
import weakref

import torch
import torch.distributed as dist

from pcie_ep import HostStagedConfig, HostStagedEPBuffer


def dump(buffer, label, handles, payloads):
    handle = buffer._last_handle
    detail = {
        "rank": dist.get_rank(),
        "label": label,
        "pool": buffer.runtime.pool_debug(),
        "live_weak_handles": [ref() is not None for ref in handles],
        "live_weak_payloads": [ref() is not None for ref in payloads],
        "last_handle_python_refcount": sys.getrefcount(handle) if handle else 0,
        "last_handle_referrers": [type(item).__name__ for item in gc.get_referrers(handle)] if handle else [],
    }
    print(json.dumps(detail), flush=True)


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if world != 2:
        raise ValueError("pool_probe requires exactly two ranks")
    handles, payloads = [], []
    config = HostStagedConfig(stage=1, payload_slots=2, descriptor_slots=2)
    try:
        with HostStagedEPBuffer(
            group=dist.group.WORLD,
            num_max_tokens_per_rank=2,
            hidden=256,
            num_topk=8,
            num_experts=256,
            config=config,
        ) as buffer:
            x = torch.full((2, 256), rank + 1, dtype=torch.bfloat16, device="cuda")
            idx = (torch.arange(8, dtype=torch.int64, device="cuda") + (1 - rank) * 128).repeat(2, 1)
            weights = torch.ones((2, 8), dtype=torch.float32, device="cuda")
            for iteration in range(6):
                dump(buffer, f"before-{iteration}", handles, payloads)
                failure = None
                try:
                    result = buffer.dispatch(x, idx, weights)
                    handles.append(weakref.ref(result[3]))
                    payloads.append(weakref.ref(result[0]))
                    result[4].wait()
                    buffer.check_errors()
                    dump(buffer, f"after-{iteration}", handles, payloads)
                    del result
                    gc.collect()
                    dump(buffer, f"after-del-gc-{iteration}", handles, payloads)
                except Exception as exc:
                    failure = f"rank {rank}: {type(exc).__name__}: {exc}"
                    dump(buffer, f"exception-{iteration}", handles, payloads)
                failures = [None] * world
                dist.all_gather_object(failures, failure)
                if any(failures):
                    print(json.dumps({"rank": rank, "failures": failures}), flush=True)
                    break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
