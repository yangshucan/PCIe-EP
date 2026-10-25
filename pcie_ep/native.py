"""Adapt the S5–S8 native API to the collective S1–S4 setup protocol."""

from __future__ import annotations

from typing import Any

import torch.distributed as dist

from .config import HostStagedConfig


class AdvancedExtensionAdapter:
    """Keep shared-memory setup and failure coordination common to all stages."""

    def __init__(self, extension: Any, config: HostStagedConfig, control_group: Any = None) -> None:
        self.extension = extension
        self.options = config.native_options()
        self.options["rank_numa_nodes"] = []
        if config.numa_mode == "relay":
            node, error = None, None
            try:
                node = extension.get_device_numa_node()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            states = [None] * dist.get_world_size(control_group)
            dist.all_gather_object(states, (node, error), group=control_group)
            if any(item[1] for item in states):
                raise RuntimeError(f"NUMA discovery failed: {states}")
            self.options["rank_numa_nodes"] = [item[0] for item in states]

    def create_segments(self, directory, world, tmax, hidden, topk, experts, slots, descriptor_slots, stage):
        return self.extension.create_advanced_segments(
            directory,
            world,
            tmax,
            hidden,
            topk,
            experts,
            self.options,
        )

    def prepare_segment(self, directory, rank, world, tmax, hidden, topk, experts, slots, descriptor_slots, stage):
        return self.extension.prepare_advanced_segment(
            directory,
            rank,
            world,
            tmax,
            hidden,
            topk,
            experts,
            self.options,
        )

    def Runtime(self, directory, rank, world, tmax, hidden, topk, experts, slots, descriptor_slots, timeout_ms, stage):
        return self.extension.AdvancedRuntime(
            directory,
            rank,
            world,
            tmax,
            hidden,
            topk,
            experts,
            self.options,
        )
