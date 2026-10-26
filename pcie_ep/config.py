"""Configuration for the single-host host-staged dispatch stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class HostStagedConfig:
    """Host transport capacity and protocol selection.

    Arguments:
        payload_slots: per-source payload ring capacity, independent of Tmax.
        descriptor_slots: per-source/per-destination notification capacity.
        channels: independently owned payload/descriptor ring channels.
        stage: supported capability level, 6 (BF16), 7 (FP8), or 8 (relay).
            All levels use the same current runtime; stages 1-5 are retired.
        timeout_ms: GPU-side protocol timeout, not a CPU completion wait.
        directory: optional parent directory for the per-buffer backing files.
        numa_mode: direct mapped reads, or S8 CPU relay to NUMA-local replicas.
        warp_specialization: dedicate GPU warps to transport control and data.
        pipeline_stages: two or three payload stages per advanced channel.
        fuse_receive_expand: scatter received payload directly to expert rows.
        receive_expand_mode: scalar preserves the original fused receive;
            vectorized and pipelined require S5+, receive fusion and warp
            specialization. Unique dispatch uses the current unique-output kernel.
        source_fusion: S6 source-side routing and payload publication fusion.
        wire_dtype: BF16 or S7 per-token scaled FP8 E4M3 transport for finite
            input. Nonzero scales have an FP32 normal-minimum floor. Input
            and output tensors remain BF16 in either mode.
        enable_graph: allocate an independent result pool for explicit S6
            capture_dispatch preparation and fixed-address replay.
        output_pool_size: number of preallocated eager result pools.
        replica_slots: bounded S8 payload capacity per NUMA replica channel.
        relay_workers_per_numa: S8 workers per source/target NUMA domain.
            Channels are partitioned between workers; at most channels workers.
        relay_cpu_affinity: optional tuple of distinct allowed CPU indices for
            S8 workers. Native pinning also intersects NUMA and process affinity.
    """

    payload_slots: int = 256
    descriptor_slots: int = 256
    channels: int = 32
    stage: int = 6
    timeout_ms: int = 30_000
    directory: Optional[str] = None
    numa_mode: str = "direct"
    warp_specialization: bool = True
    pipeline_stages: int = 2
    fuse_receive_expand: bool = True
    source_fusion: bool = True
    wire_dtype: str = "bf16"
    replica_slots: int = 256
    enable_graph: bool = False
    relay_workers_per_numa: int = 1
    relay_cpu_affinity: Tuple[int, ...] = ()
    metadata_delay_ns: int = 0
    tx_delay_ns: int = 0
    rx_delay_ns: int = 0
    scatter_delay_ns: int = 0
    output_pool_size: int = 3
    receive_expand_mode: str = "pipelined"

    def __post_init__(self) -> None:
        for name in ("payload_slots", "descriptor_slots", "timeout_ms", "replica_slots"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stage not in (6, 7, 8):
            raise ValueError("stages 1-5 are retired; stage must be 6, 7 or 8")
        if not isinstance(self.channels, int) or not 1 <= self.channels <= 32:
            raise ValueError("channels must be an integer in [1, 32]")
        if self.pipeline_stages not in (2, 3):
            raise ValueError("pipeline_stages must be 2 or 3")
        if self.receive_expand_mode not in ("scalar", "vectorized", "pipelined"):
            raise ValueError("receive_expand_mode must be 'scalar', 'vectorized' or 'pipelined'")
        if self.receive_expand_mode != "scalar":
            if not self.fuse_receive_expand or not self.warp_specialization:
                raise ValueError("Non-scalar receive expansion requires fuse_receive_expand and warp_specialization")
        if self.wire_dtype not in ("bf16", "fp8_e4m3"):
            raise ValueError("wire_dtype must be 'bf16' or 'fp8_e4m3'")
        if self.stage < 7 and self.wire_dtype != "bf16":
            raise NotImplementedError("FP8 wire transport requires S7")
        if self.numa_mode not in ("direct", "relay"):
            raise ValueError("numa_mode must be 'direct' or 'relay'")
        if self.numa_mode == "relay" and self.stage < 8:
            raise NotImplementedError("NUMA relay is an S8 optimization")
        if not isinstance(self.relay_workers_per_numa, int) or not 1 <= self.relay_workers_per_numa <= self.channels:
            raise ValueError("relay_workers_per_numa must be an integer in [1, channels]")
        if not isinstance(self.relay_cpu_affinity, tuple) or any(
            not isinstance(cpu, int) or cpu < 0 for cpu in self.relay_cpu_affinity
        ):
            raise ValueError("relay_cpu_affinity must be a tuple of nonnegative CPU indices")
        if len(set(self.relay_cpu_affinity)) != len(self.relay_cpu_affinity):
            raise ValueError("relay_cpu_affinity must contain distinct CPU indices")
        if self.stage < 8 and (self.relay_workers_per_numa != 1 or self.relay_cpu_affinity):
            raise NotImplementedError("Relay worker configuration requires S8")
        if self.output_pool_size < 2:
            raise ValueError("output_pool_size must be at least two")
        for name in ("metadata_delay_ns", "tx_delay_ns", "rx_delay_ns", "scatter_delay_ns"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def native_options(self) -> dict:
        """Options agreed collectively before allocating advanced shared rings."""
        return {
            "stage": self.stage,
            "channels": self.channels,
            "payload_slots": self.payload_slots,
            "descriptor_slots": self.descriptor_slots,
            "timeout_ms": self.timeout_ms,
            "warp_specialization": self.warp_specialization,
            "pipeline_stages": self.pipeline_stages,
            "fuse_receive_expand": self.fuse_receive_expand,
            "receive_expand_mode": self.receive_expand_mode,
            "source_fusion": self.source_fusion,
            "wire_dtype": self.wire_dtype,
            "payload_dtype": "fp8" if self.wire_dtype == "fp8_e4m3" else "bf16",
            "numa_mode": self.numa_mode,
            "replica_slots": self.replica_slots,
            "enable_graph": self.enable_graph,
            "relay_workers_per_numa": self.relay_workers_per_numa,
            "relay_cpu_affinity": list(self.relay_cpu_affinity),
            "metadata_delay_ns": self.metadata_delay_ns,
            "tx_delay_ns": self.tx_delay_ns,
            "rx_delay_ns": self.rx_delay_ns,
            "scatter_delay_ns": self.scatter_delay_ns,
            "output_pool_size": self.output_pool_size,
        }
