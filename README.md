# PCIe host-staged EP dispatch

Single-host, GPU-initiated expert dispatch with source and receiver token deduplication. Each source writes one payload to registered shared CPU memory; each target GPU reads it once. Local expert branches expand inside GPU memory. The payload path uses GPU SM loads/stores through PCIe, with system-scope READY/ACK load/store handshakes.

The implementation has one native runtime with independently selectable capabilities: multiple channels, control/TX/RX warp specialization, shared-memory receive/expand fusion with vectorized copy and optional independent RX/scatter warps, source fusion, parallel stable layout, cached layout, CUDA Graph, optional FP8 wire compression, and NUMA relay. See [the implementation notes](docs/pcie-ep/advanced-implementation.md), [API.md](API.md), and the [staged plan](docs/pcie-ep/implementation-plan.md).

The Python API follows DeepEP V2.5's `EPBuffer`, `EPHandle`, and `EventOverlap` structure and can borrow an existing PyTorch NCCL communicator. The dispatch payload path contains no NCCL, GPU peer mapping, peer copy, NVLink, or Gin operations. This is an independent package; the original DeepEP checkout is read-only reference material.

The default configuration is C32, source fusion, warp specialization and pipelined receive expansion. A 256-thread initialization CTA clears diagnostics and checks terminal flags in parallel. Receive count/prefix preparation runs in a resident RX-control warp while TX proceeds; final source/expert token ordering is unchanged. See [prefix/TX overlap](docs/pcie-ep/prefix-tx-overlap.md). Stage numbers 6/7/8 select capability gates on this same runtime; stages 1–5 are retired and rejected.

## Configuration and use

Use one process per GPU and set its CUDA device before constructing a buffer. The benchmark uses eight Gloo ranks as both DP-attention sources and EP participants. It measures dispatch; attention and expert GEMMs are outside this implementation.

```python
import torch.distributed as dist
from pcie_ep import HostStagedConfig, HostStagedEPBuffer

config = HostStagedConfig(
    stage=6,
    channels=32,
    payload_slots=256,
    descriptor_slots=256,
    warp_specialization=True,
    pipeline_stages=2,
    fuse_receive_expand=True,
    receive_expand_mode="pipelined",
    source_fusion=True,
)
with HostStagedEPBuffer(
    group=dist.group.WORLD,
    num_max_tokens_per_rank=8192,
    hidden=7168,
    num_topk=8,
    num_experts=256,
    config=config,
) as buffer:
    recv_x, recv_idx, recv_weights, handle, event = buffer.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights,
        do_expand=True, do_cpu_sync=False, async_with_compute_stream=True,
    )
    event.current_stream_wait()  # GPU stream dependency; no CPU completion wait
    # Expert kernels consume GPU expert_counts/expert_offsets and valid rows.
```

`receive_expand_mode="pipelined"` uses three dedicated scatter warps in addition to the original eight. `"vectorized"` keeps the original RX roles and uses 16-byte shared→expert copies; `"scalar"` remains an explicitly selected diagnostic variant. `"pipelined"` is the default. Unique dispatch keeps its original path. See [F2 implementation](docs/pcie-ep/f2-receive-expand.md) and [the latest prefix/TX-overlap measurements](RESULTS_PREFIX_TX.md).

`do_cpu_sync=False` is mandatory; `None` means False. Normal dispatch/replay does not read counts back to the CPU or wait for GPU completion. Initialization, explicit capture preparation, collective close, and `check_errors()` may synchronize and belong outside the steady path.

The current runtime prepares three eager output pools by default, configurable with `output_pool_size`; output tensors and metadata remain protected while referenced. Pool exhaustion raises an explicit error. Consumer work must be enqueued before the next dispatch and registered through `EventOverlap.wait()` / `current_stream_wait()` so reuse waits for its stream tail. CPU submissions are serialized.

For S6 caching, extract `cache = buffer.cache_layout(handle)` to retain only an immutable route plan, release old outputs, and call `buffer.dispatch(new_x, topk_weights=new_weights, handle=cache, do_expand=True)`. Routes, ownership and alignment must remain unchanged. Explicit Graph support uses `enable_graph=True`, `buffer.capture_dispatch(...)`, and `graph.replay()`; each buffer supports one fixed-address graph with a separate output pool. Graph outputs are overwritten on replay. Full contracts and examples are in API.md.

S7's `wire_dtype="fp8_e4m3"` is optional: external tensors stay BF16, remote payloads carry E4M3FN plus one FP32 scale per token. S8's `numa_mode="relay"` is also optional: native CPU workers copy once per target NUMA to local replica rings. The calling thread stays asynchronous, while data progress depends on those workers. Source COPY ACK and replica GPU ACK/drained are separate.

## Workspaces and build

- Clone this repository into a directory of your choice.
- Remote synchronization requires an explicit host: `bash scripts/sync_remote.sh GPU_HOST /opt/pcie-ep`.
- Use a CUDA-enabled PyTorch container with the checkout mounted at `/workspace` and a writable cache mounted at `/build`. Set `PCIE_EP_CONTAINER` to its name; the generic default is `pcie-ep`.
- Environment: PyTorch 2.13.0+cu130, CUDA 13.0, SM90, eight H20D (132 SM), PCIe Gen5 x16; GPUs 0–3 NUMA0 and 4–7 NUMA1.
- Build/cache: `/dev/shm/pcie-ep-build` on the host, `/build` in the container. No external downloads are needed.

The extension is one CUDA translation unit with implementation headers organized like DeepEP. It uses C++20 for CPU `atomic_ref` load/store handshakes. Tensor/PyObject pool ownership checks depend on the tested PyTorch C++ API; revalidate before changing PyTorch. See [code-style.md](docs/pcie-ep/code-style.md) and [REFACTOR.md](REFACTOR.md).

`bash scripts/sync_remote.sh GPU_HOST /opt/pcie-ep` pushes sources without deleting remote-only files, excludes results/build caches, and verifies SHA-256 manifests. New validation logs are local-only under the ignored `results/` directory; historical archives are kept outside the checkout. See [publication scope](PUBLICATION.md).

## Reproduction

Run GPU orchestration on your GPU host after setting up the container and mounts above. GPU jobs run sequentially. The included anonymous measurements can be recalculated on any CPU with Python 3.

```bash
bash scripts/validate_f2.sh all
PCIE_EP_F2_RUN_ID=my_run bash scripts/benchmark_f2.sh all
python3 benchmarks/report_latest.py --run-dir results/f2/runs/my_run
python3 benchmarks/report_prefix_tx.py --samples benchmarks/data/prefix_tx_samples.json --output-dir /tmp/prefix_tx_report

```

Advanced validation includes CPU unit tests, independent distributed BF16/FP8 reference checks, cache/Graph and relay combinations, forced writer/reader delays, bounded-ring stress, full-payload T8192 checks, stream/pool/communicator/error lifetime tests, and profiler audits. The stress iteration count can be changed with `PCIE_EP_F2_STRESS_ITERATIONS`; results record the actual count. Retired stage 1–5 executables and historical report generators have been removed; see [cleanup scope](CLEANUP.md).

The requested workload is DP8/EP8, 256 experts (32 consecutive experts per GPU), distinct top-k8, hidden 7168, BF16 input/output, and tokens per GPU 16/32/64/128/256/512/1024/2048/4096/8192. The deterministic route seed is 20261024. The published table uses BF16 wire; FP8 remains a separately tested optional mode.

## Timing and bandwidth

Full dispatch latency is measured with same-device CUDA events, then each round takes the maximum across eight ranks. Layout, counts/prefix, transport, local expansion and stream waiting are included. Transport timestamps span the earliest channel start through the latest channel completion on each GPU; the interval includes polling and backpressure. No cross-GPU clocks are subtracted.

The current comparison uses the requested formula `bandwidth = T*K*H*b/t`: T is input tokens per GPU, K=8, H=7168, b=2 bytes for BF16, and t is full mean dispatch latency in seconds. Divide bytes/second by 1e9 for decimal GB/s. This per-GPU logical byte count is before deduplication, includes all top-k branches, and has no extra world-size factor. [The latest report](RESULTS_PREFIX_TX.md) compares the previous and optimized pipelined implementations.

Historical raw files retain their original byte counters and bandwidth fields for traceability. For payload bytes B, U counts source tokens with a remote destination, F counts deduplicated remote token/destination-GPU pairs, and R counts relay token/target-NUMA copies, summed across eight GPUs:

- Historical remote-delivery bytes: `F*B`; the old bandwidth fields divide those bytes by full dispatch latency.
- GPU host-payload access: `(U+F)*B`.
- Additional relay CPU payload read/write: `2*R*B`, reported separately.

B is `H*2` for BF16 and `H+4` for FP8 including scales. These historical remote-delivery fields exclude metadata, padding and local GPU expert copies, while their execution time remains included. They use aggregate decimal GB/s across eight GPUs, with per-GPU averages also recorded; they are separate from the current `T*K*H*b/t` metric. CPU validation, synchronization, Gloo statistics and file writes are outside timing. Clock locking is not used.

## PCIe-only evidence and limits

NCCL P2P/SHM/NVLS/MNNVL are disabled and Socket networking is selected for control-only communicator reuse. Full NVLink telemetry covers 8 GPUs × 18 links × Tx/Rx = 288 counters before and after measurement, without disabling or resetting physical links. Source, symbol, pointer-provenance and SASS audits accompany the counters; environment flags alone are insufficient evidence.

The supported scope is one machine, one GPU/process, P≤16, top-k≤32, up to 32 independent channels, contiguous BF16 input, int64 routes, optional FP32 weights, consecutive uniform expert ownership, and alignment 1–256. Epoch exhaustion and invalid routes cause terminal errors and require rebuilding the buffer. Combine/backward, dynamic owner maps, cross-node transport, automatic direct/relay selection and concurrent CPU submissions are outside this version.

[RESULTS_F2.md](RESULTS_F2.md) contains only the latest pipelined implementation for all ten batches, using `T*K*H*b/t`. Anonymous rank timings and recalculated JSON/CSV are included in `benchmarks/data/`:

```bash
python3 benchmarks/report_latest.py
```

Machine-specific raw logs, traces, native binaries, topology dumps and historical experiment archives are excluded from Git. They can be generated locally by the validation and benchmark scripts. The numerical results are an anonymous export of the completed GPU measurements; this publication did not rerun GPU experiments. The retained measurements identify their original measured source in `benchmarks/data/f2_measured_runtime_source_sha256.json`. The current runtime was subsequently cleaned and revalidated; that historical fingerprint does not describe the cleaned source. See [cleanup scope](CLEANUP.md).
