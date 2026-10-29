# GPU transport instrumentation

This diagnostic build measures the current BF16, NUMA-direct, pipelined expanded
dispatch. It is intended to explain the communication kernel, alongside the
uninstrumented CUDA-event latency and bandwidth measurements.

## Build and read

Set the environment before importing the extension, in a fresh process:

```bash
PCIE_EP_PROFILE=1 TORCH_EXTENSIONS_DIR=/tmp/pcie-ep-profile-cache \
  torchrun --standalone --nproc-per-node=8 benchmarks/profile_dispatch.py --help
```

`PCIE_EP_PROFILE=1` adds `-DPCIE_EP_ENABLE_PROFILING=1`. Use a different extension
directory for the normal build, and omit `PCIE_EP_PROFILE` for that build. The
default build has no profiling tensor allocation, state pointer, counters, or
profiling clock reads: these are removed by preprocessing.

The instrumented native result exposes `profiling`; the Python result handle
exposes the same tensor as `handle.profiling`. Its shape is
`[channels, 11, 32]`, its dtype is CUDA `int64`, and all timestamps/durations use
the current GPU's nanosecond global timer. Read it only after completion and
after checking that the dispatch succeeded, for example:

```python
result = buffer.dispatch(x, idx, weights, do_expand=True, do_cpu_sync=False)
handle = result[3]
handle.check_errors()              # Explicit diagnostic completion wait.
records = handle.profiling.cpu()   # Outside the measured dispatch path.
```

These diagnostic CPU waits are not part of dispatch or CUDA Graph replay.
The tensor belongs to the ordinary output pool; retaining it protects that
pool from reuse, just like retaining other output tensors.

## Row ownership

Each channel has the following role rows. Only lane 0 of the owning warp writes
its row. Counters accumulate locally, and the complete row is written to HBM
once when that role exits. There is no profiling reset kernel, additional
protocol barrier, fence, or atomic operation.

| Row | Role |
| --- | --- |
| 0 | TX control and metadata |
| 1 | RX control and receive-prefix preparation/wait |
| 2–4 | TX payload worker warps 0–2 |
| 5–7 | RX host-to-shared producer warps 0–2 |
| 8–10 | Shared-to-expert scatter consumer warps 0–2 |

## Field schema

Fields not applicable to a role are zero. A zero event timestamp means that the
event did not occur, for example a channel with no remote sends.

| Index | Meaning | Applicable role |
| --- | --- | --- |
| 0 | Role entry timestamp, ns | All |
| 1 | Role exit timestamp before its profile-row write, ns | All |
| 2 | Role elapsed time, `exit - entry`, ns | All |
| 3 | Control-loop iterations, or completed worker tickets | All |
| 4 | Number of timed worker tickets | Workers |
| 5 | Accumulated timed copy/scatter interval, ns | Workers |
| 6 | Accumulated timed system-fence plus existing warp-join interval, ns | Workers |
| 7 | Accumulated timed copy/scatter plus fence/join interval, ns | Workers |
| 8 | First token's descriptor-publication completion timestamp, ns | TX control |
| 9 | Last token's descriptor-publication completion timestamp, ns | TX control |
| 10 | Total elapsed time inside cooperative health scans, ns | TX control |
| 11 | Number of those health scans | TX control |
| 12 | Actual mapped abort/relay flag loads in those calls | TX/RX control |
| 13 | Total elapsed time inside every bounded drain step, ns | TX control |
| 14 | Total drain-step calls | TX control |
| 15 | Timed drain-step calls; equals field 14 | TX control |
| 16 | Actual payload ACK loads across all `tx_drained` calls | TX control |
| 17 | Actual relay-drained flag loads across all `tx_drained` calls | TX control |
| 18 | Accumulated timed descriptor-ready probe interval, ns | RX control |
| 19 | Total descriptor-ready probes | RX control |
| 20 | Number of timed descriptor-ready probes | RX control |
| 21 | Total probes whose descriptor was not ready | RX control |
| 22 | Iterations blocked from issuing another receive because its local stage was busy | RX control |
| 23 | First TX-finished timestamp, ns | TX control |
| 24 | RX-finished timestamp, or TX guardian abort-observed timestamp, ns | RX/TX control |
| 25 | Receive-prefix preparation/wait interval, ns | RX control |
| 26 | Accumulated timed `tx_payload_available` interval, ns | TX control |
| 27 | Total `tx_payload_available` calls | TX control |
| 28 | Remote input tokens to send, or remote receive tokens expected | TX/RX control |
| 29 | Published source tokens, independent of multicast destination count | TX control |
| 30 | Sampling stride for worker/probe/availability intervals: 32 | All |
| 31 | Number of timed `tx_payload_available` calls | TX control |

The TX guardian checks local timeout on every bounded iteration. It scans
mapped flags cooperatively at most once per `min(1 ms, timeout / 8)` interval,
measured from the preceding scan's completion. Fields 10–12 time and count
every scan; instrumentation does not change the interval. Each eight-rank,
32-channel direct scan reads `8 * (1 + 32) = 264` mapped signals across all lanes.
The RX main loop uses the shared abort verdict and reports zeros for these
scan fields. Prefix preparation has its own checks inside field 25; begin's
previous-count guard is outside transport and these role records.

Every drain step is timed. It reads at most one remaining payload ACK or relay
drained signal, remembers successful confirmations, and backs off 1 us from
completion when a signal is missing. No drain step runs after TX-finished.
Fields 13–15 cover all calls and field 16 counts actual ACK loads. Historical
baseline records used the same fields for complete, repeatedly rescanned tail
tables and for serial TX/RX `advanced_failed` calls every 256 iterations; compare
the recorded source versions as well as field numbers.

Worker ticket 0 and every 32nd ticket thereafter are timed. Descriptor probes
and `tx_payload_available` calls 1, 33, and so on are timed; their total call and
not-ready counts remain exact. Divide sampled duration by its sampled-call
count to obtain a sampled mean. Multiplying that mean by the total population
is an estimate, not a measured total, and short or nonstationary workloads can
have unrepresentative samples. Field 30 does not apply to the fully timed abort
and drain calls.

## Interpretation and limits

- Only successful pipelined expanded dispatch has all 11 role rows populated.
  Scalar, unique, vectorized, and homogeneous paths are not supported by this
  complete schema. An entry failure may leave records from an older dispatch;
  reject errored results before reading profiling data.
- The dedicated guardian regression intentionally injects a fault after TX
  has entered and uses only its channel-0 role-0 timestamps 23/24/1. This does
  not make arbitrary errored or entry-failed profiling rows valid.
- TX, RX, and scatter roles run concurrently. Their elapsed times include
  waiting and overlap with each other. Neither role durations nor worker sums
  can be added to produce kernel latency. Compare timestamp intervals and the
  uninstrumented kernel duration instead.
- Copy/scatter boundaries are observations by each warp leader. They describe
  instruction issue and dependencies visible to that leader, not the instant
  at which every lane's asynchronous store reaches its final destination. The
  fence interval includes the existing warp join; copy plus fence/join is the
  more useful completed-work interval. The profiling clock has a compiler
  memory-order boundary but adds no GPU memory fence.
- Three TX, RX, or scatter worker rows cooperate on the same token. Their
  ticket counts must not be interpreted as three times as many tokens.
- Records come from one GPU's clock. Do not subtract timestamps from different
  GPUs or align them directly to a CPU or Nsight clock without calibration.
- Instrumentation adds clock reads, local counter work, register pressure, and
  row writes. Measured instruction timing includes those effects. Compare
  instrumented and release runs separately; use release measurements for the
  reported dispatch latency and `T * K * H * 2 / t` bandwidth.
- CUDA Graph replay updates the rows on the GPU using the captured build's
  instrumentation. Each successful replay overwrites its dedicated output
  records; copy the completed records before another replay if retaining
  history is required.

The field definitions live in `TransportProfileField` in
`csrc/kernels/host_staged/advanced_common.cuh`. Profiling does not change the
mapped-host single-writer handshake or introduce CPU synchronization into the
communication path.
