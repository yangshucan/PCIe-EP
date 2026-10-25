# S0 mapped publication / ACK litmus

`mapped_litmus.cu` is a standalone, single-process, two-GPU test. It allocates one
portable mapped pinned host payload and separate, 128-byte-aligned READY, ACK and
abort fields. Two host threads select their GPUs and launch one 256-thread CTA
each. They do not poll flags or copy payload in the communication loop.

The producer fills every 64-bit payload word with a function of `(sequence,
word_index)`. Every writing thread executes a system fence, the CTA joins, and
the leader publishes READY with a system-scope release store. The consumer
leader waits with an acquire load; all 256 threads validate their words and
finish before the leader release-publishes ACK. The producer acquire-waits for
ACK before reusing the single payload slot. READY and ACK start at zero;
sequences start at one. The final producer completion includes the final ACK.

Payload accesses are volatile to retain explicit mapped payload loads/stores;
volatile does **not** provide the handoff protocol. Flags use
`cuda::atomic_ref<uint64_t, cuda::thread_scope_system>` acquire/release operations.
The only RMW is `atomicMin` on CTA shared memory to identify a mismatch. The
test reports `hostNativeAtomicSupported` and `canMapHostMemory`; it does not
interpret the native-host-atomic attribute as a requirement for an unissued RMW
or claim that cross-device mapped-memory RMW is supported.

## Compile and run

For the targeted CUDA 13 / SM90 environment:

```sh
nvcc -O3 -std=c++17 -arch=sm_90 -Xcompiler -pthread mapped_litmus.cu -o mapped_litmus
./mapped_litmus --producer 0 --consumer 1 --iterations 10000
./mapped_litmus --producer 0 --consumer 1 --iterations 1000000 --bytes 14336
./mapped_litmus --producer 0 --consumer 1 --iterations 1000000 --bytes 14336 --delay 1000 --writer-delay 1000
```

`--delay` delays the last consumer thread **before** its payload loads;
`--writer-delay` delays the last producer thread before its stores. Values are
GPU clock cycles, applied to every iteration. Use payloads of at least 2048
bytes so all 256 threads participate. Smaller positive multiples of eight are
accepted for boundary checks, but may leave the delayed thread with no word.
Repeat with reversed GPU roles and GPU pairs within and across NUMA domains.

`--timeout-ms` defaults to 10000 per READY/ACK wait. It is converted using the
reported GPU clock rate, so clock variation makes it an approximate diagnostic
timeout. A timeout or validation failure publishes that GPU's abort flag, letting
its peer stop at its next poll. This is not recovery from a device fault or a
process crash. After an API/device failure, normal CUDA cleanup may still wait
for the other kernel's bounded timeout.

The program prints one JSON record and returns zero only when **both** kernels
completed every requested iteration without CUDA or payload errors. Failure
codes: 1 READY timeout, 2 ACK timeout, 3 peer aborted, 4 payload mismatch.
`host_elapsed_seconds` includes launch-thread/event overhead and final test
synchronization. Per-device CUDA event times are separately reported. This
single-slot round trip is not an EP bandwidth measurement.

## Scope and evidence to retain

- No peer access is enabled; no CUDA IPC, device-pointer exchange, GPU P2P
  memory copy, NCCL data operation or NVLink operation is issued. Both GPUs
  access the same CPU mapped allocation. Record PCIe/NVLink counters separately
  when auditing the full dispatch implementation.
- Initialization, final result collection and teardown synchronize on the CPU;
  the million-iteration kernel loop does not.
- This test covers single-process mapping and reuse. A separate multiprocess
  `mmap` / `cudaHostRegister` test is still required for the runtime's shared
  backing, and multicast requires a delayed-last-reader test of its own.
- Preserve the JSON, GPU/PCIe/NUMA topology, driver/toolkit/CCCL versions and
  generated PTX/SASS for the build used. Inspect the READY/ACK loads/stores and
  system fences; no host-mapped atomic RMW should be introduced by compilation.
- Passing stress tests is engineering evidence, not a proof of the CUDA memory
  model on arbitrary hardware. Unsupported mapping or CUDA errors fail clearly;
  there is no CPU proxy fallback.

## Recorded validation, 2026-10-24

The standalone test was compiled with CUDA 13.0 for SM90 and run in the isolated
`pcie-ep` container on `<GPU_HOST>`. Both tested GPUs reported NVIDIA
H20D, `canMapHostMemory=1`, and `hostNativeAtomicSupported=0`. CUDA runtime and
driver versions were both 13000. A CUDA 13 compatibility adjustment uses
`cudaDeviceGetAttribute(cudaDevAttrClockRate)` instead of the removed
`cudaDeviceProp.clockRate` field.

All runs used a 14336-byte payload, and both kernels completed the full requested
iteration count with error code zero:

| Physical GPU pair | Iterations | Delays per iteration | Maximum per-GPU CUDA event time | Result log |
|---|---:|---|---:|---|
| 0, 1 | 10,000 | none | 184.022247 ms | `s0_smoke.json` |
| 0, 1 | 1,000,000 | none | 18458.4316 ms | `s0_million_same_numa.json` |
| 0, 4 | 1,000,000 | none | 18090.9121 ms | `s0_million_cross_numa.json` |
| 0, 4 | 10,000 | last writer and last reader: 1000 cycles each | 299.695953 ms | `s0_delayed_cross_numa.json` |

Logs and `s0_litmus.sass` are stored in remote
`/opt/pcie-ep/results/` (container `/workspace/results/`). GPU
0/1 and GPU 0/4 were selected to cover same- and cross-NUMA GPU attachment; this
standalone allocator did not bind or inspect the actual host payload page
placement. CUDA-visible ordinals in each JSON are 0/1; cross-NUMA runs used
`CUDA_VISIBLE_DEVICES=0,4`.

The retained SASS contains 64-bit `LD/ST ... STRONG.SYS` handoff accesses and
`MEMBAR.ALL.SYS` / `MEMBAR.SC.SYS` fences. The emitted `ATOMS` operation targets
CTA shared memory; no mapped-memory global atomic RMW was observed. These runs
validate the standalone test only. They do not close the multiprocess,
multicast, error-injection, or dispatch integration acceptance requirements.
