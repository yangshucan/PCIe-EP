| Tokens/GPU | Baseline ms | Optimized ms | Baseline GB/s | Optimized GB/s | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.694 | 0.260 | 2.645 | 7.071 | 2.674x | 62.60% |
| 32 | 0.707 | 0.282 | 5.191 | 13.012 | 2.507x | 60.11% |
| 64 | 0.796 | 0.405 | 9.226 | 18.103 | 1.962x | 49.04% |
| 128 | 1.021 | 0.647 | 14.385 | 22.688 | 1.577x | 36.60% |
| 256 | 1.833 | 1.033 | 16.019 | 28.410 | 1.773x | 43.61% |
| 512 | 3.056 | 1.950 | 19.213 | 30.120 | 1.568x | 36.21% |
| 1024 | 5.905 | 3.681 | 19.889 | 31.907 | 1.604x | 37.66% |
| 2048 | 11.170 | 7.079 | 21.029 | 33.179 | 1.578x | 36.62% |
| 4096 | 21.815 | 13.558 | 21.534 | 34.649 | 1.609x | 37.85% |
| 8192 | 42.820 | 26.546 | 21.941 | 35.393 | 1.613x | 38.01% |

Bandwidth = T*K*H*2/t: T per GPU, K=8, H=7168, BF16; decimal GB/s, no extra GPU-count factor.
Each variant: two rounds x 20 samples/batch. Each sample is recomputed as the maximum across eight ranks.
Full latency is the mean of all 40 samples/batch, including layout, transport, expansion and GPU stream waits. CPU setup and diagnostic readback are outside the benchmark's timed interval.
The exported JSON/CSV also retain pooled p50/p95 and the two round means. These are observations, not confidence intervals.
Pipelined transport uses 352 threads (11 warps)/CTA in both variants. Bandwidth is logical expert volume, not measured PCIe/UPI bytes.

Caller-supplied baseline runtime SHA-256: `1456920283a6866c8c9e0165d2ac98703d94a1a586e2e29f8a92feb795833490`.
Caller-supplied optimized runtime SHA-256: `4c666f056724c03c5ecb81855d10e4f18f2653e858a95db63839f47634d7f7c8`.
