# Dispatch 控制路径优化与性能结果

2026-10-29，八卡实测。本轮落实 [profiling 改进方案](docs/pcie-ep/dispatch-profiling-20261029.md) 中的 P0 控制路径优化，以及 P1 的 counts mailbox 检查合并。默认配置仍为 C32、pipelined F2、BF16、NUMA direct，通信协议保持 PCIe-only、稳定输出顺序和主路径无 CPU 同步。

T8192 两轮平均从 **42.820 ms 降至 26.546 ms**，逻辑带宽从 **21.941 GB/s 提升至 35.393 GB/s**，加速 **1.613×**；T16 从 **0.694 ms 降至 0.260 ms**，加速 **2.674×**。

## 实现改动

1. **Warp 并行健康检查。** 原先 leader 串行读取全部 abort flags，现在由 32 lanes 分摊，再统一投票。扫描间隔按 GPU 时间取 `min(1 ms, timeout/8)`，从上次扫描完成时重新计时；本地 timeout 每个有界控制迭代都检查。没有直接采用诊断实验的“4096 轮检查一次”。
2. **每 channel 由 TX 维护健康状态。** RX 主循环只读取 shared abort。TX 即使已经发送和排空完成，仍存活到 RX 完成。统一退出时再次 acquire 本地 abort，避免 RX 尾部错误发生在 loop 顶部检查之后而漏发 mapped terminal flag。
3. **有界增量 ACK 排空。** 保存单调 cursor 和 remaining endpoint/relay masks，每次最多一个 mapped load；成功确认的 ACK 不再读取。缺失 ACK 从检查完成时退避 1 µs，期间继续检查健康状态和 timeout。所有 NUMA domain 的 relay-drained，包括零工作 domain，仍必须匹配本 epoch。
4. **Prefix 等待使用协作检查。** counts 尚未 READY 时，先保持轻量轮询，不再立即触发数百次串行 abort load。owner 的 timeout、failed READY 和 waiter 的 GPU-local 失败出口仍保留。
5. **Counts mailbox guard 合并到 begin_epoch。** 首 warp 并行确认上一轮各 GPU 的 counts_consumed，缓存已确认项。kernel 完成后，后续 channel/total counts 才能写入 mapped mailbox；删掉各 channel 和 publish_counts 的重复 guard。各 writer 的 system fence、join、counts READY 顺序保持不变。

实现细节和内存顺序见 [control-progress 设计说明](docs/pcie-ep/control-progress.md)。没有更改 payload 的 vector copy、源/接收去重、expert/source/token 行号，也没有增加 transport CTA 数量。当前默认 eager expanded dispatch 仍是 8 个 kernel；本轮收益来自减少系统内存串行访问和重复等待。

## 测试配置和口径

DP attention8、EP8、256 experts、topK8、hidden7168、BF16；每 GPU 输入 T 为下面十档。C32、payload/descriptor/replica slots=256、本地 stages=2、source fusion 和 warp specialization 开启、pipelined F2、expanded output、alignment1、eager、seed20261024。不清 padding。

两版本各两轮独立进程，每轮 5 warmup + 20 measured。逐 iteration 取八 rank 完整 CUDA-event 时延的最大值，再对 40 个样本求均值。GPU 侧 layout、transport、local expansion、stream waits 均计入；初始化、Gloo 协调、错误检查和诊断读回在计时外。正式性能不使用 profiler 或 GPU role 埋点，没有锁 GPU 时钟。

```text
bw = T * 8 * 7168 * 2 / t / 1e9  GB/s
```

这是每 GPU 的逻辑 expert 数据量，不乘 GPU 数或去重率，不等于实测 PCIe/NUMA fabric 字节吞吐。

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

T8192 原版两轮为 42.741/42.898 ms，优化版为 26.555/26.536 ms；T16 优化版为 0.260/0.259 ms。比较使用本轮新测基线，没有将上一份报告或带 profiler 的数字混入。

匿名逐 rank 样本、p50/p95、两轮均值和 CSV 位于 [benchmarks/data](benchmarks/data/control_progress_comparison.json)。源文件逐项哈希和聚合哈希位于 [runtime fingerprints](benchmarks/data/control_progress_runtime_sha256.json)。

## 收益来源与边界

第一版只做 P0（尚未合并 mailbox guard），T16/T256/T8192 分别约 0.340/1.128/26.731 ms；合并后最终分别约 0.260/1.033/26.546 ms。该阶段定位实验每档 20 samples，仅用于解释改动方向，不替代最终两轮对比。

T8192 的角色埋点中，旧版每 channel 的 RX 健康扫描中位数约 **15.777 ms**，TX 另有 **0.423 ms**；新版 RX 主循环扫描为零，由 TX 协作扫描约 **0.507 ms / 24 次**。TX 排空阶段 ACK loads 中位数从 **17,397 降至 5,694**，其累计 active time 从约 **32.938 ms 降至 10.492 ms**。新版 TX/RX role lifetime 中位数约 25.563/25.208 ms。

合并 mailbox guard 后，release CUPTI 的 prepare kernel 中位数约从 77 µs 降至 6–7 µs，publish 从约 15 µs 降至 5 µs，begin_epoch 约 7 µs。小 T 的 prefix wait 也明显减少：T16 从约 409 µs 降至 75 µs。

这些角色数据来自单独的埋点运行，旧版每档 5,120、新版 2,560 个 channel observations。CUPTI 也会影响小 T 的调度，例如新版 T16 的 transport 单 kernel 中位数为 313 µs，高于正式 release 整次时延 260 µs；不能将其直接当作正式时延组成。[匿名 profiling 汇总](benchmarks/data/control_progress_profile.json) 保留三档数据及各 kernel 中位数。

角色时间互相重叠，不能把 health/drain 的减少量相加当作整体加速。小 T 主要受固定协议成本影响；大 T 仍主要在 transport。继续优化时应关注 RX 的 outstanding mapped reads、metadata/descriptor 和 stage 复用；扩展到 C64/C128、改变 RX/scatter 配额、缩小 fence scope 尚未合入本轮。

## 验证

以下验证全部通过，匿名摘要见 [validation JSON](benchmarks/data/control_progress_validation.json)：

- 新增 49 组控制进度测试：三种 receive 模式、C1/C2/C32、Q/L=1/2 与 2/1、stages2/3；包括远端 abort、缺失 counts、缺失上一 epoch ACK、7 epoch 有界队列复用和异步提交。该专用套件 hidden256；其余主要正确性套件包含 hidden7168。
- Prefix/cache/Graph 58 组、F2 212 组、多 buffer 4 组、consumer lifetime 2 组；另有同构 scalar 两卡与单卡 pipelined 各 11 组兼容用例。
- 小环队列压力测试：三组 Q/L，每组每 rank 连续 1,000 dispatch，合计每 rank 3,000 次，hidden7168；NUMA relay 18 epochs，包括零工作、热专家、多播及延迟搬运。
- T8192、hidden7168 的 unique/expanded 全量校验，合计 **6,247,485,440 个 BF16 元素**；稳定行顺序、payload、路由、权重、counts 与 padding 按独立 reference 校验。
- 专用 GPU 埋点验证 C1/C32：observer 的 TX 已完成后仍检测到远端 abort，并在约 1–2 µs 后退出 TX role。使用同一 GPU 上的时间差，未比较不同 GPU 的绝对时钟。
- 65 个 CPU 测试，以及性能报告独立重算、匿名样本 roundtrip 和非法输入拒绝检查。同步后刷新正式目录默认 release 缓存，八卡 C32、hidden7168 的 11 组快速用例再次通过。
- Eager 三档、八 rank 的稳态 trace 与 Graph cached trace 均未发现 CPU sync、GPU scalar 读回或主路径 D2H。同步和诊断读回仅在测试边界执行。
- 源码、执行 manifest、实际加载二进制与 SASS 审计通过：mapped flags 使用 system-scope load/store，RMW 均归属 GPU-local/CPU-local 状态，未发现 peer/NCCL/NVSHMEM 数据路径。覆盖整段测试窗口的 **288 个 NVLink payload counters 增量均为零**。NVLink 物理链路保留启用；计数器精度为 KiB，这不是 PCIe TLP/UPI 字节计量。

正式 release pipelined transport 的编译资源为 72 registers/thread、1,024 B static shared、960 B stack、0 B reported local。动态 shared 仍由双 stage 的 token cache 决定；本轮没有调整 occupancy 配额。

健康扫描的逻辑响应上界是扫描间隔、一个有界控制迭代和扫描本身的耗时之和；本地 timeout 不必等待下一次远端扫描。该机制不能中断已经发出的硬件 PCIe 操作，也不能抢占测试人为插入的 GPU sleep。故障测试使用 2 秒 CPU watchdog 与外层进程 timeout；生产 dispatch 不进行这些 CPU 等待。

## 复现

八卡环境中，从仓库根目录执行默认测试两次，并为不同轮次设置不同输出路径：

```bash
PCIE_EP_PROFILE=0 torchrun --standalone --nproc-per-node=8 \
  benchmarks/benchmark.py --warmup 5 --iterations 20 \
  --output results/control/optimized_r1

torchrun --standalone --nproc-per-node=8 tests/test_control_progress.py \
  --modes scalar vectorized pipelined

PCIE_EP_PROFILE=1 TORCH_EXTENSIONS_DIR=/tmp/pcie-ep-control-profile \
  torchrun --standalone --nproc-per-node=8 tests/test_control_progress.py \
  --case remote_abort --require-profiling

# 在 CPU 上，从已公开匿名样本重算全部比较结果
python3 benchmarks/report_control_progress.py \
  --samples benchmarks/data/control_progress_samples.json \
  --output-dir /tmp/control-progress-report
```

新测原始文件也可以通过 `--baseline <r1.json> <r2.json> --optimized <r1.json> <r2.json>` 交给同一报告工具；它会校验固定配置，并重新计算 max-rank 时延与带宽。

原始 trace、日志、完整 GPU role rows 和隔离的阶段快照在未跟踪的 `results/control_20261029/`。公开结果只包含匿名数值和相对源码哈希，不包含机器名、IP、GPU UUID 或内部路径。
