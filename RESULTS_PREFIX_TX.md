# 并行初始化与 prefix/TX 重叠：性能对比

本文保留 2026-10-28 的历史结果。后续控制路径优化及重新测量的基线对比见 [RESULTS_CONTROL_PROGRESS.md](RESULTS_CONTROL_PROGRESS.md)；下表数值保持原始测量不变。

2026-10-28，在同一台 8×H20D 机器上重新测量原版本与新版本。基线为 Git `865cc49`；本表不是沿用 10 月 25 日的历史时延。

本次只优化两处：`begin_epoch` 改成 256-thread CTA，分摊诊断初始化和终止标志检查；specialized transport 的 RX 控制 warp 在 GPU 内认领 count/prefix 处理，TX 不再等待前置接收 prefix kernel。本地 token copy 排在 transport 后，最终 expert/source/token 稳定顺序保持不变。详见[实现与同步协议](docs/pcie-ep/prefix-tx-overlap.md)。

## 同配置结果

| Tokens/GPU | Baseline ms | Optimized ms | Baseline GB/s | Optimized GB/s | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.772 | 0.690 | 1.035 | 2.658 | 2.567x | 61.04% |
| 32 | 1.785 | 0.710 | 2.056 | 5.170 | 2.514x | 60.23% |
| 64 | 1.772 | 0.777 | 4.143 | 9.441 | 2.279x | 56.12% |
| 128 | 2.021 | 1.012 | 7.263 | 14.511 | 1.998x | 49.94% |
| 256 | 2.583 | 1.834 | 11.365 | 16.012 | 1.409x | 29.02% |
| 512 | 4.008 | 3.060 | 14.649 | 19.187 | 1.310x | 23.65% |
| 1024 | 7.187 | 5.884 | 16.341 | 19.961 | 1.221x | 18.13% |
| 2048 | 13.051 | 11.143 | 17.997 | 21.080 | 1.171x | 14.62% |
| 4096 | 24.495 | 21.752 | 19.178 | 21.596 | 1.126x | 11.20% |
| 8192 | 47.602 | 42.531 | 19.737 | 22.090 | 1.119x | 10.65% |

Bandwidth = T*K*H*2/t, T per GPU, K=8, H=7168, BF16; decimal GB/s.
Each variant: 2 rounds x 20 samples per batch; each sample is the maximum across 8 ranks.
Latency is the mean of all 40 samples per batch. Speedup = baseline latency / optimized latency.

## 测量口径

DP attention8 / EP8，256 experts，distinct top-k8，hidden7168，BF16 输入输出与 wire。C32、每 channel Q=L=256、双 stage、source fusion、warp specialization、pipelined F2、direct NUMA、expanded output、alignment1。正常性能测试不注入 prefix 延时，不启动 padding 清零。

每个版本两轮；每轮每个 batch 先 warmup5，再测量20次。t 是40个样本的均值，每个样本取8个 rank 的同设备 CUDA-event 完整 dispatch 时长最大值。t 包含源端路由、count/prefix、通信、本地专家展开、统计与所需 stream 等待；CPU 诊断、Gloo 统计和文件写入在计时区外。未锁 GPU 时钟。

带宽分子为每 GPU 的 `T*8*7168*2` 字节，不做 token 去重、不排除本地专家、不乘 GPU 数，除以 t 后再除1e9得到十进制 GB/s。这是逻辑带宽，不是 PCIe 物理总线流量。新 benchmark JSON/CSV 的主字段为 `logical_GBps`；wire/host-bus 字段仅保留作诊断。

T16 时延降低61.04%，T8192 降低10.65%，最大 batch 的逻辑带宽从19.737提升到22.090 GB/s。它们是两项改动共同作用的端到端结果，未进行分项性能归因。

## 正确性、重叠与数据路径

- 8卡逐行 CPU reference、单卡、空输入、不均匀 token、unique/expanded、homogeneous fallback、多 buffer 与 consumer lifetime 均通过；稳定 source/token 顺序逐行检查。
- 新定向测试58项通过；在20ms测试用 prefix 延时下，292个有发送的 rank 观测和1452个有发送的 channel 观测均满足 `prefix_start < first_payload_done < prefix_end`。该完成点覆盖全部TX payload warp及metadata；不比较不同GPU时钟。延时只用于正确性证明，不用于上述性能测量。
- F2回归212项通过，含FP8、NUMA relay、pipeline_stages=2/3、动态/缓存Graph和强制读写延时；额外relay18个epoch通过。
- 小ring三组Q/L=1/2、2/1、3/2，各1000次/卡，共3000次/卡。T8192/H7168核对6,247,485,440个有效BF16元素。
- eager与cached Graph各8rank的profiler审计均未发现主路径CPU同步、scalar readback或D2H。初始化、测试诊断、capture准备和close不属于此主路径保证。
- 两个版本四个性能round的288项NVLink Tx/Rx计数增量均为零；验证窗口同样为零。源码、原子指针来源和编译后SASS审计通过：payload仍为GPU SM→mapped CPU内存→GPU SM，新增CAS仅作用于本GPU输出池控制区。

同配置T17/H7168的小规模profiler诊断中，40个rank/iteration观测的 `begin_epoch` GPU duration中位数从331.728µs降至4.752µs。含padding时，uncached由10个kernel减至9个，cached由8个减至7个；移除的是前置prefix launch，其工作在transport内部继续执行。这些trace用于解释结构，不能替代上表的完整事件计时。

## 复算与溯源

[匿名样本](benchmarks/data/prefix_tx_samples.json)保存两个版本各400个max-rank样本、3200个逐rank样本；[JSON](benchmarks/data/prefix_tx_comparison.json)与[CSV](benchmarks/data/prefix_tx_comparison.csv)可离线复算。[runtime指纹](benchmarks/data/prefix_tx_runtime_source_sha256.json)记录实际测量的CUDA/Python源码SHA256，[验证摘要](benchmarks/data/prefix_tx_validation.json)记录测试与计数结论。最终文档和结果文件在测量后补充，不能把整棵文档树的变化解释成内核变化。

```bash
python3 benchmarks/report_prefix_tx.py \
  --samples benchmarks/data/prefix_tx_samples.json \
  --output-dir /tmp/prefix_tx_report
```

新实验可运行 `PCIE_EP_F2_RUN_ID=my_run bash scripts/benchmark_f2.sh all`；比较两组原始round JSON可使用报告工具的 `--baseline` 与 `--optimized` 参数。完整原始机器日志、trace、SASS与环境字段保留在忽略的 `results/prefix_tx/`，不进入公开源代码。
