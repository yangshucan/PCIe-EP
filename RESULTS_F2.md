# 历史流水 F2：通信时延与带宽

最新构建的重新测量及基线对比见 [RESULTS_CONTROL_PROGRESS.md](RESULTS_CONTROL_PROGRESS.md)。[RESULTS_PREFIX_TX.md](RESULTS_PREFIX_TX.md) 保留中间版本；以下保留原始 F2 实验供溯源。

2026-10-25，8 张 H20D。仅列当时的 `receive_expand_mode="pipelined"` 的结果。
沿用已完成的两轮 CUDA-event 测量，按修正后的公式重算带宽，没有重跑 GPU 实验。

## 计算口径

`bw = T × K × H × b / t`，其中 T 为每张 GPU 的输入 token 数，K=8，H=7168，BF16 的 b=2 字节。
t 使用完整 dispatch 的平均时延（秒）；GB/s 采用十进制单位。分子不做去重、不排除本地专家分支，也不额外乘或除 GPU 数。
这是用户指定的每卡逻辑通信带宽口径。

## 实验结果

| 每卡输入 token 数 T | 完整 dispatch 时延 ms | 带宽 GB/s |
|---:|---:|---:|
| 16 | 1.830 | 1.003 |
| 32 | 1.797 | 2.043 |
| 64 | 1.921 | 3.820 |
| 128 | 2.069 | 7.094 |
| 256 | 2.595 | 11.313 |
| 512 | 4.004 | 14.667 |
| 1024 | 7.156 | 16.411 |
| 2048 | 13.122 | 17.900 |
| 4096 | 24.586 | 19.107 |
| 8192 | 47.692 | 19.700 |

## 配置与测量

DP attention8 / EP8，256 experts，distinct top-k8，hidden7168，BF16。C32、Q=L256/channel、双 stage、source fusion、direct NUMA、expanded output、alignment1。
每轮 5 次 warmup、20 次测量，两轮合计 40 个样本；每次取八个 rank 的同设备 CUDA-event 时长最大值，再求均值。
时延包含布局、counts/prefix、host-staged 传输、本地专家展开与 stream 等待。初始化、CPU reference 和统计汇总在计时外。

保持 GPU SM→mapped CPU staging→GPU SM 的 PCIe-only 路径，主路径 `do_cpu_sync=False`。
原实验的 288 项 NVLink payload counter 增量均为零。原始路径审计和完整验证日志保留在本地，不随公开代码分发。

[结果 JSON](benchmarks/data/f2_pipelined_summary.json) · [CSV](benchmarks/data/f2_pipelined_summary.csv) · [匿名逐 rank 时延样本](benchmarks/data/f2_pipelined_samples.json) · [实现设计](docs/pcie-ep/f2-receive-expand.md)

公开数据只保留实验配置和数值样本，不包含主机名、IP、GPU UUID、网络接口、进程信息、绝对路径、原始日志、trace 或二进制归档。400 个 max-rank 样本与 3200 个匿名 rank 样本保留原值；本次仅脱敏导出并重算，没有重新测量 GPU。

CPU 重算：

```bash
python3 benchmarks/report_latest.py
```

新实验完成后可运行 `python3 benchmarks/report_latest.py --run-dir results/f2/runs/RUN_ID`。详见[公开范围](PUBLICATION.md)。

测量版本说明：上述数值来自清理前已完成的实验（Git commit `6a927b5`）。2026-10-26 的代码清理与回归不将这些历史数值冒充为新构建的重新测量，见 [CLEANUP.md](CLEANUP.md)。
