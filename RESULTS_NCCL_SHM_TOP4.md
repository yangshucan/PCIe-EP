# 固定非均匀 top-4：PCIe-EP 与 NCCL SHM

2026-10-29，同机八卡、同一份冻结 token 路由重测。两种偏斜下均为 **T≤128 时 NCCL 更快，T≥256 时 PCIe-EP 更快**。T8192 相比 NCCL 完整 dispatch 分别加速 **1.661× / 1.600×**；相比纯 payload AllGather 分别加速 **1.621× / 1.559×**。

本次生产 dispatch kernel 未改动，26 个运行时源码文件的 SHA256 与此前控制路径优化版本一致。修改的是固定输入、测试入口与带宽汇总；不将旧均匀 top-8 的数据混入新表。

## 路由依据

[NVIDIA TensorRT-LLM 官方文档 Table 1](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html#offline-ep-load-balancer) 给出 DeepSeek-R1/256 experts/top-8/EP32 无 EPLB 的专家 mean=128、std=164.1、(max−mean)/mean=10.948。由此取 CV 校准值≈1.282、max/mean≈11.948。它们是跨层跨迭代汇总指标，不能恢复完整概率向量；本次 K4/EP8 输入为参考这些指标的合成数据，**不是真实模型 trace**。

另保留用户指定的 51/256 专家占 80% 采样权重质量的热点对照。两组都使用固定专家 ID 置换，没有刻意集中或平衡每张 GPU 的热点。详细概率构造、抽样方法及局限见 [固定路由说明](docs/pcie-ep/fixed-routing.md)。实际 T8192 统计如下：

| 分布 | 专家 CV | 专家 max/mean | 最热51专家实际分支份额 | 每卡展开 max/mean | 每卡 unique max/mean | 跨卡 unique 投递 / AllGather |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA 统计校准 | 1.28458 | 12.000 | 50.44% | 1.315 | 1.252 | 41.68% |
| 20%/80% 热点 | 1.49412 | 4.108 | 79.59% | 1.432 | 1.330 | 41.62% |

热点组的实际热门专家份额为 79.59%，不是人为强制配平的 80%。专家负载偏斜与 GPU 负载偏斜是两种统计：calibrated 最热专家达到 12×平均，并不意味着最热 GPU 也达到 12×平均。

## 配置与固定输入

DP attention8、EP8、256 experts、topK4、H=7168、BF16；八张 H20D、PCIe Gen5 x16、双 NUMA。PyTorch 2.13.0+cu130 / CUDA 13.0 / NCCL 2.29.7。PCIe-EP：C32、Q/L256、stages2、source fusion、warp specialization、pipelined F2、NUMA direct、alignment1、eager、do_cpu_sync=False。NCCL 保留算法/协议/通道默认值，日志实际确认为 **Ring / Simple / 4 active channels / SHM/direct/direct**。

具体 top-4 ID、分支顺序和 FP32 权重已在 CPU 离线生成并保存为 NPZ；较小 T 取每个 rank 的 T8192 前缀。计时循环不随机选专家。三种实现、两轮测试共享完全一致的路由、权重、BF16 token 内容；两种偏斜之间的 token 内容也相同。准备、统计和验证均在计时外。

每组两轮，每轮 5 warmup + 20 measured；第二轮反转三种实现的执行顺序。每次取八 rank CUDA-event 时延最大值，再对40个值取平均。未锁时钟、未启用 profiler。带宽全部按 **T×4×7168×2/t/1e9 GB/s**；无 world-size 倍数，也不扣去重分子。此为逻辑带宽，不是物理 PCIe 字节吞吐。

NCCL payload 只传 BF16 输入，属于通信基线；NCCL dispatch 计入 GPU pack、payload/routes/weights AllGather、稳定专家展开。K4 的 packed row 为14384 bytes。完整 NCCL 基线输出 expert-ready tensor，不包含 PCIe-EP reusable handle/combine 接口。

## NVIDIA 统计校准：十档结果

| Tokens/GPU | PCIe-EP ms | NCCL SHM payload ms | NCCL SHM dispatch ms | EP GB/s | NCCL payload GB/s | NCCL dispatch GB/s | EP speedup vs full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.250 | 0.125 | 0.129 | 3.672 | 7.316 | 7.096 | 0.518x |
| 32 | 0.265 | 0.158 | 0.184 | 6.930 | 11.630 | 9.971 | 0.695x |
| 64 | 0.357 | 0.246 | 0.269 | 10.270 | 14.892 | 13.618 | 0.754x |
| 128 | 0.530 | 0.454 | 0.467 | 13.852 | 16.169 | 15.727 | 0.881x |
| 256 | 0.887 | 0.932 | 0.951 | 16.548 | 15.743 | 15.440 | 1.072x |
| 512 | 1.609 | 2.138 | 2.172 | 18.252 | 13.732 | 13.515 | 1.350x |
| 1024 | 3.011 | 4.253 | 4.384 | 19.500 | 13.808 | 13.393 | 1.456x |
| 2048 | 5.585 | 8.560 | 8.760 | 21.028 | 13.720 | 13.406 | 1.568x |
| 4096 | 10.973 | 17.323 | 17.741 | 21.405 | 13.559 | 13.240 | 1.617x |
| 8192 | 21.381 | 34.652 | 35.509 | 21.971 | 13.557 | 13.229 | 1.661x |

## 20%/80% 热点：十档结果

| Tokens/GPU | PCIe-EP ms | NCCL SHM payload ms | NCCL SHM dispatch ms | EP GB/s | NCCL payload GB/s | NCCL dispatch GB/s | EP speedup vs full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.250 | 0.120 | 0.144 | 3.673 | 7.662 | 6.388 | 0.575x |
| 32 | 0.265 | 0.157 | 0.183 | 6.924 | 11.676 | 10.002 | 0.692x |
| 64 | 0.359 | 0.249 | 0.274 | 10.213 | 14.747 | 13.407 | 0.762x |
| 128 | 0.563 | 0.455 | 0.466 | 13.032 | 16.129 | 15.743 | 0.828x |
| 256 | 0.906 | 0.932 | 0.955 | 16.207 | 15.751 | 15.377 | 1.054x |
| 512 | 1.676 | 2.145 | 2.193 | 17.516 | 13.685 | 13.390 | 1.308x |
| 1024 | 3.132 | 4.282 | 4.383 | 18.749 | 13.713 | 13.398 | 1.399x |
| 2048 | 5.912 | 8.568 | 8.766 | 19.864 | 13.706 | 13.397 | 1.483x |
| 4096 | 11.342 | 17.415 | 17.788 | 20.708 | 13.488 | 13.205 | 1.568x |
| 8192 | 22.318 | 34.793 | 35.700 | 21.049 | 13.501 | 13.158 | 1.600x |

最后一列为 NCCL 完整 dispatch 时延 / PCIe-EP 时延，大于1表示 PCIe-EP 更快。NCCL 的传统 busbw 单独按 7×T×H×2/t 计算，K4 时等于本表 payload 逻辑带宽的 7/4。

## 结果解读

T8192 的两组跨卡 unique 投递量几乎相同，但热点组每卡 unique 最大/平均由 1.252 增至1.330，PCIe-EP 时延增加 4.38%。这与目的卡负载不均导致尾部等待的方向一致，尚不能作为细粒度 profiling 的因果结论。NCCL AllGather 仍对所有目标传输，纯 payload 时延约34.7–34.8ms，对专家概率变化不敏感。

PCIe-EP 只向实际需要 token 的 GPU 投递，接收后再按专家展开；这两组的去重后跨卡 unique 投递数量约为 AllGather 的41.6–41.7%。大 batch 可利用这部分流量节省。小 batch 则仍受多 kernel、flag 握手和启动成本影响，NCCL 更快。不能将这两组优势外推到所有真实 router trace。

PCIe-EP 两轮同 T 均值差：校准组≤0.991%，热点组≤0.61%。小 batch NCCL 有轮间波动，热点 T16 完整路径两轮均值差约17%；但两轮的胜负边界都保持在 T256。JSON 同时保留40个八卡最大值、各 rank 原始样本、p50/p95与两轮均值。

## 验证与复现

每组480项 rank 输出验证完成，包括320项完整专家展开检查；所有 BF16 内容、counts、weights、source/token/branch 元数据与稳定输出顺序通过独立 CPU reference。两组各16份 NCCL 日志对应64条数据连接，全部为 SHM；各288个 NVLink Tx/Rx 数据计数器增量均为0。生产主路径保持 GPU stream completion、do_cpu_sync=False。GPU 测试后八卡均为空闲。

热点组第一次尝试在第二轮启动时测试容器被移除；该次不完整尝试全部排除，热点表来自新容器中从头完成的两轮。原始诊断记录保留于私有 results，不进入匿名表。

57项相关 CPU 测试通过：固定输入生成/加载及篡改拒绝19项、计时汇总/SHM解析/路径检查边界/driver38项。独立复算两组 raw rounds 与匿名结果逐字段一致。

| 分布 | 固定路由 manifest | NPZ SHA256 |
|---|---|---|
| NVIDIA 统计校准 | [nvidia_calibrated_routes.json](benchmarks/data/top4_skew/nvidia_calibrated_routes.json) | `1abaa4737fb53e4fdf0bcec544fb630968533b37c1b04237266036128fa6c2be` |
| 20%/80% 热点 | [hotspot_20_80_routes.json](benchmarks/data/top4_skew/hotspot_20_80_routes.json) | `71e2645fbb106bb548a6153466d062abcc41b7bdd902cfdea546d6d906131de9` |

- [校准组原始匿名样本与完整统计](benchmarks/data/top4_skew/nvidia_calibrated_comparison.json) / [CSV](benchmarks/data/top4_skew/nvidia_calibrated_comparison.csv)
- [热点组原始匿名样本与完整统计](benchmarks/data/top4_skew/hotspot_20_80_comparison.json) / [CSV](benchmarks/data/top4_skew/hotspot_20_80_comparison.csv)
- [路径证据汇总](benchmarks/data/top4_skew/path_audit.json)、[测量时源码指纹](benchmarks/data/top4_skew/measured_source_sha256.json)、[软件指纹](benchmarks/data/top4_skew/software_sha256.json)

```bash
bash scripts/benchmark_nccl_shm.sh results/top4_nvidia \
  --topk 4 --routing-fixture benchmarks/data/top4_skew/nvidia_calibrated_routes.json
python3 benchmarks/report_nccl_shm.py \
  --samples benchmarks/data/top4_skew/nvidia_calibrated_comparison.json \
  --route-fixture benchmarks/data/top4_skew/nvidia_calibrated_routes.json \
  --output-dir /tmp/top4_recomputed
```

重测时直接加载仓库中的既有 JSON/NPZ；不要重新抽样替换。host/IP、机器路径和原始进程日志没有写入这些公共数据。
