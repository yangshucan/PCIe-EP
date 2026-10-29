# NCCL SHM AllGather 与 PCIe-EP 性能对照

> 本文保留均匀 top-8 对照。最新固定非均匀 top-4 结果见 [RESULTS_NCCL_SHM_TOP4.md](RESULTS_NCCL_SHM_TOP4.md)。

2026-10-29，同机八卡、相同 token / routes / weights 实测。**T≤256 时 NCCL SHM 更快；T≥512 时当前 PCIe-EP 更快。**

T8192 的新测 PCIe-EP 为 **26.607 ms / 35.311 GB/s**，NCCL 纯 AllGather 为 **34.914 ms / 26.910 GB/s**，PCIe-EP 加速 **1.312×**、时延降低 **23.79%**。包含动态路由展开的 NCCL 基线为 **36.018 ms / 26.085 GB/s**，PCIe-EP 加速 **1.354×**、时延降低 **26.13%**。T16 则是 PCIe-EP 267.5 µs、纯 AllGather 123.2 µs、完整 NCCL 路径 136.8 µs，小 batch 仍没有优势。

本轮已停止占用 GPU 的模型服务，确认八卡空闲后顺序测试。PCIe-EP 也重新测量，未混用上一份控制路径优化报告的旧样本。源码和报告已同步本地及远端，生产 dispatch 本身没有改动。

## 配置与计时

DP attention8、EP8、256 experts、topK8、hidden7168、BF16，每卡 T=16/32/64/128/256/512/1024/2048/4096/8192。八张 H20D，PCIe Gen5 x16、双 NUMA。软件为 PyTorch 2.13.0+cu130、CUDA 13.0、NCCL 2.29.7。实际库、扩展及测量源码 SHA256 见 [软件指纹](benchmarks/data/nccl_shm_software_sha256.json) 和 [源码指纹](benchmarks/data/nccl_shm_measured_source_sha256.json)。

PCIe-EP 为 C32、Q/L256、stages2、source fusion、warp specialization、pipelined F2、BF16 wire、NUMA direct、expanded alignment1、eager、do_cpu_sync=False。NCCL 主表使用默认算法、协议、通道与 host allocator；实际调度日志确认 **Ring / Simple、4 个 active channels、SHM/direct/direct**。

各版本每轮 5 warmup + 20 measured，共两轮；第二轮反转三条路径的执行顺序。每个 iteration 取八 rank CUDA-event 时延最大值，再对40个最大值求均值。初始化、CPU reference、Gloo 协调、验证和错误读回均在计时外；不锁 GPU 时钟，不启用 profiler 或 GPU role 埋点。正式数据与诊断 trace 独立。

所有路径带宽统一为用户指定的每 GPU 逻辑专家数据量：

```text
bw = T * 8 * 7168 * 2 / t / 1e9 GB/s
```

不额外乘 GPU 数或去重率。它不等于实测 PCIe、CPU 内存或 UPI 的物理字节吞吐。

## 十档正式结果

| Tokens/GPU | PCIe-EP ms | NCCL SHM payload ms | NCCL SHM dispatch ms | EP GB/s | NCCL payload GB/s | NCCL dispatch GB/s | EP speedup vs full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.268 | 0.123 | 0.137 | 6.859 | 14.893 | 13.417 | 0.511x |
| 32 | 0.291 | 0.156 | 0.186 | 12.617 | 23.459 | 19.733 | 0.639x |
| 64 | 0.412 | 0.249 | 0.262 | 17.798 | 29.493 | 27.978 | 0.636x |
| 128 | 0.657 | 0.457 | 0.475 | 22.352 | 32.107 | 30.877 | 0.724x |
| 256 | 1.048 | 0.934 | 0.964 | 28.027 | 31.438 | 30.454 | 0.920x |
| 512 | 1.966 | 2.148 | 2.230 | 29.865 | 27.332 | 26.329 | 1.134x |
| 1024 | 3.700 | 4.288 | 4.410 | 31.737 | 27.390 | 26.631 | 1.192x |
| 2048 | 7.095 | 8.573 | 8.838 | 33.106 | 27.396 | 26.578 | 1.246x |
| 4096 | 13.541 | 17.447 | 18.004 | 34.693 | 26.926 | 26.093 | 1.330x |
| 8192 | 26.607 | 34.914 | 36.018 | 35.311 | 26.910 | 26.085 | 1.354x |

最后一列为 NCCL 完整路径时延 / PCIe-EP 时延，大于1表示 PCIe-EP 更快。纯 AllGather 的加速比在 T512/1024/2048/4096/8192 分别为 1.093/1.159/1.208/1.288/1.312×。

T8192 两轮时延分别为：PCIe-EP 26.597/26.617 ms，NCCL 纯 AllGather 35.000/34.828 ms，NCCL 完整路径 36.308/35.728 ms。[匿名原始 rank 样本、p50/p95、轮均值及全部指纹](benchmarks/data/nccl_shm_comparison.json) 与 [CSV](benchmarks/data/nccl_shm_comparison.csv) 可独立重算。

## 完全一致的 token 路由

直接复用 `benchmarks/benchmark.py` 的 routes()：CPU seed 为 `20261024 + rank*104729 + T`，先选不同的 top-8 专家，再用同一 generator 生成并归一化 FP32 weights。每档仅创建一次 x、idx、weights，三条路径共用相同的 GPU tensor；x 使用原来的 GPU generator seed `20261024+rank`。

逐 rank 对 int64 expert IDs、FP32 weights、BF16 payload 的原始字节计算 SHA256，再按 rank 顺序合成总指纹。三种路径、两轮，以及补充的32通道和Simple实验，输入指纹全部一致；并核对 [原始路由 fixture](benchmarks/data/nccl_shm_routes.json)。topK 的槽位顺序也一致。

## 两种 NCCL 工作量

| 路径 | 计时内工作 | 完成后的有效输出 |
|---|---|---|
| PCIe-EP | 动态路由、计数/prefix、去重通信和本地展开 | expert/source/token/branch 顺序的专家输入及现有 handle |
| NCCL payload | BF16 [T,H] 一次 AllGather；每 token 发送一份 hidden state | [8T,H] 全部源 token，尚未按专家展开 |
| NCCL dispatch | GPU pack → 一次 uint8 AllGather → GPU 动态计数、prefix、稳定 scatter | 相同 expert-ready payload、expert IDs、weights、来源元数据、counts/offsets |

完整 NCCL 路径每 token 打包 BF16 payload、8个 int64 expert IDs 和8个 FP32 weights，共14,432 bytes，比纯 payload 多96 bytes（0.67%）。打包每轮计时，没有提前广播或缓存路线，没有按 topK 重复发送 token。三个本地展开 kernel 使用有序 tile scan、prefix 和128-bit向量拷贝，不依赖 CPU count readback。该最小基线不提供 PCIe-EP 的 unique layout、可复用 handle 或 combine API。

纯 AllGather 是通信下界，完整 NCCL 路径用于比较相同 expert-ready 输出。NCCL-tests 的 AllGather algbw 用 P*send_bytes/t，busbw 用 (P-1)*send_bytes/t；本次 P=K=8，纯 payload 的 algbw 恰与主表逻辑带宽数值相同。[NCCL-tests 带宽定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allgather)

## 通道和协议敏感性

为排除只比较并发数或默认协议的影响，另测完整十档、各两轮的两组补充实验。T8192 结果：

| NCCL 配置 | 对应新测 PCIe-EP ms | 纯 AllGather ms | NCCL 完整路径 ms |
|---|---:|---:|---:|
| 默认自动配置 | 26.607 | 34.914 | 36.018 |
| MIN/MAX_NCHANNELS=32 | 26.582 | 39.964 | 41.087 |
| NCCL_PROTO=Simple | 26.555 | 34.855 | 35.979 |

设置 MIN/MAX_NCHANNELS=32 后 communicator 初始化32个通道；实际每次 collective 根据消息大小使用28–32个 active channels，大 batch 为32个。该配置的大 batch 比默认4通道慢约14%，不能用增加通道数解释为必然提速。[32通道完整样本](benchmarks/data/nccl_shm_c32_comparison.json)

显式 Simple 与默认的大 batch 结果相差不足0.2%，实际调度日志确认默认本就选择 Simple。[Simple 完整样本](benchmarks/data/nccl_shm_simple_comparison.json) 两组补充均保持 T≤256 NCCL 更快、T≥512 PCIe-EP 更快。没有将不同配置逐 batch 的最优值拼接成主表，也未穷举所有 NCCL 参数。

协议采用每次 collective 的 INFO/TUNING 日志判断，不能根据 profiler 的 `ncclDevKernel_AllGather_RING_LL` 符号判断 LL。该版本将不同 AllGather 组合映射到共用入口，再按实际 funcId 调度。[NCCL 入口映射](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/generate.py#L139-L150)，[实际算法/协议日志](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L740-L743)

## PCIe-only 与正确性证据

显式禁用 P2P、NVLS、MNNVL、CollNet、IB，并启用 SHM。额外设置 **NCCL_NET_DISABLE_INTRA=1**：初次校验发现仅设置 P2P_DISABLE/SHM_DISABLE，NCCL仍将跨NUMA的部分数据连接选为 NET/Socket；那组诊断数据未进入正式结果。该参数关闭节点内 NET 选择，保留 bootstrap 网络发现。[NCCL 拓扑选择源码](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/graph/paths.cc#L492-L519)

- 主实验16份日志、两个完整八 rank communicator，共64条数据连接，全部 SHM/direct/direct。
- 32通道补充共512条数据连接，Simple补充共64条，全部 SHM；逐连接匿名记录和真实算法/协议选择见 [路径审计](benchmarks/data/nccl_shm_path_audit.json)。
- 三组正式测试各自覆盖全部288个 NVLink Tx/Rx payload counters，前后增量均为零。没有关闭物理NVLink、没有重置计数；计数器分辨率为KiB。
- 每组均校验全部源 payload 字节 SHA256、独立 CPU 路由 reference、所有有效专家输出的逐元素 BF16、weights、expert IDs、counts/offsets 和稳定来源顺序。
- 独立 trace 覆盖 T16/T8192 × 三条路径 × 八 rank，共144个 steady markers；CPU sync、GPU scalar readback、相关 D2H 均为零，允许 GPU stream wait 依赖。[匿名 trace 审计](benchmarks/data/nccl_shm_profile_audit.json)
- CPU reference 24例、helper CPU 模型198例，以及报告/审计边界/日志解析30个 CPU 测试通过。报告拒绝不同输入、不同配置、缺失样本或未按八卡最大时延计算的数据。

## 为什么大小 batch 结论不同

小 T 时，当前 PCIe-EP 仍有8个 eager kernel 以及 epoch、mailbox、layout、prefix、ACK和polling的固定成本；NCCL 纯 AllGather 仅一个通信 kernel，完整基线为5个 kernel。T16 的 PCIe-EP 仍慢于完整 NCCL 路径约1.96倍，后续应继续降低固定协议和 launch 开销。

大 T 时，精准路由和去重减少了实际搬运。给定这套路由，远端 token/GPU deliveries 只有 AllGather 全远端广播的65.6%–66.6%；T8192 为303,584 / 458,752 =66.18%。PCIe-EP 每源 token 只写一次 host buffer，目标GPU只读一次，再在GPU本地展开；AllGather 需要让全部GPU获得全部token。该通信量差异与具体实现共同形成实测优势，不能将1.31/1.35倍全部归因于单个kernel效率。

这些结论限定于当前硬件、路由分布、BF16、PCIe-only、NCCL SHM/direct/direct 和本轮三组参数。更强的路由局部性、不同拓扑、使用NVLink或其他NCCL调优需要重新测量。

## 复现与同步

从仓库根目录运行；入口会检查八卡空闲，生成INFO日志、两轮数据及路径审计：

```bash
bash scripts/benchmark_nccl_shm.sh results/nccl_shm

NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 \
  bash scripts/benchmark_nccl_shm.sh results/nccl_shm_c32

NCCL_PROTO=Simple bash scripts/benchmark_nccl_shm.sh results/nccl_shm_simple

python3 benchmarks/report_nccl_shm.py \
  results/nccl_shm/round1.json results/nccl_shm/round2.json \
  --output-dir results/nccl_shm/summary
```

无需 GPU，也可从公开匿名样本重算主表：

```bash
python3 benchmarks/report_nccl_shm.py --samples benchmarks/data/nccl_shm_comparison.json \
  --output-dir /tmp/nccl_shm_report
```

主表及两组补充的匿名逐 rank 样本保留在 benchmarks/data/。原始日志、trace、完整执行manifest和阶段诊断位于本地与远端未跟踪的 results/ 中，公开文件不含机器名、IP、GPU UUID 或内部绝对路径。
