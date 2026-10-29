# Dispatch 细粒度 profiling 与改进方案

后续进展：本文的 P0 控制路径优化与 P1 counts mailbox guard 合并已实现并通过测试，当前结果见 [控制路径优化报告](../../RESULTS_CONTROL_PROGRESS.md)，协议见 [control-progress](control-progress.md)。本文保留优化前的 profiling、诊断消融和原始数据，文中的“当前”指该次被测版本。

测量日期：2026-10-29。被测默认算法：`4597caf`（begin_epoch 并行化、RX prefix 与 TX 重叠后的版本）。本次增加的是可关闭的诊断工具，默认传输协议没有采用消融实验中的优化。

## 结论

当前大 batch 中，**高频、串行的 mapped-host abort 全表检查是已被消融验证的主要开销**，其中 RX 角色的累计检查时间明显更高。RX 数据通路和 ACK 排空是下一步候选；它们之间的因果优先级尚未完全确定。仅减少 kernel launch 次数或提高 HBM/scatter 吞吐，无法解释本次观察到的大幅改善。

- T=8192 默认 dispatch 约 **42.53 ms、22.09 GB/s**；同一次时间线中最慢 rank 的 transport 约占完整 dispatch 的 **98.7%**。
- 每个 RX 控制 warp 在每 256 轮检查一次全部 264 个 abort flags。诊断版中，每 channel 累计检查时间的中位数约 **15.78 ms / 40 次**，约占该 RX 角色存活时间的 41%。这是角色内的实测时间，不等于删除后必然节省 15.78 ms。
- TX 控制的 `tx_drained` 从头重复扫描最后一圈 ACK，累计约 **32.94 ms / 34 次、17,397 次 ACK load/channel**。它与 RX 重叠，不能与上面的 15.78 ms 相加。
- 无埋点消融把 transport 的全表检查周期由 256 轮改为 4096 轮，T=8192 降到约 **29.0 ms**；叠加 drain cursor 约 **27.4 ms**。这证明控制面存在很大的可优化空间。这两项结果是诊断实验，**没有作为默认性能发布，也没有完成生产故障响应验收**。
- 增加 channel 确实有利于大 batch，但当前支持范围是 1–32。C8→C32 使 T=8192 从约 99.2 ms 降到 42.5 ms；小 batch 趋势相反。应先处理随 channel 数增长的重复系统内存读取，再扩展 CTA 数量。

## 配置与测量口径

8 × H20D，DP attention=8、EP=8、E=256、K=8、H=7168、BF16。每 GPU 输入 T 为 16/32/64/128/256/512/1024/2048/4096/8192；随机且每 token 专家不重复，seed=20261024。C32、payload/descriptor slots 均为 256、本地流水深度 2、source fusion 与 warp specialization 开启、pipelined F2、NUMA direct、expanded output、alignment=1、eager；不清 padding。

正式时延使用无 profiler 的 CUDA events：每轮取 8 rank 的最大值，再求轮间均值。基线两轮独立进程，每轮 5 warmup + 20 measured，共 40 个 iteration-max 样本/T。

主带宽严格采用用户指定口径：

```text
bw = T * K * H * 2 / t / 1e9   GB/s
```

不乘 world size，不乘去重率。物理 host payload 读写量和该逻辑带宽另列：T8192 基线约为 124.4 GB/s 的八卡 aggregate host payload 读+写，不能把它和单卡逻辑 22.09 GB/s 混用，也不能直接当作某条 PCIe 链路的总线吞吐。

数据路径仍为 GPU SM → mapped host RAM → GPU SM，无 peer mapping、GPU P2P、NVLink 或 NCCL 数据面。Gloo 只用于初始化、计时外协调和结果收集。主路径 `do_cpu_sync=False`；trace 的稳态 marker 内没有 CPU sync、GPU scalar readback 或关联 D2H。profiling 完成后的错误检查和读回显式放在 marker 外。

## 无 profiler 基线

| T / GPU | 平均时延 ms | 逻辑带宽 GB/s |
|---:|---:|---:|
| 16 | 0.680 | 2.697 |
| 32 | 0.703 | 5.222 |
| 64 | 0.811 | 9.048 |
| 128 | 1.021 | 14.382 |
| 256 | 1.826 | 16.082 |
| 512 | 3.052 | 19.243 |
| 1024 | 5.878 | 19.978 |
| 2048 | 11.177 | 21.014 |
| 4096 | 21.749 | 21.599 |
| 8192 | 42.527 | 22.092 |

编译期 profiling 关闭后，又对十个 batch 完整回测；数据见 [匿名测量摘要](../../benchmarks/data/dispatch_profile_20261029.json)。没有将带埋点时延混入这张表。

## 分层定位结果

### 1. Kernel 时间线：大 batch 应先改 transport

使用 PyTorch/CUPTI 对真实八卡并发运行采集 T16、T256、T8192，每卡 10 次 dispatch。采集启动后额外执行两次 dispatch，并在 marker 外完成 GPU 等待与 CPU barrier，减少 profiler 初始化造成的首轮偏斜。按 CUDA runtime correlation 把 kernel 关联到 CPU marker，实际每次为 8 个 kernel。

下面为全 rank × iteration 的 **单 kernel 时长中位数，单位 µs**。不同 kernel 的中位数不可直接相加或除以 max-rank event 均值来推导严格贡献率；分析器另保留每轮最慢 rank 的完整 breakdown。

| Kernel | T16 | T256 | T8192 | 作用与解释 |
|---|---:|---:|---:|---|
| begin_epoch | 4.74 | 4.99 | 5.02 | 清设备诊断、推进 epoch、并行检查 terminal flags |
| route_flags | 5.02 | 22.66 | 25.76 | token 路由、GPU mask |
| stable_ordinals | 2.08 | 2.78 | 40.67 | 稳定 expert/source/token 行号与计数 |
| prepare_channels | 76.77 | 79.47 | 82.16 | channel compaction、histogram、旧 counts mailbox ACK |
| publish_counts | 14.78 | 14.98 | 15.14 | 发布 rank/expert counts 和 READY |
| transport_pipelined | 527.86 | 1,625.34 | 41,598.76 | prefix、TX、RX、scatter、ACK/credit、故障检查 |
| copy_local_tokens | 11.71 | 25.34 | 342.16 | 本 GPU 专家分支展开 |
| reduce_statistics | 15.94 | 17.78 | 22.40 | 聚合 channel 诊断与 transport 时间戳 |

T8192 同轮临界 rank 的 transport 均值为 41,957.95 µs，完整 event 均值为 42,511.82 µs，即 98.7%。stable scan 的复杂度虽高，当前仅约 41 µs；本地展开约 342 µs。外围 fusion 对小 T 有意义，无法单独解决大 T 的几十毫秒。

小 T 仍需关注 prefix 与 mailbox。GPU 内部 RX prefix prepare/wait 中位数约 T16 409 µs、T256 441 µs、T8192 549 µs。此区间包含等待其他 rank counts、owner 处理及其检查；本次没有进一步分离这些来源，不能把整段称为 prefix 算术时间。

### 2. Warp 角色：控制读取与 RX 流水线是重点

诊断版 `handle.profiling` 为 GPU int64 `[C,11,32]`。11 个 warp 分别为 TX control、RX control、3 TX workers、3 RX producers、3 scatter consumers。每角色只在退出时由 lane 0 写自己的 HBM row；没有新增协议 barrier/fence/atomic。abort 与 drain 每次调用完整计时，worker/probe/availability 每 32 次采样。完整 schema 见 [埋点说明](profiling-instrumentation.md)。

以下 T8192 数字为 8 rank × 20 iterations × 32 channels，即 **5,120 个 channel 观测的中位数**，不是某个最慢 CTA 的非重叠阶段分解。worker 区间列出各类 worker0 的“每 channel 采样均值的中位数”，没有把三个合作 warp 的时间合池。

| 观测 | 实测 | 含义 |
|---|---:|---|
| TX control 总存活 | 40.45 ms | 包含发送、等待所有消费者及等待 RX |
| RX control 总存活 | 38.92 ms | 包含 prefix、descriptor、stage/ACK/scatter 回收、故障检查 |
| RX abort 检查总时间 | 15.78 ms / 40 次 | 每次串行读取 8×(1+32)=264 个 mapped signals |
| TX abort 检查总时间 | 0.423 ms / 1 次 | TX/RX loop 速度不同，触发频率不等 |
| TX drain 总时间 | 32.94 ms / 34 次 | 与 RX 读取/检查并发，主要是重复扫描与等待消费者 |
| TX drain ACK loads | 17,397 / channel | 远多于只确认一次所需的 endpoint ACK 数 |
| TX worker0 的 sampled copy+fence/join | 9.41 µs / ticket | 三 warp 合作同一个 token，不能把三份时长相加 |
| RX producer0 的 sampled host-copy+fence/join | 18.48 µs / ticket | 当前数据通路比 scatter 慢 |
| scatter consumer0 的 sampled copy+fence/join | 5.01 µs / ticket | 不是全 HBM 带宽饱和的证据 |

`copy` 时间只描述 leader 观察到的 issue/依赖边界；`fence` 区间包含原有 warp join。因此 copy+fence 更适合比较，不能把它等同于纯 PCIe TLP 时延或纯硬件 fence 时延。

当前 Q256/C32、T≤8192 时每 channel 最多 256 个输入 token，不触发 payload ring 的本轮槽位回绕；descriptor 也不需要本轮容量回绕。ACK 成本主要集中在最终 drain。扩大 ring 不能直接解决这里的主要问题。

RX descriptor-not-ready 的 channel 中位数为 0，local-stage-busy 观测很多；这说明很多时候 descriptor 已有，接收端还在等待本地 stage 完成。busy 是**次数**，不能换算成等待时间占比，也不能把所有 busy 归因于 scatter。RX copy、ACK 回收和控制 warp 的检查都会影响 stage 复用。

### 3. 硬件指标：没有整体带宽满载的证据

Nsight Systems 2026.4.1，GH100 metrics，10 kHz。只统计每轮八卡 transport 时间区间的交集，合并八个 GPU metric source，避免初始化和大部分外围 kernel。

| GPU 级指标，平均 % | T16 | T256 | T8192 |
|---|---:|---:|---:|
| SMs Active | 16.17 | 22.90 | 23.42 |
| SM Issue | 2.98 | 5.18 | 5.22 |
| DRAM Read | 0.08 | 0.43 | 1.00 |
| DRAM Write | 0.39 | 1.23 | 1.01 |
| PCIe RX | 5.35 | 21.17 | 25.43 |
| PCIe TX | 1.36 | 5.10 | 8.58 |

C32 最多覆盖 132 SM 中的 32 个，即 24.24%；SM active 约 23.4% 与此吻合。但 active 包含轮询，不能理解成高效 payload 执行。SM issue 约 5.2% 和低 HBM 吞吐支持“等待/控制受限”的判断。

PCIe 百分比是 profiler 自己的硬件归一化指标，不是用户逻辑带宽；八卡平均值也不能排除某个 PCIe root complex、NUMA 跨 socket 链路或瞬时突发受限。本次没有获得跨 NUMA fabric/每 root-port 的独立计数，因此不把 NUMA direct 的剩余开销全部归因于 UPI 或 CPU DRAM。

采样中的全部 NVLink 指标为 0；正式基线及后续验证窗口的 NVLink 计数差分也为 0。

## 因果消融与 channel 扫描

这些实验使用 `4597caf` 的隔离副本，均不带 GPU 计时埋点。正式源码没有保留多份历史 transport 实现。每项每轮 5 warmup + 20 measured，代表 T16/T256/T8192；消融第二轮反转运行顺序。数值不用于替换十 batch 默认基线。

| 无埋点变体，40 samples/T | T16 ms | T256 ms | T8192 ms | T8192 GB/s |
|---|---:|---:|---:|---:|
| 默认基线 | 0.680 | 1.826 | 42.527 | 22.092 |
| drain_once | 1.189 | 2.094 | 42.580 | 22.065 |
| drain_cursor | 1.211 | 2.108 | 41.041 | 22.892 |
| abort_4096 | 0.688 | 1.446 | 28.963 | 32.438 |
| cursor_abort_4096 | 0.678 | 1.409 | 27.344 | 34.360 |

T8192 两轮分别为：drain_once = 42.389/42.771 ms；drain_cursor = 41.029/41.053 ms；abort_4096 = 28.981/28.946 ms；cursor_abort_4096 = 27.353/27.335 ms。

变更定义：

1. `drain_once`：`tx_finished` 置位后不再调用 drain，TX-control 继续存活到 RX 完成并处理故障。
2. `drain_cursor`：在上一项上，成功确认一个尾部 token 后推进本 channel cursor，不再从第一项 ACK 重扫。诊断版仍可能重读当前 token 已确认的 endpoints；后续 remaining-mask 可进一步优化。
3. `abort_4096`：仅把 transport 中 `(++rounds & 255)` 改成 `(++rounds & 4095)`；prefix/mailbox 检查没有改变。
4. 组合：cursor + abort_4096。

`abort_4096` 同时改变 TX/RX 的 transport 检查周期，没有单独做 RX-only/TX-only 消融；“RX 更重”来自角色计时，不能把组合收益全部归到 RX 一项。

单独去掉 drain 冗余，小 batch 反而恶化：TX 空转更快，而故障检查按 loop 次数触发，可能更早/更频繁进入几百微秒的全表扫描。组合后改善。因此不能机械地把某个 busy loop 删掉，再把所有后果归因于 payload。正式方案应同时解决等待节奏、检查频率和有界故障响应。

| Channels / CTAs | T16 ms | T256 ms | T8192 ms |
|---:|---:|---:|---:|
| 8 | 0.429 | 3.471 | 99.201 |
| 16 | 0.470 | 2.242 | 61.198 |
| 24 | 0.583 | 1.861 | 48.901 |
| 32 | 0.680 | 1.826 | 42.527 |

C8/C16/C24 每项为单轮 20 samples，C32 为基线两轮 40 samples。

当前 Python/native 配置上限均为 C32。C64 请求被配置校验拒绝，没有实际 GPU 测量；没有把 C64/C128 写成已测结果。C8/C16 对小 T 更好、对大 T 更差，适合后续做按 workload 选 channel 的候选，不能据这三个 T 就发布自动选择规则。

## 开发改进顺序

| 阶段 | 具体改动 | 预期收益与约束 | 验证与准入 |
|---|---|---|---|
| P0-1 | warp 协作扫描 abort；用 GPU 时间或明确工作量限制检查节奏 | 先解决 264 次串行依赖读取；所有 lane 必须在一致分支参与，不能在仅 lane0 分支内直接调用 warp 集体操作。保留 timeout 与 mapped abort 单写者 | 对照本次 4096 消融；检查 p50/p95 故障发现与退出时间，禁止依靠无限延后检查获得分数 |
| P0-2 | 单调 drain cursor + remaining endpoint/relay mask；完成后不再扫描 ACK，保留有界健康检查并等待 RX | 每个匹配 ACK 只确认一次。保留所有 NUMA domain 的 relay_drained，包括零工作 domain；TX-control 仍负责 abort 直到 RX 完成 | 小 Q/L 回绕、多 epoch、慢消费者、relay replica ACK、abort；十 T 均不退化后才能合入 |
| P0-3 | 减少同 GPU 重复健康扫描，考虑一个已驻留 warp 维护 device-local health/epoch | 当前各 channel、TX/RX 重复扫描随 C 增长近似 O(P×C²)。可先由每 channel TX 负责并向 RX 发本地 abort，再评估每 GPU monitor | 证明 monitor 先驻留且存活到全部参与者退出；避免等待未调度 CTA；验证 terminal error 永不清除 |
| P1-1 | RX active-source mask、warp 协作 probe；调整 2/3 stage、TX/RX worker 配额 | 用固定 expert/source/token ordinal 保持输出顺序；优化有用的 outstanding payload 读取，减少已耗尽 source 探测 | 先补 issue→host-readers-done→ACK→stage-free ticket trace；独立 sweep，避免同时改多项 |
| P1-2 | 小 T 的 prefix/mailbox 快路径；统一旧 counts mailbox 可写检查 | prepare 的约 80 µs 与 prefix 约 0.4–0.55 ms 对小 T 重要。先分离 owner 的 counts wait/abort scan/fence，不直接删除 counts_consumed guard | 包含零 token、零 channel 流量、unequal T、跨轮 mailbox 复用和 prefix 人为延迟 |
| P1-3 | 处理控制面的 C 扩展成本后，再支持更多 CTA、调整 token/channel 映射 | C32 只有 32 个 SM；更多 CTA 有提高 RX outstanding 的可能，但目前没有 >32 的实测证据 | 每 root complex/NUMA 分组统计、C32/64/128 实测；同时关注 host pinned footprint、metadata/flag traffic |
| P2-1 | 仅本地 scatter 的 fence scope 审计、ACK/descriptor 批处理 | sampled scatter ~5 µs，小于 RX ~18.5 µs，优先级低于控制面。不能全局删除 system fence；TX payload/metadata→READY、host readers→ACK 的跨 PCIe 顺序必须保留 | 用内存顺序证明和强制 delay/slot reuse 验证；descriptor credit、payload ACK、tile FREE 三种生命周期仍分开 |
| P2-2 | route+channel fusion、稳定 histogram/prefix 算法、本地复制向量化、finalize fusion | 减少小 T 固定开销。stable expert 全量 scan 为 O(E×T×K)，可用 tile histogram + tile prefix + 稳定 tile 内 ordinal；不能改成抢占式 atomic 行号 | 按位比对 expert/source/token 顺序，独立比较 cached/uncached/Graph；大 T 收益上限受当前 transport 占比约束 |
| P2-3 | NUMA relay 与 direct 做独立拓扑对照 | 当前结果仅 direct；后台 relay 增加 CPU 内存读写和线程调度，不能假定一定更快 | 分 NUMA 的 pinned 放置和 CPU affinity、relay queue 深度/延时；传输主路径仍不做 CPU sync |

所有候选默认保持 PCIe-only、发送/接收去重、稳定输出顺序、GPU 侧无锁握手、主路径无 CPU 同步。每一步单独提交并独立 A/B，避免把控制面、CTA 数量和 F2 stage 数同时修改后无法归因。

## 扰动、证据边界与复现

同一 focused driver 的完整 GPU 埋点开/关对照如下，每项 20 iteration-max samples：

| T | 埋点关 ms | 埋点开 ms | 相对变化 |
|---:|---:|---:|---:|
| 16 | 0.777 | 0.760 | -2.21% |
| 256 | 1.844 | 1.848 | +0.21% |
| 8192 | 42.659 | 42.747 | +0.21% |

小 T 的约 -2% 不是加速结论；本次没有对埋点开关做多轮交替置信区间检验。focused driver 自身的 marker/诊断提交节奏也不同于正式 benchmark，因此不要用该表替换正式性能。

发布版 transport 静态资源为 64 registers/thread、944-byte stack；诊断版为 116 registers/thread、960-byte stack。静态 `LOCAL:0` 和 stack 数值不能证明动态 spill 有/无。埋点会改变指令与调度，尤其小 T；正式性能一律取无埋点版本。worker sampled 时间也不能当作 release 的逐指令精确时延。

当前 Nsight Compute 为 2025.3.1，CLI 没有新版本的 communicator/lockstep 参数。通信 kernel 必须与其它 GPU 同时推进，并修改 mapped-host epoch/ACK；本次没有对它做不安全的单 kernel 多 pass replay。因此**没有声称取得 long-scoreboard、membar、local-memory spill 或源行级硬件 stall 的实测比例**。若要进一步验证这些假设，应先升级并核实支持 mandatory-concurrent 的采集模式，再采少量硬件指标；不能照搬最新版选项到旧工具。参见 [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/) 和 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)。

在已具备八卡 CUDA/PyTorch 环境的机器，从仓库根目录执行：

```bash
# 真实配置的无埋点十 T 基线
torchrun --standalone --nproc-per-node=8 benchmarks/benchmark.py \
  --warmup 5 --iterations 20 --output results/profile/release

# 独立编译 cache，GPU 内 role 计时
PCIE_EP_PROFILE=1 TORCH_EXTENSIONS_DIR=/tmp/pcie-ep-role-cache \
  torchrun --standalone --nproc-per-node=8 benchmarks/profile_dispatch.py \
  --profile-kind gpu --tokens 16 256 8192 --warmup 5 --iterations 20 \
  --output results/profile/roles

# 发布版 CUPTI kernel 时间线；不要和 nsys 同时开启 CUPTI
PCIE_EP_PROFILE=0 TORCH_EXTENSIONS_DIR=/tmp/pcie-ep-release-cache \
  torchrun --standalone --nproc-per-node=8 benchmarks/profile_dispatch.py \
  --profile-kind torch --tokens 16 256 8192 --warmup 5 --iterations 10 \
  --output results/profile/torch

python benchmarks/analysis_dispatch_profile.py \
  --torch-dir results/profile/torch --roles-dir results/profile/roles \
  --output results/profile/analysis.json

# 单个代表 T 的八卡并发 Systems + GPU metrics
PCIE_EP_PROFILE=0 TORCH_EXTENSIONS_DIR=/tmp/pcie-ep-release-cache \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
  --gpu-metrics-devices=all --gpu-metrics-set=gh100 --gpu-metrics-frequency=10000 \
  -o results/profile/nsys_T8192 \
  torchrun --standalone --nproc-per-node=8 benchmarks/profile_dispatch.py \
  --profile-kind nsys --tokens 8192 --warmup 5 --iterations 10 \
  --output results/profile/nsys_data

nsys export --type sqlite --output results/profile/nsys_T8192.sqlite \
  results/profile/nsys_T8192.nsys-rep
python scripts/analyze_nsys_metrics.py results/profile/nsys_T8192.sqlite \
  --iterations 10 --output results/profile/metrics.json
```

GPU metrics 为设备级采样，运行时需排除其它 GPU 工作负载；不将其与其它计数器 profiler 同时采集。100 µs 的采样间隔尤其限制小 T 指标的精度。没有锁 GPU 时钟，所有候选仍需在匹配条件下复测。

本次新增回归通过：诊断版正确性 22 cases、诊断版 prefix/cache/Graph 58 cases、无埋点正确性 22 cases；四个消融副本各 22 cases，均为八卡参考校验；CPU API 39 tests。同步后重建远端默认 cache，另通过八卡 11-case smoke，资源仍为 64 registers/thread。稳态 trace 为 3T×8rank×10次，全部通过 CPU sync/scalar/D2H 审计。最终无埋点十 T 回测 T8192=42.493 ms、22.110 GB/s。两段最终 NVLink 审计均完整读取 288 个计数并确认全部差分为 0。消融正确性通过不等于其延迟检测/超时策略已满足生产故障验收。

原始 trace、SQLite、role rows、运行日志和隔离消融源码放在未跟踪的 `results/profiling_20261029/`，公开数据只保留匿名数值和相对源文件哈希，不含机器名、IP、内部路径、GPU UUID。后续开发优先读 `advanced_dispatch.cuh` 的 control/drain、`advanced_common.cuh` 的 abort helper，以及 `fused_receive.cuh` 的 RX producer/consumer，再读本报告的验收项。
