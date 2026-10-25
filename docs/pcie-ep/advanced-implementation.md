# S5–S8 实现与验收记录

本文件说明 S4 之后的可选路径。配置与源码实现同步更新；交付总结见 [S5–S8 实测与验收](../../RESULTS_F2.md)，完整测量数据另见 高级性能报告（本地生成：`../../results/s5_s8/report/REPORT.md`），只有实际执行过的结果才作为验收证据。旧 S1–S4 kernel 与结果保留供回归比较。

## 配置与阶段

| 阶段 | 配置 | 行为 |
|---|---|---|
| S5 | `channels=1..32` | 每 channel 独立 source payload ring、目标 descriptor/credit 和 ACK；一个 token 只由一个 channel 发送 |
| S5 | `warp_specialization=True` | TX/RX control 各一个 warp，TX/RX payload 各三个 warp，分别推进本地有界 stage |
| S5 | `pipeline_stages=2/3` | 当前 payload 与下一 token 的 metadata/credit 准备重叠；独立于 host Q 和 descriptor L |
| S5 | `fuse_receive_expand=True` | expanded 模式从 host 读入 shared tile，再写多个本地专家行；unique 模式保持 compact 输出 |
| S5 | `receive_expand_mode` | scalar 原 F2；vectorized 使用 uint4 本地展开；pipelined 再增加 3 个专用 scatter warp，形成 11-warp CTA |
| S6 | 并行稳定布局 | token 并行构造 route flags，按 rank/expert 用 CUB scan 生成稳定 ordinal，按 channel compact 远端 token |
| S6 | `source_fusion=True` | 控制 warp 使用本轮路由和权重，复用必要的 counts/稳定 ordinal，减少源端中间 metadata 依赖 |
| S6 | cached handle、Graph | 不变布局可复用；Graph 在 capture 外分配，执行时由 GPU 生成新 epoch |
| S7 | `wire_dtype="fp8_e4m3"` | 输入/输出为 BF16；远端 token 使用 E4M3FN 字节和每 token 一个 FP32 scale；默认仍为 BF16 wire |
| S8 | `numa_mode="relay"` | 原生 CPU worker 将 source staging 复制到目标 NUMA 的 replica ring，每个目标 NUMA 一份 |

`payload_slots` 和 `descriptor_slots` 是每 channel 容量，`replica_slots` 是每 source/channel/目标 NUMA 的副本容量。增大 channels 同时增大总 host 窗口；消融实验须区分固定总容量与固定每 channel 容量。修改通道数、容量或 NUMA 模式必须排空并重建 buffer。

F2 新增两条路径的角色表、READY 代际与 shared-memory 上限管理见 [F2 实现补充](./f2-receive-expand.md)，最新流水版实测见 [RESULTS_F2.md](../../RESULTS_F2.md)，带宽按 `T*K*H*b/t` 计算。原 specialized helpers 的标量算法仍保留，安全加固由各路径共同使用。

## S5 发布与回收

每个 progress CTA 对应一个 channel。源 token 分片和输出布局是两套索引：channel 决定谁搬运，原始 `(source_rank, token_index, topk_slot)` 的稳定 ordinal 决定输出行。多个 CTA 不通过 grid barrier 等待未驻留的 CTA。

控制 warp 快照路由、权重与输出位置，检查 ring credit，给 payload worker 提交任务。所有 metadata/payload writer 完成自己的 system fence 后，通过 CTA 域的 release/acquire 交接；control 最后发布 host READY。接收控制先 acquire READY 并快照 descriptor，再归还 descriptor credit。所有必需 host 字段与 payload 读完后才归还 host ACK；shared stage 要等全部本地 reader/专家 scatter 完成才 FREE。这三种回收彼此独立。

收发角色的循环不使用要求全 CTA 参加的 barrier。TX 等待 credit 时，RX 仍可搬运和回 ACK。完整 dispatch completion 覆盖全部本地输出、发送 ACK、接收任务和本地 stage 排空。测试专用延迟用于验证最后一个 writer/reader 的完成条件。

## S6 缓存与执行状态

S5 保留单线程稳定路由作为参考；S6 使用逐 token flag、每 rank/expert 的 CUB BlockScan 和 per-channel compact。该并行化没有改变 source/token/slot 的输出顺序，也没有消除 counts/prefix 的必要 GPU 执行边界。

可缓存的是路由布局，不是 epoch、sequence 或 ACK。缓存须绑定 buffer generation、group/owner、原始 route tensor 的身份、shape、版本和 alignment；路由修改后必须重新生成布局。每次 dispatch 仍读取当前 x 与 top-k weights。Graph 固定容量和指针，路由动态变化时必须在 replay 内重新构建布局；静态缓存版本须拒绝失效路由。

输出池、CUDA stream 依赖与 handle 生命周期沿用 S4 的语义。Graph 的固定输出池与 eager 池分离，复用前等待已登记的 consumer stream；初始化和 capture 的同步不进入正常 dispatch/replay 的稳态区间。

## S7 传输格式

每个非零 token 的 FP32 scale 为 `max(max(abs(x.float())) / 448, 2**-126)`，全零行 scale 为 1；下限避免 BF16 极小值的 FP32 倒数溢出；归一化值按 E4M3FN 最近偶数舍入并做有限值饱和，接收后乘 scale 并转换回 BF16。格式要求有限 BF16 输入；NaN/Inf 没有约定的量化精度保证。scale 和 payload 使用同一个 READY/ACK 生命周期。本地目标直接使用原 BF16，因此 reference 必须区分本地旁路与远端量化路径。

BF16 wire 的有效字节为 `B=H*2`；FP8 wire 为 `B=H+4`，包含 scale，不包含 metadata 和对齐空隙。FP8 的数值误差与通信内容错误分别检查，压缩后带宽和 BF16 等效带宽分别报告。

## S8 两级握手

源 GPU 只写自己的 NUMA staging。对于同 NUMA 目标，GPU 直接读取源 ring；对于其他 NUMA，source descriptor 指向目标 NUMA endpoint。绑定 CPU 核的原生 worker acquire 该 READY，在目标 NUMA 的 replica ring 写一份 payload、scale 与 metadata，并发布各目标 GPU descriptor。

复制完成后 worker 写 source `COPIED_ACK`；源 slot 在本地直接 reader ACK 与所有目标 NUMA 的 COPIED_ACK 完成后可以复用。replica slot 要等该副本全部目标 GPU 的 `CONSUMED_ACK`。worker 发布本轮 drained epoch 之前必须等副本排空，GPU completion 也必须观察所有 channel/NUMA flow 的 drained（包括零任务），防止下轮覆盖 CPU 尚未消费的 counts mailbox，不能把“复制完成”当成“接收完成”。

replica payload 的每个 slot 独占 4KiB 对齐页，由初始化的 `mbind` 放在目标 NUMA；metadata、scale 和控制页仍在源 NUMA。实际驻留位置通过 `move_pages` 查询，受权限限制时使用 `get_mempolicy(MPOL_F_NODE|MPOL_F_ADDR)` 查询相同采样页，不用 policy 值冒充实际位置。

CPU/GPU 共享的 mapped signal 使用自然对齐的 acquire/release load/store；不在 mapped host 内存做跨设备 RMW。多 CTA 的统计和错误竞争只使用本 GPU 内存原子操作。销毁先排空并协调所有 rank，再停止/join worker，最后 unregister/unmap。

relay 继续保持 `do_cpu_sync=False`：调用线程不等待 GPU counts 或 CPU copy；数据前进依赖后台 CPU worker。它适合跨 socket 读延迟较高的机器，是否有收益由 direct/relay 对照测量决定。

## 测量口径

对每轮用同 GPU CUDA event 测完整 dispatch，取八个 rank 的最大值，再统计多个样本。transport timer 取所有 channel 的最早开始和最晚结束，包含轮询/背压，不能称为纯 PCIe 硬件带宽。

设 U 是源端至少有一个远端目标的 token 数，F 是去重后的 `(source token, destination GPU)` 数，R 是 relay 复制的 `(source token, destination NUMA)` 数，均对八卡求和：

- 用户可见通信量为 `F*B`，通信带宽为 `F*B / 完整 dispatch 时延`。
- GPU 访问 host 的 payload 字节为 `(U+F)*B`；这是算法字节，不是总线计数器。
- relay 额外 CPU payload 读写为 `2*R*B`，另列，不并入 GPU 有效通信带宽。

Gloo 只用于初始化、测试 reference 和计时区间外的统计。CUDA event 的 CPU 等待、错误诊断、数据核对和文件写出放在计时区间外。所有主性能表使用 BF16 wire；FP8、缓存和 Graph 作为独立对照。

PCIe-only 验证同时审查 host mapped 指针来源、传输指令和完整 NVLink 计数：8 GPU × 18 link × Tx/Rx 共 288 项必须齐全且增量为零。物理 NVLink 可以保持启用，但数据路径不使用 peer GPU pointer 或 NVLink/NCCL payload 通信。
