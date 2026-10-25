# PCIe Host-staged EP Dispatch：可行性与详细设计

2026-10-25：本文保留原始协议推导和阶段目标；当前 S5–S8 的具体路径、FP8 格式、Graph 合同与 NUMA payload/control 放置边界见 [实现记录](./advanced-implementation.md) 和 [API](../../API.md)。

## 1. 结论与范围

该方案在单机、各 GPU 均可映射同一份 pinned host 物理内存的环境中具有实现合理性，适合探索缺少 NVLink/P2P 条件下的 EP。它保留逐 token 精确路由，发送端一次写 host、接收 GPU 一次读 host，再在显存内展开到多个专家。

性能收益是待验证假设。GPU thread 发起的 PCIe host read 具有较高延迟，必须靠合并访问和并发 outstanding read 隐藏；CPU 内存带宽、PCIe root complex、跨 NUMA 链路、flag 轮询都会形成瓶颈。不能仅凭少一次软件发送就断言优于 NCCL 或 DMA。

首个可集成版本（S4）范围：Linux、NVIDIA CUDA、单机、一进程一 GPU、同一 EP group、BF16、任意有效的每 rank token 数（包含零）、固定容量预分配、连续均匀专家 owner。owner 在内部显式表示，非连续自定义 owner 映射作为独立后续扩展；不要求 DP attention group 与 EP group 完全相同，但所有输入源和目标专家必须在当前 EP group 内。跨节点、TP 内 token 重建、专家副本动态选择、combine/backward、FP8、CUDA Graph、多并发 dispatch 分阶段处理。

默认 direct 模式中，CPU 负责初始化、共享内存映射和资源管理。主路径明确采用 `do_cpu_sync=False`：路由、counts 交换、prefix、payload 和 READY/ACK 握手全部由 GPU 完成，正常 dispatch 不等待 CPU 读取计数或确认进度。GPU 发起不等于零 kernel launch，CPU 仍可异步提交工作。可选 NUMA relay 模式增加后台 CPU 数据搬运，仍不在调用线程读回 counts 或等待完成，但数据前进依赖 relay 线程。`do_cpu_sync=True` 仅作为未来可选兼容/调试路径，不属于首版交付要求。

### 1.1 分阶段实现：先简化，再优化

本文第 4–9 节是能力逐步完善后的设计参考，不要求最小原型同时实现完整 ring、fusion、Graph 和 NUMA relay。开发顺序以 [分阶段实现计划](./implementation-plan.md) 为准，阶段编号统一为 S0–S8，替代旧 M0–M6。每阶段在前一版本上增加可独立验证的能力，并保留参考路径。

| 阶段 | 最小新增内容 | 对应本文设计 |
|---|---|---|
| S0 原语验证 | 固定 payload、READY/ACK、双 GPU/多进程映射 | 第 4、6 节的最小子集 |
| S1 最小传输 | 双 GPU、C=1、固定整批 staging，固定位置稀疏输出 | 暂不实现第 5 节 ring，不声称正式 API 兼容 |
| S2 功能基线 | 多 GPU/top-k 去重、GPU counts/prefix、独立专家展开 | 第 3、7 节语义；仍用整批 staging |
| S3 有界流水 | payload/descriptor ring、背压、收发/ACK 推进 F1 | 第 5–7 节最终 direct 协议 |
| S4 集成交付 | DeepEP API/comm、事件/生命周期、压力验收 | 第 2、8 节；首个 BF16 direct 可集成版本 |
| S5 搬运优化 | 多 channel、warp specialization、访问/轮询调优、接收展开 F2 | 第 7.1–7.2、7.5–7.6 节 |
| S6 提交优化 | 源端 F3、cached layout、CUDA Graph，逐项开关 | 第 7.3–7.4、8 节缓存语义 |
| S7 可选压缩 | FP8/scales 与布局优化 | payload ABI 和独立量化验收 |
| S8 可选分级 | CPU NUMA relay 和两级 ring/ACK | 第 4.1–4.6 节 |

S1 为减少变量，按源 token index 预留 Tmax 个 host slot，整批写完再由接收 kernel 读取，输出用 GPU valid mask 标记有效位置；暂不需要 compact counts、descriptor 或 ring credit。S2 增加多 GPU 共享单份 payload、每 peer batch ACK、compact/专家输出，保留整批 buffer 作为后续正确性对照。S3 才把整批存储换成可循环复用的 Q-slot 窗口；T>Q 时必须同时推进收发，不能继续使用“整批发送完再接收”的执行方式。

所有阶段始终保持 `do_cpu_sync=False`、预分配容量和合法发布/回收内存序。最小实现的简化针对性能结构与支持范围，不是先省略同步正确性。S0–S4 是主线交付，S5–S8 是可选增量优化；S7/S8 可直接基于 S4 单独验证，不依赖其他可选优化已完成。具体输入限制、GPU 等待顺序、验收门槛见实现计划各阶段。

## 2. 本地 DeepEP 基线与复用边界

基线目录：`<DeepEP checkout>`；commit `b5a33eefda39485f438dc764695b71ba7b9249b6`。本次仅阅读，未修改该仓库。

| 源文件 | 已观察行为 | 本方案处理 |
|---|---|---|
| `deep_ep/buffers/ep.py:242` | 构造 EPBuffer、创建 comm handle、检查 NVLink、初始化现有 runtime | 保留外层习惯；新增 host-staged runtime，绕开 NVLink/Gin 初始化假设 |
| `deep_ep/buffers/ep.py:562` | dispatch；CPU sync、expand、cached handle、延迟 epilogue | 对齐参数及返回语义，按里程碑逐项支持 |
| `deep_ep/comm/handle.py:44` | 按 group 缓存；存在 `_comm_ptr` 时借用，否则新建 managed NCCL comm | 优先复用，明确 borrowed/owned 生命周期 |
| `deep_ep/utils/event.py:39` | 当前 CUDA stream wait；可运行一次 epilogue hook | 复用事件与延期收尾约定 |
| `deep_ep/include/deep_ep/impls/ep/dispatch.cuh:102`、`:340` | 按目标 rank 去重；现有数据传输依赖 Gin 等 | 参考路由语义；重写 host ring transport |
| `deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh` | recv 后本地专家展开；含 TMA、PDL、固定 layout 假设 | 复用语义，首版用普通 CUDA load/store 单独实现 |
| `deep_ep/include/deep_ep/layout/ep/token.cuh` | payload/scales/topk/source 等打包 | 参考字段，host wire ABI 独立定义 |
| `csrc/buffers/ep.hpp` | host workspace 为每进程分配；主要通信区由 Context 管理 | 不能把现有 mapped workspace 直接当作跨进程共享 payload |
| `tests/ep/test_ep.py`、`deep_ep/utils/refs.py` | dispatch reference、展开折叠一致性、事件组合 | 复用参考语义与测试用例思路 |

建议先增加独立 `HostStagedEPBuffer`，验证后由 `EPBuffer(..., transport="host_staged")` 门面转发；保留原 transport 的默认行为。不能只替换 `gin.put`：原计数通知、barrier、拓扑、显存分配、slot 回收也要替换。

## 3. 路由和去重的精确定义

设 rank 数为 P，源 rank s 的输入为 `x_s[T_s,H]`；每个 token 有 K 个 top-k expert ID 和 weight。

`owner[e] -> ep_rank`；token 身份为 `(buffer_instance, epoch, src_ep_rank, src_token_idx)`。去重按身份进行，不能按向量数值或 token 文本合并；两个 attention rank 即使持有相同内容也视为不同 token，除非上层明确提供一致的全局身份及去重协议。

对源 token t：

```text
D(t) = unique(owner[e] for valid e in topk[t])
R(t) = D(t) - {s}
if R(t) nonempty: host payload 写一次
for d in R(t): 发一个 descriptor；GPU d 读 payload 一次
if s in D(t): 直接从 x_s 做本地路径
每个 d 再将一份 unique token 展开给本 GPU 的相关专家
```

例：GPU0 的 token 选了 GPU1 上两个专家、GPU2 上一个专家、GPU0 上一个专家。GPU0 写 host 一次，GPU1/GPU2 各读一次；GPU1 内部展开两行，GPU0 直接处理本地行。源端去重不只是“每个目的 GPU 发一次”，而是“所有远端目的 GPU 共用一次 host 写入”。PCIe 不会因此自动进行硬件 multicast，远端仍分别读取。

输入约束：`-1` 表示无效路由；其他负值及超范围 expert 报错；同 token 重复有效 expert ID 首版报错；weight 可以为零但不自动删除路由；weight 与原 top-k slot 绑定。通信去重不能丢失专家分支及其权重。首版不在 dispatch 时把 weight 乘到 x，展开返回每专家行 weight，由后续专家流水线按约定应用一次。

接收有两种布局：

- `do_expand=False`：每 `(s,t,d)` 一行 `recv_x`，返回本地化 top-k ID（非本地位置为 -1）及对应权重。
- `do_expand=True`：按本地专家分段，每有效 `(s,t,e)` 一行；先读进 GPU unique buffer，再本地 scatter。每专家 `align(n_e,A)` 容量，实际 count 单独存，padding 不可当有效 token。

首版稳定布局规定：unique 按 `(src_ep_rank,src_token_idx)`；展开按 `(local_expert,src_ep_rank,src_token_idx)`。布局预处理生成确定位置，避免原子到达顺序引入不稳定。后续可加非确定模式。内部 `expert_offsets[E_local+1]` 和 `expert_counts[E_local]` 分开保存；对外适配本地 DeepEP 的 `psum_num_recv_tokens_per_expert` 特殊编码，不能假定其永远是普通 exclusive prefix sum。

## 4. 共享内存与拓扑

单进程 PoC 可使用 `cudaHostAllocMapped` 并为不同 device 获取映射。正式多进程采用 POSIX shared memory 或 memfd：创建物理共享 backing -> ftruncate -> mmap(MAP_SHARED) -> NUMA 绑定/first-touch -> 各进程 `cudaHostRegister(..., cudaHostRegisterMapped)` -> `cudaHostGetDevicePointer`。注册与注销必须按进程配对；初始化期间交换 SHM 名字或经 Unix socket 传 FD，不能把整数 FD 直接广播当成另一个进程的 FD。

`cudaHostAllocPortable` 表示 CUDA context 可用性，不会自动把分配变成多进程共享内存。CUDA device IPC handle 也不是共享 host memory 的替代品。不同进程的 CPU/GPU 虚拟地址可以不同，消息仅传 segment ID、offset、长度、generation，不能传裸指针。

默认 direct 模式中，每 source rank、每 channel 独立 payload ring，其物理页放在靠近 source GPU 的 NUMA node。这样保持全远端一次 host payload 写；读方可能跨 socket。可选 NUMA relay 模式为被路由到的远端 NUMA node 建副本，源 GPU 仍只写一次，但会增加 CPU host-to-host 复制，必须单独统计，详见第 4.1–4.6 节。

控制区与 payload 分页布局，控制字段至少 8 字节自然对齐，按 64/128 字节隔离不同 writer，实际 cacheline 参数可调；不要使用 write-combined 控制页。启动检查 canMapHostMemory、注册能力、memlock/SHM 配额、GPU UUID 与 PCI BDF、NUMA、EP rank 到设备映射、同机共享 backing 可达性。共享主机不共享 IPC namespace/SHM 权限时明确拒绝。

### 4.1 可选 NUMA relay：按目标内存域复制

模式分为 `numa_mode="direct"`（默认，无稳态 CPU payload 搬运）与 `numa_mode="relay"`（显式启用的后台 CPU 搬运）。不默认启用动态自动切换；模式和路由规则在初始化时集体确定。

双 NUMA 示例：源 GPU 靠近 NUMA0，远端多个目标 GPU 靠近 NUMA1。源 GPU 写 NUMA0 一份 payload，relay 在收到通知后主动复制到 NUMA1 一份，目标 GPU 只轮询本地副本就绪标记，不先尝试远端 payload read：

```text
源 GPU -> NUMA0 SourceRing[s,c] -> NUMA0 本地目标 GPU
                    |
          RELAY_READY（发布给 relay 的源端通知）
                    |
             CPU relay 跨 NUMA 复制一次
                    v
            NUMA1 ReplicaRing[s,c,NUMA1]
                    |
               REPLICA_READY
                    v
          NUMA1 各目标 GPU 各读一次 -> 本地专家展开
```

“NUMA0/NUMA1 各一个 ring”是内存域划分的简化表述：实际每个 NUMA 内存池中按 `(source_rank, channel)` 分片；源端 `SourceRing[s,c]` 放在 `source_numa(s)`，远端 `ReplicaRing[s,c,n]` 放在目标 NUMA n。每分片仅有一个 producer，避免一个全局多生产者 ring。双向通信有不同 source 分片；推广到多个 NUMA node 时，只为本 token 真实命中的远端 node 复制一份，不广播给无目标 node。

这里以实际 GPU-to-memory locality 为依据，不假设“一 socket 等于一 NUMA node”。初始化记录 GPU PCI BDF、NUMA distance 和内存节点；目标 GPU 的归属不明确时要求显式配置或退回已验证的 direct 模式。

### 4.2 副本不依赖自动缓存同步

两个 ring 是不同物理内存，不存在硬件自动维护的 payload 副本一致性。relay 是显式复制服务；每份 payload 发布后只读，到对应 ACK 收齐再复用，不实现一般性的可变缓存或双写同步。NUMA 页放置负责分配位置，不能代替副本复制与发布。[Linux NUMA 内存策略](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)

共享 backing 的 NUMA policy 必须按实际分配方式验证：先放置/first-touch，再 pin/register；用系统页位置查询确认结果，不仅根据线程绑核推断。CPU relay 读写的 payload 和控制区采用 CPU 可正常缓存的内存，首版不使用 write-combined 页或非临时流式 store；引入这些优化前需另行证明 fence 与 GPU 可见性。

### 4.3 两级通知、ACK 与 slot 生命周期

源端与副本 ring 有独立容量、sequence、credit 和消费者集合。副本 header 保留原 `(buffer_instance, epoch, src_rank, src_token_idx)` 身份，并额外包含源 payload sequence、副本 sequence 与目标 NUMA。不能把源 slot index 当作副本 slot index，也不能复用旧副本 READY。

1. 源 GPU 非阻塞检查源 payload 空位，以及本地 GPU descriptor 和各目标 NUMA relay descriptor 的 credit。一次写完整 payload 后，分别 release 发布通知。远端 GPU 不再接收 direct descriptor；同一目标不能同时走两条路径。
2. 每个目标 NUMA 的 relay worker acquire 观察源通知，检查副本 ring 空位；无空位时继续推进其他队列及副本回收，不占着发送循环无限等待。
3. relay 从源 ring 复制 payload、top-k/weight、ordinal 等后续必需元数据到本地副本。保留本 NUMA 目标集合，且预先取得这些 GPU 通知队列的 credit，避免发布半套无法收尾的事务。
4. CPU 复制和必要内存序完成后，relay release 发布各目标 GPU 的 `REPLICA_READY`。目标 GPU acquire 观察到本次 sequence 后只读取本地副本。
5. relay 确认不再读取源 payload/header 后，release 写独立 `RELAY_COPIED_ACK[s,c,source_slot,n]`。此后不能再访问旧源 slot。源 GPU acquire 收齐本地直接消费者 ACK 和所有目标 NUMA 的 COPIED_ACK 后即可回收源 slot，无需等远端专家展开完成。
6. 各目标 GPU 完整读取副本及必要元数据后，分别 release 写 `REPLICA_CONSUMED_ACK[s,c,n,replica_slot,d]`。接收/展开 fusion 可在读进 shared 后提前 ACK，但 completion event 仍须覆盖输出写入。relay acquire 收齐本副本目标集合内所有 ACK 后，才覆盖副本 slot。

各队列 descriptor credit 仅表示 descriptor 不再被访问，与上述 payload ACK 分开。无目标 NUMA 不参与复制/ACK；源 GPU 上的本地专家仍直接读原输入 HBM。每个 COPIED_ACK 和 CONSUMED_ACK 均保持单 writer，禁止跨 GPU/CPU 共享 refcount RMW。

源 slot 已释放不代表本轮远端接收完成。每个 relay worker 在本 epoch 所有通知处理完、所有副本被目标 GPU ACK 后，发布 `RELAY_DRAINED_EPOCH`；本地 completion 除原收发条件外，还须 GPU acquire 观察本 rank 发往各目标 NUMA 的 drained epoch。这样保留第 7 节“本 rank 所发数据均已被目标接收”的完成语义，且销毁或后续 epoch 不会提前复用 relay 状态。

### 4.4 CPU/GPU 内存序与前进性

第 6 节原子访问封装扩展为 host/device 共用的 `load_acquire` / `store_release`。CPU 与 GPU 必须访问同一个自然对齐 64-bit 控制对象，验证目标平台上的 CPU/GPU 原子互操作、对象生命周期和编译后指令；不通过把任意 `std::atomic` 与 CUDA atomic 类型互相 reinterpret 来假设兼容。禁止 CPU 普通 load/store 与 GPU 原子访问同一控制字段，禁止以 volatile 或普通 memcpy 完成就代替 READY 的发布语义。

新增 litmus：GPU payload 写 -> CPU acquire 读取/复制 -> CPU release 发布 -> GPU acquire 读取；反向还需验证 GPU ACK -> CPU acquire 回收、CPU COPIED_ACK -> 源 GPU acquire 回收。正文第 6 节直接 GPU-GPU 协议的测试不能替代这些跨 CPU/GPU 测试。

首版每目标 NUMA 设固定数量的 native relay worker，绑定该 node 的 CPU core，轮询多个 source/channel 队列；一个分片固定归属一个 worker。worker 运行于 C++ runtime，稳态不依赖 Python/GIL、不为每 token 分配内存。采用有界任务和轮询退避，在批处理吞吐与小消息等待间调参；NUMA0/NUMA1 双向搬运互不等待对方先排空整个 ring。

relay 不调用逐 token CUDA synchronize、D2H count 读回或 dispatch 调用线程上的完成等待，`do_cpu_sync` 仍为 False。不过 GPU 会等待 CPU relay 发布 READY，因此该模式依赖 CPU 调度和 CPU 内存复制，不能宣传为“稳态完全不需要 CPU 参与”。初始化启动 worker、退出时 drain/join 可以有 CPU 协调；故障诊断可在正常主路径外同步。

relay 卡顿、进程退出或副本 ring 满会产生背压，必须纳入 GPU timeout 和 peer error。错误槽保持单 writer；中途失败不静默切换 direct，不伪造 COPIED/CONSUMED_ACK；销毁前确保 GPU 与 relay 均不再访问共享页，再 unregister/unmap。

### 4.5 配置、容量与流量模型

配置草案（新增设计，尚未实现）：

```python
HostStagedConfig(
    channels=1, payload_slots=256, descriptor_slots=256,
    numa_mode="relay",                 # 默认是 "direct"
    replica_slots=256,                  # 每 (source, channel, target NUMA)
    relay_descriptor_slots=256,
    relay_workers_per_numa=1,
)
```

CPU core affinity 与 GPU->NUMA 映射可显式配置并导出到拓扑报告；ring 大小按在途窗口调优，不等于 dispatch 总 token 数。副本 payload 容量为 `sum_(s,c,n!=source_numa(s)) Q_replica[s,c,n] * aligned_token_bytes`，只包含配置可达的目标 NUMA；另计两级 descriptor、ACK、worker 和 drained epoch 控制区。Q_source/Q_replica/L 可不同，不能假设同大小就能共享 sequence。

设 token t 在远端 NUMA n 上有 m_n 张目标 GPU，payload 大小为 B：direct 的逻辑远端 GPU payload 读取为 `m_n*B`；relay 的跨 NUMA payload 复制为 B，再由 m_n 张 GPU 本地读取 `m_n*B`。relay 每个目标 NUMA 增加一次 host payload 读和一次 host payload 写；源 GPU 的原始 host 写仍是一份。该模型不计缓存命中、预取、write-allocate、控制消息和拓扑差异，实际跨 socket 链路及内存流量以硬件测量为准。

relay 在多个远端目标 GPU 共用 payload 时可能受益；m_n=1 或小消息时，额外 CPU copy、轮询与两级发布可能增加时延。提前搬运需要足够流水窗口：relay 必须在目标消费前有机会推进，不能保证每次 GPU 轮询时副本已经就绪，也不能消除首次数据的跨 NUMA 延迟。

### 4.6 与融合、接口及 Graph 的关系

NUMA relay 是 transport 内的分级复制，不改变 token 身份、top-k 路由、GPU 接收去重或专家权重。F2 接收融合读取的地址由 direct 的源 ring 切换为本 NUMA 副本 ring，随后仍可经 shared memory 展开；源端 F1/F3 要把远端通知视作 relay 队列，并遵循两级完成语义。counts/prefix 保持 GPU 驱动，首版仍直接通过原 mailbox 握手，不声称全部控制流量也已 NUMA 本地化。

Python 五项返回值与有效 GPU counts 不变；runtime/handle 记录 numa mode、topology generation，切换模式须排空并重建，旧 handle 失效。CPU relay 在 capture 之外启动，Graph 运行时持续推进；Graph replay 的 epoch/sequence、超时和 worker 生存期需专项测试，不能因 direct 支持 Graph 就自动宣称 relay 也支持。

## 5. 队列组织：一份 payload，多份引用

本节描述默认 direct 模式。relay 模式用第 4.3 节两级 ring/ACK 扩展；每个 ring 内仍遵守单 payload producer、每消费者独立 ACK 和版本化 sequence。

每 `(s,c)` 一个单生产者、多消费者 payload ring，容量 Q token slots。每 `(s,d,c)`（s != d）一个 SPSC descriptor ring，容量 L。GPU s 的 channel c 只有一个 producer owner；GPU d 的 channel c 只有一个 consumer owner。多个 warp/CTA 不得绕过 owner 并发写同一队列。

```text
source GPU s ---- 一次 payload 写 ----> host PayloadRing[s,c]
                                          ^          ^
                                          |          |
                              GPU d1 一次读      GPU d2 一次读
                              unique[d1]        unique[d2]
                                  |                 |
                            本地专家 scatter   本地专家 scatter

DescriptorRing[s,d,c]：payload slot + payload sequence + dst unique offset
                       + epoch + 路由/索引所需引用
Ack[s,c,slot,d]：仅 GPU d 写，GPU s 读
```

slot header：ABI version、buffer generation、epoch、64-bit payload sequence、source token index、payload length、目标 rank bitset/稀疏列表、top-k IDs/weights（后续含 scales）。目标集合可多字 bitset，不能默认只支持 32/64 ranks。

控制字段：descriptor `published_seq`、consumer credit/consumed sequence、每 payload slot 每目标 `ack_seq`。head/tail 的本地副本保存在所属 GPU；跨设备字段只通过 system-scope 原子 load/store 访问。首版 token 一个 slot；后续可按小 batch 发布，但逻辑路由仍逐 token。

队列 credit 与 payload 回收是两件事：descriptor 被消费不等于 payload 已读取。ACK 必须明确对应 payload sequence。slot 仅在其上一次目标集合内所有接收方均 ACK 后复用，不等待非目标 rank。ACK 槽不靠写零重置，读取时比较本次期望 sequence；sequence 从 1 开始，0 为初态，64-bit wrap 前停用并重建实例。

## 6. 内存序与握手

关键区别：自然对齐的 system-scope atomic load/store 与需要跨设备 read-modify-write 的 atomicAdd/CAS 不是同一能力。当前 NVIDIA 内存模型允许满足条件的 mapped memory 对齐原子 load/store；host native atomic 属性影响更广泛的 RMW 能力。因此本设计不把 `hostNativeAtomicSupported==0` 当作 load/store 协议的直接否决条件，也不因此假设任意 host atomicAdd 都可用。实现前对所选 CUDA/CCCL 版本、8 字节对齐映射对象及生成指令完成核对。[NVIDIA CUDA 内存模型](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)、[CCCL 内存模型](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html)。

所有 ready/credit/ack 通过封装的 `cuda::atomic_ref<uint64_t,cuda::thread_scope_system>` load/store 或经审查的等价 PTX 实现。CPU 不并发混用普通 load/store 操作这些字段。不使用 `volatile` 代替 acquire/release；不靠一次 `__threadfence_system()` 修复其他线程尚未完成的写；不默认把 `__ldg` 等只读缓存路径用于可变映射 payload。

发布次序：

1. producer 非阻塞检查 payload slot 可回收，且所有目标 descriptor queue 有 credit；不足则本轮继续处理接收/ACK。
2. 一个 CTA 内的 worker 写完整 payload/header。首版 payload 不跨 CTA 分片。
3. 所有写线程完成各自必要的 system fence，并经过 CTA barrier；publisher 收齐完成关系后，向各目标 descriptor 的 sequence 执行 release store。descriptor 的其他字段必须在该 release 前写完。具体 fence/barrier 组合由 litmus 和 PTX 审查确认。
4. consumer acquire 读到预期 descriptor sequence，再读取 payload。检查 generation、epoch、payload sequence、边界和目的 rank。
5. consumer CTA 合并读取 host 到显存 unique row；所有 reader 完成数据读取及显存落地、完成必要同步后，leader release 写 `ack_seq`。随后 release 归还 descriptor credit。
6. producer acquire 读到旧目标集合全部 ACK 才覆盖 slot。release ACK 同时约束先前读操作，防止 payload 在读取完成前被覆盖。

不引入 host 共享 refcount 或 ACK bitmap 的多写者 RMW。每目标 ACK 独占字段，多接收方无需争抢 `fetch_sub`。所有等待带退避、超时、错误状态和可诊断的 `(s,d,c,epoch,seq)`。

伪代码只表达协议，不能直接复制作为可编译 CUDA：

```text
while local_send_remaining OR local_recv_remaining OR outstanding_payloads:
    progress_incoming_descriptors_round_robin()  # acquire -> copy -> release ACK/credit
    progress_payload_reclaim()                  # acquire all required ACKs
    if next_token_has_all_credits():
        write_one_host_payload_cooperatively()
        join_all_writers_and_order_to_system()
        publish_descriptors_release()
    check_abort_and_timeout()
```

该循环不在单个发送动作内无限等待 credit，否则全体 rank ring 满时无法执行接收。payload ring 有慢消费者背压，因此只能称 mutex-free、有界缓冲协议；不能声称严格 lock-free 或 wait-free。

## 7. GPU 预处理与执行流水

本节描述 S3/S4 的 ring direct 基线；更早的 S1/S2 使用第 1.1 节与实现计划所述整批 staging，尚不套用这里的全部步骤。基线采用较清晰的多 kernel 结构，先保证正确性，再在 S5/S6 融合。

1. **route/layout**：GPU 验证 top-k；构造远端目标集合、每 `(s,d,c)` 去重 token 索引表、专家 counts 和稳定 ordinal。token 固定分配到 channel，源端按 token 顺序处理。小规模可暂存 dense mask，正式路径避免 `T*E` dense routing map，使用 `O(T*K)` 稀疏结构。
2. **count publish**：有限时长 kernel 写共享 count mailbox，包括 epoch、每 channel 每目标 unique 数、每专家 counts；release 发布。每 source 独占其 mailbox，按 collective epoch 串行使用。
3. **count receive / prefix**：读方等待所有 source 的本轮 count（包括零），GPU prefix 分配接收区间及专家段。所有 rank 的 count publish 必须排在等待 kernel 前，不能把未发布的 count kernel 排在该 rank 的等待后。
4. **capacity / valid ranges**：输出和 workspace 在初始化或显式 prepare 阶段按最大容量预分配；主路径 `do_cpu_sync=False`，由 GPU counts/offsets 描述有效区间，不在本轮读回 Python int 或按实际 count 分配精确 shape。容量不足时通过 GPU 错误状态处理，正常路径不加入 CPU 等待。下游专家 kernel 必须直接消费 GPU counts/offsets，不能先取回 Python list 再启动 GEMM。
5. **transport**：每 GPU launch C 个有限生命周期 progress CTA，每 CTA 对应一个 channel，负责本 channel 发、收和 ACK。每 descriptor 只有一个接收 owner，天然接收去重；收到显存的 unique row 后 ACK，本地路径直接 GPU copy。
6. **expand epilogue**：独立 CUDA kernel 读 unique HBM，按预计算专家位置 scatter，生成本地 expert ID/weights、反向映射，按需清 padding；transport 完成后同 stream 顺序保证可见性。
7. **completion**：只有本地发送全部被 ACK、接收全部完成、epilogue 完成后才记录返回事件。输出独立于 host ring，后续重用 ring 不破坏已有结果。

S3/S4 不在稳态通过 NCCL all-gather 整个 routing map。初始化 group 集合通信可交换共享段信息；每轮只交换 GPU 发布的小型 counts，再经目标描述符精确路由。路由变化时重新算 layout，不能在 decode 中盲目复用旧 handle。

调度是正确性问题：禁止先 launch 阻塞 send grid 再在同 stream launch recv grid；禁止依赖未驻留的 CTA 做“全局 barrier”。首版 C=1 起步，一个 CTA 内同时推进收发；扩展 C 前按实际寄存器/shared memory 检查可驻留数，并为 progress 保留执行资源。即使 C 小于 SM 数，也不能保证与任意占满 GPU 的外部 persistent kernel 并发时有前进性；需定义调度契约并测试，优先级 stream 不提供抢占保证。

每轮 expected receive 数来自 counts，不依赖“ring 暂时为空”。本地无 token、所有路由 -1、无远端目标均必须发布 count 并正常完成。count mailbox 下轮覆盖前，需通过本轮全体完成/初始化控制屏障或独立 epoch slot 确认所有消费者已读取；首版串行 collective 及单个在途 epoch，并为 count mailbox 设置每接收方 consumed epoch 握手，不能仅凭本地 kernel 返回就覆盖。

### 7.1 Kernel fusion 的目标与优先级

上述分阶段流水是 correctness baseline，后续优先融合通信内部工作。fusion 必须继续满足源端全远端共用一次 host payload 写、每目的 GPU 一次 host payload 读，不能因多个专家 warp 各自读取 host 而破坏接收端去重。

| 优先级 | 融合点 | 预期收益 | 必须满足的条件 |
|---|---|---|---|
| F1 | 发送、接收、ACK/credit 推进 | 在同一个 progress kernel 中推进双向通信，减少 launch，并让发送背压期间仍能接收 | 每 channel 单 owner；不能在发送子任务内无限等待 credit |
| F2 | host 读取、本地专家展开 | 省去 unique HBM 中转读写和独立 scatter launch | 先得到专家输出位置；共享暂存与 ACK 生命周期明确 |
| F3 | token 路由去重、host 写入、descriptor 发布 | 减少中间路由表读写及部分预处理 launch | 稳定布局所需 counts/ordinal 仍须先建立；所有 payload writer 完成后才能发布 |

F1 是 S3 有限 ring transport 的前进性基础结构，不要求 S1/S2 整批传输原型提前实现它。S2–S4 保留独立 expand epilogue 用作对照；S5 验证 F2 后再作为默认优化候选，S6 单独探索 F3，避免为省一次 launch 引入跨 CTA 或跨 rank 的循环依赖。收益均待目标 GPU 实测。

### 7.2 接收与专家展开融合

针对 `do_expand=True`，由唯一 consumer CTA 将 token 完整读入 shared memory，再向该 GPU 上命中的多个专家输出位置写入：

```text
baseline：host payload -> unique HBM row -> 多个专家 HBM row
                transport kernel             expand kernel

fused：   host payload -> shared token tile -> 多个专家 HBM row
                        同一个接收 CTA
```

设 payload 为 B bytes、本 GPU 命中 m 个专家。相对 baseline，理想融合省去 B bytes unique HBM 写和 B bytes 后续 unique HBM 读；m*B 的专家输出写仍存在。这里不把 baseline 多专家读取的缓存行为算成额外确定收益。只有无需对外保留 unique tensor 的 expanded 模式才可以完整省去此中转；`do_expand=False` 仍需把 unique 输出写到 HBM。

融合接收步骤：acquire descriptor -> 合并读取 host payload/路由 metadata 到 shared -> 确认所有 host reader 完成 -> 发布 payload ACK -> 从 shared 写各专家输出和反向映射。此时 ACK 仅表示 host slot 可回收，不表示专家输出已经完成；返回 completion event 必须覆盖全部输出写入。ACK 之前必须捕获后续仍要使用的所有 host metadata，不能 ACK 后再读取原 header。descriptor credit 也只能在其字段全部读完且不再引用该 queue slot 后归还。

shared tile 在所有 scatter reader 完成前不能被下一 token 覆盖。多 warp 分工需用 CTA barrier 或合法的分阶段共享内存协议；异步复制需显式等待其完成，不能以 warp 已发出指令代替数据到达。大 H 或多级 pipeline 导致 shared memory 不足时，可以分 tile 读一次、展开到所有本地专家；必须等最后一个 tile 的 host 读取完成才 ACK。shared memory、寄存器和 barrier 增加可能降低 occupancy，因此保留 baseline 回退路径并通过 benchmark 决定适用区间。

### 7.3 输出位置与源端融合边界

确定性 expanded 输出需要先算出专家段起点、每 source 的专家分段，以及 token 在该 source/expert 内的 ordinal。源端 layout 可把 ordinal 随路由 metadata 传递；接收 CTA 据此直接写唯一位置。不能仅知道专家总 count 就声称已知道每 token 的稳定位置。

另一个可选方案是在接收 GPU 的 HBM 上对专家 cursor 做 device-local atomic 分配行位置；该 atomic 不访问 host ring。它适用于允许不稳定到达顺序的模式，仍须先确定专家容量/段起点，检查越界并保存逆映射，不能默认替换 deterministic 模式。

源端可在同一 producer CTA 中完成 token 的 top-k rank 去重、一次 host 写和各目标通知；必要时重算很小的 rank set，以减少从 RoutePlan 读取 mask 的开销。不过稳定位置和收发总数仍依赖 layout/count 阶段，不能在未完成这些依赖时消除预处理。CTA 内使用 warp 分工也必须保持单 payload owner、单队列 producer 和完整发布次序。

### 7.4 推荐收敛结构：两个主要阶段

```text
阶段 A：Layout / notify
  路由与去重 -> counts 发布/接收 -> prefix / 专家位置 / 容量确认

阶段 B：Fused dispatch
  源端写 host + descriptor 发布
  接收轮询 + host 读取 + 本地专家展开
  ACK / credit / 回收推进 -> 全部输出完成 -> completion event
```

这是两个逻辑阶段，不承诺每次调用恰好两个 kernel launch。阶段 A 的全 GPU prefix、跨 rank counts 依赖可能仍需多个有限 kernel 边界，这些是 GPU 执行依赖，不要求 CPU 同步。主路径不包含 count 读回与精确 shape 分配。padding 清零也可保留独立 kernel，除非已证明融合清零与有效行写互不重叠。通信 progress 不能被清零或大规模本地 scatter 长时间阻塞，应在有界任务间轮询其他 peer。

后续 decode 固定容量模式或有效 cached layout 可进一步探索 F3，减少阶段 A 的开销；路由变化仍需重算。普通 kernel 不允许假设所有 CTA 同时驻留，也不能等待只有该 kernel 结束后才会执行的其他 kernel。融合 expert GEMM 不列入此轮优化目标，先通过事件与通信重叠，单独评估共享内存和 SM 资源竞争。

### 7.5 Fusion 的验证要求

逐级对比 baseline、F1、F1+F2、F1+F2+F3，确认 payload/weight/metadata、专家 counts、padding 和稳定顺序一致；用 Q=1/2 与故意延迟 scatter 的测试验证“提前 ACK 后 source 覆盖 host”不会破坏 shared 中的数据。记录 kernel launch 数、unique HBM 中转流量、host 实际读写、shared/register 用量、occupancy、poll 等待与端到端延迟；不能只用 kernel 数下降判定优化成功。

### 7.6 Warp specialization

2026-10-25 明确补入 S5 的优化。当前 S4 虽把收发放在一个 progress kernel 中，但采用 thread 0 控制、全 CTA 顺序参与搬运和 barrier；它没有 control/metadata 与 payload warp 并发流水。前文笼统的“多 warp 分工”不足以定义该优化，本节规定独立实现与验收边界。

**DeepEP 参考方式。** 固定 commit 的 `deep_ep/include/deep_ep/impls/ep/dispatch.cuh:78、:258` 将 notify warps 和 dispatch warps 分开：前者做路由计数、交换和 prefix；后者在 `:286–356` 先发起从本 GPU HBM 到 shared 的异步 TMA load，再准备 top-k、metadata、rank 去重和 slot，等数据到达后继续 store/put。这既包含不同 warp 的角色划分，也包含同一 warp 的控制处理与异步搬运重叠。PCIe 后端借鉴其分工、预取和阶段同步原则，不复用 Gin/NVLink/RDMA 数据路径，也不假设 mapped host 地址支持同样的 TMA/cp.async 操作。

**与其他优化分开。** 多 CTA/channel 提高跨 SM 并发；cooperative metadata load 让 warp lanes 合并处理 metadata；warp specialization 让控制准备与 payload 工作在不同 warp 上前进。仅把 metadata 从一个 lane 改为多个 lanes 加载，仍不等于已经实现 producer/consumer 流水。S5 先保留现有 GPU RoutePlan/ReceivePlan，S6 再把更多路由计算接入控制角色；布局所需的 counts/prefix 依赖始终保留。

候选分工如下；8-warps 配比仅为实验起点，可按读写不对称和资源用量调整：

| 角色 | 候选 warp | 职责与所有权 |
|---|---|---|
| TX control / metadata | 0 | 检查源 payload slot 和各目标 descriptor credit，预留任务；协作准备/写入 metadata；lane leader 在全部 writer 完成后发布每目标 READY；推进发送 ACK 回收 |
| RX control / metadata | 1 | 有界轮询各 source READY，快照 descriptor/metadata 与稳定输出位置，向 RX workers 提交任务；在正确完成点由单一 owner 返回 ACK/credit |
| TX payload workers | 2–4 | 按任务从本 GPU 输入写 source host payload；每个 token 只有一个 channel/owner 和一份 host payload，组内 lanes 按不重叠字节分工 |
| RX payload workers | 5–7 | 从对应 source host payload 读取一次到 unique HBM；F2 启用后改为 shared tile 并本地展开，多专家不得重复读 host |

同一字段仍只有一个 producer/consumer owner。worker 不自行分配队列 head/tail，不独立发布远端 READY 或 ACK；不同 lanes 可以写同一 metadata/payload 的不重叠部分。多 CTA 时必须先按 channel 分片队列与状态，不能让多个 CTA 复制原来的私有计数器后争用同一 ring。

任何 host metadata 或 descriptor body 预写前，TX owner 必须已经取得对应 slot/credit。首版按每 `(source,destination,channel)` 的预留顺序分配并发布 descriptor sequence；即使本地 stage 先后完成次序不同，也先等待该队列的前序发布，不能让流水回绕覆盖未消费字段。channel 队列序号与原稳定输出 ordinal 分开保存，不能用到达顺序重排专家输出。

**本地有界流水。** TX、RX 分别设置至少两个 local stages；stage 存放任务、metadata 快照和完成状态，首版 payload 仍用已验证的 SM load/store direct 路径，不强制增加 shared payload 中转。这样控制 warp 可以准备 token n+1，同时 TX workers 搬运 token n，RX workers 也可处理已就绪的其他任务。F2 才按需增加接收 shared tiles。Q、L 与 local-stage 数分别计量；stage ticket 带 `(epoch, token identity, stage generation)`，回绕不能误认旧 DONE。

```text
TX：预留 credit/slot、准备 metadata -> workers 搬 payload -> 汇合全部 writer DONE -> 发布 READY
        同时准备下一 token              不同 warp 并发                同 token 必须有依赖
RX：快照 descriptor/metadata -> workers 完成 host 读取 -> host ACK -> 本地展开完成 -> stage FREE
```

metadata 与 payload 可以并行写，但控制 warp 只有在两者的所有 writer 都完成必要 system fence，并通过合法本地完成握手汇合之后，才能发布对应 token 的 READY。不能靠 publisher 自己的一次 `__threadfence_system()` 代替其他 writer 的 fence；“已发出 load/store”也不代表搬运完成。预写 metadata 或 descriptor body 不对消费者发布完成，READY 必须最后 release-store。

**角色间同步与前进性。** 角色进入不同循环后，不能继续复用原来要求全 CTA 参与的 `__syncthreads()`。使用参与线程数明确的 named/subset barrier、CTA 范围 release/acquire stage 状态或经过验证的共享队列；`__syncwarp()` 不能独自承担跨 warp 交接。全 CTA barrier 只用于全体收敛的初始化和最终汇合。所有 worker 的完成关系和 generation 必须覆盖参与该任务的实际线程，不能只等待一个 lane。

TX 缺少 credit 时不能阻断 RX/ACK 角色；控制轮询与任务提交有界，避免占用全部 stages 后互等。控制准备阶段也须继续检查已完成任务和错误，不能为了预取下一 token 无限等待。空输入、无目标、尾批和错误路径必须通知所有角色退出等待并收敛，不伪造 host ACK 或 credit；不得依赖其他尚未驻留 CTA 才能解开等待。主路径不引入 CPU completion sync 或 CPU payload relay。

**三个独立回收点。**

1. Descriptor credit：必需字段全部快照，且没有任何角色还会访问该 descriptor slot 后才能返回。
2. Host payload ACK：全部 host payload reader 和后续必需 host metadata reader 已结束后才能返回；任务刚入本地队列不能算读取完成。源 slot 仍等待所有命中目标的 ACK。
3. Local-stage FREE：最后一个本地 worker/metadata reader/可选 scatter reader 完成后才复用。F2 中 host ACK 可以早于专家输出完成，但 shared tile 不能因此提前回收。

发送 task/stage 已复用不代表 host slot 已回收，独立 channel 状态必须继续保存对应 sequence/ACK 追踪。完成事件覆盖本轮预期接收数、全部本地输出、所有 stages 排空及源端发送全 ACK；不能以控制 warp 已提交完任务代替 dispatch 完成。

**参数和验收。** 为 control/TX/RX warp 配比、stage 数和 specialization 开关设计独立 kernel variant/config；这些是后续配置草案，当前 API 仍明确拒绝不支持模式。固定角色不自动降低每线程寄存器占用，需核查编译资源、local stack、shared memory 和实际驻留能力。分别与同构 CTA、仅 cooperative metadata、多 channel 及 F2 对照，保持总窗口和输入路由一致；记录控制/数据重叠、stage 空满等待、READY miss、ACK/credit wait、有效 warps、barrier/访存等待、时延与有效带宽。

用延迟 metadata writer、最后 payload writer、最后 host reader 和最后 scatter reader 的测试验证发布与回收次序；覆盖 Q/L=1/2 且不相等、stage=2/3 的回绕、热点多播、零/不等长输入、连续 epoch、timeout、销毁与 stream 依赖。对照 S4 核对稳定行序、weights、padding 和发送/接收去重计数，保留同构路径作为正确性与性能回退版本。

## 8. Python API 与 communicator 复用

S4 独立后端门面示例（完整导入、初始化与生命周期见[实现 API](../../API.md)）：

```python
buffer = EPBuffer(
    group=ep_group,
    num_max_tokens_per_rank=Tmax,
    hidden=H,
    num_topk=K,
    num_experts=E,
    transport="host_staged",        # 新增
    host_staged_config=HostStagedConfig(
        channels=1, payload_slots=256, descriptor_slots=256,
    ),
)
recv_x, recv_idx, recv_weights, handle, event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_experts=E, expert_alignment=1,
    do_expand=True, do_cpu_sync=False,
    async_with_compute_stream=True,
)
event.current_stream_wait()          # 当前 GPU stream 建依赖；不等于 CPU synchronize
```

直接实现类可先命名 `HostStagedEPBuffer`。支持与 `EPBuffer.dispatch` 相同的五项返回值；`defer_epilogue=True` 时返回 EventOverlap，首次 `.wait()` 返回四项结果，参照本地 `event.py` 的一次 hook 语义。各参数的支持进度见实现计划；不支持项应明确拒绝。

host-staged 后端的 `do_cpu_sync` 默认值为 `False`；兼容签名保留 `None` 时也必须解析为 `False`，不能继承原后端首次 dispatch 默认 CPU sync 的行为。首版显式传 `True` 应报不支持。返回 tensor 的 shape 表示已分配容量，实际有效 unique/expanded 行数和每专家 counts/offsets 保存在 GPU handle 中；主路径不填充需要读回的 CPU count list。延期 epilogue 的 `.wait()` 仅建立 GPU stream 依赖并提交 GPU 收尾工作，不等待 CPU 获取真实计数。固定容量本身不代表已实现 CUDA Graph，Graph 仍按后续里程碑独立验证。

`num_qps=0` 在该 transport 无意义，允许默认值，非零显式报不支持。原 `allow_hybrid_mode` 的 RDMA/NVLink 含义不套用到 host-staged，不借此偷偷初始化 Gin。`get_buffer_size_hint` 应分别返回 host payload/control/descriptor 与 device workspace/output 估计，避免误用原 symmetric buffer 大小计算。原基于 NVLink/RDMA 的理论 SM/QP 计算不可直接复用。

**复用 NCCL 的具体含义：**

- 传入既有 `dist.ProcessGroup`，保存 group-local/global rank 映射；不假设 `global_rank % local_world_size` 正确。
- 本地 `comm.get_nccl_comm_handle(group)` 已通过 `_get_backend(cuda)` 和 `_comm_ptr()` 借用已有 communicator；保留该兼容适配。存在时 `managed=False`，EP 销毁不销毁借用 comm，并持有 group/handle 强引用直到操作完成。
- 缺少 `_comm_ptr` 时，上游现有行为是新建 managed communicator；增加 `require_reuse_nccl_comm=True` 严格模式，不可复用时明确报错。默认 fallback 要输出可查询状态 `comm_reused`，不能把重新创建叫作复用。
- host payload 不经 NCCL；借用 comm 不等于借用 stream 或同步状态。group 用于初始化/控制元数据，按实际需要调用，不能要求纯 PCIe 路径建立 Gin device communicator。
- 使用已有 group 的控制 collective 必须与应用保持所有 rank 一致顺序。transport 运行时避免等待排在 progress 后的 NCCL kernel，防止 SM/stream 依赖环。

事件持有输入、routing、输出、device workspace 与 runtime 强引用，或正确 record_stream。`previous_event`、comm stream、当前 stream 的依赖严格对齐本地接口；有 previous_event 时遵守 allocate_on_comm_stream 要求。成功 destroy 先停止新调用、排空在途操作、集体确认无人再读共享段，再各进程 unregister/unmap，最后 owner unlink；出错时禁止在其他 GPU 仍访问时立即解除映射。

`HostStagedEPHandle` 保存：buffer UUID/generation、group identity/rank map、路由 shape/version、T/H/K/E、owner map version、unique/source mapping、top-k slot mapping、专家 offsets/counts、展开索引及 completion 生命周期。cached handle 只能复用 layout，不能复用旧 READY/ACK epoch 或旧 payload 数据；每次重新生成传输 sequence。路由 tensor 原地修改、resize、buffer 重建或 owner map 变化使 handle 失效。原 DeepEP combine 不能消费此 handle，首版 combine/backward 明确报不支持。

## 9. 带宽、容量与收益模型

忽略 metadata，BF16 每 token `B=2H` 字节。令 U 为至少一个远端目标的源 token 总数，F 为所有 token 的远端目标 GPU 数之和，M 为所有远端 token-expert 分支数。

```text
Host-staged multicast：GPU -> host = U*B；host -> GPU = F*B
逐目的 GPU staging：GPU -> host = F*B；host -> GPU = F*B
逐专家 staging：GPU -> host = M*B；host -> GPU = M*B
```

发送写节省 `(F-U)*B`；接收相对逐专家节省 `(M-F)*B`。`F>=U`、`M>=F`。如果每 token 仅一个远端专家，payload 流量无 multicast 节省；小消息收益可能来自减少 CPU 调度，也可能被轮询成本抵消。

以 H=7168、BF16、单 token 发往三张远端 GPU 为例，payload 为 14,336 bytes：本方案 host 写 14,336、host 读 43,008，总 57,344 bytes；逐目标 staging 总 86,016 bytes。该例是流量算术，不是实测性能。

粗略容量：payload `P*C*Q*aligned_token_bytes`；ACK `P*C*Q*(P-1)*ack_stride`；descriptor `P*(P-1)*C*L*descriptor_stride`；另加 counts、generation、error。若 P=8、C=2、Q=256、H=7168，payload 未含 metadata 为 56 MiB；ACK 若每目标 64B 隔离为 1.75 MiB。控制区不能只算 8 字节有效字段忽略 padding。

设备容量上界：每目标 unique 行 `sum_s Tmax_s`；均匀 E_local 专家情况下 expanded 行不超过 `sum_s Tmax_s * min(K,E_local)`，另加每专家最多 `A-1` padding。热点下所有源都可路由到同一 GPU，不能按平均负载分配。给定用户更小容量时，在写入前检查并集体失败，不截断或越界。

理想大消息 pipeline 下，时间下界由 source 写、各 dest 读、CPU 内存 `(U+F)*B`、跨 NUMA 流量及 root complex 竞争的最大限制决定；小消息还受读延迟、发布次数和 kernel launch 约束。需单独测 SM read/write 有效带宽，不能代入 memcpy 带宽或 PCIe 标称双向总带宽。

## 10. combine 的接口预留

本次仅实现 dispatch（multicast/scatter），不执行数值 reduce。未来 combine：先在每专家 GPU 将同一源 token 的本地专家输出按约定加权归并，再每目的源 GPU 回传一份 partial，源 GPU 在 HBM 做跨专家 rank 求和。host ring 保留单 producer 字段，不让多个专家 GPU 对同一 host payload 做 atomicAdd。若 weight 已在专家路径应用，combine 不得二次应用。dispatch 保存 source/top-k/专家展开映射，为此提供逆路由；没有 combine/backward 时不能宣称完整训练支持。

## 11. 目标机器验证与尚未确认项

2026-10-24 已在 `<GPU_HOST>` 独立目录 `/opt/pcie-ep` 完成 S0–S4。环境为八张 H20D、PCIe Gen5 x16、双 NUMA、PyTorch 2.13.0+cu130 / CUDA 13.0。S0 单进程 GPU0/1 与 GPU0/4 各完成百万轮 publication/ACK 验证；跨进程映射、多个接收 GPU、有限队列回绕和延迟 ACK 由 S2–S4 分布式测试覆盖。结合生成指令审查与 NVLink payload 计数核对后，本次实际运行支持 PCIe-only 数据路径结论；这些实验不能代替一般内存模型证明。详见[实际结果](../../RESULTS_F2.md)与路径审计（本地生成：`../../results/pcie_path_audit.md`）。

实现作为独立 `pcie_ep` 后端保留，未修改本地 DeepEP；复用其接口形状和现有 NCCL communicator，payload 不经过 NCCL。实测采用 C=1、Q=L=256，主路径无 CPU 完成同步，S4 保留独立专家展开。S1 复用了 S2 的 compact stable layout，属于原稀疏原型的功能扩展。该段记录 S4 初版；后续已完成 S5–S8、多 channel、F2/F3、cached layout、Graph、FP8 和 CPU relay，见 [高级实现](./advanced-implementation.md)。F2 又增加了 shared→expert 向量化与独立 RX/scatter warp 流水，详见 [F2 实现补充](./f2-receive-expand.md)及[新实测](../../RESULTS_F2.md)。

需要实测决定的参数：C/Q/L、vector 宽度、poll backoff、每轮批量 token 数、descriptor padding、NUMA 策略。保持逐 token 路由，但允许后续聚合通知降低 flag 事务；首版不预先承诺吞吐或延迟目标。

补充公开资料：[CUDA mapped memory](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#mapped-memory)、[PTX memory consistency](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)。接口事实以本地固定 commit 为准，公开上游 main 可能变化。
