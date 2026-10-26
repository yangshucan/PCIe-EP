# PCIe EP Dispatch：从最小实现到性能优化

> 2026-10-26：本文的阶段拆分保留为设计推导，不代表仓库仍保存多套历史实现。当前只有一个 native runtime，S1–S4 Runtime、整批传输、S5 串行布局均已移除；默认使用 C32/pipelined F2。当前支持与清理范围见 [CLEANUP.md](../../CLEANUP.md) 和 [API.md](../../API.md)。

本文件规定开发顺序；[design.md](./design.md) 保留最终协议与优化细节。阶段编号统一使用 S0–S8，替代此前按组件划分的 M0–M6；F1/F2/F3 与 R0–R3 仅作为 fusion/relay 子任务名。S0–S4 已完成目标机器测试，原始结果保留在 [RESULTS.md](../../RESULTS_F2.md)。2026-10-25 已按阶段落地 S5–S8 的独立路径；实际配置、协议实现与测量口径见 [advanced-implementation.md](./advanced-implementation.md)，新验收和性能数据保存在 `results/s5_s8/`。

S4 的 thread 0 控制、全 CTA 顺序搬运保持为参考。S5 新增真正的 control/metadata、TX、RX warp 并发流水；S6 并行稳定布局、F3、缓存和 Graph 分别开关；S7 使用可选 FP8 wire；S8 增加 CPU relay。阶段开关与原有路径并存。

基线：`<DeepEP checkout>`，commit `b5a33eefda39485f438dc764695b71ba7b9249b6`。后续开发前检查仓库指令和工作区状态；当前项目 `sources/` 保持只读。

## 1. 各阶段共同约束

- 从最小实现开始，主路径均为 `do_cpu_sync=False`；`None` 解析为 False，首版显式 True 报不支持。不先实现 CPU 读回 count 的同步版本。
- 初始化/prepare 预分配容量，运行时使用 GPU valid mask、counts、offsets。CPU 不等待本轮接收数量，不按实际数量创建精确 shape；下游 kernel 直接消费 GPU 元数据。
- payload 和 READY/ACK 使用经过验证的访问与内存序；早期实现也不能以 volatile 代替 acquire/release，不能省略 slot 回收握手。
- 一 buffer 同一时间只允许一轮逻辑通信；连续调用可以在同一 stream 排队，由 GPU epoch/ACK 保证物理复用安全。未支持的多 stream 并发明确拒绝。
- 每阶段保留可运行的参考版本和独立配置，下一阶段与上一阶段对比；正确性、超时和容量边界从最小版本开始验证。
- 初始化、销毁、故障诊断及测试读取结果可以进行 CPU 协调/同步，必须放在正常 dispatch 主路径和稳态计时区间之外。

## 2. 总体路线与依赖

| 阶段 | 当阶段新增能力 | 暂不加入 | 阶段产物 |
|---|---|---|---|
| S0 | mapped memory、发布与 ACK 验证 | EP 路由、ring、性能优化 | 可复现的通信原语测试 |
| S1 | 双 GPU、固定整批 staging、单 channel | ring、compact、融合展开 | 最小异步传输原型 |
| S2 | 多 GPU、top-k 双端去重、GPU counts、独立专家展开 | 流式 ring、性能 fusion | 功能正确的 dispatch 基线 |
| S3 | 有限 payload/descriptor ring、背压、收发推进 F1 | 多 channel、F2/F3、relay | 固定 host 窗口的流式 dispatch |
| S4 | DeepEP API、comm 复用、事件/生命周期、压力验收 | Graph、FP8、性能优化 | 首个可集成 BF16 direct 版本 |
| S5 | 多 channel、warp specialization、vector/poll 调优、接收展开 F2 | F3、relay | PCIe/HBM 数据搬运优化版 |
| S6 | 源端融合 F3、cached layout、Graph | FP8、relay | 提交与布局开销优化版 |
| S7 | FP8 payload/scales 和布局 | combine | 可选带宽压缩模式 |
| S8 | NUMA relay、CPU worker、两级 ACK | 自动模式切换 | 可选跨 NUMA 分级复制模式 |

主链为 S0 -> S1 -> S2 -> S3 -> S4。S4 通过后即可交付 BF16 direct；S5–S8 均是增量优化，不阻塞这个版本。S6 的 F3、cached layout 和 Graph 分别开关、分别验收；S7、S8 可在 S4 基础上独立开发，并针对启用的优化组合回归，不强制等待其他可选分支。将 S8 排后是建议实施顺序，不是 FP8 依赖。

## 3. S0：先证明 mapped memory 通信原语

**实现范围**：不做 EP、不做完整队列。先用单 GPU 检查映射，再双 GPU 单进程，最后双 GPU 多进程共享同一物理 backing。使用一个固定 payload 区及自然对齐 64-bit READY/ACK，sequence 从 1 开始。

发送 GPU 写包含 sequence 的整行图样 -> 所有 writer 完成并正确发布 -> 接收 GPU acquire 观察 -> 检查全部 cacheline -> release ACK -> 源 GPU 确认后覆盖。只在初始化创建/register 内存，稳态不读回 CPU。多进程交换 segment 信息和 offset，不交换裸指针。

**验收**：延迟最后 reader、多 writer thread、反复复用至少 100 万次；无旧数据、撕裂和提前覆盖。记录 GPU/PCIe/NUMA/driver/CUDA/CCCL/PyTorch/NCCL 与生成指令审查结果。测试通过是工程证据，不能替代内存模型证明。能力不满足时明确不支持，不自动引入 CPU proxy。

## 4. S1：双 GPU 固定整批 staging，先不做 ring

**简化选择**：P=2、C=1、BF16 contiguous、固定 H/K/Tmax、连续均匀 expert owner、单轮在途、direct 模式、无后台 CPU 搬运。先支持 `do_expand=False` 的稀疏原型输出；该原型尚不声明 DeepEP dispatch API 兼容。

每 source 预分配 `Tmax * aligned_token_bytes` 的 host staging，token t 使用固定 slot t。本阶段内存随 Tmax 增长，目的是省掉动态 slot 分配、descriptor queue、credit 和跨 batch ring 回收状态。

```text
有限 send kernel：按 top-k 判断是否有远端目标，写 slot[t]，发布本轮 batch header + READY
receive kernel：等待对方 READY，按 header/mask 读有效 token 到固定 GPU 输出位置，最后 ACK
本地 GPU copy：本地目标直接从 x 进入固定 GPU 输出位置
```

header 记录 epoch、实际 T 和路由信息。即使 T=0 或没有远端 token，也发布 READY；对方只在全部需要的 payload/header 读完后 ACK。所有 rank 先在同一通信 stream 排有限 send，再排 receive，send 不等待对方本轮 receive。接收和本地 copy 完成后，GPU 还需确认本 rank 源 batch 已被对方 ACK，再记录完成事件；Python 提交返回不等待该事件。下轮复用前由 GPU 检查上一轮 ACK；上一轮 receive 已排在各 rank 的下轮复用检查之前，避免依赖环。不加入同 stream “阻塞发送等接收 kernel”结构。

输出按 `(source_rank, token_idx)` 固定位置放置，配 GPU valid mask 和本地化 top-k/weight；不需要先交换精确接收 counts，也不 compact。对没有目标的 slot 不读取未写 payload。局部专家信息保留，输出用于核对传输，不直接当 compact GEMM 输入。

**验收**：双向同时传输、本地加远端、多 top-k 落在同一个远端 GPU、零输入、全 -1、T 不同和连续 epoch；每个远端 token 源端只写一次、接收 GPU 只读一次，输入/输出和权重按位一致。超时、边界检查、资源释放必须已有。此阶段不要求性能或固定 256-slot 窗口。

## 5. S2：多 GPU 去重与专家展开，建立功能基线

**在 S1 上增加**：P>=2 的源端单份整批 staging、多目标共享读取、每接收 rank 独立 batch ACK；源 staging 仅在所有需要读取本批 header/metadata 的 peer ACK 后复用。为简化发现流程，本阶段所有 peer 都检查每 source 的 batch header 并 ACK，包括无目标 peer；它们不读无关 payload。只等目标 peer 的选择性通知在 S3 实现。

GPU layout 先构造每 token 的目的 rank 集合、每 `(source,dest)` 稳定 token ordinal 和每专家 ordinal。有限 kernel 发布 counts/READY 后，接收 GPU 读取 GPU counts 并做 prefix；所有 source 的发布必须排在本 rank 的等待之前。输出/workspace 仍按最坏情况预分配，host staging 仍按 Tmax 固定槽位。

数据路径保持分离便于核对：

```text
route/layout -> batch payload/counts publish -> counts receive/prefix
             -> host-to-unique HBM receive -> 独立 compact/专家 scatter -> completion
```

`do_expand=False` 输出稳定 compact unique 行；`True` 经 unique HBM 展开到专家段。每个目标 GPU 从 host 只读一次 token，本地多个专家从 HBM 复制。先做 alignment=1，再补 alignment/padding。以 `(source_rank,token_idx)` 身份去重，禁止按内容合并；重复有效 expert ID 报错，-1 无路由，其他非法 ID 报错。

**验收**：2/4/8 GPU（按实际硬件）、多 GPU multicast、本 GPU 多专家、本地加远端、不等 T、热点、空输入。对比独立 CPU/PyTorch reference（仅测试路径允许 CPU 收集）；unique/expanded 内容、counts、weights、稳定顺序、padding 完全正确。为下一阶段保存流量与时延基线。

## 6. S3：将整批 staging 替换成有限 ring

**本阶段才引入最终队列协议**：[design.md 第 5–7 节](./design.md#5-队列组织一份-payload多份引用)。保留 S2 layout 和独立 epilogue，只替换 transport。初始 C=1，Q/L 可配置，256 为起始配置而非性能结论。

新增每 `(source,channel)` payload ring、每 `(source,dest,channel)` SPSC descriptor ring、每 slot/目标独立 ACK、GPU 本地 head/tail、64-bit sequence 和 generation。一个 token 写一次 payload，各目标 descriptor 引用同一份；不等非目标 ACK。descriptor credit 与 payload ACK 分离，counts mailbox 有 consumed epoch 回收握手。

有限窗口不能沿用 S1/S2 的“整批先写完再接收”：T>Q 时发送会被 ring 填满。因此从本阶段开始，必须采用 F1 收发/ACK/credit 融合 progress kernel，一个 CTA 同时推进本 channel 的发送、接收和回收；credit 不足时继续处理其他接收任务。F1 是有限缓冲前进性基础，不作为可推迟的纯性能优化。

**验收**：T 远大于 Q；Q/L=1、2、256 及不相等、非二次幂值；同时多播、慢 reader、ring wrap、连续 epoch、计数发布和错误退出。不得出现全 rank 堵在发送而无法接收。payload write=U*B、read=F*B，metadata 独立统计；ACK 仅在 host 读取完成后发布。

本 rank 发送全 ACK、接收和独立 epilogue 完成后才记录 completion event；Python API 提交返回不等待 GPU 完成。压力测试至少 100 万次 slot 重用。

## 7. S4：接口与可靠性完善，交付 BF16 direct

按本地 `ep.py` 对齐五项 tuple、GPU handle metadata、EventOverlap、previous_event、默认/非默认 stream、allocate_on_comm_stream、延期 epilogue；门面 `EPBuffer(transport="host_staged")` 可在独立类稳定后接入。host-staged handle 标记 transport，不与原 combine 混用。

复用既有 torch.distributed group，保存 group-local/global rank/GPU 映射；优先借用 `comm.get_nccl_comm_handle(group)`，暴露 comm_reused，严格复用模式在无 `_comm_ptr` 时明确报错。管理 borrowed/owned 生命周期，不为纯 host 路径初始化 Gin，不通过移除原 transport 全局 NVLink 检查接入。

**API 范围**：BF16、top-k、可选 FP32 weights、unique/expand、deterministic 稳定布局、alignment/zero padding、预分配输出、GPU counts/offsets、异步事件。`num_qps` 等无关参数的显式非默认值、`do_handle_copy=True`、`do_cpu_sync=True`、FP8、Graph、cached replay、combine/backward 在尚未支持时明确拒绝。不要求此阶段兼容原后端所有扩展选项。

**可靠性验收**：非连续 group ranks、同成员不同 group、双 buffer 隔离、buffer 重建旧 handle、容量不足、peer 超时/退出、反复 construct/destroy、输入和 workspace 保活。排空并确认无人再读才 unregister/unmap；失败 runtime 不再使用，不伪造 ACK，不承诺 crash 自动回滚。

至少十万轮 dispatch、百万次复用，包含不同 T、热点与延迟 reader。profiler 检查正常路径和延期 hook 无 `.item()`、`.cpu()`、stream/device synchronize 或 count 就绪的 CPU 等待；专家 kernel 直接消费 GPU counts。记录 API 提交时间、GPU 完成延迟及测试同步位置，不能把它们混为一个指标。

**交付边界**：S0–S4 完成即为首个可集成 BF16 direct 版本；host 窗口有界、双端去重、主路径无 CPU 同步、CPU 不搬 payload。后面的优化均不改变该版本的正确性契约。

## 8. S5：多 channel、warp specialization 与接收展开融合 F2

先建立可复现 benchmark，再逐项增加下列能力。多 CTA/channel 解决跨 SM 并发，warp specialization 解决 CTA 内控制与数据流水，cooperative metadata load 解决单线程 metadata 访存；三者分别开关、分别测量，不能互相替代。

1. **多 channel 基线**：每个源 token 只归属一个 channel，独立分片 payload ring、descriptor/credit、ACK 和进度状态。先保留原同构 CTA 流程，扫描 1/2/4/8/16/32 个 channel；分别固定总窗口与每 channel 窗口，区分并发收益与容量收益。检查真实 occupancy 和前进性，不依赖未驻留 CTA 的 grid barrier。
2. **Warp specialization**：同一 progress CTA 内，control/metadata warp 准备任务、预取/写入 metadata、检查 credit 和发布通知，TX payload warps 写 host，RX payload warps 读 host；角色以有界本地任务队列交接。至少使用双 stage，让 token n 的 payload 搬运与 n+1 的 metadata/credit 准备重叠，TX/RX 角色也可独立前进。先保留独立专家展开，隔离该优化的收益。
3. **访问与轮询调优**：metadata 由控制 warp 的 lanes 协作加载/写入；调整 vector 宽度、Q/L、轮询退避、每轮 bounded task、通知批量和 abort 检查频率。批量不改变逐 token 目标集合；不得提前发布 READY，也不能饿死接收或无限延迟错误检测。
4. **接收展开融合 F2**：`host -> shared tile -> 多个专家输出 HBM`，在 expanded 模式省去 unique HBM 中转与独立 scatter launch；non-expand 仍保留 unique 输出。在已验证的 RX stage 上接入展开，单独测量新增 shared/register 占用及收益。

F2 已按两个增量落地并各自保留开关：

- `receive_expand_mode="vectorized"`：同组 3 个 RX warp 仍依次接收和展开，shared→expert 使用 uint4，H%8 非零时整体标量回退。
- `receive_expand_mode="pipelined"`：复用 vector scatter，RX warp 5–7 只接收，新增 warp 8–10 专做展开；shared 双/三 stage 经 generation READY 和独立 reader/scatter 完成条件回收。

新增实现、混合 buffer 初始化回归见 [F2 实现补充](./f2-receive-expand.md)；[实际结果](../../RESULTS_F2.md)仅列最新流水版，带宽按 `T*K*H*b/t` 计算。S5–S8 阶段编号保持不变。

### 8.1 Warp specialization 的角色与交接

详细协议见 [design.md 第 7.6 节](./design.md#76-warp-specialization)。scalar/vectorized 的 8-warp CTA 划分为 TX control 1 warp、RX control 1 warp、TX payload 3 warps、RX payload 3 warps，通过 `warp_specialization=True` 启用；同构 CTA 路径单独保留。pipelined 使用上述角色外加 3 个 scatter warp；任意 warp 配比尚未作为公开 API，后续可依据资源和吞吐再扩展。

控制角色包含 metadata 处理及其 host 写入/本地快照，worker 角色负责 payload 搬运。为隐藏延迟，可先准备后续 token 的 metadata，再在该 token 的全部数据完成后发布 READY；不能把“metadata 可以提前写”理解为“接收方可以提前读取未完成 payload”。

本地 stage 携带 `(epoch, token identity, stage generation)`，只有无使用者后才复用；host ring 的 Q、descriptor 的 L 和本地 pipeline stage 数独立设置。metadata 准备和 payload 搬运可以重叠，同 token 的 READY 仍须等待全部 metadata/payload writer 的 system-fence 完成关系。本地主从交接使用合法的 CTA 范围 release/acquire 或明确参与者的 barrier，publisher 自己执行一次 fence 不能代替其他 writer 的 fence。

不同角色的循环不能继续套用现有全 CTA `__syncthreads()`。采用参与线程数明确的 named/subset barrier、阶段握手或已验证的共享内存队列；只在全体线程收敛的初始化/最终汇合处使用全 CTA barrier。控制角色等待发送 credit 时，RX/ACK 角色仍须前进；零 token、尾批、错误与 timeout 分支都必须能排空或一致退出，不能伪造远端 ACK。

分清三种回收：descriptor 的字段已快照且无人再读取后归还 credit；payload 和后续必需 host metadata 都读完后归还 host ACK；本地 worker/展开 reader 全部完成后释放 local stage。返回 completion 还必须覆盖全部输出写入、所有 stage 排空、本轮接收完成及本 rank 发送全 ACK；主路径继续 `do_cpu_sync=False`。

F2 使用 S2 预计算的稳定专家位置。host payload/header 的后续必需字段全部读进 shared 后可 ACK，shared 在全部 scatter reader 完成前不能复用，completion 晚于全部输出写。大 H 可分 tile，但最后一个 host tile 读完前不能 ACK。padding 可保留独立清零，不为减少 launch 制造写竞争。

### 8.2 独立验收与性能对照

**正确性验收**：故意延迟 metadata writer、最后 payload writer、最后 host reader 和 scatter reader，验证 READY/ACK/stage FREE 不会提前；覆盖 Q/L=1/2 且不相等、本地 stage=2/3、stage 回绕、多个 channel、零输入/全本地/无目标/不等长、热点多播、alignment、连续 epoch、timeout、退出与 stream 依赖。不得 ACK 后再读 host header；与 S4 逐项对比 payload、weights、稳定布局和双端去重计数。

**消融对照**：在相同 channel 数、总窗口、输入路由和计量口径下，分别测同构 CTA、仅 cooperative metadata、warp-specialized 双 stage、不同 stage/warp 配比、再叠加 F2。检查 READY miss、payload ACK wait、descriptor-credit wait、空轮询及 local-stage wait，确认 metadata 准备与 payload 搬运实际发生重叠。不能把 CTA 增多、窗口增大和 warp 分工的收益混为一项。

记录 p50/p95/p99、tokens/s、actual host bytes、unique HBM 中转、launch 数、shared/register、occupancy、poll 等待和 GEMM 重叠后的 step 时间。理想省 B unique HBM 写和 B 后续读不等于物理 DRAM 流量或时延等比例下降。资源占用导致回退时保留未融合路径。

## 9. S6：布局、源端融合和提交开销优化

以下三项分别实现、分别验收，不捆绑为一次大改：

- **F3 源端融合**：将更多 token 路由/去重准备接入 S5 的 control/metadata warp，并与 payload worker 流水；消除不必要的路由表读写，保留生成 counts、稳定 ordinal、输出位置和容量所需步骤。源端一次 host 写，全部 metadata/payload writer 完成后才发布通知；不把 warp specialization 推迟到 S6，也不为它提前省略布局依赖。
- **cached layout**：只在路由不变时复用 RoutePlan/ReceivePlan，每次使用新 epoch/sequence 和新输入 payload；校验 tensor version、shape、owner map、buffer generation。decode 不天然意味着路由可复用。
- **CUDA Graph**：所有分配/注册在 capture 外，固定输出容量，GPU 执行时递增 epoch。验证首次/重复 replay、路由变化、全 rank replay 顺序及 eager 交错；未实现的多 stream 并发仍拒绝。

目标收敛为 layout/notify 与 fused dispatch 两个主要阶段，不强制两个 launch。跨 rank counts、全局 prefix、padding 保留必要 GPU 执行边界，主路径不加入 CPU 同步。与 expert GEMM 的融合不在本轮范围。

**验收**：新旧路由、cache 失效、连续 replay 无旧 READY、S4/S5 对照，分别记录实际 launch、预处理时间和端到端变化；没有独立收益的优化不强制启用。

## 10. S7：可选 FP8 带宽优化

BF16 路径稳定后新增 FP8 payload/scales 打包、源写与接收展开；scales 与对应 token 一同发布和回收，一次写/每目标读一次。量化位置、scale dtype/分块布局、padding 与 TMA-aligned SF 格式明确约定，不仅改变 tensor dtype。

**验收**：payload/scale 对齐和映射正确，与量化 reference 对比、反量化误差在约定阈值内；不同 H/K/专家分支与已启用 fusion/Graph 组合回归。单独报告压缩后 payload/scales/control 流量及端到端收益，不能把量化误差等同于通信错误。

## 11. S8：可选 NUMA relay

细节见 [design.md 第 4.1–4.6 节](./design.md#41-可选-numa-relay按目标内存域复制)。先在 S4 direct 基线上验证 relay，再与 F2/Graph/FP8 逐项组合；relay 不依赖这些优化已完成。

- **R0**：确认 source/replica 实际页位置和 GPU locality，验证 GPU 写 -> CPU copy -> GPU 读及反向 ACK 的跨 CPU/GPU 原子互操作、对齐、发布语义。
- **R1**：每 source/channel/目标 NUMA 独立副本 ring，native worker 固定 owner/绑核；实现 relay credit、COPIED_ACK、每 GPU CONSUMED_ACK 和 drained epoch。每 token 只复制到实际命中的目标 NUMA，每 NUMA 一份。
- **R2**：新增 numa_mode/replica_slots/worker 配置和诊断；稳定 identity 与 GPU 输出位置不变，mode/topology 改变必须排空重建。F2 接收地址可切为副本 ring，counts/prefix 仍由 GPU 推进。
- **R3**：双向多 source、每远端 NUMA 命中 1/2/多 GPU、Q_source/Q_replica/L=1/2 且不相等、延迟 worker/最后 reader、worker 退出、epoch 与 destroy 压力验收。

source 只在所有本地直接 ACK 和远端 COPIED_ACK 完成后复用；副本等全部目标 GPU ACK 后复用。事件完成还须观察本轮 relay drained，源 slot 回收不能冒充远端全部消费完成。中途失败不静默切 direct，不伪造 ACK。

**性能验收**：对比同 NUMA direct、跨 NUMA direct、relay 的 p50/p95/p99、跨 socket 实测流量、CPU 核数/利用率、复制/排队/背压时间与内存容量。单独统计 relay host 读写，不把源 GPU 一次 host 写描述为系统仅有一次写。

relay 保持 `do_cpu_sync=False`、调用线程不等待 counts，但稳态数据前进依赖 CPU worker；明确区分“无调用线程 CPU 阻塞同步”与“无 CPU 数据搬运”。Graph 支持单独验证，worker 在 capture 外启动且 replay 时存活。

## 12. 代码按阶段增添

下列路径相对 DeepEP 仓库，文件名为建议，不要求一次建立所有模块。

| 首次需要阶段 | 建议模块 | 职责 |
|---|---|---|
| S0 | `csrc/comm/host_shared_memory.hpp`、`host_staged_sync.cuh`、`tests/host_staged/` | 共享 backing、注册、原子封装、litmus |
| S1 | `csrc/buffers/host_staged_ep.hpp`、最小 CUDA launch、`host_staged_ep.py` | 双 GPU 整批 staging、固定输出、READY/ACK |
| S2 | `host_staged_layout.cuh`、`host_staged_epilogue.cuh` | 稀疏路由、GPU counts/prefix、独立 compact/专家展开 |
| S3 | `layout/ep/host_staged.cuh`、`host_staged_dispatch.cuh` | ring ABI、descriptor/credit、F1 progress |
| S4 | `deep_ep/comm/host_staged.py`、`deep_ep/buffers/ep.py`、bindings/build | group/comm、正式 API、事件、生命周期和门面 |
| S5 | transport/epilogue 的可选 kernel variants、本地 stage/mailbox 与角色同步 | 多 channel、warp specialization、vector、F2；保留 baseline |
| S6 | RoutePlan cache、Graph/epoch、source fused variants | F3、cached layout、Graph 独立配置 |
| S7 | payload/scale layout variants | FP8 与 SF 格式 |
| S8 | `csrc/comm/numa_relay.hpp/.cpp` | native worker、绑核、复制与两级回收 |

将 RoutePlan（路由与 ordinal）、ReceivePlan（接收/专家位置）和 TransportState（epoch/sequence/credit）分开；前两者可缓存，TransportState 每轮更新。最小阶段可仅实现必要字段，不预先引入 relay、JIT specialization 或所有参数组合。沿用仓库许可要求和实际构建规则。

## 13. 统一测试与交付记录

每阶段测试边界必须写在报告里：当前支持的 P/T/H/K/E、owner map、布局、容量、channel、dtype、模式和不支持项。S2 起用 reference 对比，S3 起加 ring 强制回绕，S4 起作为集成发布门槛；后续每个优化都回归这些测试。

benchmark 先测 mapped read/write、双 GPU ping-pong 和多 GPU 共读，再测 dispatch 与 expert GEMM 重叠。比较对象按实际依赖选择 direct、逐目标 staging、DMA pipeline、NCCL 路由分发、可用时 P2P/RDMA；算法带宽与真实总线 bytes 分开，跨 GPU 时间戳不直接相减。固定软硬件、路由分布及计时方式，新增变化才扩展测试。

开发前记录目标 GPU/PCIe/NUMA、CUDA/PyTorch/NCCL、Tmax/H/K/E、owner 与 prefill/decode 重点。没有目标环境数据时保留可配置参数，不凭空设定吞吐/延迟达标值。

2026-10-24 的 S0–S4 交付使用本项目根目录，未改动原 DeepEP。保留的 S1 与 S2 共用 GPU stable layout，输出 compact unique 行，替代原最小稀疏输出建议；S1 的双 GPU/整批 staging、S2 的多 GPU/展开、S3 的有限 ring、S4 的 API/生命周期阶段边界保持清晰。八卡主路径、三组各十万轮/卡 ring 压力、最大 batch 全量内容与跨 stream 生命周期检查通过；已测齐用户指定的十个 batch，性能数据与 PCIe-only 证据见实际结果。S4 是功能基线，后续优化收益必须另行实测。
