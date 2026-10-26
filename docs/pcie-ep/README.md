# PCIe Host-staged EP Dispatch 设计包

设计日期：2026-10-23；实现更新：2026-10-25。S0–S4 基线和 S5–S8 可选路径均已落地，目标环境为 `<GPU_HOST>` 的八张 H20D；阶段验收、调参与全部指定 batch 的实际记录保存在 `results/s5_s8/`。历史实验材料移出工作目录；当前源码不保留旧 Runtime。

本设计以本地 `<DeepEP checkout>` 为接口与源码基线，commit：
`b5a33eefda39485f438dc764695b71ba7b9249b6`（读取时工作区无变更）。
该仓库 README 明确标注 V2.5，提供 `EPBuffer`、`EPHandle`、`EventOverlap`。
当前项目的 `analysis/hybrid-ep` 只是补充参考，其中 `pcie.cu` 走 RDMA，不能等同于本文传输路径。

- [分阶段实现计划：从最小原型开始](./implementation-plan.md)
- [详细设计：架构、协议、API 与优化](./design.md)
- [S5–S8 实现细节与计量](./advanced-implementation.md)
- [F2 向量化与独立 RX/scatter warp 流水](./f2-receive-expand.md)
- [最新流水 F2：十个 batch 实测与修正带宽](../../RESULTS_F2.md)
- [源码与复现说明](../../README.md)
- [S5–S8 实际性能与验收](../../RESULTS_F2.md)
- [原始 S0–S4 性能结果及验收](../../RESULTS_F2.md)
- [重构后的代码地图与验收](../../REFACTOR.md)
- [代码风格与同步约定](./code-style.md)

建议先按阶段计划开发，详细设计用于查协议，避免一次实现全部功能：

| 阶段 | 新增内容 |
|---|---|
| S0 | 映射内存、READY/ACK 正确性验证 |
| S1 | 双 GPU、固定整批 staging、固定位置输出，暂不做 ring |
| S2 | 多 GPU 双端去重、GPU counts、独立专家展开 |
| S3 | 有限 ring 与背压、收发/ACK 融合推进 |
| S4 | DeepEP API、comm 复用、事件与可靠性，交付 BF16 direct |
| S5 | 多 channel、warp specialization、访问调优、接收/专家展开融合 |
| S6 | 源端融合、cached layout、CUDA Graph，分别验证 |
| S7 | 可选 FP8/scales 带宽优化 |
| S8 | 可选 NUMA relay 分级复制 |

S0–S4 为主线；S5–S8 按测量收益逐项添加，FP8/relay 可基于 S4 独立验证。最终 direct 路径使用源端单份 payload ring、多目标描述符和接收方独立 ACK；早期整批 staging 版本保留作正确性与性能对照。

主路径默认 `do_cpu_sync=False`，从 S1 开始按最大容量预分配输出，以 GPU counts/offsets 描述有效区间，下游专家 kernel 直接读取；稳态不读回计数、不等待 CPU。原计划 S1 为固定位置稀疏输出，保留实现复用了 S2 的 stable layout 并直接返回 compact unique 输出；双 GPU 整批 staging 的阶段边界不变。首版显式 `do_cpu_sync=True` 报不支持，未来可选兼容模式单独处理。

Kernel fusion 路线：S3 先融合收发与 ACK/credit 推进；S5 增加 control/metadata、TX payload、RX payload 的 warp specialization，以本地双 stage 重叠不同 token 的准备与搬运，再验证 host 接收和专家展开 F2；S6 将更多源端路由去重接入该流水。目标收敛为 layout/notify 与 fused dispatch 两个主要阶段，保留计数与 prefix 所需的 GPU 执行边界；详见[设计文档第 7.6 节](./design.md#76-warp-specialization)和[实现计划 S5](./implementation-plan.md#8-s5多-channelwarp-specialization-与接收展开融合-f2)。当前 S4 仍使用原始顺序 CTA；S5 已提供 warp specialization 与独立开关。

NUMA relay 为显式可选模式：源端 ring 放在源 GPU 邻近 NUMA，后台 native CPU worker 将 payload 按需复制一份到每个目标 NUMA 的副本 ring，再由本地目标 GPU 共读。采用复制完成 ACK 与 GPU 消费 ACK 两级回收，不依赖两份 ring 自动缓存一致。该模式保持 `do_cpu_sync=False`，但稳态数据前进依赖 CPU worker；默认 direct 模式仍完全由 GPU 搬运。详见设计文档第 4.1–4.6 节及实现计划的 NUMA relay 子任务。

核心边界：普通 device-scope atomic 不足以完成此协议；不使用 host 上跨 GPU 的共享引用计数。采用自然对齐的 system-scope acquire/release load/store，依次证明 payload 发布、读完成和 slot 回收。稳态 ring 协议采用无互斥锁握手，慢接收方仍会造成背压；初始化的 kernel shared-memory 属性另由进程级 mutex 保护，不进入 dispatch/replay。

当前代码清理和兼容性变更见 [CLEANUP.md](../../CLEANUP.md)。
