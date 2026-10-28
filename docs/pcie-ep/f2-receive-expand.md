# F2 接收与专家展开：向量化与独立 warp 流水

2026-10-28：当前运行时进一步实现[并行初始化与prefix/TX重叠](prefix-tx-overlap.md)，最新重新测量见 [RESULTS_PREFIX_TX.md](../../RESULTS_PREFIX_TX.md)。下文保留F2接收/展开实现说明。

本文记录现有 F2 的两个增量实现；最新 pipelined 实现的十个 batch 时延和按 `T*K*H*b/t` 重算的带宽见 [F2 实测报告](../../RESULTS_F2.md)，匿名时延数据在 `benchmarks/data/`；完整机器相关数据仅保留在本地 `results/f2/`。阶段编号仍为 S5–S8；F2 是接收/展开融合子功能，两者不是同一层级。

## API 与选择规则

`HostStagedConfig` 新增 `receive_expand_mode`，取值为 `"scalar"`、`"vectorized"`、`"pipelined"`，默认值仍为 `"scalar"`。提高 stage、开启接收融合或使用 FP8/relay 都不会自动切换到新模式。

```python
config = HostStagedConfig(
    stage=6,
    channels=8,
    warp_specialization=True,
    pipeline_stages=2,
    fuse_receive_expand=True,
    receive_expand_mode="pipelined",
    source_fusion=True,
)

result = buffer.dispatch(
    x, topk_idx, topk_weights,
    do_expand=True,
    expert_alignment=8,
    do_cpu_sync=False,
)
```

这里的 `buffer` 必须由该 config 构造；输入、process group 和 CUDA device 的准备沿用现有 API 合同。`vectorized` 和 `pipelined` 要求 `stage >= 5`、`fuse_receive_expand=True`、`warp_specialization=True`；配置不满足时会明确拒绝。`pipeline_stages` 仍为 2 或 3。配置经 `native_options()` 进入已有 Gloo 初始化一致性检查，不能在不同 rank 使用不同模式。

`do_expand=False` 保留 unique API 语义：即使 buffer 配置为新模式，也自动运行原 `advanced_transport`，不创建接收 tile 的动态 shared-memory 区域，诊断返回 None。它不会因为配置了新的接收模式而拒绝 unique dispatch。

实现位置：

- `pcie_ep/config.py`：`HostStagedConfig.__post_init__()`、`native_options()`。
- `csrc/comm/advanced_shared_memory.hpp`：`AdvancedConfig::ReceiveExpandMode` 及字符串解析。该枚举仅参与 CPU launch 选择，没有增加 `AdvancedDeviceState` 字段。
- `csrc/buffers/advanced.hpp`：`bind_state()` 用 `expand && config_.fuse_receive_expand` 生成本轮融合标志；`launch_dispatch()` 仅在实际 fused expanded 路径选择新 kernel；`configure_shared_memory()` 设置对应 kernel 的动态 shared-memory 上限。
- `tests/advanced_options.py`：分布式测试的 `--receive-expand-mode` 参数桥。

## 两个增量实现

`vectorized` 保留原 RX 分工：仍由同一组 RX warp 完成 host read、跨 warp reader join、expert scatter，再处理下一个 ticket。它将 shared tile 到 expert HBM 的 BF16 拷贝显式改为 `uint4`，没有引入独立 scatter warp 组。

`pipelined` 在相同 vector scatter 基础上分离 RX producer 与 scatter consumer。producer 可以开始读取下一 token，同时 consumer 仍在展开上一 token；每个本地 stage 的回收条件保持完整，不能通过提前释放 tile 获得重叠。

入口为 `csrc/kernels/host_staged/fused_receive.cuh` 中的 `advanced_transport_vectorized()` 和 `advanced_transport_pipelined()`。两者复用 `advanced_dispatch.cuh` 的 `advanced_tx_control()`、`advanced_rx_control()`、`advanced_tx_worker()` 和既有 host ring 协议，避免复制两套控制逻辑。

### uint4 与不对齐 hidden 的处理

`scatter_rx_tile_vectorized()` 在 `H % 8 == 0` 时，每个 work unit 拷贝 8 个 BF16，即一个 16-byte `uint4`。三个 scatter warp 共 96 个线程，以 worker index 为起点、96 为步长遍历 `H / 8` 个向量。tile 起点按 16 byte 对齐，且此分支中每个 tile 和 expert row 的长度均为 16 byte 的整数倍，因此行起点也满足对齐要求。

`H % 8 != 0` 时，整个本地 scatter 改用逐 BF16 的标量循环，覆盖全部 H 个元素；当前实现不是“前段 vector + 尾部 scalar”混合循环。该回退保留两种新模式各自的 warp 分工和 stage 生命周期。接收 FP8 时仍先解码到 BF16 tile，再使用相同的 BF16 vector/scalar scatter。

上述选择不改变 token 去重、路由顺序、专家行号或 host 传输字节数。新 kernel 也省去了原 kernel 的 homogeneous 分支，因此资源编译结果不同；完整变体的性能差异不能全部归因于拷贝宽度。

## Warp 分工

下表以 `warp_specialization=True` 为前提；每个 channel 对应一个 progress CTA。

| 角色 | scalar / vectorized：256 threads、8 warp | pipelined：352 threads、11 warp |
|---|---|---|
| TX control | warp 0 | warp 0 |
| RX control | warp 1 | warp 1 |
| TX payload workers | warp 2–4 | warp 2–4 |
| RX host-read producers | warp 5–7；随后同组执行 scatter | warp 5–7；只退休 host read |
| 专用 scatter consumers | 无独立组 | warp 8–10 |

scalar 在 `warp_specialization=False` 时仍运行原 homogeneous CTA，不采用此表中的角色划分。unique 回退使用原 256-thread kernel，RX 数据直接写入 unique GPU 输出，不分配 BF16 shared tile。

pipelined 增加 3 个 warp、96 个线程。动态 shared-memory 公式不变，但寄存器使用、可驻留 CTA 数和 scheduler 行为可能变化，不能由 warp 数量直接推导性能收益。

## Shared 双/三缓冲与独立回收条件

`pipeline_stages=2` 是双缓冲，`pipeline_stages=3` 是三缓冲。本地 ticket t 使用 `t % pipeline_stages` 对应的 RX stage 和 BF16 tile。它们独立于 host payload ring 的 Q 与 descriptor ring 的 L；增加本地 stage 不会扩大 host Q/L。

`advanced_shared_memory_bytes(H, stages, fused)` 的计算保持为：

```text
aligned sizeof(AdvancedLocalQueues) + (fused ? H * 2 * stages : 0)
```

例如 H=7168 时，tile 部分在双缓冲下为 28 KiB，三缓冲下为 42 KiB，另加控制队列空间。三种 fused 模式使用同一 helper 和原 `AdvancedLocalQueues`，没有增加 device-state 或 shared-queue ABI 字段。

单个 RX stage 的正常状态流为：

```text
FREE: ready == 0
  → RX control acquire host descriptor READY
  → 将 descriptor / metadata / scale 快照到 shared stage
  → 归还 descriptor credit
  → release ready = ticket + 1
  → 3 个 producer warp acquire 对应 generation，host payload → BF16 tile
  → 各 producer 完成读取、fence 和 warp join，分别发布 host_readers_done
  → host_readers_done == 3：RX control 归还 host payload ACK
  → 3 个 consumer warp acquire READY 与 reader join，tile → 所有本地 expert rows
  → 各 consumer 完成输出写入、fence 和 warp join，分别发布 workers_done
  → ACK 已发且 workers_done == 3：RX control release ready = 0，stage 才能复用
```

descriptor credit、host payload ACK、本地 stage FREE 是三种不同的回收条件。descriptor 完整快照后即可归还其 credit；host payload 必须等所有 host reader 完成；shared tile 则必须等所有 scatter consumer 完成。pipelined 的 consumer 读取 shared metadata/tile 和本 GPU 的布局数据，写入本 GPU 的 expert 输出，不再访问可能已经被 ACK 并覆写的 mapped host payload。

对应函数为 `advanced_rx_control()`、`fused_receive_copy_host_payload()`、`fused_receive_producer()`、`fused_receive_consumer()`。vectorized 的相同三组 RX warp 在 `fused_receive_vectorized_worker()` 中依次完成 producer 和 consumer 工作，仍遵守相同的 reader join 和 FREE 条件。

## READY 代际与入口一致性修复

specialized TX/RX stage 现在用 `ready = ticket + 1` 发布任务，0 表示 FREE。worker 的准入检查是一次 acquire 读取并比较 `ready == ticket + 1`，不再采用“先看到旧 boolean READY，再读取可能已经更新的 plain stage.ticket”的组合。后者在 slot 复用时存在 ABA 窗口。control/debug 仍可保存 plain ticket，但它不再决定 worker 是否可以读取本轮 stage。

TX control 在发布 host READY 前，仍需等待全部 payload writer 和 metadata writer 各自完成必要的 fence；这一修复没有放宽 host payload 的发布顺序。worker leader 的 acquire 之后还用 `__syncwarp()` 将顺序传递给实际读取 shared metadata/tile 的其他 lane，不能把 `__shfl_sync()` 的寄存器广播当作内存同步。

原 `advanced_transport()` 和新入口都先由线程 0 采样 GPU error 状态，再用 `__syncthreads_or(entry_failed)` 得到不可变的 CTA 一致判定。只有该判定失败时才统一退出。不能在初始 barrier 后重新依据可变的 `queues.abort` 决定某个 warp 是否进入角色循环，否则快速角色写 abort 后，其他 warp 可能退出而让已进入的角色困在末尾 CTA join。

这些安全 guard 同样应用于原 specialized helpers。scalar 仍保持原拷贝和推进算法；`advanced_homogeneous()` 没有采用 generation READY 准入，其函数内容保持不变。角色循环内部不插入要求全 CTA 参与的 barrier；正常完成和 abort 都需让各角色到达统一的末尾 join。

## 同进程多 buffer 的 shared-memory 上限

CUDA 的 `MaxDynamicSharedMemorySize` 属于当前 device/context 中的 kernel function，不能当作某个 buffer 的私有设置。初始化较小 buffer 时直接降低该值，会让仍存活的大 buffer 在下次 dispatch 失败。修复前，scalar H7168/三 stage 与新模式小 buffer 混用，以及新模式 H8192/三 stage 与同模式小 buffer 混用，四组测试均复现 `CUDA invalid argument`。

`AdvancedRuntime::configure_shared_memory()` 现在在进程级 mutex 内先读取 kernel 当前属性，仅在本 buffer 需求更大时提高上限，避免不同线程的 get/set 竞争。没有统一强设设备最大 shared 配额，实际 launch 仍按本轮 shape 和 expand 状态申请字节。get/set 和锁均只在初始化执行，dispatch/replay 不新增 CPU 操作。

永久回归为 `tests/test_f2_buffers.py`，包括两个 H7168 跨模式用例和两个 H8192 同模式、不同 shape/stage 用例；后者的动态 shared 需求为 52928 字节，覆盖 opt-in 边界。修复后四组全部匹配 CPU reference；原始失败与修复后结果分别保存在 `results/f2/regression/pre_cap_fix/` 和 `results/f2/validation/post_cap_fix/all/buffers.json`。

此初始化修复前后的 native 文件哈希不同，但全部 25 个 GPU 函数的完整 SASS、指令编码与资源记录逐字节相同，证据在 `results/f2/kernel_review_final/`。长压力和最终构建复验分别保留执行来源，不把不同 native 文件混记为同一构建。

## Graph、缓存路由、FP8 与 NUMA relay

- **CachedLayout**：新模式仍使用同一份稳定 route plan；每轮 token values 和 weights 来自当前输入。epoch、ACK 和阶段进度不会被缓存。路由 Tensor 身份、版本、shape、owner/group、buffer generation、alignment 和 expand 模式的原有校验不变。
- **CUDA Graph**：kernel 选择在 capture 时固定，由不可变的 buffer config 决定。每次 replay 仍由 GPU 生成新 epoch，并初始化本次 kernel 的本地队列与诊断；无需 CPU count 回读。一个 buffer 仍只支持一份固定地址 Graph，Graph 输出池与 eager 池分离，下一次 replay 会覆盖上一轮 Graph 输出。动态路由图仍每轮重建布局，缓存图仍要求路由不变。
- **FP8 wire**：新模式仅改变解码后的本地接收/展开组织。`copy_rx_payload()` 沿用每 token FP32 scale 将远端 E4M3FN 解码为 BF16 tile；同 GPU 的目标仍从原 BF16 输入旁路拷贝。有限 BF16 输入及 scale 下限 `max(maxabs/448, 2**-126)` 的合同不变。
- **NUMA relay**：`receive_descriptor()`、`receive_metadata()`、`receive_scale()`、`receive_payload()`、`receive_ack()` 仍选取既有 direct/replica 路径。CPU COPIED_ACK、replica GPU ACK 和全 flow drained 的顺序不变；新的 overlap 原子操作只作用于 GPU-local tensor，不在 mapped host 内存增加 RMW。
- **主路径**：配置解析、分配和 kernel attribute 设置发生在初始化；正常 dispatch/replay 不新增 CPU 同步、Gloo collective 或 D2H count/诊断读取。跨 stream consumer 登记和 GPU stream-tail 等待沿用原输出池复用合同。

## 诊断的精确定义与开销

`HostStagedEPHandle.receive_expand_overlap` 仅在新模式实际执行 fused expanded kernel 时提供，为 CUDA `int64[channels, 2]`。scalar 和 unique 返回 None。vectorized 每 CTA 将两个字段置零，之后不更新；pipelined 使用：

| 字段 | 含义与写入者 |
|---|---|
| `[:, 0]` | 保守 overlap 次数。RX producer 组内 warp 0 的 lane 0 开始 ticket t>0 时，若 acquire 读取的指定 scatter 完成 marker `< t`，则加 1。只有这一个线程更新每 channel 的计数。 |
| `[:, 1]` | 专用 scatter 组的相对 warp 2 完成本轮 ticket t 的有效输出写入和 warp join 后，以 GPU device-scope atomic release 写入 `t+1`；producer 以同 scope atomic acquire 读取。它是内部完成 marker，不是通信量或第二个 overlap 计数。 |

这里的 ticket 是单 channel 的接收任务序号，既不是原 token ID，也不是跨 dispatch 的全局 epoch；每次 kernel/replay 开始都清零。这一计数观察“当前 RX 任务已经开始时，尚未观测到前一 ticket 的指定 scatter 完成 marker”，按 marker 可见顺序反映任务退休关系；不证明 HBM load/store 在同一硬件周期执行，不表示完整下一 payload 已读完，不表示时间比例，也不是 PCIe 带宽或 stall timer。计数为零不能排除真实重叠。

`fused_receive_scatter_delay()` 延迟的是 `min(95, work_units-1)` 这个最后仍有工作的 worker，而不是最后一个 vector 的拥有者。H=7168、BF16 vector scatter 时有 896 个 work unit，worker 95 属于相对 scatter warp 2，确实处理 vector 95、191、…、863；延迟它可以针对 marker 对应 warp 构造保守重叠验证。小 H 选择实际有工作的线程，但 marker warp 2 可能没有有效拷贝，因此小 H 的计数允许假阴性。

诊断 tensor 在每个 `OutputSet` 初始化时预分配，逻辑存储为每池 `16 * channels` byte，实际 allocator 对齐/池化开销另计，并纳入 `output_tensors()` 的所有权检查。新 kernel 的每 CTA 初始化完成归零，不增加独立 zero kernel。pipelined 每个非首 ticket 增加一次 GPU-local marker acquire load、每个 scatter ticket 增加一次 release store，正样本还增加一次单 writer 计数更新；这些操作会进入实测时间，不应视为免费。vectorized 只有初始化写入；scalar/unique 不增加诊断初始化 launch。

eager 的诊断 Tensor 与其他输出一样，在外部引用存续时保护所属池；Graph 使用专用池，仍遵守跨 stream consumer 等待和下一 replay 覆写语义。读取诊断到 CPU 只能用于计时外验证或报告，不能由正常 dispatch 自动完成。

实现位置：`fused_receive_initialize()`、`fused_receive_producer()`、`fused_receive_consumer()`；`AdvancedRuntime::allocate_workspace()`、`output_tensors()`、`make_result()`；`HostStagedEPHandle.__init__()`。对应验证入口为 `tests/test_api.py`、`tests/test_f2.py`；实际记录见 `results/f2/validation/`。

## 编译资源与测量边界

SM90/CUDA 13 实际构建的资源与机器码如下，完整输出和 SHA-256 在 `results/f2/kernel_review/`。

| 变体 | 线程/CTA | 寄存器/线程 | Stack frame bytes | 静态 shared bytes | H7168 双 stage 动态 shared bytes |
|---|---:|---:|---:|---:|---:|
| scalar F2 | 256 | 72 | 160 | 1024 | 32448 |
| vectorized F2 | 256 | 56 | 96 | 1024 | 32448 |
| pipelined F2 | 352 | 56 | 96 | 1024 | 32448 |

新两核的 vector 分支实际生成 `LDS.128 → STG.E.128`，原 scalar scatter 为 `LDS.U16 → STG.E.U16`。单个专家行仍搬运 14336 字节；H7168 的理想有效 warp 数据 load/store issue 数从每种操作 224 条降为 28 条。这是拷贝循环的指令量核算，不是 8 倍性能承诺，也不是物理内存事务计数。新增 warp、GPU-local 诊断、shared-memory 访问、控制轮询和 fanout 均进入实际时延。没有采集动态 occupancy/warp-stall 计数，不能由寄存器数量直接推导 occupancy 或主瓶颈。

正式对照固定 C32、Q=L256、双 stage、source fusion、BF16 wire、direct NUMA、expanded/alignment1、相同路由 seed；仅切换当前 scalar F2、关闭 F2、vectorized 和 pipelined。标量和关闭 F2 对照共同包含本次 READY/入口和初始化属性安全修正，保留其原算法与拷贝方式；修正前的原始源码与二进制另存 `results/f2/baseline/`。因此本次结果不是旧二进制未经修改的重放。

完整 dispatch 时延包含布局、counts/prefix、host-staged 传输、本地专家复制以及所需 stream 依赖。F2 的 transport 区间包含融合 scatter，关闭 F2 的 transport 不包含后续独立 scatter；两者只能用 full latency 做端到端等价比较。`full - transport` 仅为两个统计量之差，不能当作独立 scatter timer。

当前正式结果仅展示最新 pipelined 实现，按用户指定的 `bw = T * K * H * b / t` 计算。T 为每张 GPU 的输入 token 数，K=8、H=7168、BF16 的 b=2 字节，t 为完整 dispatch 平均时延（秒）；GB/s 为十进制。分子在去重前计算，包含全部 top-k 分支，不排除本地专家，也不额外乘或除 GPU 数。公开结果在 `benchmarks/data/f2_pipelined_summary.json` 与 `.csv`，从两轮各 20 个原始时延样本合并重算，未重跑 kernel。

历史原始文件中的 `F * H * 2` 远端去重投递量、`U * H * 2` 源端逻辑 host 写量和 `(U+F) * H * 2` host payload 总访问量仍保留作为独立计数，其旧带宽字段不代表当前公式。两轮测试反转执行顺序，每条路径每个 batch 各 5 warmup、20 measurement；原始样本不变。

2026-10-26 清理说明：默认已改为 pipelined，所有模式共用同一 Runtime。历史独立 Runtime 与串行布局不再编译；旧测量和源码快照移出工作目录。参见 [CLEANUP.md](../../CLEANUP.md)。
