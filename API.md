# PCIe host-staged EP Python API（当前统一运行时）

本实现提供单机、单进程单 GPU 的 BF16 token dispatch。跨 GPU payload 由 GPU SM 写入和读取共享 mapped host memory，不使用 CUDA peer access、NVLink 或 NCCL 数据传输。S8 可增加后台 CPU worker，将跨 NUMA payload 复制到目标 NUMA 的 replica ring。初始化、Graph 准备和关闭允许 CPU 协调；正常 dispatch 和 Graph replay 不把 counts 读回 CPU，也不阻塞调用 CPU 等待 GPU 或 relay 完成。

## 最小示例

下面脚本由 `torchrun --standalone --nproc-per-node=8` 启动。专家连续均匀分配：group-local rank `r` 拥有全局专家 `[r * E/P, (r+1) * E/P)`。

```python
import os
import torch
import torch.distributed as dist
from pcie_ep import HostStagedConfig, HostStagedEPBuffer

dist.init_process_group("gloo")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

with HostStagedEPBuffer(
    group=dist.group.WORLD,
    num_max_tokens_per_rank=8192,
    hidden=7168,
    num_topk=8,
    num_experts=256,
    config=HostStagedConfig(stage=6, payload_slots=256, descriptor_slots=256),
) as buffer:
    # x: contiguous CUDA BF16 [T, 7168]
    # topk_idx: contiguous CUDA int64 [T, 8]
    # topk_weights: optional contiguous CUDA FP32 [T, 8]
    recv_x, recv_idx, recv_weights, handle, event = buffer.dispatch(
        x, topk_idx, topk_weights,
        do_expand=True,
        expert_alignment=8,
        do_zero_padding=True,
        do_cpu_sync=False,
    )
    # 默认已给当前 compute stream 添加等待。下游 GPU kernel 直接读取
    # handle.expert_counts 和 handle.expert_offsets，不能先 .cpu()/.item()。
    run_expert_kernel(recv_x, handle.expert_counts, handle.expert_offsets)

dist.destroy_process_group()
```

示例中的 `x`、路由和下游 kernel 由应用提供。构造时必须提供 `num_experts`；构造时预分配输出容量，第一次 dispatch 不隐式扩容。

## 构造与配置

`HostStagedEPBuffer` 也导出为 `EPBuffer`；`HostStagedEPHandle` 也导出为 `EPHandle`。这是一套独立后端，其 handle 不能传给原 DeepEP 的 combine。

| 参数 | 含义 |
|---|---|
| `group` | 现有 EP process group；默认 WORLD。保存其 group-local/global rank 对应关系。 |
| `control_group` | 可选的现有 Gloo group，成员及顺序必须与 EP group 相同。EP group 本身为 Gloo 时直接使用；否则创建辅助 Gloo group。 |
| `num_max_tokens_per_rank` | 每个源 rank 的最大 T；所有 rank 相同。实际每 rank T 可以不同，也可以为零。 |
| `hidden`, `num_topk`, `num_experts` | 固定 H/K/E；E 必须能被 EP world size 整除。 |
| `config` / `host_staged_config` | 二选一，传 `HostStagedConfig`。 |
| `require_reuse_nccl_comm` | 为 True 时必须能通过现有 NCCL backend 的 `_comm_ptr` 借用 communicator，否则明确报错。 |

`HostStagedConfig` 的默认值为：

```python
HostStagedConfig(
    payload_slots=256,
    descriptor_slots=256,
    channels=32,
    stage=6,
    timeout_ms=30000,
    directory=None,
    numa_mode="direct",
)
```

Q/L 表示每个源 ring 和每个源→目标 descriptor 队列的容量，不限制单次 dispatch 的总 token 数。host payload 使用有界 ring，不再保留整批 staging。stage 6/7/8 仅表示同一运行时的能力门槛；stage 1–5 已删除并会报错。

| 配置 | 支持阶段与含义 |
|---|---|
| `channels=32` | 可使用 1–32 个独立 channel。 |
| `warp_specialization=True` | 可分工执行 TX/RX control 与 payload copy。 |
| `pipeline_stages=2` | 支持 2 或 3 个 GPU payload stage。 |
| `fuse_receive_expand=True` | 可将接收与 expert expansion 融合。 |
| `receive_expand_mode="pipelined"` | F2 可选 `scalar` / `vectorized` / `pipelined`；后两者要求开启 F2 和 warp specialization。 |
| `source_fusion=True` | S6+ source-side 路由与发送融合。 |
| `enable_graph=False` | S6+ 预分配独立 Graph 输出池。 |
| `wire_dtype="bf16"` | S7+ 可选 `"fp8_e4m3"`，每 token 使用 FP32 scale；输入和输出仍为 BF16，本地旁路保持原始 BF16。 |
| `output_pool_size=3` | eager 输出池个数。 |
| `numa_mode="direct"` | S8 可选 `"relay"`，跨 NUMA 由后台 CPU worker 主动复制。 |
| `replica_slots=256` | S8 每 channel、每目标 NUMA domain 的 replica ring 容量。 |

高级阶段可组合此前阶段的优化，例如 S8 relay 可与 FP8、缓存路由和 Graph 同用。功能仍需显式启用，单独选择更高 stage 不会自动打开所有开关。

specialized transport 由 TX control warp 持续监测健康状态，即使发送已完成，也保持存活到接收完成。所有 lanes 协作读取 mapped abort flags，扫描间隔为上次扫描完成后的 `min(1 ms, timeout_ms/8)`；本地 timeout 每个有界控制迭代检查。接收 prefix 等待另有协作检查。实际错误响应还包含当前控制迭代和扫描耗时，不能中断已发出的 PCIe 操作。这些策略由运行时管理，没有新增公开调参项；主路径仍无 CPU 同步。增量 tail ACK 与上一轮 counts mailbox 回收协议见 [control-progress](docs/pcie-ep/control-progress.md)，当前十档测量见 [性能结果](RESULTS_CONTROL_PROGRESS.md)。

`receive_expand_mode="vectorized"` 将 shared→expert BF16 拷贝改为 16-byte 向量；`"pipelined"` 再用独立 RX/scatter warp 组流水，CTA 从 256 增至 352 threads。`do_expand=False` 始终回到原 unique 路径。默认改为 pipelined，模式在构造时固定，各 rank 须一致；H 不能被 8 整除时本地 scatter 完整回退标量。详见 [F2 实现与生命周期](docs/pcie-ep/f2-receive-expand.md)和[十个 batch 的当前实测](RESULTS_CONTROL_PROGRESS.md)。

FP8 wire 支持有限 BF16 输入，不保证 NaN/Inf 的传输语义。非零 token 的 scale 为 `max(maxabs/448, 2**-126)`，零 token 的 scale 为 1。`2**-126` 是 FP32 最小正规数；该下限防止 BF16 极小值或 subnormal 产生无穷大的倒数，进而使零分量在量化时变成 NaN。解码按 FP8 值乘以 scale 后舍入回 BF16。

### S8 worker 与 CPU affinity

```python
config = HostStagedConfig(
    stage=8, channels=4, numa_mode="relay",
    payload_slots=256, descriptor_slots=256, replica_slots=256,
    relay_workers_per_numa=2,
    relay_cpu_affinity=(0, 2, 64, 66),
)
```

`relay_workers_per_numa` 默认 1，必须是 `[1, channels]` 内的整数。每个 source rank 为每个目标 NUMA domain 创建这些 worker，并将 channel 分配给不同 worker；worker 数 2/4 的对比应保持 channel 数至少为 2/4。增加 worker 不增加 payload 或 replica ring 容量。

`relay_cpu_affinity` 默认空 tuple，表示从该 NUMA node 上当前进程允许使用的 CPU 中选择。非空时必须为互不重复的非负 CPU 编号 tuple；native 再取指定 CPU、目标 NUMA node 和实际允许 affinity 的交集，然后固定 worker 的 CPU。交集为空会在集体初始化时报错。示例 CPU 编号需要按机器实际拓扑替换。

非默认 worker 数或非空 affinity 仅支持 S8；S8 direct 模式允许保存这些配置，但仅在 relay 模式创建 worker。所有 rank 需使用一致的配置。初始化会收集每个 rank 实际 GPU 对应的 NUMA node，支持非连续 GPU 选择和子 group。实际 page placement 可用显式 `buffer.runtime.relay_diagnostics()` 查询；查询和 CPU 数据读取应放在计时区外。

默认在 `/dev/shm` 中创建每 buffer 独立的目录。`directory` 可指定父目录；具体共享 backing 子目录仍由实现创建。初始化先由各 source owner 进行 NUMA 放置和 first-touch，再经 Gloo 屏障由所有 GPU 注册。NUMA 绑定受操作系统权限和部署配置约束，不能仅根据请求的 node 推断实际页位置。

构造、配置和关闭必须由 group 内所有 rank 按相同顺序调用；禁止在部分 rank 构造成功后让其他 rank 进入另一套 collective。

## 输入与输出

`dispatch(x, topk_idx, topk_weights=None, ...)` 默认返回五项：

```python
recv_x, recv_topk_idx, recv_topk_weights, handle, event
```

输入的 T/H/K、dtype 和连续性在 CPU 上按 tensor metadata 检查，不读取 GPU 内容。`topk_idx` 使用全局专家编号；`-1` 表示该 top-k slot 不路由。其他负值、越界专家或同一 token 重复选择同一专家会触发 GPU 错误与 abort。不同专家落在同一 GPU 是正常情况。

| 布局 | `recv_x` | `recv_topk_idx` | `recv_topk_weights` |
|---|---|---|---|
| `do_expand=False` | `[unique_capacity, H]` | `[unique_capacity, K]` | `[unique_capacity, K]` |
| `do_expand=True` | `[expanded_capacity, H]` | `[expanded_capacity]` | `[expanded_capacity]` |

输入未提供 weights 时，返回的 `recv_topk_weights` 为 None。输出专家编号为目标 rank 的**本地专家编号**；unique 布局中不属于当前 rank 的 top-k slot 为 `-1`，对应权重为 0。

输出 shape 表示预分配容量，不能直接作为有效 token 数。有效范围由 GPU handle 描述：

| Handle 字段 | GPU 含义 |
|---|---|
| `unique_count` / `num_recv_tokens` | `[1]`，unique 布局前多少行有效，包含本地旁路和远端接收。 |
| `expert_counts` | `[E/P]`，每个本地专家的真实行数，不包含 padding。 |
| `expert_offsets` | `[E/P+1]`，考虑 alignment 的 exclusive offsets。专家 e 的有效行范围为 `[offset[e], offset[e]+count[e])`。 |
| `expanded_count` / `num_expanded_tokens` | `[1]`，真实专家分支总数，不包含 padding；有 padding 时不能把此前缀长度当作所有有效行。 |
| `source_offsets` | `[P+1]`，unique 输出按源 rank 分组的 exclusive offsets。最后一个值等于 unique_count。 |
| `psum_num_recv_tokens_per_scaleup_rank` | source_offsets 去掉首零后的 inclusive prefix `[P]`。 |
| `psum_num_recv_tokens_per_expert` | 每专家 padded 起点加真实 count；与 DeepEP expand 的 prefix 语义一致。 |
| `source_meta` | `[capacity,3]`。unique 行为 source group-local rank、source token index、`-1`；expanded 行第三项为原始 top-k slot。 |
| `global_ranks` | CPU tuple，将 metadata 中的 group-local source rank 映射到全局 rank。 |
| `error` | GPU `[8]` 错误状态。正常全零；出现错误后本 buffer 不应继续使用。 |
| `stats` | GPU `[8]`，每轮清零。前三项为 source host payload 写 token 数 U、remote payload 读 token 数 F、逻辑目标 GPU 投递数。S5+ 的 `[3:8]` 依次是 metadata/payload 重叠事件、RX not-ready 探测、TX credit miss、TX local-stage miss、实际 endpoint descriptor 发布次数；这些是事件计数，不是停顿时间。S8 的 endpoint 可聚合多个目标 GPU，因此最后一项可少于 F。 |
| `transport_timestamps_ns` | GPU `[2]`，底层 transport 区域的 `%globaltimer` 时间戳。与完整 dispatch 时延分别报告。 |
| `receive_expand_overlap` | 新 F2 expanded 路径的 CUDA int64 `[channels,2]`：列 0 为保守 RX/scatter 任务交叠计数，列 1 为指定 scatter warp 的完成 ticket 水位。vectorized 为零，scalar/unique 为 None；不是时长或利用率，CPU 读取仅用于计时外诊断。 |
| `prefix_tx_timestamps_ns` | CUDA int64 `[channels+1,2]`。行 0 为接收 prefix producer 开始/结束时间；行 `channel+1` 为该 channel 首次 TX payload 开始、以及全部 payload workers 和 metadata 写入完成的时间。无 remote payload 时对应 TX 行为零；homogeneous 不记录 TX 行。使用 GPU `%globaltimer`，仅用于计时外诊断。 |

specialized 路径在 RX 控制 warp 中准备接收 prefixes，与 TX 并行；本地 token copy 随 transport 完成后执行，保持原输出行号和顺序。时间戳不保证每轮自然观测到重叠；首次 TX 完成早于 prefix 结束才是该轮的直接重叠证据。内部 `receive_prefix_state` 是每个输出池私有的 GPU claim/ready，不是 handle 字段；详细协议、缓存与 Graph 语义见 [接收计数与 TX 并行](docs/pcie-ep/prefix-tx-overlap.md)。

`num_recv_tokens_per_expert_list` 始终为 None。下游 kernel 需要直接消费 GPU counts/offsets。`do_cpu_sync=None` 与 False 相同；显式 True 会报不支持。

支持 `expert_alignment` 1–256。`do_zero_padding=True` 仅可与 expand 同用，会清零专家段间 padding，padding 的专家编号和 source metadata 标为 `-1`。容量尾部不承诺初始化。

## 预分配结果池

运行时默认预分配三个 eager 输出池，可用 `output_pool_size` 在 2–8 之间调整。结果仍被外部 Tensor 引用时不会覆盖。所有输出池都被占用时，再提交 dispatch 会得到 `pool exhausted`，不会在 CPU 上等待，也不会隐式扩容。

仅保留某个 weights、metadata 或 count Tensor 同样会占用其池。需要长期保留时，在正确的 CUDA stream 依赖下 clone 所需数据，然后释放原始结果、handle、event 和 tensor 引用。常规循环只保留最后一轮结果即可轮换使用池。输出容量与最大 T/K/H 有关；启用 Graph 时还会额外预分配一个专用输出池。

`topk_idx` 不能在 handle 使用期间原地修改。Python version counter 可以检测普通 in-place 修改，无法保证检测 `.data` 或 raw pointer 写入。

## S6 缓存路由与 CUDA Graph

```python
first = buffer.dispatch(x, topk_idx, topk_weights, do_expand=True)
cache = buffer.cache_layout(first[3])   # 同 first[3].cached_layout
del first
result = buffer.dispatch(x, topk_weights=topk_weights,
                         handle=cache, do_expand=True)
```

`CachedLayout` 只保留独立的 native 路由计划、原路由 Tensor 及其版本和 buffer 身份，不保留首轮输出池或完整 handle。可长期持有它进行重复 dispatch；`x` 和 weights 可更新。复用时须省略 `topk_idx`，原路由身份、版本、T/H、expert ownership、group、buffer generation、alignment 和 expand 模式必须一致。原路由为 inference tensor、缺少版本计数时不能缓存。

独立路由计划也来自有界预分配池，其容量为 `output_pool_size`。长期持有多份不同布局的 cache 可能耗尽路由计划池；这与占用输出池是两种不同的资源限制，均不会隐式扩容。

```python
# 构造时设置 HostStagedConfig(stage=6, enable_graph=True, ...)
graph = buffer.capture_dispatch(x, topk_idx, topk_weights, do_expand=True)
result = graph.replay(async_with_compute_stream=False)
```

每个 buffer 支持一个固定输入/输出地址的 Graph。显式 capture 可同步 CPU，并会集体报告准备或捕获错误；后续 replay 只提交 GPU 工作，返回标准五项。动态 Graph 每次重新计算路由，允许在 producer stream 上原地更新 x、weights 和路由内容，但不能改变其地址、形状或 stride。

静态缓存图使用 `buffer.capture_dispatch(x, topk_weights=weights, handle=cache, ...)`；原路由的 PyTorch 版本每次 replay 都需保持一致。GPU kernel、raw pointer 或另一个已捕获的 Graph 直接改写路由内容不会自动更新 CPU 版本计数，此类写入违反缓存路由的不可变合同；动态路由应使用未缓存的 Graph。

Graph 可与 eager dispatch 交错，使用相同的 producer/consumer stream 依赖保护。其结果使用专用输出池，**下一次 replay 会覆盖上一轮 Graph 输出**；保留历史结果需自行 clone。需要不同地址或另一份 capture 时重建 buffer。`graph.close()` 是允许显式同步的销毁操作。

## Stream 与 EventOverlap

每个 buffer 有独立的 communication stream，`buffer.get_comm_stream()` 返回该 stream。默认 dispatch 让 communication stream 等待当前 compute stream，再让 compute stream 等待 completion event；调用 CPU 不等待 GPU 完成。

```python
result = buffer.dispatch(x, topk_idx, topk_weights,
                         async_with_compute_stream=True)
do_independent_compute()
result[4].current_stream_wait()  # 在当前 GPU stream 上添加等待
consume(result[0], result[3].expert_counts)
```

输入由其他 producer stream 生成时，可以在 producer 上 `HostStagedEPBuffer.capture()`，传 `previous_event=...`；必须同时设置 `allocate_on_comm_stream=True`。也接受原生 torch CUDA Event 或提供 `current_stream_wait()` 的事件对象。传入 previous_event 后，应由应用确保该事件覆盖全部输入准备依赖。

EventOverlap 支持 `.wait()`、`.current_stream_wait()` 和 `with event:`；上下文退出时才给当前 stream 添加等待。`event.event` 是依赖事件；完整时延应使用应用自行创建的 `torch.cuda.Event(enable_timing=True)` 测量，不能假设依赖事件启用了 timing。

在独立 consumer stream 上等待 EventOverlap 后，该 stream 会登记到 buffer。后续 dispatch 在复用结果池前，通过 GPU `wait_stream` 等待已登记 consumer 的当前队尾；这同时保护已经排队读取、但 Python Tensor 引用已释放的情况。只做 `record_stream` 不能阻止向预分配内存直接覆写，因此两类保护都会保留。stream 登记持续到 close，仅持有 stream，不保留历史结果。CPU 调用必须串行：先排入所有需要保护的 consumer GPU 工作，再提交下一次 dispatch；跨 CPU 线程并发提交不属于 S4 合同。

跨 stream 消费必须使用 EventOverlap 的等待方法登记 consumer；直接对 `event.event` 调用原生 `stream.wait_event` 不会登记，无法提供结果池复用保护。

使用 previous_event 时，当前 compute stream 若已登记为旧结果的 consumer，仍需额外等待其当前队尾：旧结果消费可能排在 previous_event 之后。该保守保护可能减少部分重叠，S4 优先保证池复用正确性；更细的 per-pool 依赖属于后续优化。

`defer_epilogue=True` 要求 `async_with_compute_stream=True`，此时 dispatch 直接返回 EventOverlap。首次 `.wait()` 返回 `(recv_x, recv_idx, recv_weights, handle)`，之后的 wait 只建立依赖并返回 None。S4 的 GPU 工作已在 dispatch 时提交，deferred hook 不进行 CPU count 回读。

## 错误诊断与销毁

`buffer.check_errors(handle=None)` 和 `handle.check_errors()` 是**显式诊断操作**，会等待 CUDA event 并把 error 状态读回 CPU，不能放入正常主路径或测量区间。检测到 GPU 错误后 runtime 保持 abort 状态，后续 dispatch 明确拒绝；应集体关闭并重新构造 buffer。

`buffer.close()` / `.destroy()` 集体排空并注销映射，再由 owner 删除共享 backing。它们不销毁借用的 NCCL communicator。不要依赖 Python 析构函数发起 collective；推荐显式 close 或 `with HostStagedEPBuffer(...)`。

如果 GPU 等待或 peer barrier 失败，不能假设其他 GPU 已停止读取，也不能直接删除或解除映射。S4 的错误处理不是跨进程 crash 恢复协议。

## NCCL group 与支持边界

传入现有 NCCL group 时，仅通过现有 backend `_comm_ptr` 借用原始 communicator，并保留 group 强引用。实现不创建 managed NCCL communicator、不调用 NCCL payload collective，也不把控制组切换到 NCCL。`comm_reused` 与 `comm_reuse_status` 可查询实际状态；公开 PyTorch 版本缺少 `_comm_ptr` 时会记录 unavailable，不能称为已经复用。

当前不支持 combine/backward、非默认 `num_sms` / `num_qps`、handle copy 和 DeepEP FP8 scale-factor layout。S7 的 FP8 仅改变 wire payload，仍接受和返回 BF16 Tensor。stage 1–5 不再提供运行时实现；FP8 与 relay 分别要求 stage 7+ 与 stage 8。
