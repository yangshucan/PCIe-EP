# 接收计数与 TX 并行

当前 specialized dispatch 将接收计数交换和输出前缀计算放进 transport 的 RX 控制 warp，使 TX 不必等待全部源的接收布局就绪。改动保留既有 host ring、发送与接收去重、稳定输出顺序和 Python dispatch 接口。主路径只使用 GPU stream 依赖和 GPU/host flag 握手，不等待调用 CPU 同步。

## 执行顺序

源端路由、稳定 ordinal、channel 计划和 `publish_counts` 仍在 transport 前执行；这些源端计数尚未与 payload 发送融合。本次重叠的是**接收侧 count 等待和 prefix 计算**与 TX。

```text
begin_epoch：单 CTA、256 threads
  → route_flags / stable_ordinals（缓存布局时跳过）
  → prepare_channels_parallel
  → publish_counts
  → transport
      ├─ TX control + TX workers：立即发送
      └─ RX control warps
          ├─ 一个已驻留 warp 认领 prefix producer
          │   → 读全部源计数、计算 prefix、发布 prefix_ready
          └─ 其余 RX 等待 prefix_ready
              → RX host read / expert scatter
  → local token copy
  → 可选 padding 或独立 expert scatter
  → statistics reduction
```

`advanced_begin_epoch` 由原单线程改为 256 threads，分摊诊断清零和终止 flag 检查。先完成清零并同步，再记录本轮新错误和更新 epoch；mapped abort flag 保持终止语义，不随诊断清零而复位。

本地 token copy 移到 transport 后，在同一 communication stream 上执行。expanded 输出仍按本地专家、source rank、原始 token 顺序排列；unique 输出仍按 source rank、原始 token 顺序排列。本地与远端 source 的输出区间由同一组 prefixes 确定，因此执行先后调整不改变行号，也不会覆盖彼此。

## Prefix 生产者与 mailbox 握手

每个输出池私有预分配一个 GPU `uint64[2]` 控制区 `receive_prefix_state`：

| 位置 | 含义 |
|---|---|
| `[0]` | claim。初始为 0；首个通过本 GPU CAS 的 RX 控制 warp 写入当前 epoch，成为 producer。 |
| `[1]` | ready。初始为 0；成功时发布当前 epoch，失败时发布独立的失败标记。 |

它不是 Python handle 的公开字段，也不是 mapped host memory。CAS 和 READY 原子操作都仅作用于当前 GPU，不引入跨 GPU 或跨 CPU/GPU 的 RMW。

producer 由已经执行到认领点的 warp 产生，不依赖某个固定 CTA 被调度。它为每个 source 执行：

1. 以 system-scope acquire 等待该 source 的 `counts_ready == epoch`。
2. warp 各 lane 合作读取 rank、expert 和**全部 channel** counts，写入本 GPU 的 prefixes 和 `expected_receive_counts`。
3. 所有 lane 完成读取及其相关写入，经过 fence 和 warp join 后，再由 leader release 发布该 source 的 `counts_consumed`。

不能让各 channel 分别读取自己的 mapped counts，却在它们读完之前提前 ACK；ACK 允许 source 在后续 epoch 复用 mailbox。其余 RX 控制 warp 等待的是 GPU 内的 READY，随后读取已保存的 GPU counts，不再次访问这份 mapped mailbox。

所有 prefix 写入完成后，各 lane 执行 device fence 并 join，由 leader 以 device-scope release 发布 `ready=epoch`。等待者的 leader acquire 匹配 epoch 后执行 `__syncwarp()`，将可见性传给本 warp 的 prefix 读取者。后续 RX stage 的 release/acquire 握手继续将布局依赖传给 payload 与 scatter workers。

TX 不读取 receive prefixes：它依赖已经准备好的本源 route/channel 计划、输入 payload 和 epoch。因此其他 RX 等待 READY 时，各 CTA 的 TX 控制与 payload workers 可以继续运行；host payload 和 descriptor credit 不变，仍受原来的环形队列反压约束。

## 失败、缓存与生命周期

producer 在 count 等待中检测 timeout 和终止信号；失败时记录 GPU error 并发布失败 READY。每个 RX 控制 warp 随后设置本 CTA 的 `abort` 和 `rx_finished` 后退出，由该 channel 的 TX-control leader 独占发布 mapped channel abort。prefix producer 不另写 transport 期间的 mapped channel abort，避免增加写者。各角色仍通过完成或 abort 退出后到达最终 CTA join。

控制区属于输出池，多个 buffer 不共享它。每次 dispatch 或 Graph replay 都先运行 begin_epoch，重置 claim/ready，并在 GPU 上增加 epoch；READY 匹配 generation，不能把前一轮完成状态当成本轮结果。输出池、consumer stream 和 Graph 专用池沿用原有生命周期规则。

本 rank 的 cached layout 只复用其源端路由、稳定 ordinal 和 channel 计划，并不保证其他 rank 的路由不变。因此 cached dispatch 仍发布和接收本轮 counts，重新计算接收 prefixes。普通 dispatch、动态 Graph 和缓存 Graph 使用同一接收侧握手。

`T=0`、全本地路由和单 rank 也要完成 prefix/count 流程，以得到正确的零计数并消费本轮 mailbox。NUMA relay 仍由 `counts_ready`、endpoint counts 和 `relay_drained` 协调，TX 提前运行不改变 source/replica ring 的所有权和 ACK 条件。

## Homogeneous 路径

`warp_specialization=False` 的 homogeneous transport 没有独立 RX 控制 warp，其主循环使用全 CTA barriers。因此保留 transport 前独立的 prefix kernel，使用同一 warp 协作计数算法，再进入 homogeneous transport。

specialized scalar、vectorized 和 pipelined 路径共用 `advanced_rx_control`，均在 transport 内准备 prefixes。unique 输出也沿用这项重叠，不要求启用 expert expansion。FP8 量化仍是 transport 前的独立 kernel。

## Kernel 数量

下表为 `T>0`、BF16、specialized fused expanded 路径的原生 launch 数量。padding 指 `do_zero_padding=True` 时启动的尾部 kernel；没有 padding 时关闭该选项。表中不包含应用的下游计算或可选累计专家统计。

| 路由计划 | 不启动 padding | 启动 padding |
|---|---:|---:|
| 每轮重新计算 | 9 → 8 | 10 → 9 |
| cached layout | 7 → 6 | 8 → 7 |

省掉的是前置 receive-prefix kernel，prefix 工作本身进入 RX 控制 warp。begin_epoch 和本地 copy 的 launch 仍保留。FP8、空输入、独立 scatter 和 homogeneous 配置的数量需要按实际分支计算，不能直接套用此表。

## GPU 诊断

`handle.prefix_tx_timestamps_ns` 是 CUDA int64 `[channels+1, 2]`，使用同一 GPU 上的 `%globaltimer` 纳秒时间戳：

| 行 | 列 0 | 列 1 |
|---|---|---|
| `0` | prefix producer 开始 | producer 完成计数/prefix 处理、即将发布 READY |
| `channel+1` | 该 channel 第一 token 的指定 TX worker 开始 payload copy | TX control 观察到第一 token 的全部 payload workers 和 metadata 写入完成 |

TX 完成时间记录于 descriptor READY 发布之前，表示 payload 与 metadata 已完成写入和相应 fence，不等于接收端完成时间。prefix 区间不包含前置源端 route/count 计算。没有 remote payload 的 channel，TX 两列保持 0；homogeneous 路径不记录这些 TX 行。

若第一 token 的 TX 完成时间位于 prefix 开始与结束之间，可直接证明这轮实际数据发送与 prefix 准备发生重叠。正常运行不保证每轮都观测到这种关系：计数已到达、调度和工作量不同都可能使 prefix 很快完成。时间戳不是累计停顿、带宽或利用率；完整 dispatch 时延仍需用覆盖整个 dispatch 的 CUDA events 测量。

native `set_test_prefix_delay_ns()` 仅是定向测试 hook，用来延长 producer 区间并检验 TX 在此期间完成，以及 READY 在后续 eager/Graph 调用中正确复位。它不属于公开 `HostStagedConfig`，生产使用保持为零。诊断读取到 CPU、错误检查和测试断言都放在计时区外；正常 dispatch 不为这些诊断执行 CPU 同步。

实现入口为 `csrc/buffers/advanced.hpp`，producer 和 READY 协议位于 `csrc/kernels/host_staged/advanced_layout.cuh`，TX/RX 角色位于 `advanced_dispatch.cuh` 与 `fused_receive.cuh`。接口与输出池规则见 [Python API](../../API.md)。本文件说明执行和同步逻辑，不给出未测量的性能结论。
