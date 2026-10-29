# PCIe-EP：CPU host buffer 实测

2026-10-27，在单机 8 张 H20D 上测试。当前默认 direct 配置实际占用 **2,494,201,856 B = 2.323 GiB** 的共享 CPU backing，每个源 segment 为 **297.332 MiB**。从每卡 T16 到 T8192，占用保持不变，close 后对应映射和 backing 全部释放。

## 被测版本和条件

- 本地版本：`865cc495eafbd63e9745dd0a81c9d03d824afc89`，测试前工作区干净；新增测量脚本，生产实现未修改。
- 测试在独立工作目录和容器中完成；具体机器路径与容器标识保存在未纳入版本管理的本地原始记录中。
- DP8/EP8、256 experts、top-k8、hidden7168、BF16、Tmax8192、C32、双 stage、source fusion、pipelined F2、expert alignment1、output pool3。
- 每组构造一次，按8卡最大值报告构造/释放时长；不包含扩展编译、模块加载、CUDA context 的预热。注册阶段包含 GPU 输出工作区分配和 Gloo 初始化协调，不是纯 cudaHostRegister 耗时。
- 每个 batch 运行1次预热和3次测量以触达数据路径，再做错误检查、全量 expert/unique/expanded 数量核对及每个非空专家首尾行的完整 hidden 内容核对。共96个 rank×batch 内容检查通过。本次着重测内存，不替代正式通信性能测试。

## 三组实测

`Q/L/R` 分别是 payload、descriptor、replica 的每 channel 槽位数。

| 配置 | 逻辑副本域 | 每源 segment MiB | 整机唯一 backing GiB | 构造最大耗时 s | close 最大耗时 s |
|---|---:|---:|---:|---:|---:|
| 默认 direct，S6，Q=L=R=256 | 1 | 297.332 | **2.323** | 0.628 | 0.503 |
| direct 容量对照，S6，Q=L=R=64 | 1 | 74.348 | **0.581** | 0.164 | 0.122 |
| relay 容量对照，S8，Q=L=R=256 | 2 | 446.996 | **3.492** | 0.810 | 0.726 |

缩容对照和 relay 对照使用 T128；默认配置覆盖原来的10个 batch。它们用于比较容量，不据此推荐缩容后的吞吐或评估 relay 性能。三组在构造后与 dispatch 后均为全量 backing 驻留，释放后对应文件字节、mapping RSS/PSS 和映射数量均为0。

## 默认配置的实际组成

布局字节分解由独立计算生成，其总量与原生诊断 `segment_bytes`、8个 backing 文件的 fstat 大小和实际分配块数校验一致。

| 区域 | 整机 MiB |
|---|---:|
| Source payload | 1024 |
| 预留 replica payload | 1024 |
| Source ACK + replica ACK | 136 |
| Source descriptor + replica descriptor | 136 |
| Source metadata + replica metadata | 58 |
| Scale、计数、控制信号及剩余对齐 | 0.656 |
| **合计** | **2378.656 MiB = 2.323 GiB** |

H7168/BF16 的有效 payload 为14336 B，槽位 stride 为16384 B。stride 的对齐增加内存容量，payload 拷贝仍只搬有效数据。

**修正此前估算：** 硬件有两个 NUMA 节点，但最新 `AdvancedExtensionAdapter` 仅在 relay 模式收集节点拓扑。direct 模式传入空节点列表，原生配置构造1个占位副本域。因此此前按两个副本域估算的3.492 GiB对应本次 relay 对照，不能作为当前默认 direct 的占用。

direct 下仍预留、清零并注册 replica 区。该区域的 payload、metadata、ACK、descriptor、scale 合计 **1,238,630,400 B = 1.154 GiB，约占默认容量49.7%**。如果未来仅裁掉这些未使用区域，按布局估算可降到约1.169 GiB；本次没有修改实现，也没有实测这个裁剪方案。

## 默认配置：十个 batch 的容量稳定性

| 每卡输入 T | 实际分配 backing GiB |
|---:|---:|
| 16 | 2.323 |
| 32 | 2.323 |
| 64 | 2.323 |
| 128 | 2.323 |
| 256 | 2.323 |
| 512 | 2.323 |
| 1024 | 2.323 |
| 2048 | 2.323 |
| 4096 | 2.323 |
| 8192 | 2.323 |

## 共享物理内存与进程 RSS

- 8个 source 文件实际分配块数合计：**2,494,201,856 B**。按文件 inode 计算一次，不把8个映射进程重复累计。
- 每个进程的这8段 mapping RSS：**2.323 GiB**；8进程简单相加会得到 **18.583 GiB**，这是对共享页的重复计数。
- 8进程 mapping PSS 合计：**2,494,169,088 B**，与 backing 仅差32 KiB的内核 KiB舍入，支持唯一物理占用约2.323 GiB。
- 构造后和 dispatch 后的 Size/RSS/PSS一致，Swap为0，close 后这些目标 mapping 全部消失、backing 文件全部删除。
- 原生 cudaHostRegister 全段注册成功，但此次 `/proc/status` 的 VmLck/VmPin 和 smaps Locked 都为0。这些字段不能用来判断 NVIDIA 驱动是否成功 pin；此处分别报告 backing、驻留、PSS和注册成功事实，不把 VmLck 当作 pinned-byte 计量器。
- 上述大小专指共享 buffer 的 backing，不包括驱动登记/页表元数据、Python/PyTorch 匿名内存和 GPU 输出工作区，不能当作整个程序的 CPU 内存总开销。

## 边界与验证证据

本次容器的 `mbind` 返回 EPERM，所有 rank 都保留该原始诊断；没有修改容器权限绕过它。因此 relay 对照验证的是容量和内容正确性，不能据此声称副本页已经成功绑定目标 NUMA 或推导本地化性能收益。原始 `/proc/self/numa_maps` 也已保存。

所有 dispatch 调用使用 `do_cpu_sync=False`。为了读取内存与验证输出，脚本在生产调用之外显式等待 CUDA、进行 Gloo 汇总和 CPU 内容检查。CPU buffer 测量代码没有加入生产路径。

最终执行退出码0，74文件起止源码 manifest 完全一致；8 GPU×18 link×Tx/Rx 的288项 NVLink payload counter 均完整、单调且增量为0。原生二进制 SHA-256为 `34794daef19168f2054f8ca89e9a827e1d46af0285029682f678e2c8925e33c1`。

两次早期仪器修正的记录保留在 `attempt1/`、`attempt2/`：第一次数值测量完成后，文件删除检查缺少外部观察屏障；第二次默认和缩容已通过，relay测试误用S6被API拒绝。最终脚本增加 close 后的测量屏障，并将relay设为S8，完整三组重新通过。生产实现始终未修改。

原始汇总 JSON、逐进程记录、源码快照与 NVLink 审计仅保存在本地 `results/host_buffer/20261027/`，不上传 GitHub；公开报告保留上述汇总数值和[测量脚本](benchmarks/host_buffer.py)。

在具有八卡 CUDA/PyTorch 环境的机器上，从仓库根目录使用新输出目录复现：

```bash
torchrun --standalone --nproc-per-node=8 benchmarks/host_buffer.py \
    --output results/host_buffer/reproduce_$(date -u +%Y%m%dT%H%M%SZ)
```

复现命令只运行测量脚本；前后 NVLink 审计可使用仓库内的 `tests/capture_path_evidence.sh` 和 `scripts/check_path_counters.py`，原始输出应保留在被 Git 忽略的 `results/` 目录中。
