# 当前代码与历史路径清理

2026-10-26。当前只编译一个 native 运行时 `AdvancedRuntime`。默认配置为
stage 6、32 channels、source fusion、warp specialization、双 stage 和
`receive_expand_mode="pipelined"`。stage 6/7/8 是同一运行时的能力门槛，
不是三份版本；stage 1–5 已删除并会在构造时明确报错。

## 已移除

- 独立 S1–S4 `Runtime`、其 pybind 导出及 Python 选择分支。
- 整批 staging 和旧单 CTA ring transport；旧运行时的映射/注册辅助函数。
- S5 串行 route/channel layout，统一使用当前并行布局。
- 旧 batch/ring 的未使用 host 内存区，以及旧 route plan 的未使用 GPU 数组。
- 一次性执行脚本、旧阶段 benchmark/validation runner、历史报告生成器和模板。
- 重复的 distributed correctness 入口，合并到 `tests/test_correctness.py`。

## 保留的当前能力

unique 与 expanded 输出、GPU counts/metadata、scalar/vectorized/pipelined 接收选项、
可选独立 scatter、source fusion 开关、同构/warp specialization、缓存布局、
CUDA Graph、FP8 和 NUMA relay 都仍由当前 API 直接调用并有回归测试覆盖。
它们共享同一运行时和 ring 协议；选择某种配置时暂未执行的分支不等于历史死代码。
NCCL communicator 借用和 `torch.distributed` group 复用保持不变。

API 迁移：原 `HostStagedConfig(stage=4)` 等历史配置须改为
`HostStagedConfig()`，或显式使用支持的 stage 6/7/8。`pipeline_stages=2/3`
仍然表示 GPU 本地流水缓冲数，与开发阶段编号无关。

## 内存变化

在 P=8、E=256、H=7168、BF16、Q=L=256 的配置下，旧公共布局中未使用的
batch/ring 区占 4,317,184 字节/源，精简后的 count mailbox 为 4,096 字节/源。
共移除 4,313,088 字节/源，八卡合计 32.90625 MiB 的无用 host 注册内存。
有效 payload/descriptor ring 容量未减小；Tmax 语义保持不变。

共享布局和 kernel 参数 ABI 随清理更新。必须重新编译扩展、重新构造所有 rank
的 buffer，不能让新旧二进制同时使用同一 shared-memory segment。

## 实验材料与测量版本

历史日志、原始数据、源码快照与旧二进制整体移到仓库外的私有归档目录，
不再混在工作目录中。Git 历史保留，未重写已发布提交。
公开仓库仅保留匿名时延样本和最新指标口径表。

现有性能表来自清理前的已测提交 `6a927b5`。
`benchmarks/data/f2_measured_runtime_source_sha256.json` 标记的是该测量版本，
不把历史数值或 fingerprint 冒充为清理后新构建的测量结果。
清理版另行执行构建、功能和同步回归；原始验证输出保留在本地 `results/`。

## 清理版验证

- CUDA 扩展重新构建通过，旧 `Runtime` 确认不再导出。
- 57 项 CPU API、缓存/Graph、FP8 和报告合同测试通过。
- 212 条 F2 内容、延迟、交叠、Graph/缓存与 BF16/FP8/direct/relay 组合记录通过。
- 默认配置与显式 scalar/source-unfused、homogeneous/独立展开通过逐元素参考核对。
- 4 组同时存活的不同大小 buffer、NCCL communicator 借用、输出池、consumer stream、超时回归通过。
- Q/L=1/2、2/1、3/2 各 1000 次/卡队列压力通过。
- T8192/H7168 的 unique+expanded 共核对 6,247,485,440 个 BF16 元素。
- eager 与 cached Graph 共 16 个 rank 的 profiler 稳态区域无 CPU 同步、scalar 回读或 D2H memcpy。
- 新构建的 mapped-host 路径审计通过，回归窗口 288 项 NVLink payload 计数增量全部为零。
