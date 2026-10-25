# DeepEP 风格可读性重构

日期：2026-10-24。重构保持 S0–S4 协议、公共 API、输出池和无 CPU 同步主路径，改善目录职责、命名、排版与协议注释。

| 入口 | 职责 |
|---|---|
| [runtime.cu](./csrc/runtime.cu) | 薄 pybind 导出 |
| [host_staged.hpp](./csrc/buffers/host_staged.hpp) | workspace、输出池、输入检查、launch 与返回组装 |
| [host_shared_memory.hpp](./csrc/comm/host_shared_memory.hpp) | 共享段、NUMA 放置、注册与解除映射 |
| [common.cuh](./csrc/kernels/host_staged/common.cuh) | 协议 ABI、DeviceState、原子封装与错误码 |
| [layout.cuh](./csrc/kernels/host_staged/layout.cuh) | 路由去重、counts、prefix、本地 token |
| [dispatch.cuh](./csrc/kernels/host_staged/dispatch.cuh) | 整批基线、ring 收发推进、专家展开 |
| [ep.py](./pcie_ep/buffers/ep.py) | Python buffer 门面、stream 依赖与生命周期 |
| [handle.py](./pcie_ep/buffers/handle.py) | GPU 输出元数据与路由版本 |
| [event.py](./pcie_ep/utils/event.py) | EventOverlap 与 consumer stream 登记 |
| [comm.py](./pcie_ep/comm.py) | 借用 NCCL、Gloo 控制与集体初始化 |

完整约定见[代码风格说明](./docs/pcie-ep/code-style.md)。`pcie_ep.buffer`、顶层导出和底层 Runtime 签名保持兼容。

实现保留参考 kernel 与对应测试，最新性能数据见 [RESULTS_F2.md](RESULTS_F2.md)。机器相关日志、源码归档和编译产物保留在本地。

## 复验

`bash scripts/validate_remote.sh` 可在配置好的 GPU 容器中复验参考路径；`bash scripts/validate_f2.sh all` 复验 F2 路径。通过 `PCIE_EP_CONTAINER` 设置容器名称，checkout 须挂载到 `/workspace`。

参考验收覆盖输出池、consumer stream、NCCL communicator 借用、有限队列、超时与错误路径、逐元素内容验证以及主路径无 CPU 同步。完整结果在运行后写入本地 `results/`，不纳入公开 Git 仓库。

同步使用 `bash scripts/sync_remote.sh GPU_HOST /opt/pcie-ep`，以 checksum 传输源码并比较 SHA-256。GPU 主机和工作目录由使用者提供。
