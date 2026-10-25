# 代码组织与风格

本项目参考本地 DeepEP V2.5 的 `csrc/buffers/ep.hpp`、`deep_ep/include/deep_ep/impls/ep/dispatch.cuh`、`deep_ep/buffers/ep.py` 和 `deep_ep/utils/event.py` 组织代码。使用自己的 `pcie_ep` 命名空间和包，不引入 DeepEP 的 NVLink/Gin 数据路径。

## CUDA 与 host runtime

- `csrc/runtime.cu` 只负责 pybind 导出；构建仍只有一个 CUDA translation unit。
- `csrc/buffers/host_staged.hpp` 管理设备 workspace、输出池、输入检查和 kernel launch；`csrc/comm/host_shared_memory.hpp` 管理共享 backing、NUMA first-touch、映射与注册。
- S5–S8 的 `advanced.hpp`、`advanced_shared_memory.hpp`、`numa_relay.hpp` 和 `advanced_{common,layout,dispatch}.cuh` 分别管理高级资源、共享布局、CPU relay、GPU 协议和布局/搬运；旧 S1–S4 模块保持独立。
- `csrc/kernels/host_staged/common.cuh` 定义协议 ABI、具名错误码、系统域原子操作和 DeviceState；`layout.cuh` 负责 GPU 路由与接收布局；`dispatch.cuh` 负责整批基线、ring progress 和专家展开。
- 缩进为 4 个空格，namespace 为 `pcie_ep::host_staged`；类型使用完整的 PascalCase 名称，函数和字段使用有含义的 snake_case 名称，固定容量常量使用 `kNum...`。常见的 `x`、`topk_idx` 等公开接口名称保留。
- 一行表达一个步骤，较长条件与参数分行。函数边界按资源或通信阶段划分，避免把初始化、检查、分配、launch 和返回组装塞进同一个长函数。
- 协议注释解释发布顺序、回收条件和前进性。尤其保留所有 writer 的 system fence、CTA barrier、READY acquire/release，以及独立的 payload ACK 与 descriptor credit。
- ABI 的大小、对齐与字段偏移使用 static_assert 检查；协议数组的 error/statistics/timestamp 字段使用具名索引。改名不能改变共享段布局、错误码或 Python 返回语义。

`.clang-format` 固定 4 空格、140 列和展开控制流的默认规则；协议屏障附近的修改仍需要人工复核。

## Python

配置放在 `config.py`，通信组与初始化放在 `comm.py`，buffer 和 handle 放在 `buffers/`，stream event 放在 `utils/event.py`。`buffer.py` 保留原 import 路径的兼容导出，`_C.py` 只负责惰性构建与加载。

方法签名、属性名和返回布局保持与既有 API 一致；CPU 初始化、正常 GPU 提交、显式诊断和销毁分别处理。EventOverlap 的 wait 是 GPU stream 依赖，不添加 CPU 完成等待。输出池复用既检查 Tensor 所有权，也等待已登记的 consumer stream。

benchmark 的计量、文件写出和作图分别封装；报告正文放在 Markdown 模板中。Python 文件统一使用 Black 的 120 列配置；本次 34 个文件完成格式化前后的 Python AST 等价检查，最终 Black `--check` 全部通过，接口模块的关键 GPU 提交与生命周期代码另外完成结构对照。记录见 `results/s5_s8/code_style/`。

## 同步与验收

本地开发目录为 `<PCIe-EP checkout>`，远程实验目录为 `<GPU_HOST>:/opt/pcie-ep`。运行 `bash scripts/sync_remote.sh GPU_HOST /opt/pcie-ep` 推送源码，并自动比较两端 SHA-256；该脚本保留远程独有文件、排除结果和构建缓存。远程额外的源码文件会导致哈希检查失败，需要人工确认。

`scripts/validate_remote.sh` 在远程既有容器中顺序编译和验收，结果写入 `results/refactor/`。原始性能记录和审计不覆盖；原始源码另存为 `results/baseline_sources_20261024.tar.gz`。

高级验收使用 `scripts/validate_s5_s8.sh`，channel 消融与正式测量使用 `scripts/benchmark_s5_s8.sh`。CPU 原子 load/store 使用 C++20 `atomic_ref`；CUDA block-scope shared counters 与本 GPU 错误原子操作允许 RMW，mapped host signal 只允许单写者 acquire/release load/store。
