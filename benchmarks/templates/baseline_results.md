# PCIe EP S0–S4 实现与性能测试结果

本页保留可读性重构前的正式性能记录。对应源码快照见 `results/baseline_sources_20261024.tar.gz`；重构后结构与验收见 [REFACTOR.md](./REFACTOR.md)。

实验日期：2026-10-24（Asia/Shanghai）。机器：`<GPU_HOST>`。独立目录：`/opt/pcie-ep`；容器：`pcie-ep`。

已完成 S0–S4 的实现及下述实际验收。当前结果为 BF16 direct 的功能基线，未加入 S5/S6 的多 channel、接收展开/源端融合，也未使用 CUDA Graph、FP8 或 CPU NUMA relay。源码入口见 [README](./README.md) 和 [API](./API.md)。

## 配置与计量

8 张 H20D，GPU0–3 邻近 NUMA0，GPU4–7 邻近 NUMA1；源端共享页的实际 NUMA 放置已通过 `/proc/*/numa_maps` 检查。PyTorch 2.13.0+cu130 / CUDA 13.0。DP attention=8、EP=8、总专家=256（每卡连续32个专家）、top-k=8、hidden=7168、BF16。这里输入已在 GPU 上，未执行 attention、专家 GEMM 或 combine。

主路径 `do_cpu_sync=False`，固定容量、GPU counts/offsets。Q=L=256、channel=1、每 GPU 一个通信 progress CTA、独立 HBM 专家展开；所有 batch 共用 Tmax=8192 的预分配输出池。路由是 seed=20261024 的均匀随机 distinct top-k expert，每轮重算布局，不复用 cached layout。每个 batch 预热5次、测量20次。

Dispatch 时延由本地 CUDA events 测量，每轮取八卡时长最大值，再计算均值/分位数。包含 GPU 路由、计数、传输、专家展开及 stream 等待；正常异步提交时也可能包含 CPU 提交间隙。Transport 时延由同一 GPU 的 `%globaltimer` 记录传输 kernel 开始/结束，每轮同样取八卡最大值，不包含 layout/scatter。不跨 GPU 相减时间戳；未锁定GPU时钟。CPU提交耗时和p99/原始20个样本保留在JSON。原始JSON的`timing_note`中“counts ... outside timing”措辞不精确，计时外的是CPU计数读回与流量统计，GPU count publish/prefix包含在完整Dispatch中；保留原始文件及其hash，计量范围以本段和实际代码为准。

B=7168*2=14,336 bytes；U为有远端目标的源 token 数，F为去重后的远端 token/GPU 对数。远端交付通信量=`F*B`，host 写量=`U*B`，host 读写通信量=`(U+F)*B`。下面带宽均为**八卡合计、十进制 GB/s**，以实际路由通信量除以对应时延；不包含本地专家拷贝、metadata/flag、padding 或硬件一致性开销，因此不是全物理总线事务带宽。每卡平均值为合计除以8。

## 全部 batch 结果

本表带宽统一使用 Dispatch 平均时延作分母：

{{DISPATCH_TABLE}}

## 仅传输 kernel 的结果

本表带宽使用 Transport 平均时延作分母，与上表分开：

{{TRANSPORT_TABLE}}

## 正确性和路径证据

- S0：14336-byte 单槽 READY/ACK，GPU0/1和GPU0/4各100万轮，另有 smoke 与延迟读写线程测试，全部零错误。SASS 记录 system-scope load/store 与 fence，不依赖 mapped-memory RMW。
- S1：双 GPU、多进程整批 staging 测试通过；S1共用S2的GPU stable layout，输出为compact unique，属于原稀疏原型的能力扩展。
- S2/S3/S4：八卡去重、稳定布局、权重、unique/expanded、alignment/padding、空输入、不等长、热点、本地与多播通过；S4覆盖Q/L=1/1、2/3、3/2、256/256，以及非连续子group和同成员不同group。
- S4压力：三组Q/L各10万轮/卡，共30万轮/卡、八卡累计240万次dispatch；两buffer、延迟reader、逐轮错误快照均通过。每组每卡至少102万次payload slot写入，协议压力H=128；完整H=7168另由S0和最大batch验证。
- 最大batch：T=8192、H=7168，完整检查 unique 与 expanded 两模式共 {{PAYLOAD_ELEMENTS}} 个有效 BF16 元素（{{PAYLOAD_GB}} GB），连同路由、weights、metadata和counts全部通过，没有跨rank收集payload。
- API：异步stream、previous_event、deferred epilogue、输出池保活、单独保留weight、非法/重复路由、sticky abort、安全close与已存在NCCL communicator借用已验证。额外验证独立consumer stream和早于旧输出消费的previous_event，释放Python引用后连续复用pool仍保留原始payload；原始记录见consumer_lifetime JSON。Python接口单元测试27项通过。
- Profiler：expanded稳态采样区域未见CPU CUDA同步、scalar读回或D2H memcpy；初始化、诊断与profiler结束的同步不计入稳态。超时/释放event后诊断测试另行通过。
- PCIe-only：CUDA runtime代码与二进制未包含peer copy、peer enable、Gin/NVSHMEM/NCCL payload路径；指针来源为本GPU tensor和registered host mapping。正式benchmark前后全8卡288项NVLink payload计数均增量0 KiB，Tx总增量={{TX_DELTA}}、Rx总增量={{RX_DELTA}}。没有重置计数、关闭链路或改动现有业务容器。硬件计数精度为KiB，完整证据和限制见audit。

长压力测试在Python consumer依赖修复前完成，CUDA协议代码及二进制随后未改变。修复后最终顺序流水线重跑了consumer生命周期、API/NCCL借用、timeout、expanded profiler、完整S4正确性、最大batch以及全部十组性能测试，exit=0；十万轮长压力未重复执行。旧性能与路径审计以`pre_streamfix`名称归档，不参与以上表格。

## 结果解释与后续工作

当前实现首先满足S4的功能、去重、有限窗口和无CPU同步要求。单channel/单progress CTA限制并发，路由/prefix采用简单实现；metadata处理、逐token通知/ACK和system fence均未做性能优化，独立scatter也按预分配容量上界launch。因此这些结果不是PCIe Gen5 x16或整机host-memory带宽上限。

Transport与全Dispatch的差值可区分预处理/展开开销；进一步归因仍需专项profiling，不能仅从该表宣称某条硬件链路已饱和。下一步应按S5优先验证更多channel、合并metadata访问及减少无效轮询/fence，再测F2展开融合；S6再处理源端融合、cached layout和Graph。NUMA relay留在S8，不能将本次跨NUMA direct结果描述为已实现relay。

## 原始文件

- [性能JSON](./results/dispatch.json)、[性能CSV](./results/dispatch.csv)、[结果图](./results/performance.png)
- [S4正确性](./results/s4_correctness.json)、[压力](./results/s4_stress.json)、[最大batch](./results/large.json)
- [接口](./results/integration.json)、[Consumer生命周期](./results/consumer_lifetime.json)、[超时](./results/timeout.json)、[Expanded profiler](./results/profiler_expanded/audit.json)
- [PCIe-only审计](./results/pcie_path_audit.md)、[审计JSON与哈希](./results/pcie_path_audit.json)、[NUMA页位置](./results/numa_maps_stress.txt)

复现命令见README。CUDA代码和Python API均保留在该独立目录；实验GPU进程已退出，保留空闲容器和build cache供复现。
