# 固定 top-4 非均匀路由对照

本次固定 DP8、EP8、256 experts、每卡连续放置 32 experts、H=7168、BF16；K 从 8 改为 4。十个 T 为 16/32/64/128/256/512/1024/2048/4096/8192。生产 dispatch kernel 不变，只修改实验输入与汇总。

## 一手实测依据与适用范围

[NVIDIA TensorRT-LLM 官方 EP 技术文档，Table 1](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html#offline-ep-load-balancer) 给出 DeepSeek-R1、256 experts、top-8、EP32、GB200、机器翻译工作负载、关闭 EPLB 时的统计：

| 对象 | mean | std | (max−mean)/mean | 推导 CV=std/mean | 推导 max/mean |
|---|---:|---:|---:|---:|---:|
| expert slot | 128 | 164.1 | 10.948 | 1.28203125 | 11.948 |
| GPU rank | 1024 | 491.6 | 1.564 | 0.480078125 | 2.564 |

原表各项是逐层、逐迭代指标的平均，不能恢复完整概率向量，也不能照搬 EP32 的每卡负载到 EP8。这里的 std/mean 比值是合成校准值，不等同于逐层逐迭代 CV 的平均；保留的小数仅用于确定生成器参数，不代表原始测量具有相同精度。这里以专家统计约束合成路由，实际每卡负载另行统计。当前公开一手材料不足以断言“真实路由普遍遵循 20%/80%”。

[《Patterns behind Chaos》，III-C/Fig. 8–9](https://arxiv.org/html/2510.05497v5#S3.SS3.SSS1) 也观察到与任务、语言有关的热点；其“top 10% 承担 60–80%”说的是专家对共激活，不能当作单专家负载份额。论文的[逐 token 路由数据集](https://huggingface.co/datasets/core12345/MoE_expert_selection_trace) 当前需要申请访问，因此本实验没有冒充或回放该数据。

## 两组输入

- `nvidia_calibrated`：构造单专家选择边际，使理论专家负载 CV=164.1/128、max/mean=11.948。不是 NVIDIA 原始 trace；只匹配单专家边际负载的两个汇总指标，不复现真实专家共激活、时间相关性、跨源/跨域相关性或 router scores。
- `hotspot_20_80`：保留用户指定的合成热点对照。51/256≈19.92% 的专家占采样权重总质量的 80%；无放回 top-4 后的最终热门分支份额以实际直方图为准。

两组都通过预先固定的专家 ID 置换放置概率。不会刻意将热点集中在少数 GPU，或人为平衡每卡流量。专家偏斜、GPU 偏斜、去重后的唯一 token 负载分别报告。

## 冻结与公平性

CPU 离线生成八个 rank 的最大 T=8192 路由，每个较小 T 取对应 rank 的前缀。保存实际 `[rank, token, branch]` 的 int64 专家 ID 和 FP32 权重到 NPZ，附带 JSON manifest。分支权重是离线归一化的合成数值，不是模型 router scores。每个 token 选中四个不同专家；branch 顺序、权重和 token 顺序都固定。manifest 记录数据文件和每个 T/rank 的 SHA256、概率构造、专家 ID 放置及实际负载直方图。

GPU benchmark 仅加载、校验并上传已保存的张量；计时循环不生成路由、不采样、不改变权重、不收集负载统计。PCIe-EP、纯 NCCL payload AllGather、包含 pack/AllGather/专家展开的 NCCL dispatch，以及正反两轮测试，都必须匹配相同的输入 hash。源 token 内容也通过 BF16 原始字节 hash 核对。全量输出内容和稳定 expert/source/token/branch 顺序在计时外验证。

此前均匀 top-8 benchmark 也只在计时外生成一次路由，并非每次通信随机重选；本次进一步将具体选择持久化，允许独立复查与重复测试。

## 计时与带宽

每组执行两轮，每轮 5 次 warmup、20 次测量，第二轮反转三种实现顺序。时延取每次八卡最大 CUDA-event 时延，再对 40 个值求平均。计时包含完整 eager dispatch；主路径 `do_cpu_sync=False`，NCCL completion 使用 GPU stream 依赖。准备、验证、测量结束后的 completion wait 不属于主路径。

统一逻辑带宽为 `T*4*7168*2/t/1e9` GB/s，不乘 world size，也不因去重减少分子。纯 AllGather 的 NCCL 传统 `algbw=8*T*H*2/t`、`busbw=7*T*H*2/t` 与逻辑带宽分开；K=4 时 busbw 是逻辑带宽的 7/4，不能沿用 K=8 时的 7/8。

PCIe-EP 使用 mapped CPU staging；NCCL 所有 GPU 数据连接必须为 SHM。每组独立检查初始化日志及全部 288 个 NVLink Tx/Rx 数据计数器的零增量。环境变量本身不是路径证明。此对照不包含专家 GEMM、combine 或端到端模型吞吐。

## 重现

```bash
python benchmarks/routing_fixture.py --profile nvidia_calibrated \
    --output benchmarks/data/top4_skew/nvidia_calibrated_routes.json
bash scripts/benchmark_nccl_shm.sh results/top4_nvidia \
    --topk 4 --routing-fixture benchmarks/data/top4_skew/nvidia_calibrated_routes.json
```

生成器拒绝覆盖已有 fixture。仓库保存的是正式测试所用的 JSON/NPZ，重测时直接加载已有文件；需要生成新样本时使用新目录。将 profile 改为 `hotspot_20_80` 可生成热点对照。正式测试在同一 NumPy 2.3.5 环境中生成、加载；loader 对概率模型有严格浮点校验，跨软件或架构可能提前拒绝，不能绕过完整路由 hash 校验。
