# 新服务器单 Segment windowed 验证：2026-09-09

## 范围与身份

- 新实例入口端口10411；GPU：A800-SXM4-80GB。
- GPU UUID：`GPU-50565073-5eb1-2da9-cd87-6e7e7f88d3a1`。
- Mistral revision：`c170c708c41dac9275d15a8fff4eca08d52bab71`。
- Torch 2.2.1+cu121 / Transformers 4.40.2 / vLLM 0.4.1 / xformers 0.0.25。
- 原补丁 SHA：`f63a3d93a37ab593c91141811e9644d9194ddda7ed61722f90f7a2e624d028b6`。
- 原补丁 tree：`91ec6f46f1cb2cdee29a4c90643ccb1e98c391ba`。
- 只运行合成单 Segment 正确性与诊断；没有真实 QA 收益验证、多 Segment、Qwen、Profile 冻结、qualification 或 locked test。
- 此处 fixed15 是强制固定 Source 的诊断臂，不是生产 selector/FinalCommit 成功证据。

## 已修复的证据缺口

提交 `21743a9`：

1. `fully_ready()` 必须覆盖全部预期层，pending 层不得被忽略。
2. all-ready 对照主动提交 pending，再等待全部层，而非只等待已存在的 events。
3. qualification windowed 模式只在全部层完成且 before/destination/after 摘要一致后标记通过；未完成时不是 verified。
4. 正确性入口不再将 `prefetch_window` 静默改为 eager。
5. 无 Prefix 的 working composite 覆盖全部层，而非仅最初预取窗口。
6. copy 提交成功后才从 pending 移除；提交异常先 fence，不发布半成品。
7. online immutable 仍不执行完整 KV 摘要；窗口模式不改变 Source 的租约要求。

## GPU 观测

输出根目录：`/root/autodl-tmp/probekv_stage2/artifacts/`。

| 输出目录后缀 | 执行 SHA | Segment/window | r=1 | 32位置 logit relative-L2 |
| --- | --- | --- | --- | --- |
| `native-mistral-21743a9-newgpu-window1-128` | 21743a9 | 128/1 | PASS | 0 |
| `native-mistral-86b76fe-newgpu-window1-512` | 86b76fe | 512/1 | PASS | 0 |
| `native-mistral-554ba75-newgpu-window2-512` | 554ba75 | 512/2 | PASS | 0 |

两条 r=1 路径均实际执行 Source reuse；greedy token IDs 与无缓存 dense 完全一致。
128-token任务核验了256-token原生 Prefix命中及32层 Source/目标/Source 摘要。
512-token任务明确跳过有界 eager CFO reference，不宣称该任务 CFO sentinel PASS。

保留失败目录 `native-mistral-21743a9-newgpu-window2-512`：完整 Source 请求超过512-token CFO eager reference上限，任务在 r=1 前退出。没有删除失败证据。后续入口将该参数冲突提前到模型加载前拒绝。

### 非 profiler 单次诊断 TTFT（ms）

| Segment/window | 原生 Prefix+dense | resumable dense | fixed15 streaming | fixed15 all-ready | prepared-dense |
| --- | --- | --- | --- | --- | --- |
| 128/1 | 30.925 | 53.632 | 51.192 | 49.080 | 57.400 |
| 512/1 | 53.270 | 78.067 | 63.969 | 62.123 | 83.265 |
| 512/2 | 53.359 | 78.119 | 64.017 | 62.457 | 83.521 |

这些不是重复配对统计或 matched-quality 结果，不能宣称收益。复用仍慢于主要原生 Prefix 基线；all-ready在修复后确实报告32层ready。

## Event envelope 不等于硬件 overlap

- 原 `overlap_trace` 只算 CUDA Event 包围的时间区间；区间包含CPU提交间隙，不能证明 kernel busy。
- `86b76fe` 增加独立 instrumented arm；`554ba75` 修复 Kineto `cuda_runtime` external-ID关联。
- 新分析从标记范围内的真实 `gpu_memcpy` 与 `kernel` activity计算交集，按device/stream匹配，取区间并集避免重复计费。
- 初始 eager copy、decode不在本次逐层统计范围内；profiler TTFT不进入上表。

| 512-token窗口 | 已归属 H2D events | prefill kernels | H2D并集 ms | kernel并集 ms | 实际交集 ms |
| --- | --- | --- | --- | --- | --- |
| 1 | 62 | 1,389 | 2.739 | 27.101 | 0 |
| 2 | 60 | 1,389 | 2.653 | 27.103 | 0 |

结论仅为：这两次**带profiler**运行没有观察到硬件copy/kernel overlap。不能把0外推为所有无profiler请求，也不能继续把Event envelope交集当硬件并行证据。

window1原始trace SHA：`2876a3d2b9469e68aa899255648b74939bbd53d47d213ed3f036cbaf79b72c7f`。
window2原始trace SHA：`15922d58a3f1108a65fa2eaa39de407712530f4465fb6b110768ed325b9974478`。
window1首次汇总因未解析runtime外部ID而没有归属H2D；该文件保持不变，修正分析另存 `hardware-trace/reanalysis-554ba75.json`，绑定原trace摘要及分析SHA。

## 归档与后续边界

本地和服务器均保留 `server10411-windowed-20260909.tar.gz`，摘要：
`f556a210111f8dd01f512ece1402e64350677a1a013f3c2ddbfb7875e2eac356`。
包含原始tokens/logits、trace、摘要、失败记录和日志；排除可再生成的store，服务器原store仍保留。

下一个诊断仅针对执行时序：独立0013补丁延后层计时取值、避免debug GPU标量同步，保留Source依赖等待。必须新补丁SHA/tree、新代码SHA、新输出目录重新验证r=1；不得原地替换历史补丁。

保持：`gpu_runtime_qualified=false`、`online_trace_execution_allowed=false`、`paper_evidence=false`、`locked_test_accessed=false`。没有启用slack repair候选。实例费用单价未核验，费用记为未知而非零；没有自动关闭实例。

## 延后层计时：独立补丁复验

代码 `56a100ed071f0e9db0c1ba40d457bc8c561f9313`；新增独立、默认关闭的0013补丁。
组合patch SHA：`72687e66e33f7b78984cfb6dfdd438f861cf36de3187eb6d3274a21b6672744c`。
组合tree：`418b06282672261779ca622a0f646fbdc94665ea`。
目录：`CacheBlend-deferred-56a100e`；旧目录和旧补丁不覆盖。

优化只针对层执行时序，不改变数学运算或repair ratio：

- 层计时保留真实CUDA事件，执行中`gpu_ms=null`，prefill完成后读取；不填0、不减少计时证据。
- 已知CPU token positions用于等价的debug比较，避免为debug执行GPU `torch.equal()`同步。
- Source-ready依赖等待仍然保留，没有删除资源、lease或FinalCommit合同。
- 新模式显式要求新补丁；未启用时保留历史计时模式。

环境拦截：第一次新目录导入校验发现editable安装把旧vLLM插到环境路径前面，停止于模型加载前。失败日志 `native-mistral-56a100e-deferred-window1-512.log`保留。第二次使用只对该进程生效的显式sys.path入口，核验新目录、组合SHA、tree和扩展库摘要后执行，没有修改全局环境安装。

`native-mistral-56a100e-deferred-window1-512-attempt2`：

- r=1 greedy token一致，32位置logit relative-L2=0。
- 硬件H2D事件62个、prefill kernels 1,235个。
- H2D并集2.769ms、kernel并集26.753ms，copy/kernel交集2.558ms。
- 硬件trace SHA：`f20787650a637beaf5bcb38212aff04c4ba5939616d0a6147b1b0705122303779`。
- 非profiler诊断TTFT：native Prefix 53.252ms，resumable dense 71.621ms，fixed15 streaming 55.812ms，all-ready 56.823ms，prepared dense 76.316ms。

这证明该带profiler的运行发生了真实跨stream copy/kernel overlap，但不证明总体收益。single-sample fixed15仍慢于native Prefix，且未执行真实QA或生产选择闭环。

本地736项测试：735通过，1项ijson缺失跳过；22套配置、compileall、合同validator和diff检查通过。

同一执行代码和补丁下的window2复验目录：`native-mistral-56a100e-deferred-window2-512`。

- r=1 greedy一致，32位置logit relative-L2=0。
- H2D 60 events、kernel 1,240 events；H2D并集2.671ms、kernel并集26.732ms、交集2.445ms。
- trace SHA：`3f8e4017548df9692438e1d694bb421a504afe56e41b144f8b1fc1a9a5f03fb1`。
- 非profiler TTFT：native Prefix 53.330ms、resumable dense 71.257ms、streaming 55.114ms、all-ready 56.391ms、prepared dense 75.811ms。

当前结论：单 Segment 的窗口加载正确性及真实逐层copy/kernel overlap得到小规模诊断证据；不等于全系统gamma准入、QA质量合格、或者统计显著收益。当前仍不进入多 Segment。

新补丁阶段归档：`server10411-deferred-20260909.tar.gz`，SHA256：
`3bc5881bd901960474460b333053ed0de85134f76353aab37abfd9865abe42eb`。
包括失败启动日志、两个成功窗口任务、raw logits、硬件trace及独立patch audit；同样排除store，不删除原始文件。

下一步优先项：

1. 在同一代码/补丁下交错重复显式sync与deferred计时对照，避免把单样本差异当稳定收益。
2. 分离request setup、Prefix shadow/composite构造、逐层位置索引、Source加载和真正selective compute成本；保留真实原生Prefix基线。
3. 优先减少重复CPU→GPU索引构造和可避免的host同步，不降低gamma、不引入slack repair调参。
4. executor存在稳定收益空间后，再接入真实Source选择和QA匹配质量实验；只有单Segment闭环成功后扩展多Segment。

## 三次 matched 重复（window=1，512 tokens）

同一代码、补丁、GPU和请求规格下交错执行三次；每次仍包含完整诊断控制臂。

| 指标 | mean ms | sample std ms |
| --- | ---: | ---: |
| native Prefix + dense remainder | 53.331 | 0.110 |
| resumable dense remainder | 71.704 | 0.248 |
| fixed15 Source reuse | 55.659 | 0.256 |
| fixed15 all-ready control | 56.858 | 0.206 |
| prepared-dense control | 76.363 | 0.133 |

因此 fixed15 相对**同一 resumable执行器的 dense remainder**节省约22.4%；相对优化的 native Prefix + dense remainder 仍慢约2.33ms（约4.4%）。前者是执行器增量收益，后者是最终用户可见的强基线差距，两者都必须报告。这里仍没有加入真实 Source selection、QA质量或多请求并发，故不能称为最终系统收益。

三次输出目录为 `native-mistral-56a100e-matched-r{1,2,3}-512`，GPU已再次确认空闲。
