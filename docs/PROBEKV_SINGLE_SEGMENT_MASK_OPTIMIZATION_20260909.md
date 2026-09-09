# 单 Segment Prefix + fixed15：CPU mask 优化与 A800 复验

## 范围与代码

- 执行 SHA：`86446f80d41f02a3d20ae1df959df3aa1d5b7454`。
- 对照执行 SHA：`56a100ed071f0e9db0c1ba40d457bc8c561f9313`。
- 模型：Mistral-7B-Instruct-v0.3，revision `c170c708c41dac9275d15a8fff4eca08d52bab71`。
- GPU：A800-SXM4-80GB，UUID `GPU-50565073-5eb1-2da9-cd87-6e7e7f88d3a1`。
- 未修改 CacheBlend 补丁；组合 SHA `72687e66e33f7b78984cfb6dfdd438f861cf36de3187eb6d3274a21b6672744c`，tree `418b06282672261779ca622a0f646fbdc94665ea`。
- 保留 deferred layer timing、合法 boundary=2、CPU pinned winner backing、fixed15。
- 请求：256-token 原生 Prefix + 32-token dense 区 + 一个 non-prefix Segment + 32-token suffix。
- Segment=512 或640；相应 prefetch window=1或2。同一长度的所有对照保持相同 window 配置。

128–640 是当前 canonical 长度合同。本次没有放宽合同到1024/2048，也没有将长文档伪装成单个合法 canonical Segment。更长单 Segment 只能另行定义为合同外执行器压力测试，不能混入本次结果。

## 修复

`commit_segment_reuse` 原先在每个 active token 上重复构造 `set(segment)` / `set(repair)`。
改成提交时各构造一次，并在循环中复用。gradual support 的 pending membership 同样移出逐token循环。

没有改变 token 顺序、repair 数量、Source 身份、Prefix 排除、Source-ready 等待、数学运算或准入阈值。
新增128/512/640长度测试，验证集合构造次数与token数无关，以及dense区域、repair行和suffix的执行归属。

本地CPU微测（仅mask membership，不是GPU TTFT）：512 tokens旧约2.679ms，新约0.016ms。
737项unittest完成：736通过、1项因缺ijson跳过；compileall、合同validator、22套配置和diff检查通过。

## 新 SHA 的三次重复

每个长度执行三次完整诊断，512/640交替运行；每次内部控制臂顺序固定，没有做随机化顺序实验。
表中为非profiler首token host wall-clock均值与样本标准差；不使用更短的boundary-to-first-token代替请求端点。

| 路径 | 512 tokens，mean ± std ms | 640 tokens，mean ± std ms |
| --- | ---: | ---: |
| 原生 Prefix + dense remainder | 53.286 ± 0.061 | 57.871 ± 0.239 |
| resumable Prefix + dense remainder | 71.414 ± 0.299 | 78.242 ± 0.249 |
| 原生 Prefix + fixed15 streaming | 51.959 ± 0.381 | 51.885 ± 0.539 |
| fixed15 all-ready control | 53.116 ± 0.396 | 54.819 ± 0.711 |
| prepared-dense control | 76.089 ± 0.213 | 82.433 ± 0.389 |

相对原生 Prefix + dense remainder，节省分别为2.49%、10.34%。
旧SHA的512-token三次fixed15均值为55.659ms；新均值51.959ms，下降约3.70ms。
这是跨运行批次比较，不是随机化代码A/B。旧640-token单次值56.121ms只作为探索性对照。

所有同长度控制臂的请求token哈希、Prefix命中数、block size与sampling signature核验一致。
原始JSON observation摘要重新计算通过；Prefix实际命中256 tokens，fixed15实际发生selective reuse，未退化dense。

## 正确性与证据限制

- 六个新SHA任务均通过r=1 greedy token完全一致；共同teacher的32位置logit relative-L2=0。
- r=1 Source before、destination、Source after三个logical digest一致，绝对位置mask检查通过。
- online-immutable成本臂不执行每请求full-KV hashing。
- 六个fixed15成本臂生成的token也与各自native Prefix基线相同，但输入是构造诊断文本，不等同于真实RAG QA认证。
- 原生Prefix teacher控制臂相对无缓存dense的logit relative-L2分别约0.007570（512）、0.025747（640）；resumable Prefix及r=1为0。此差异保留为待诊断项，不能宣称所有执行路径logits都等价。
- 本次Source是固定指定的诊断winner，不含真实多Source Residual-K选择成本，不是生产准入成功。
- 两档成本比均高于0.8，不满足以原生Prefix基线计算的FinalCommit节省20%门槛。
- 512与640还使用不同prefetch window，不能把跨长度差异全部解释为长度效应。
- 未执行真实QA、CFO eager对齐（本次显式跳过）、新硬件profiler、正式Profile或多Segment实验。

## 原始产物与后续

服务器目录前缀：`/root/autodl-tmp/probekv_stage2/artifacts/`。

- `native-mistral-86446f8-maskopt-512-w1-r{1,2,3}`
- `native-mistral-86446f8-maskopt-640-w2-r{1,2,3}`
- `native-mistral-56a100e-640-window2-attempt2`（优化前探索性控制）

每个目录保留原始tokens/logits、对照结果、传输审计和完整日志。归档排除可再生成store，不删除服务器原store。
归档：`server10411-maskopt-86446f8-20260909.tar.gz`，SHA256：
`f0f001d3676f39e8e2268ec837dd37ad7c60d384a3c3113fe652d7f1fc3f9ca1`。
GitHub连接暂时失败，新代码通过经验证Git bundle同步并按精确SHA检出；没有手工修改服务器源码。

下一步仍为单Segment：诊断native Prefix数值差异，测量剩余逐层CPU索引/composite/setup成本，再接真实Source选择和真实QA。
不通过扩大Segment合同、降低repair质量或改变gamma掩盖开销。

本次结束GPU进程清空、显存0MiB；实例未自动关机，费用单价未知而非零。
保持 `gpu_runtime_qualified=false`、`online_trace_execution_allowed=false`、`paper_evidence=false`、`locked_test_accessed=false`。
