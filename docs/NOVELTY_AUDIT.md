# ProbeKV 系统性创新点文献审计

审计冻结日期：2026-07-27

审计对象：公开可检索的论文、会议论文和专利文本

结论等级：**有条件的新颖性候选，不是“已证明绝不雷同”**

## 1. 结论先行

截至冻结日期，本轮检索**没有发现一项公开工作完整披露下面全部要素的交集**：

> 对同一 exact non-prefix segment，由不同历史前文分别执行 full prefill，形成多个只读、上下文相关的 canonical KV variants；在线对当前请求先计算目标模型的少量早层，用当前 K/V/hidden/query 状态比较这些 variants，预测每个 variant 的保守安全修复比例与加载、修复总成本；只有在校准后的候选区间明确分离且总成本低于 full recompute 时才选择，否则 abstain 并重算。

但是，宽泛版本的原创新表述已经与现有工作重叠：

- **“同一 chunk 保存多个历史 KV variants 并在线选一个”不是新颖点。** Cache-Craft 已明确保存多个 cache variants，并以 CFO 选择预计重算量最低的版本。
- **“使用当前 query 改善 KV repair”不是新颖点。** Cache-Craft、ProphetKV 和 QCFuse 已覆盖 query-aware repair 或 recomputation selection。
- **“使用 hidden state 指导 repair”不能单独主张。** SpecCache 已使用轻量 speculative model 的深层 hidden-state norms 选择目标模型的关键重算 token。
- **“动态选择 repair/reuse ratio、按硬件成本调节加载与重算”不是新颖点。** CacheTune 已将比例选择建模为硬件感知的端到端 TTFT 优化。
- **“异步预取、逐层 overlap、CPU/SSD tier、调度”只能是系统支撑机制。** 论文与专利中均已有相邻设计。

因此，ProbeKV 的可投稿核心必须收窄为：

> **Target-model current-early-state-guided, variant-specific conservative safe-cost selection among multiple canonical historical KV variants of the same exact non-prefix segment, with calibrated abstention.**

中文建议表述：

> **基于当前请求目标模型早层状态的历史 KV 版本选择：在同一精确非前缀片段的多个上下文相关 canonical Source 之间，估计版本特异的保守安全修复总成本，并在不可信或不经济时拒绝复用。**

这是一个“组合交集仍未发现、但邻近技术密集”的结论，风险为**黄色**，不是可以脱离实验结果直接宣称的绿色结论。

## 2. 审计边界与方法

### 2.1 检索范围

本轮使用以下公开入口进行题名、摘要、全文关键词及专利权利要求检索：

- arXiv；
- ACL Anthology；
- ACM/SIGMOD 论文的作者公开版本；
- USENIX 官方论文页；
- Google Patents 中的中文、美国和 PCT 公开号；
- 仓库中已下载的 Cache-Craft、QCFuse、SparseX、CacheTune、ProphetKV、KVCOMM 全文。

主要检索式族包括：

- `KV cache reuse + multiple variants/sources/historical contexts`；
- `current state/query/hidden state + cache selection/repair/recomputation`；
- `safe repair ratio + uncertainty/conformal/abstention`；
- `hardware-aware reuse recompute cost + storage tier/prefetch`；
- `同一片段/多版本/历史 KV/当前状态/选择/重算/修复/成本模型`；
- 上述关键词的论文题名追踪、参考文献回溯和相邻工作的交叉核验。

### 2.2 本审计不能证明什么

“绝不雷同”是无法由一次公开检索严格证明的：未公开投稿、未公开专利申请、检索系统尚未收录的 2026 新作、不同术语表述，以及无法访问的付费数据库都可能形成漏检。本轮未完整覆盖 CNKI、万方、Scopus、Web of Science、Derwent 和各国专利法律状态数据库；Google Patents 的机器翻译也不能替代专利代理人的 freedom-to-operate 检索。

本文件可以支持的准确说法是：

> **截至 2026-07-27，在本文件列明的公开来源与检索式范围内，未发现完整覆盖 ProbeKV 最窄权利要求交集的单项工作。**

## 3. 权利要求要素拆分

| 编号 | 必要要素 | 是否必须保留 | 原因 |
|---|---|---:|---|
| C1 | 同一 tokenizer 下 token-identical 的重复 non-prefix segment | 是 | 将问题与普通 prefix cache、语义缓存区分开 |
| C2 | 同一 segment 有至少两个由不同 preceding contexts 独立 full prefill 得到的 canonical KV variants | 是 | 定义历史 Source 选择空间；repair 结果不得晋升 |
| C3 | 当前请求在目标模型上 fresh-compute 少量早层，并抽取 K/V/hidden/query 状态 | 是 | 与 Cache-Craft 的历史元数据 CFO、纯 query probe 区分 |
| C4 | 对每个 variant 预测其满足质量约束所需的保守 repair budget，而非只预测相似度或 token importance | 是 | 把“选相似版本”变成“选最低安全修复成本版本” |
| C5 | 目标函数包含 probe、比较、可见加载、修复及继续重算的 total cost | 是 | 防止选择质量好但系统上不经济的 Source |
| C6 | 校准区间不分离、质量覆盖不足或经济边界失败时 abstain/full recompute | 是 | 将均值预测与可执行的保守决策区分 |
| S1 | 动态 `L_probe`、动态 `L_reuse` | 支撑 | 已有早停、分层和硬件自适应相邻工作，不能单独宣称首创 |
| S2 | 预取、CPU/SSD tier、A/B/Hybrid scheduler | 支撑 | 属于实现端到端收益的系统机制，不是核心算法新颖性 |

只要删掉 C2、C3、C4 或 C6 中任一项，ProbeKV 都会明显靠近已有工作，论文创新防线会显著变弱。

## 4. 最接近工作的逐要素对照

符号：`✓` 为明确覆盖；`△` 为部分或相邻覆盖；`—` 为未发现覆盖。

| 工作 | 多个上下文 variants | 当前在线信号 | variant-specific safe budget | 保守拒绝 | 系统成本 | 雷同风险 |
|---|---:|---|---:|---:|---:|---|
| [Cache-Craft](https://arxiv.org/abs/2502.15734) | ✓ | 历史 CCI、当前/旧 prefix overlap、顺序；query attention 用于 repair early stop | △，CFO 近似重算需求 | — | ✓ | **最高**：已经覆盖多版本保存与选择 |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | — | 逐层 KV deviation | — | — | △，load/recompute pipeline | repair backend 的直接先例 |
| [ProphetKV](https://arxiv.org/abs/2602.02579) | — | 当前 user-query-to-context attention | — | — | △ | query-aware token repair 先例 |
| [QCFuse](https://arxiv.org/abs/2606.05875) | — | query-conditioned chunk anchors + critical layers | — | — | ✓，pipeline-constrained selector | 早期/压缩视图 query probe 先例 |
| [SpecCache](https://aclanthology.org/2026.acl-long.859/) | — | speculative model 深层 hidden-state norms | — | — | △ | hidden-state-guided token repair 先例 |
| [CacheTune](https://arxiv.org/abs/2605.24022) | — | 离线频域 token importance | —，采用统一质量下限 | — | ✓，硬件感知 ratio 与 TTFT 模型 | 动态 repair ratio 和 tier 优化先例 |
| [SparseX](https://arxiv.org/abs/2606.01751) | — | Sparse-Q token selection | — | — | ✓，segment reuse runtime | repair backend 和 interleaved serving 邻近 |
| [KVCOMM](https://arxiv.org/abs/2510.12872) | △，bounded anchor pool | 当前 prompt 与历史 anchor 的表示关系 | — | — | △ | 多 anchor/cross-context 选择邻近 |
| [KVShare](https://arxiv.org/abs/2503.16525) | — | cross-request sharing 与 dual-stage KV deviation | — | — | ✓，cache-aware scheduler | 不是同一 exact segment 的多历史版本选择 |
| [KVLink](https://arxiv.org/abs/2502.16002) | — | trainable link tokens | — | — | △ | 独立文档 cache 连接，需训练 |
| [EPIC](https://arxiv.org/abs/2410.15332) | — | 静态 attention sparsity/position-independent cache | — | — | △ | position-independent reuse 邻近 |
| [RAGCache](https://arxiv.org/abs/2404.12457) | — | prefix-aware knowledge tree | — | — | ✓ | 分层缓存、调度和淘汰先例 |
| [Prompt Cache](https://arxiv.org/abs/2311.04934) | — | schema/module identity | — | — | △ | 模块化 KV reuse 先例 |
| [CacheSlide](https://www.usenix.org/conference/fast26/presentation/liu-yang) | — | 位置/层相关修正 | — | — | ✓ | 分层和 spill-aware 复用邻近 |
| [CN120338090A](https://patents.google.com/patent/CN120338090A/en) | △，跨用户/请求历史池 | segment UUID/hash match | — | — | ✓，成本模型决定查缓存或重算 | **高**：RAG 历史 KV 池与经济选择专利先例 |
| [CN120371524A](https://patents.google.com/patent/CN120371524A/en) | △，multi-KV/document fusion | 语义标识、attention/gradient difference | — | — | ✓，异步加载、选择性 prefill、tier | **高**：宽泛系统机制专利先例 |
| [CN120163246A](https://patents.google.com/patent/CN120163246A/en) | — | 前 M 层 look-ahead、hidden-state window/attention | — | — | ✓，逐层加载重叠 | 早层观察和传输重叠专利先例 |

### 4.1 Cache-Craft 是必须正面处理的最近邻

Cache-Craft 不能只作为“repair baseline”描述。其正文明确讨论同一 chunk 来自多个 sources 的 cache、为一个 chunk 保存多个 variants、记录 CCI 和旧 prefix，并在请求到来时计算 CFO 后选择最低者。它还根据当前 question 对 chunk 的 attention 在层间提前终止重算。

ProbeKV 与它唯一可守的核心差异不是“有多个 Source”，而是：

1. Cache-Craft 用历史 cache 元数据和 prefix/order 关系估计候选；ProbeKV 用**当前目标模型 fresh early state**形成 variant-specific observation。
2. Cache-Craft 的 CFO 是启发式重算需求代理；ProbeKV 要输出满足预注册质量定义的**保守 safe repair budget**及其 total-cost interval。
3. ProbeKV 只有在区间分离、质量覆盖和 admission 同时通过时选择，否则 abstain；这必须通过尾部违规率而不是平均质量证明。

### 4.2 组合显而易见性风险

即使没有单项工作覆盖 C1–C6，审稿人仍可能提出：

> ProbeKV 只是 “Cache-Craft 多版本选择 + ProphetKV/QCFuse/SpecCache 当前状态信号 + CacheTune 成本模型 + 通用 conformal calibration”。

所以“没有完全相同论文”不等于一区/二区创新成立。必须用机制和实验说明，把这些组件直接拼接不能得到相同决策：当前早层状态应当预测的是**同一 segment 不同 context-conditioned variants 的安全成本排序**，而不是一般 token importance；而保守校准必须显著降低尾部错误，不能只是包装一个均值回归器。

## 5. 可主张、不可主张与论文措辞

### 5.1 禁止作为首创点

- first multi-version / multi-source KV cache selection；
- first query-aware KV reuse or selective recomputation；
- first hidden-state-guided KV repair；
- first dynamic repair ratio or hardware-aware reuse–recompute trade-off；
- first layered prefetch/copy–compute overlap/heterogeneous KV storage；
- first adaptive early termination。

### 5.2 推荐贡献表述

论文摘要与 Introduction 不建议使用绝对的 `the first`。推荐写成：

> ProbeKV formulates source selection for non-prefix KV reuse as a variant-specific safe-cost decision. For multiple canonical KV variants of the same exact segment, it uses fresh early-layer states from the current target-model request to estimate conservative repair-cost bounds, and abstains whenever quality confidence or end-to-end economy is insufficient.

Related Work 中必须主动写明：

> Cache-Craft already stores and selects among multiple chunk-cache variants using CCI, prefix overlap, and order-derived CFO. ProbeKV addresses the remaining oracle gap by conditioning variant ranking on fresh target-model early states and by selecting against calibrated safe total-cost bounds rather than a metadata heuristic.

如果投稿时仍希望使用 `to our knowledge`，只能放在完整的 C1–C6 交集前，且必须在投稿前重新检索；不能把它放在“multi-source selection”或“current-state-guided repair”前。

## 6. 实验上必须建立的创新防线

以下不是普通消融，而是决定创新点能否成立的 gate：

1. **先证明选择问题真实存在。** 至少 25% case 的四个 Source 之间 `r_safe` spread ≥10 个百分点，Oracle-K4 相对 Latest/K1 的安全成本至少降低 10%。
2. **正面对比 Cache-Craft。** 实现真实 CFO，并给它完全相同的 repair backend、cost model、admission、prefetch 和 scheduler；只允许选择信号不同。当前请求 prefix 不得被放入候选历史 Source；Cache-Craft 和 ProbeKV 必须共享同一批由不同过去请求形成的 cache variants。
3. **隔离当前早层状态的增量。** 对比 metadata-only、current-query-only、K/V/hidden-only、current-state+metadata，并报告 paired normalized regret、排名相关性和安全成本。
4. **隔离“版本选择”与“token 选择”。** ProphetKV、QCFuse、SpecCache-style 信号至少要作为同一后端中的 selector/feature baselines，不能只引用其论文数字。
5. **隔离校准与拒绝机制。** 比较 mean prediction、uncalibrated quantile、calibrated bound、calibrated+abstention；同时报告 coverage、abstention、尾部违规和 TTFT。
6. **证明不是 CacheTune 的比例调优收益。** 用 CacheTune-style hardware-aware ratio/admission 作为相同系统层基线，再比较有无 variant-specific current-state ranking。
7. **必须有自然 trace。** 受控四版本构造只能解释机理，主结论必须在真实重复检索或可审计 replay 中成立，并按 `content_hash/document_id` 隔离 split。
8. **端到端核算所有开销。** probe、每层检查、summary 传输、错误预取、winner 加载、repair、调度干扰和 HBM 峰值必须全部计入。

建议把 H2 的最强同栈对照明确命名为：

`Cache-Craft-CFO + identical safe-budget calibrator + identical cost/admission/runtime`。

只有 ProbeKV 相对该对照仍显著降低 regret/安全成本并最终改善 TTFT，才能说明收益来自 C3–C6，而不是系统栈差异。

## 7. 当前风险判定与 Go/No-Go

| 判定对象 | 当前状态 | Go 条件 | No-Go 条件 |
|---|---|---|---|
| 多版本选择空间 | 未经 A800/真实数据证明 | H1 按预注册阈值通过 | Source spread 稀少或 Oracle 改善 <5% |
| 当前早层状态增量 | 假设成立，尚无论文证据 | 相对公平 Cache-Craft-CFO regret 至少下降 20%，CI 排除 0 | CFO 已接近 Oracle，或多数请求探到 25% 仍不可信 |
| 保守安全预算 | 本地机制已实现，尚无正式质量证据 | 非劣效与 pooled tail gate 同时通过 | 只能改善平均质量，尾部违规失控 |
| 端到端经济性 | 尚无 A800 证据 | 全成本计入后满足 `T_reuse <= 0.8 T_full` 且主要 workload 改善 | 收益 <3% 或被加载/HBM/并发干扰抵消 |
| 投稿创新等级 | 黄色 | H1、H2、质量和端到端 gate 全通过 | 只在受控合成集成立，或增益来自通用调度 |

## 8. 提交前更新协议

为避免 2026 年快速演进造成审计过期：

1. 每月运行一次题名/摘要增量检索，重点跟踪 `KV cache reuse/fusion/repair/variant/current state/query-aware`。
2. 投稿前 30 天做一次全文、引用和被引文献扩展；投稿前 7 天再做一次 arXiv、ACL/EMNLP/NSDI/OSDI/SOSP/FAST/SIGMOD/VLDB 增量扫描。
3. 对 Cache-Craft、QCFuse、ProphetKV、SpecCache、CacheTune、SparseX 和 KVCOMM 做 forward-citation 检索。
4. 若准备申请专利，另请专利代理人检索 CNIPA、WIPO、USPTO、EPO 和同族专利；本学术审计不能替代法律意见。
5. 每次更新记录检索日期、新增命中、逐要素差异及是否需要修改核心表述。阈值和 test split 不得因新文献或测试结果而事后调整。

## 9. 审计结论

ProbeKV 目前**不能**宣称“多历史 Source 选择”是首创，也不能承诺“绝不与任何人雷同”。它仍有一个可验证的、较窄的创新机会：

> 从当前目标模型的早层内部状态中识别同一 exact segment 的哪一个历史上下文版本最容易被安全修复，并把该预测转换成带拒绝机制的保守端到端成本决策。

这个机会是否达到 SCI 一区或二区，不由措辞决定，而由 H1/H2 的 oracle gap、对 Cache-Craft 的公平优势、尾部质量保证和 A800 端到端收益共同决定。若任一关键 gate 失败，应按研究契约停止或重构，而不是增加调度模块掩盖核心假设失败。
