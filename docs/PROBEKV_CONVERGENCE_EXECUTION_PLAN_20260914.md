# ProbeKV 收敛执行总计划：先证明 Source 价值，再优化选择与执行

日期：2026-09-14。审计代码：`c7730ad5c48c448864693a2234fbd203e9ef8de0`。
分支：`codex/probekv-schema10-evidence-integration`。

**本文是下一阶段修复与实验计划，不是完成报告。** 本次只审计、核对原论文和更新文档；不修改运行时代码、不启动 GPU、不推送 GitHub。本计划更新下一步执行顺序；历史代码、Profile、结果和失败证据保持原解释，不追溯改成通过。

## 1. 先回答三个尚未得到证据的问题

| 问题 | 当前结论 | 需要什么证据 |
|---|---|---|
| 同一 exact Segment 保存多个历史 Source，是否覆盖更多未来请求？ | **尚未完成项目内真实工作负载验证** | 非前缀重复频率、Source×request×quality 矩阵、因果 Coverage(K) |
| d1 或 d1/d2 是否选到真正合适的 Source？ | **有代码与 shadow 接口，未完成多 Source 真实 QA/cost 验证** | K>1、不同历史前文、独立 Source Oracle、held-out 排名及成本遗憾 |
| 加入 SparseX/QCFuse 是否更快、更稳？ | **候选已登记，不是已集成并通过的生产能力** | 分别接入、独立消融、计入全部成本，再判断是否组合 |

最新 `run_schema10_native_online_closure.py` 只从 `requests["source"]` 注册一份历史 Source。容量设为16不代表实际比较16份。真正单候选在 d1 锁定，只验证单候选规则，不能证明 d1 优于 d2、legacy 或其他 selector。

同样，CPU 状态机测试、d1-half 的离线函数、命名为 `qcfuse_anchor_d1_d2` 的实验规格，都不是相应 GPU 路径或效果证明。`development_experiment_spec()` 本身仍标记 `live_dispatch_integration_complete=false`。

### 1.1 当前可以确认与不能确认的边界

- 指定 Mistral 路径已经产生 Prefix/K-hook/r=1、digest 和 cost-probe 证据；这些资格只覆盖对应 SHA、补丁、参数和输入，不能推广到所有执行器分支。
- 单 Segment live replay 已接通查池、选择、准备、FinalCommit 和 dense fallback；但最近四次 cache replay 没有生产 reuse commit，不等于收益闭环已通过。
- 已有逐段完整 TTFT 账本、取消尚未提交 copy、若干 setup/lookup 优化。它们有用，但并未证明最终端到端快于匹配基线。
- native-dense-continuation 曾出现 teacher-logit 超阈值，仍是**失败的 opt-in 候选**，不得因 greedy token 相同直接启用。
- Qwen、任意多 Segment、真实并发、完整 Profile 与 H1–H5 没有因此获得新资格。

### 1.2 时间解释必须纠正

`native-efc2876-server46068-v55` 中：native Prefix dense 为58.043363 ms；fixed15 的35.093678 ms是 **boundary→first-token**，不是完整请求 TTFT。不能据此宣布已超过20%端到端收益目标。

`online-c7730ad-cache-20260913` 中四次完整 TTFT 为167.518581、137.631380、91.063125、110.778334 ms，均未 commit。它们不是严格随机交错的基线 A/B，不能据此给出稳定加速比。

其中一次91.063125 ms的墙钟分解为：

| 互不重叠的区间 | ms |
|---|---:|
| 排队 | 0.008788 |
| request context 初始化 | 5.393841 |
| context open→selection closed | 7.137350 |
| 间隙 | 0.000142 |
| ready 检查 | 1.062127 |
| FinalCommit planner | 4.868954 |
| cancellation/finish dispatch | 0.025331 |
| remaining prefill submission | 72.184349 |
| 其余 bookkeeping/logits/first-token | 0.382243 |
| **总计** | **91.063125** |

7.137350 ms包含共享层执行和准备，不能全叫 Source comparator 开销。72.184349 ms是主机提交区间，不是已证明的纯 GPU kernel 时间。FinalCommit 当前返回的是最终剪枝后路径成本，不能拿它替代剪枝前候选 reuse 成本。

## 2. 不再扩大的大框架

主线保持：

```text
Native Prefix Cache + 完整 token/Segment inventory
→ exact content lookup（非前缀部分）
→ 真实当前状态比较，默认 full_compare
→ 质量代理合格的 Source/repair 计划
→ Source freeze + lease + 资源 reservation
→ winner-only layerwise preparation
→ 请求级 FinalCommit
→ selective reuse 或正确 dense continuation
→ 请求完成后受预算约束的 exact-dense 物化、LRU
```

不恢复固定5%选择时间拒绝门。选择、准备、planner 和其他实际开销全部进入请求时间。
Gate1 保留当前 source-local futility `gamma=1.0`，不是旧版本0.8；资源 reservation 不得删除；FinalCommit 保持请求级 `gamma=0.8`。

只有 exact dense full prefill 可创建 canonical Source；一份 BF16 Artifact、exclusive CPU/SSD backing、GPU hot replica、Source freeze 后不能换第二名、lease/copy/execution 保护继续保持。

CFO 已退出主路径和 readiness 依赖。旧消融/旧结果可以读取，但这轮不再修 CFO extractor 或重开 CFO 实验。Source 状态放得下就一次向量化比较；放不下才有界 microbatch。

**收敛不是把所有候选都装进去。** 默认只保留一条有证据的主选择路径、一条修复路径和必要的 dense/legacy 回退；没有净收益的模块停留在独立实验分支或退役。

## 3. 总顺序与每阶段停止点

| 阶段 | 核心问题 | 主要交付 | 进入下一步的条件 |
|---|---|---|---|
| S0 | 账本/资格是否可信？ | 候选与剪枝成本审计、冻结输入和基线 | 端点一致、总账闭合、无失效资格 |
| S1 | 已知 winner 的执行器为什么慢？ | fallback 修复、固定 winner 对照、overlap/host 分解 | 正确性通过；收益空间已量化，或给出明确负结果 |
| S2 | 多历史 Source 的选择场景是否真的存在？ | 自然 trace 机会审计、Source 价值矩阵 | 观察互补性和可复现收益空间；否则缩小方法主张 |
| S3 | d1/d2 能否捕获该空间？ | full-candidate depth shadow 与真实 selector 对照 | held-out 质量、排名和开销证据；否则 legacy/dense |
| S4 | Source×repair 是否优于固定15%？ | 九点真实 QA/cost 表和阈值候选 | 质量达标且包括选择开销的完整收益成立 |
| S5 | 哪个外部机制值得加？ | SparseX-inspired / QCFuse-inspired 独立消融 | 单模块收益超过测量不确定性，否则不合入主路径 |
| S6 | 是否在实际工作负载中成立？ | chunking、因果池、存储、单 Segment 正收益闭环 | 包含 fallback/物化的 trace 验证通过 |
| S7 | 能否推广？ | Mistral 多 Segment、Qwen 独立复验、后续并发 | 每个扩展单独正确性与收益通过 |
| S8 | 是否具备论文证据？ | 最终 Profile、资格、H1–H5 | 冻结后独立评估，不用开发调参结果代替 |

S0后可准备S2的数据并执行最小 fixed15 Source质量矩阵；**不要求把S1优化到某个人为漂亮加速比才允许检验研究前提**。但S2的强制Source诊断不等于生产commit，S1正确性不通过时不得测收益。

单 Segment 是当前主线。不得以“多 Segment 应该更快”为由跳过S0–S3；若目标单 Segment 确实不存在足够收益空间，先报告break-even负结果并重新确认工作负载边界，不偷偷切到更有利场景。

## 4. S0：先补账本和实验隔离

### 4.1 FinalCommit 必须输出所有实际决策路径

修改 `v8_schema6_planner.py` 的实现及schema10调用方，保留旧schema读取兼容。新审计记录至少包含：

```text
decision_start_elapsed_ns / decision_end_elapsed_ns
candidate_reuse_subset / boundary_vector / mask_digest
candidate_joint_future_ms / candidate_total_ms
each_pruning_step: removed_segment, resulting_subset, future_ms, total_ms
post_prune_dense_or_partial_total_ms
matched_dense_total_ms / gamma
measurement_key / measurement_row_digests / planner_snapshot
realized_disposition / realized_TTFT / realized_overrun
```

`candidate_total = actual_sunk_at_snapshot + predicted_joint_future`。
如果 planner 自己耗时数毫秒，snapshot至commit的时间必须明确进入模型或提交前最终核验，不能以决策开始时的旧sunk掩盖planner时间。固定一次审计清楚的开销预算/更新步骤，不能无限递归重算planner。

返回 dense 后的成本大于0.8 dense，不代表此前候选成本也是该值。测试必须覆盖 candidate PASS、candidate FAIL后dense、partial pruning、缺cost和stale snapshot。

### 4.2 计时与基线

- 主指标：同一 arrival→实际 first-token callback 墙钟；以整数ns验证各互斥区间之和等于总值。
- 模型前段、comparison、load、repair-check、future、logits各自标明端点，CUDA服务时间不能直接相加成TTFT。
- 同SHA/补丁/GPU/request/sampling/实际Prefix命中行数/热身与backing状态配对。
- 区分 uncached dense、native Prefix+dense、resumable dense、已知winner source-only诊断、真实生产selector。不能将 source-only 时间写为系统TTFT。
- CUPTI/逐层instrumentation只作归因；性能臂关闭高扰动采样，并做仪表开/关配对。
- 初步2次warm-up+5次采样只用于smoke与ETA；收益验证预注册至少20个交错paired repetitions，保留慢样本，报告配对差及区间。依赖/related requests按content group聚合，不能把重复timing当独立质量样本。
- 重放exact相同请求时，native Prefix基线获得同等历史缓存机会；不能只让ProbeKV受益于上次请求。

### 4.3 最近 selector cache 默认不作主性能臂

审计 `v8_schema10_online_backend.py` 的实际集成，而不是只测试独立的 `selection_result_cache.py`。

需核对key中的完整输入/位置/Segment边界、模型和tokenizer数学签名、选择Profile、完整候选identity/digest/SelectionState版本；确认实际metadata digest字段存在。LRU访问不是语义变化，但候选增删/质量配置变化必须失效。

只缓存语义观测/排序，不复用旧经济PASS；命中仍重新核验Source资格、资源、成本、snapshot及FinalCommit。snapshot/restore、A/B、K容量replay必须明确隔离cache。增加实际backend集成测试，禁止第二个replay自动变成未标记的新策略。

若维护成本大于收益，关闭这一可选cache。它不能替代未知请求的当前状态选择，也不是多Source贡献的必要条件。

**S0交付：** `accounting_contract.json`、配对baseline manifest、cost-decision事件schema、cache审计与回归结果。没有真实新测量时support仍为pending。

## 5. S1：修复执行器，先定位而不是猜原因

### 5.1 Dense fallback

对照原生dense、当前resumable剩余层、opt-in native continuation的逐层hidden/residual/QKV/attention输出。首个偏离层定位到norm、position、attention mask、KV布局、Prefix shadow或数值kernel路径。

沿用greedy token完全一致和至少32个共同teacher位置relative-L2≤1e-4的正确性合同；出现超阈值停止性能比较。若原生Prefix本身与no-cache参考不同，单独保留该控制，不借此放松切换路径阈值。

优先优化**既有正确数值路径**中的重复metadata construction、位置索引、tensor分配、projection、同步、逐层Python/setup。未经等价验证不能换attention kernel。dense fallback复用已完成hidden/residual，不从第一层重算；同时不能把probe/准备已花的时间删掉。

### 5.2 已知 winner 与真正 live selector 分开

同输入和Prefix条件分别跑：

1. native Prefix+dense；
2. 已知winner、请求前GPU-resident、固定mask；
3. 已知winner、pinned CPU streaming、固定mask；
4. live selector、相同winner/后端；
5. live selector拒绝后的dense。

记录真实executed rows、层调用数、H2D bytes、ready/wait、workspace/reservation、host launch gaps。GPU-resident必须用零请求归属H2D和完整resident layers证明，不能用“all ready”名称代替。

对候选预算：`允许前置开销 = 0.8*T_dense - matched_reuse_future`。右侧非正时该形状没有当前计划的20%准入空间；不得改gamma造收益。fixed-winner是省去选择的乐观执行对照，不自动称为数学最优下界。

### 5.3 Overlap 与开销逐项处理

依次检查同步、setup、mask构造、copy提交、planner查询、日志、hash/pinning。每项必须“单一改动→新SHA→正确性→同条件配对”，不能从不同SHA或不同GPU的两条均值推断因果。

维护pending/submitted/ready/consumed全生命周期；拒绝后取消未提交copy，已提交copy保留lease直到completion。qualification验证source-before/destination/source-after；online_immutable不做每请求full-KV SHA256。正式CPU路径已pinned，SSD staging使用有界双缓冲，不静默整Artifact pinning。

overlap以真实copy/kernel交集和关键路径节省报告，不能以人为延长copy获得更高重叠率。

**S1停止点：** 正确性通过、完整TTFT可解释；固定winner和live差额量化。若仍无收益，明确是backend、选择成本、短请求空间或成本支持问题；不一律归因“selector40ms”。

## 6. S2：首先验证多 Source 场景是否存在

### 6.1 两种证据必须分开

**自然机会审计：** 从冻结development partition按原请求顺序检查同exact Segment的重复出现、左侧context/位置变化、native Prefix已覆盖比例和未来再访问次数。输出non-prefix reuse opportunity，不假设人工重复fixture代表真实RAG频率。

**受控因果实验：** 可以用真实文档构造相同Segment+不同历史前文，固定合成规则；必须标注controlled-context，而非自然trace。它检验机制存在性，不能代替自然部署收益。

重要因果约束：decoder-only causal self-attention中，Segment之后的question变化不会改变该Segment自身的current K。仅变suffix的问题组应是负对照；要检验Residual-K对历史上下文的识别，必须有真正不同的左侧prefix。query-aware辅助probe是另一机制，不能把后置query信号写成Segment early-K已经包含。

### 6.2 最小预注册 Source-quality 矩阵

建议首轮从三个既有development数据源各选4个content groups，共12组：每组4份**在评估请求之前构建**的canonical历史Source，再选6个未来请求。有效数据不足如实pending，不复制单个问题充数。

- 72个request cells×(dense+4个Source)，fixed15、统一后端/Prefix和统一actual boundary；共360个基本执行action，重复计时另列。
- 先全部GPU-resident隔离Source质量；CPU/SSD访问成本后续加入，不能让tier差异掩盖“Source是否更合适”。
- d1/d2比较的公共boundary首轮可固定first-reuse-layer=3；必须先完成对应repair-check。legacy统一边界另做影子诊断，不能强制第8层才选出的Source在第3层复用。
- 受控组划分在测量前冻结fit/validation；同content/document group不能跨集合。首轮样本只是可行性，不当正式Profile或论文统计样本。

已有 `run_native_source_oracle()` 支持fixed15/r1与真实QA，可先复用；它目前拒绝teacher/capture和其他ratio，扩展必须分离QA生成、teacher诊断及timing臂，不能在原API未支持时宣称九点Oracle完成。

对每个请求，保存所有Source的QA、teacher-logit误差诊断、实际Source-only耗时、actual boundary、digest、repair count、bytes；reference答案和scorer版本必须存在。

定义固定boundary/ratio下：

\[
\mathcal S_Q(q)=\{s:F1(q,s)\ge F1(q,dense)-0.02\}.
\]

这是开发期QA合格集合，不是概率安全证书。无参考答案时不能把Residual-K当QA。KL阈值尚未冻结时只报连续误差，`safe_hit=null`。

Source-only质量约束Oracle在这个集合中选实测快者；QA最高Source另报，避免“最快=质量最好”混淆。实测近并列按预注册噪声带处理，不用计时抖动定义唯一winner。

### 6.3 必须比较的策略和判据

- latest、deterministic random、fit-only最佳固定Source（validation不重新选）、full-candidate Oracle；CFO只作为将来最近邻论文比较，非本轮资格前置。
- 检查不同未来请求是否需要不同Source，是否存在某个固定Source无法覆盖但其他Source可覆盖的请求。
- 报告Coverage(K)、Coverage Gain、额外存储和物化成本；不仅报告Variant之间距离大。
- Oracle池的嵌套集合可以有单调覆盖上界；不同容量LRU因果重放的Operational Coverage不保证单调，负marginal gain保留。

**S2判定：** 有真实非前缀机会、有互补Source、相对强单Source策略有覆盖/repair-cost空间，才继续把多Source作为核心贡献。如果没有，先检查数据/位置控制；合法负结果不能靠只挑有利case翻转，转入K=1或较小容量方案并收窄主张。

## 7. S3：验证 d1、d1/d2，而不是让分数自己证明分数

### 7.1 固定观察语义与实验臂

\[
K_{obs}^{(d)}=W_K^{(d+1)}Norm_{d+1}(h^{(d)}).
\]

d表示完成block数；d=0仅负对照。d1只有一次观察，不能触发“连续两checkpoint稳定”。

先固定参考trim=.15、fixed15 repair，不同时调所有参数。Mistral shadow捕获{0,1,2,4,5,8}；Qwen随后用{0,1,2,4,5,7}。

实验臂：`d1_only`、`d1_d2_full_rescue`、`legacy_full_compare`；成立后再单独加`d1_half→d2`。K≤8默认全部比较；K=16也先全量。flat/tied d1不任意砍一半。

两个任务分开：

1. **表示判别力：** 在共同可执行boundary的Oracle标签下比较不同depth排名；继续dense的完整shadow不计在线TTFT。
2. **真实dispatch：** 每臂使用其合法实际boundary/repair-check/early-exit，真实测选择到首token。shadow节省不能充当在线剪枝时间。

d1/d2未通过则保留legacy；legacy也可能失败，最后dense，不把legacy称为自动保证质量的证明。

### 7.2 指标和分母

- StateAvailability：所需Source×checkpoint状态可用数/应可用数。
- SelectionCoverage：合法Source locks/至少1个correctness-eligible Source的请求；另报全工作负载coverage。
- QA-safe hit：所选Source属于实测QA合格集合的比例；空集合单独统计，不填100%。
- tie-aware winner/near-best hit、质量违例、选错后实际成本损失。
- `wrong_early_lock_residual`：与完整候选相同参考depth残余分数Oracle不同；`wrong_early_lock_quality_cost`：与独立QA/cost标签不合格/非近优。两者分开，不把Residual-K Oracle当QA真值。
- shortlist recall、绝对regret和normalized regret。归一化分母floor及tie带在fit阶段冻结；无有效Oracle或零方差排名时为null。
- 真实selection wall-clock、完整请求TTFT、Fallback税和GPU状态scratch峰值。

保留既有development选择门槛：coverage≥80%、对应明确定义的wrong-lock≤5%、mean normalized regret≤0.10；不据此声称尾部质量已认证。**旧5%选择耗时硬门不恢复**，最终以完整E2E成本为准。

来源只有1个时margin=null，走single-candidate路径；原eligible>1而只比较1个时不能声称全池最好。线上freeze后不再比较；早锁错误只能由另跑shadow得到。

**S3交付：** depth source-quality/cost报告、真实执行depth轨迹、选择开销与收益预算、失败回退选择。阈值只用fit，validation失败不能现场重调后称原方案通过。

## 8. S4：Source×repair联合决策与I/O balance

### 8.1 两步建立，避免初期混杂

先完成固定参考trim=.15的Source选择验证，再对每Source的同一层token drift排序一次，通过prefix sums得到九点剩余漂移：

\[
\mathcal R=\{.05,.075,.10,.125,.15,.175,.20,.25,.30\},\quad
m(r)=\min(N-1,\lceil rN\rceil),\quad
R_s^{(d)}(r)=\frac{\sum_{j\notin TopK_{m(r)}}\delta^K_{s,j}}{N-m(r)}.
\]

一次排序得到九个proxy，不等于免费得到九次QA结果；d1和d2仍需各自计算。不同Source曲线会交叉，不能选15%分数最优后声称它对所有ratio最优。

扩展真实Oracle的ratio支持，测量对应QA/执行成本，形成每模型×depth×ratio×明确长度支持域的经验threshold。先固定selector再验证Source×ratio候选，不循环反复调整selection与repair获取validation最优。

候选集合中选择质量代理合格且有实测成本支持的`(source,ratio,access_plan)`；缺测量逐候选排除，不能让未测SSD Source把可支持CPU候选一起否决。freeze后只允许同Source访问重规划。

### 8.2 实际 repair metric

维护三种独立、明示名称：历史V-only兼容、K-only、规范化KV-deviation候选。先锁定本项目CacheBlend patch实际metric、layer、normalization和Top-K，再做V/KV配对；**不能因为论文写KV deviation就给V-only实现改名冒充一致**。

selector的`source_score_trim_indices`绝不能直接成为runtime repair mask。winner repair使用当前位置对齐后的K/V、明确RoPE语义和可复现绝对位置tie-break。r=1正确性端点不产生canonical Source。

### 8.3 I/O余量不是一律降低修复率

- compute已比load慢：增加ratio只会更慢；只在质量支持范围内尝试降低ratio。
- load更慢：可以增加ratio利用被隐藏的计算时间，但只在实测joint completion不增加且支持集合允许时做。
- 请求/层采用统一ratio主候选，多Segment的repair时间用union rows实测；不逐Segment独立相加。
- gradual/no-reentry时`M(l+1)⊆M(l)`。后来I/O变慢不能重新加入先前已丢弃token；最多保留当前support。新策略不得悄悄违反这一点。
- 九点是质量/Source选择网格；20/30/50/75/100%的历史runtime I/O-grid作为独立测量扩展，未经QA支持不能仅因已测耗时上线。

效率优先默认使用最小合格且经济的ratio，并只利用无额外关键路径成本的余量。质量优先单独实验，不能混入效率臂。FinalCommit的0.8是预测准入，不是每个未来请求实际零超支的保证。

## 9. S5：SparseX 与 QCFuse 如何真正接入

### 9.1 原论文核验与借鉴边界

- [QCFuse v1](https://arxiv.org/html/2606.05875v1)：compressed chunk-anchor条件化query、少量critical layers用于重算token定位；实现基于SGLang。
- [SparseX v2](https://arxiv.org/html/2606.01751v2)：利用非复用位置的Sparse-Q指导重算，并组织full+sparse执行。
- [Cache-Craft](https://arxiv.org/html/2502.15734v1)：已有多cache variants与候选选择，不能把“我们有多个Source”单独宣称首创。
- [CacheBlend](https://arxiv.org/html/2405.16444v3)：修复与加载流水线的直接基线；不直接拿其论文加速倍数当本项目同条件目标。

本次核对原文版本和机制；QCFuse/SparseX的arXiv记录不等于本次已确认同行评审发表状态。下面是**ProbeKV改造提案**，不是作者已证明适用于本项目的结论。

### 9.2 优先做最小SparseX-inspired repair实验，不立即移植整个后端

条件：S2/S3有可评估的Source标签，现有后端正确，固定winner控制可用。

保持同Source、相同ratio、同实际boundary和既有执行器；新增repair metric候选：从当前请求**原本必须计算且已合法获得**的非复用Q位置产生attention-based token importance，选待修复Segment内部token。

建议新对象`QueryProbeState`记录request/snapshot、producer layer、绝对Q位置、heads/geometry、position semantics、score digest；`RepairMaskProvider`只返回合法绝对位置mask。它不注册第二份Artifact，不加载非胜者full KV。

需要post-RoPE Q/K、GQA mapping、causal可见性、FP32累加和eager小输入核验；不能用pre-RoPE residual差替代attention。若所需query尚未计算，额外probe或更晚boundary必须计费，不能读取未来状态。

对比winner V/KV-deviation与Sparse-Q-inspired mask；固定ratio先看质量，质量matched后再比TTFT。必要边界token仍由系统execution ownership负责，不把原文其他机制一次性全部打开。

只有mask候选有价值、且S1证明剩余瓶颈在稀疏执行时，才启动独立SparseX-style executor feasibility：许可、模型/位置/布局/Prefix/block生命周期核验，之后r=1及masked-path正确性、GPU timeline。当前执行器已存在active rows时，必须量化还有什么可减少，不能因更换名字声称收益。

### 9.3 QCFuse分成两类，不混淆“token选择”和“Source选择”

**Q-R（更直接的借鉴）：** 对已冻结winner，使用有预算的anchor视图和query-conditioned信号提出repair mask；与Sparse-Q及KV-deviation单独配对。这是修复特征候选，可能更适合后置question决定证据重要性的场景。

**Q-S（本项目扩展，必须额外证明）：** 在K较大且完整SelectionState读取确实昂贵时，使用每Source独立anchor摘要做Source shortlist，然后对保留者做现有真实Residual-K。对同content的不同Source，仅保存相同文本anchor ID没有区分能力；必须包含Source-specific状态并绑定digest、checkpoint和原位置。

首轮不写死16→8；fit-only确定anchor预算和保留数，full-candidate Oracle核验QA合格/近优计划recall。不可直接把作者token-localization结果当历史Source-ranking资格。

先用静态/轻量anchor构建验证通路；原作者更复杂的构建或额外query forward若要移植，单独计物化、存储、all-layer anchor传输和临界层cost。压缩视图不必然比当前只读1–2层K便宜。

原文critical层不必是d1/d2；若需要较深层，应计入独立probe成本或合法等待，不能把压缩query的状态冒充主请求已完成深度。模型、SGLang/vLLM差异使直接替换不成立。

### 9.4 组合和退役规则

默认`full_compare`，独立选择消融为`d1_half`或`anchor_shortlist`，**不串联CFO→anchor→d1-half→d2**。当前规格中的旧组合名称不是开启许可；执行manifest要拒绝多个有损shortlist同时开启。

Sparse/QCFuse repair实验也先各自替换一个mask provider，不同时改变Source选择、ratio和backend。只有单模块稳定收益后，才做小型组合消融验证叠加是否仍有收益。

实际selection/repair关键路径未下降、QA变差、Oracle recall不足，或者新增元数据成本抵消收益：默认关闭。加入文档和接口≠必须留下运行时机制。

## 10. S6–S8：把之前未做的事情按依赖收口

### 10.1 Chunking与真实数据

固定分块512/1024/2048和语义窗口±50/±100为独立数据处理消融；优先文档离线canonicalize，避免每请求动态切法破坏content identity。

段落→句子→子句是优先级，不保证任意超长句永不切断。硬max、无标点、Unicode、short-tail必须有确定性fallback并审计；不同策略生成独立manifest/namespace。阈值不能只在固定512训练就直接适用于浮动长度，增加held-out非anchor长度验证。

纳入tokenization/chunking在线真实开销；静态离线预处理计入建设成本。不得每请求按“今天哪个切法快”改变同一文档的canonical边界。

### 10.2 动态池、Coverage与存储

K={1,2,4,8,16}各自从相同初始条件、相同全局byte budget独立因果replay；请求只能看到arrival之前已发布Source。fixed16最终池只能用来报告明确标记的未来信息Oracle上界。

输出四层coverage：proxy-compatible、selected、committed、QA合格且实际节省为正。分母同时报全trace请求和non-prefix opportunity；报告Coverage Gain、LRU变化、Variant寿命、warm-up、写放大、CPU/SSD占用和trace总用时。

保留exclusive CPU/SSD backing，逐tier和per-content LRU按`last_request_use_epoch`；comparison/迁移不刷新，grace、lease、copy、execution保护规则不改。迁移目标完成验证后才能切换backing。

budget-truncated dense可受配额约束物化，但`no_compatible_stored_variant_proven=false`；必须完整exact dense，不能把局部dense或r=1 reuse入池。

### 10.3 Gate1 A/B

先保留explicit barrier。相同静止pool/placement/LRU/grace/Prefix恢复做少量真实paired enabled/bypassed，用实测校验无副作用shadow。shadow预测不能替代实际contention。

使用既有PreparationPolicy的误差/额外开销标准判断是否fused advisory；通过率95%或99%不是删除依据。资源reservation与FinalCommit不删除。页缓存不受控则标注该证据限制。

### 10.4 多 Segment、Qwen与并发

先完成Mistral单Segment真实production commit、匹配质量收益与自然/受控证据区分。多Segment按2→5→10，37只作容量/执行归属压力，不能把构造压力样本混入自然质量均值。

按当前schema10 barrier调度语义做联合mask/边界，不恢复旧A/C为新的必测主模式。请求级shared dense fallback、committed固定路径、union repair不可重复计费；检查每层Q/hidden/position数量和Suffix保留。若未来需要新调度模式另建合同，不能声称浅层选择自动解决因果依赖。

Qwen随后独立tokenizer/Source/threshold/repair/runtime验证；公共代码修改后在最终SHA回归受影响Mistral路径。只有两个模型单活跃请求均成立后，才接多请求并发1/2/4及独立资源生命周期测试。

### 10.5 最终H1–H5与资格

| 实验 | 正式问题 | 前置开发证据 |
|---|---|---|
| H1 | 多Variant必要性、Coverage(K)、Growth成本 | S2、因果池replay |
| H2 | d1/d2/legacy、fullcompare/pruning与Source×ratio | S3/S4 |
| H3 | V/KV/Sparse-Q/QCFuse repair、fixed/static/I/O-aware | S4/S5独立消融 |
| H4 | storage、overlap、Gate1、planner、并发、摊销 | S1/S6/S7 |
| H5 | locked end-to-end matched-quality比较 | 全部正式Profile与资格通过 |

正式基线必须包括native Prefix+dense、single-Variant、匹配后端CacheBlend以及最近邻Cache-Craft；CFO作为比较基线不是重新放回ProbeKV主路径。外部移植结果标记adapted，不冒称作者原实现。

每模型最终只发布一个dispatch时runtime qualification是140×2=280；若发布多个可部署dispatch，每个单独资格。质量风险认证与runtime任务分开：至少每模型300个独立唯一request单位且零违规才可能满足单侧95%上界<1%；同content强相关时还需按分组依赖设计，不能仅凭行数套独立二项式。

90-case development只能选参数和报告观测上界，不能证明1%尾部风险。正式Profile依次冻结selection/admission，再repair/runtime/preparation，最终一致性只通过或失败，不回头重调selection。locked test仍禁止在这些步骤前访问。

## 11. 代码修改清单（本次不实施）

| 优先级 | 现有模块/入口 | 明确修改 |
|---|---|---|
| P0 | `v8_schema6_planner.py`、`v8_schema7_planner.py`、`v8_schema10_cost_provider.py` | candidate与post-prune成本分离、decision耗时、subset重查和快照 |
| P0 | `v8_schema10_online_backend.py`、`v8_schema10_evidence.py` | 完整账本、候选审计、缓存隔离、当前与历史经济决策分离 |
| P0 | `v8_schema10_native_adapter.py`、`cacheblend_continuation.py`、固定补丁 | fallback首偏离层诊断；保持正确数值路径的setup优化 |
| P0 | `run_schema10_native_correctness.py`、`run_schema10_native_online_closure.py` | 相同Prefix配对、仪表控制、cache显式实验臂、至少K>1 fixture |
| P1 | `v8_schema10_native_oracle.py`、`v8_schema10_experiments.py` | fixed15多Source矩阵；再分离teacher/QA/timing并扩展九点ratio |
| P1 | `source_policy_native_capture.py`、`source_policy_replay.py`、`source_policy_development.py` | d1/d2完整shadow、因果控制、独立QA标签、去掉自动组合shortlist |
| P1 | `v8_schema10_selector.py`、`v8_schema10_slack_repair.py`、repair模块 | fit/validation隔离；Source×ratio候选、repair metric与trim强类型隔离 |
| P2 | 新`query_probe_state.py`、`repair_mask_provider.py`（拟名） | Sparse-Q-inspired/QCFuse-inspired repair候选，复用既有executor |
| P2 | 新`source_anchor_state.py`（拟名）及capture/replay | 有预算、Source-specific anchor shortlist，仅独立消融 |
| P2 | `v8_schema10_storage.py`、`v8_schema10_staging.py`、`v8_schema10_metrics.py` | 实际因果K replay、exclusive backing、物化/写入成本 |
| P3 | 统一handoff/manifest与模型adapter | 通过单Segment后扩多Segment、Qwen、并发和正式证据 |

每项都要有：问题ID、假设、最小改动、旧证据、新SHA、新输出目录、正确性测试、配对结果、采纳/退役裁决。没有测量依据不生成cost=0或伪Profile SHA。

### 11.1 之前讨论事项的追踪清单

| 事项 | 本次核验状态/限定 | 安排 |
|---|---|---|
| non-prefix多Source场景与Coverage Gain | 最新online fixture仅1份Source；未见完整自然机会与QA矩阵结论 | S2优先 |
| d1、d1/d2、legacy选择质量 | 有capture/replay与剪枝函数；真实QA/cost结论未完成 | S3 |
| Source选择开销与实际fallback开销 | 有账本，旧40.7ms和35.09ms解释曾混淆端点 | S0/S1 |
| CacheBlend相同后端对照 | 已有适配与诊断；`ResidentRepairPlan`当前拒绝非零cached Prefix，不能据此声称该adapter已完成Prefix匹配 | S1逐入口确认支持范围 |
| Source-only Oracle扩展 | 已有fixed15/r1真实QA；当前API拒绝teacher/capture及其他ratio | S2先用现有合法路径，S3/S4扩展 |
| K/V/KV repair | 候选合同与历史V-only路径并存，不等于KV主路径已验证 | S4固定metric再配对 |
| 一次排序九点ratio | 有development候选设计；不是九点真实QA/成本证书 | S4 |
| I/O-aware统一层ratio | 有controller相关模块；当前目标线上净收益仍需独立验证 | S4、S6 |
| 自适应semantic chunking | 有开发组件；非anchor长度质量、成本和命中影响未完成 | S6 |
| CPU SelectionState独立budget | 存储目标不能等同于所有Source状态已常驻；需检查落盘/命中/传输 | S1归因后，S6实现与实测 |
| SSD优质Source未来晋升CPU | 与当前服务分离；队列、带宽、完整写入校验及LRU不刷新都须计账 | S6 |
| exclusive backing / per-content LRU / grace | 有代码与测试；连续真实trace及保护/失败恢复证据需完整 | S6 |
| Gate1可否融合 | 无证据支持仅凭高通过率删除 | S6真实paired A/B |
| SparseX、QCFuse | 规格登记不等于生产实现 | S5，分别验收或退役 |
| GDS、多请求并发 | 非单Segment前置，不能借硬件不可用拖住主验证 | staging主路径通过后独立扩展 |
| 双模型Profile、资格与H1–H5 | 开发数据不能自动解锁 | S7/S8 |

完成状态以后以原始artifact digest和对应SHA更新，不以“已写进计划”或函数名判断。

## 12. 明确的收敛标准和止损规则

分开报告，不合成一个模糊passed：

```text
runtime_correctness_passed
matched_timing_accounting_verified
multisource_opportunity_observed
multisource_complementarity_observed
early_selector_heldout_validated
production_reuse_commit_observed
matched_quality_trace_net_gain_observed
selected_architecture_frozen
```

系统收敛至少要求：正确性、端点可信、真实多Source价值、可泛化选择、生产commit、包含fallback及物化成本的预注册trace净收益、失败路径资源无泄漏、默认策略和适用范围冻结。不能用“内核某一段更快”代替。

每个工作负载仍使用FinalCommit0.8预测约束，但报告actual-overrun与全部请求P50/P95；不能承诺未知请求TTFT永不增加或答案永不下降。质量阈值的经验验证与数学正确性分别陈述。

止损：

1. 没有自然重复机会：报告不适用，不人工制造频率作为部署结论。
2. 多Source没有互补性：默认K=1/较小cap，不继续堆selector复杂度。
3. 有互补性但early-K不能识别：legacy或独立query-aware候选；仍失败则dense，不声称d1已有效。
4. fixed-winner有空间但live没有：按账本修选择/setup/准备，不把总时间藏起来。
5. 固定执行本身无空间：定位后端或声明当前形状不适用；不通过加多个Segment掩盖负结果。
6. SparseX/QCFuse没有净收益：不合入默认系统。
7. 一项修复完成后进入下一问题；没有新的假设或控制，不重复相同全dense合成replay。

## 13. 紧接着执行什么，以及时间如何管理

**下一轮只做三件事：**

1. P0账本：输出候选reuse与剪枝dense的分别成本，审计planner开销与旧cache。
2. fallback首偏离层/剩余执行开销定位；同条件固定winner与native Prefix完整TTFT对照。
3. 准备并运行最小真实4-Source价值矩阵，以及d1/d2全候选shadow。只有固定15%路径正确后才执行；不等待QCFuse/SparseX实现。

2026-09-14 P0/P1服务器诊断结果已记录于[结果报告](P0_P1_SERVER46068_RESULT_20260914.md)：补丁树重建和cost-probe门通过，但当前d1在线重放两次均为dense fallback，逐层resumable remainder约68 ms，是下一轮唯一优先瓶颈。该结果不等于多Source或生产reuse成功。

先CPU测试与manifest冻结，再在已确认实例做分阶段A800任务。每阶段6-case canary实测ETA，之后按剩余任务×实测单任务耗时估算；按model加载、Source构建、答案decode、重复次数分别计算。不预先保证“几小时必有正收益”，也不把旧4小时上限静默改为无限租用；开始长任务前记录本次授权范围与费用。

本轮计划阶段不启动/停止实例。正式Profile、GPU runtime qualification、H1/H2执行资格、paper evidence、locked-test accessed仍均为false。

相关历史文档：[统一方案](PROBEKV_UNIFIED_SOURCE_SERVING_PLAN_20260910.md)、[单Segment Oracle计划](PROBEKV_SINGLE_SEGMENT_ORACLE_NEXT_PHASE_PLAN.md)、[时间解释纠偏](SERVER46068_TIMING_CORRECTION_20260913.md)、[fallback与失败numerical候选](SERVER46068_FALLBACK_ACCOUNTING_20260913.md)。
