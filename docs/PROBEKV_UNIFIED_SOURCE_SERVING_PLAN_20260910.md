# ProbeKV 统一方案：质量约束下的历史 Source 选择、修复与分层服务

状态：**统一框架 + 用户确认后的无卡候选实现；不是已冻结的 GPU 执行合同。**

日期：2026-09-10。审计基线：`73ec107`；最近已记录的后端实验运行 SHA：`3b365bc`。

首次分析只阅读材料并核查证据。随后用户要求实施七项补充，本次已增加无卡策略组件、离线分块入口和测试，详见第13节。尚未切换生产默认、改动历史补丁或旧Gate，未连接服务器、未启动GPU、未提交或推送Git。

**2026-09-11 方案修订：**第14节纳入 CFO 按收益启用、九点比例曲线、Source×repair 联合选择、I/O 余量利用与冗余机制简化。与前文不同之处以第14节的下一轮计划为准；第13节的实现记录和旧实验合同不被追溯改写。此次仅更新方案，不代表新候选已实现、Profile 已通过或 GPU 已获授权。

## 1. 输入、证据边界与结论

按用户要求先读文本，再读 PDF 全部 32 页，并对公式和关键页面做视觉核对。

| 输入 | SHA256 |
|---|---|
| 文本附件 `b07b062e-9d33-4fa7-8f40-d42079fe4889/pasted-text.txt` | `e384501a5cc2b3152943d16d2c15d15be11979abc1e88ad8b7462fc877a8eda8` |
| `计算机实验 - 连接Github分析代码.pdf` | `59b061d553b161c0ded9e7dc905fda4d372d370c5012e79a3d3277f719d54f97` |

附件是设计讨论材料，不是实测证明。PDF 页码以下均指本文件第 1–32 页，不使用其原聊天导出的 51–82 页页脚。

建议统一主线为：

> 在相同修复后端上，根据当前左侧上下文状态，从同一 exact Segment 的历史 KV Variants 中选择质量代理合格且访问成本合理的 Source；对胜者执行有质量证据支持的修复，并由请求级最终成本准入决定 reuse/dense。另用独立、受预算约束的存储晋升改善未来请求。

核心创新候选是 **Source-aware matched-quality efficiency**，而不是“对同一 Source、同一 mask 比 CacheBlend 的修复 kernel 更快”。是否形成有价值的贡献，必须先通过 Source-quality/repair-cost 实验，不能用系统复杂度代替证据。

### 1.1 建议的采纳范围

| 材料中的建议 | 处理 | 必须补充的限定 |
|---|---|---|
| 多历史 Source 改善质量—延迟关系，PDF 1–11 页 | 采纳，作为主问题 | 必须胜过 Latest/Random/CFO 的同后端比较，计入选择开销 |
| 一次排序计算多个比例的剩余漂移，文本 | 采纳 | 这是 proxy，不是 QA 安全证明；K 选择分数与实际 V/KV repair 排名不能混用 |
| 5–30% 在线比例候选，文本 | 作为新策略网格采纳 | 未经质量/成本支持的点不可上线；30% 是策略上限，不是所有更高比例都不经济的定理 |
| 有余量多修复，文本 | 有条件采纳 | 主效率策略只利用不增加关键路径的余量；主动花完延迟预算作为独立质量优先模式 |
| 质量合格后按总成本选 Source，PDF 12–17 页 | 采纳 | 只声称当前比较集合与冻结参考修复方案下的预测最优，不声称全局真实最优 |
| CPU/SSD 的价值看 exposed load，PDF 18–22 页 | 采纳 | 不以裸带宽、线性 repair 比例或 PDF 的估算替代测量 |
| 当前服务与未来晋升分离，PDF 23–32 页 | 采纳 | 后台搬运也计资源/字节/时间；不得暗中让未胜者完整 KV 进入 GPU |
| 热 content 保留 1/2/4 个 CPU Variants | 采纳为软目标 | 全局容量优先，不能承诺每个 content 至少 4 个；不改成新的价值分数淘汰 |
| SSD Source 的 SelectionState 常驻 CPU | 采纳为有预算的数据面目标 | SelectionState 与完整 KV backing 解耦，独立计字节；不能假定所有 content 的状态无限驻留 |
| 根据并发负载决定访问成本，文本 | 保留到后续阶段 | 先完成单活跃请求；未知并发测量单元不外推、不给资格 |

## 2. 当前系统真实起点

### 2.1 已有内容与缺口

| 模块 | 已有 | 本方案需要补齐 |
|---|---|---|
| `cacheblend_continuation.py` | 一次性移交真实 dense 层状态，避免重复执行已完成层 | live SelectionState 比较、冻结/租约、生产 FinalCommit；之后分别接 Prefix 和 streaming |
| `v8_schema10_selector.py` | absolute threshold、早停、最终 residual band + 成本选择 | 新模式最终选择覆盖全部 proxy-qualified 候选；使用一致的访问计划成本，不能用 Gate1 乐观下界代表最终最快 |
| `v8_schema10_online_backend.py` | 真状态比较、单活跃请求、Source Pool 接口 | 比较接入 continuation；保留排序/尾和；候选缺成本应逐候选处理，不能由一份未测 SSD Source 连带否定所有热 Source |
| `v8_schema10_slack_repair.py` | 固定胜者、一次排名、多个嵌套 mask、逐计划查询成本 | 目前仅 development shadow，外部传入 quality support；尚不是在线质量准入器 |
| `v8_schema10_cost_provider.py` | shape 查询与 PlannerSnapshot 分离、缺测量返回 UNSUPPORTED | 现有 exact-mask 支持范围窄；逐步引入经过 anchor 验证的 shape 类别，不静默泛化 |
| `v8_schema10_storage.py` | 单请求静止状态下的 exclusive backing、迁移、LRU | SSD SelectionState 目前仍落盘；独立 CPU state budget、受控 promotion queue、事务化后台执行 |
| `v8_schema10_native_oracle.py` | 真 dense/Source QA、静止池恢复 | 当前主要固定 0.15/1.0，需完整比例矩阵、teacher-logit 诊断、Source-quality 曲线 |

### 2.2 最新记录不能被过度解释

`docs/PROBEKV_CONTINUATION_HANDOFF_20260909.md` 记录：Mistral、单个 512-token Segment、零 cached Prefix、GPU-resident 固定 Source、两次 warm-up 后三次测量。

| 首个复用层 | dense 均值 | continuation 均值 | matched pinned loop 均值 |
|---|---:|---:|---:|
| 2 | 73.923 ms | 50.474 ms | 48.537 ms |
| 3 | 73.918 ms | 51.466 ms | 49.464 ms |

这说明近期重复执行/setup 优化有效，且同后端差距明显收窄；不能证明真实 Source ranking、Prefix 组合、CPU/SSD 加载、自然 QA 或多请求并发已通过。这里的 matched loop 是经过审计的适配路径，不是未经修改的 upstream CacheBlend。此前测试记录为 764 项中 763 通过、1 项可选依赖跳过；本轮未重新运行测试，也不据此新增通过声明。

## 3. 稳定大框架与不变量

保持现有身份结构：

```text
canonical Segment provenance
→ exact token content bucket
→ 最多 16 个历史 Source Variants
→ 每 Variant 一个 lossless BF16 canonical KV Artifact
→ CPU 或 SSD 一个 authoritative backing
→ 按需 GPU execution replica
```

1. canonical Source 只来自目标模型完整、无缓存的 exact dense full prefill；repair 派生、r=1 reuse、Prefix+仅剩余部分 dense 不得晋升。
2. Source identity 绑定历史 prefix、位置、模型数学签名与 occurrence；迁移不改变身份。
3. canonicalizer provenance 不混入 token 内容正确性身份；不按历史缓存长度重新切 Segment。
4. Prefix 优先：完整覆盖的 Segment 不比较；部分覆盖的 Segment 剩余尾部 dense，不临时重切以制造匹配。Prefix/mandatory suffix 不进入 repair pool。
5. Source 一旦冻结不可切换。胜者准备、质量确认或 FinalCommit 失败后 dense，不改选第二名。下一请求可重新选择。
6. 物化在当前请求的 lookup/coverage 记录完成后发布，禁止未来信息泄漏。
7. 当前仍为单活跃请求；多 Segment 和多请求并发不是同一能力。
8. 不恢复固定 5% 选择时间硬门。选择成本仍须进入总账，并可基于剩余收益预算提前停止比较。

## 4. 选择与修复：统一算法，但不混淆质量代理

### 4.1 四个量必须分开

| 量 | 含义 | 是否等于实际 repair |
|---|---|---|
| `source_residual_trim_ratio`，rho_trim | 多 Source 排名时去掉最大的部分 K 漂移 | 否 |
| `r_reference` | 资格阈值及 Source 成本预览所绑定的统一参考修复方案 | 不一定；首轮固定 0.15 |
| `r_min_proxy` | 胜者在有限测量网格中通过残余代理规则的最小点 | 不是数学安全下界 |
| `r_execute` / layer support | 最终获质量与成本支持、实际执行的修复比例/位置 | 是 |

不引入学习型 per-Source r_safe 模型。绝对阈值和 ratio support 是固定的 development Profile/查表规则，不是在线训练；但依然需要独立验证，不能因为“无训练”就免除质量审计。

### 4.2 Source K 分数

对 exact Segment 的 N 个 token：

\[
\delta^K_{s,j}=\frac{\|K^{cur}_j-K^{src}_{s,j}\|_2}{\max(\|K^{cur}_j\|_2,\epsilon)}.
\]

\[
m_{score}=\min(N-1,\lceil\rho_{trim}N\rceil),\qquad
J_s=\frac{\sum_{j\notin TopK_{m_{score}}}\delta^K_{s,j}}{N-m_{score}}.
\]

所有分数相同的 Top-K 按绝对位置确定性打破平局。N=1 不进入新在线 ratio-tail 路径，走 dense；r=1 作为独立正确性端点，不计算空残余集合均值。

深度仍定义为已完成的 Transformer blocks：`K_obs[d]` 是完成 d 层之后观察到的第 d+1 层输入 K。d=0 只作负对照。Mistral legacy 为 `{1,2,4,5,8}`，Qwen 为 `{1,2,4,5,7}`。不得修改历史 adapter 常量后反向重释旧实验。

重要限制：如果问题 Q 在 Segment C 后面，causal self-attention 下 C 的 K 不含未来 Q。因此本方法是 current left-context-conditioned selection，不能直接称作 query-aware selection。实验要加入“相同 C/左侧上下文、不同后置问题”的压力项来揭示此限制，而不是通过重排提示词偷偷改变任务。

### 4.3 固定参考比例的 Source 选择基线

固定参考比例基线分两步，避免为比较 16 个 Source 加载 16 份完整 KV/V；第14节新增联合 Source×ratio 候选，与此基线实测比较，不直接删除此路径：

1. 用各候选独立的轻量 pre-RoPE K SelectionState，筛出在冻结参考修复方案下 proxy-qualified 的候选。
2. 在这些候选中，使用同一 `r_reference`、boundary、计时端点的访问计划，选择预测总成本最低者。

\[
\mathcal S_{proxy}=\{s\in\mathcal S_{compared}:J_s\le\theta_{model,d,\rho,reference-policy}\},
\]

\[
s^*=\arg\min_{s\in\mathcal S_{proxy}\cap\mathcal S_{cost-supported}}
\widehat C_q(s,r_{reference}).
\]

默认参考路径先用 fixed15；阈值必须与这个完整 repair policy 绑定。不能以 fixed15 的资格阈值声称 5% repair 也安全。也不能说该策略找到了全部 Source×ratio 的全局最优：未经比较的 Source、仅在另一修复方案下合格的 Source，都不在其声明范围。

新策略最终层不再把相对最小 J 的 narrow band 作为唯一主选择范围。只要绝对代理质量资格成立，就允许选择 J 略高但当前访问更便宜的 Source。旧 residual-band 策略保留为隔离基线。

早停仍保留，但不能只因 SSD Source 的 residual margin 很大，就跳过当前已知更经济的合格候选。第一版 d1 早停要求 residual 决策与支持成本的选择一致、且通过冻结早停规则；不一致则进入 d2/legacy 的下一 checkpoint。最终层才按完整的当前合格集合决策。整个新决策过程需重新评价 wrong lock/regret，不能继承旧策略资格。

CFO 只负责预算不足时的排序/shortlist；不替代 Residual-K。真正唯一 correctness-eligible Source 可走 single-candidate 路径；16 个 eligible 但只比较 1 个不能伪装为确定胜者。部分候选不合格不能证明整个存储池没有兼容 Source。

### 4.4 胜者确认与一次排序、多比例代理曲线

freeze 后仅准备胜者。当前固定 CacheBlend patch 的兼容执行仍是 V-only，因此历史结果必须继续标为 V-only；本轮新增的 CacheBlend-aligned 主候选改为 winner-specific **KV deviation**（K 与 V 的联合漂移），K-only 与旧 V-only 仅作为隔离对照。不能把尚未实施的 KV metric 反向重释成已有 V-only 结果。

记所选 repair metric 的每 token 漂移为 e_j，按从大到小排序为 e_(1)...e_(N)：

\[
m(r)=\min(N,\lceil rN\rceil),\qquad
R_{tail}(r)=\frac{\sum_{j=m(r)+1}^{N}e_{(j)}}{N-m(r)}.
\]

对新的在线候选网格：

\[
\mathcal R_{online}=\{0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30\}.
\]

一次排序加尾和即可得到所有九个候选点；**不是在线执行九次 Transformer repair**。离线质量验证仍须真正执行相应 repair 路径。每个候选 mask 仍需要合法的成本查询，查询与 mask 构建 CPU/GPU 开销都进入账本。此网格为最新方案，已有生成器/历史 Profile 尚不因此自动具备九点支持。

`R_tail <= theta` 只能给出 proxy pass。正式可用集合还必须满足：对应模型/depth/metric/support policy 的质量验证、非空 support 规则、真实成本支持和资源可用。

尤其注意：R_tail 是**在修复前的观测中剔除拟重算 token 后**得到的统计，不是真正执行后再次测得的 KV 误差；后续 attention 会继续传播状态变化。论文和字段不能将它标成“已证明的修复后误差”。

\[
\mathcal R_{feasible}=\mathcal R_{proxy-pass}\cap\mathcal R_{quality-supported}\cap\mathcal R_{cost-supported}.
\]

如果没有不超过 30% 的合法计划，当前新模式 dense。30% 以上仍保留离线修复曲线及历史路径；这不是“更高比例绝对没有意义”的结论。r=1 永久保留正确性端点。旧 `{0.10,0.12,...,1.0}` Profile 不得静默改成新网格。

K SelectionState 的 trim indices 不能直接用作 V-only 或 KV repair mask；阈值也不能跨 metric 复用。更多 repair 不保证每个 QA case 单调变好，因此 quality support 是显式集合，不是看到最小点通过就默认所有更大点通过。

ratio 是名义比例，实际数量使用 ceil，必须同时记录 `repair_count` 和 `effective_ratio=repair_count/N`。不在冻结长度支持范围内的短 Segment 走 dense，不能将小 N 的巨大取整偏差说成已经验证了30%上限。

阈值冻结只在 group-isolated fit 集上枚举预注册 residual 阈值候选，结合真实 QA 标签选择；验证集只评估、不重调。固定参考比例基线先在 fixed15+explicit Gate1 下确定 Source selection/absolute admission，再固定它们评价 winner ratio 与存储成本。联合候选则遵守第14节的 Source×ratio 矩阵与独立验证顺序，不能先选赢家再事后调阈值使其合格。最终一致性失败保留失败，不反复使用同一验证集调参。这是固定规则的经验验证，不是数学安全证明或独立尾部风险认证。

### 4.5 不把“花完余量”与“更快”混为一谈

保留同一框架下两个明确命名的模式：

- `efficiency_first`：在合格计划中最小化预测完整请求时间；在预注册测量误差容忍范围内时间等价时，选更多 repair。作为主系统效率结论的默认模式。
- `quality_within_budget`：在同样通过 FinalCommit 的合格计划中选择更大 ratio。适合质量优先/低负载，作为独立结果报告；不冒充延迟最优策略。

若 dense=55 ms，gamma=0.8，总预算是44 ms，不是50 ms。50 ms 即使比 dense 快，也不能通过本系统现行 FinalCommit。

### 4.6 Gradual repair 不允许状态重入

第一轮先使用所选 initial ratio 的 fixed-support continuation，单独识别 Source 与 ratio 的收益。之后再启用经过 oracle 验证的 gradual。

首个 selective layer 前可在 5–30% 中选初始 pool；之后：

\[
M_{l+1}\subseteq M_l,\qquad r_{l+1}\le r_l.
\]

不能已经只保留15% hidden states，后层 I/O 变慢后又直接升到30%。若未来要允许 reentry，必须设计额外状态保存或重算并独立计费，本方案不包含它。

逐层质量规则不可在 commit 后才发现其要求高于现存 support；必须在 commit 前选取有完整后续支持证据的计划。若实际运行发生超时，记录成本误差；若发生正确性异常，失败停止，不能称为无成本 dense 回滚。

## 5. 成本、overlap 与准入

### 5.1 不增加新的经济 Gate

保留三种职责，不重复计算三次 Source winner：

1. Gate1：source-local futility screening，保持当前 gamma=1.0，避免明显不值得的准备。
2. Atomic Preparation Admission：租约、HBM、staging、copy/waste 预算，不是新的质量证明。
3. FinalCommit：首个不可逆 selective layer 前的完整请求经济准入，gamma=0.8。

无卡阶段不根据推测把 Gate1 融合；是否改为 advisory，仍需真实 paired A/B 和浪费收益证据。freeze 后 FinalCommit 拒绝就 dense。

### 5.2 统一总式

\[
\widehat T_{reuse,q}=T_{elapsed,q}^{actual}+\widehat F_{joint,q}(s,r,\mathcal I,\mathbf b,\mathbf M,\mathbf S),
\]

\[
\widehat T_{reuse,q}\le0.8\widehat T_{dense,q}^{matched}.
\]

- elapsed 从同一到达/计时起点算，包含排队、early layers、comparison、planner、已暴露的准备与等待；不是多个重叠事件持续时间之和。
- joint future 是完整请求尚未执行的关键路径，包含 mandatory rows、选定 repair、必要 full KV attention、ready 等待、调度与收尾。
- 不能将多个 Segment 的完整 remaining TTFT 相加。
- matched dense 是相同 token、Prefix 条件、sampling、计时端点及可比外部负载下“不做本次 ProbeKV 选择/准备”的参考。
- 不将已浪费的 ProbeKV 时间也加到 dense 分母，从而制造通过；也不只比较当前 remaining 而把历史选择时间抹掉。
- 并发时可以使用同系统状态的 remaining 比较作为局部诊断，但不能替代到达时基线的整体 TTFT 报告。共同外部排队必须对齐；本算法导致的排队属于本算法成本。

预测准入不等于测得的性能保证。所有实际超支、dense fallback、负收益请求都进入报告。

### 5.3 真实 layer-wise overlap

\[
stall_l=\max(0,t_{ready,l}-t_{need,l}).
\]

保存每层 enqueue/start/end、t_need/t_ready、compute start/end、stall、copy bytes、path。t_need 必须由真实依赖时间线递推，包含此前等待，不能机械相加独立事件得出总时间。

`max(load_next, repair_current)+nonoverlap` 仅是被 anchor 验证的简化模型。后续层还包含 attention/MLP、mandatory suffix、launch/setup、竞争；不能把 pure repair 当作全部 prefill。

PDF 的512-token/15%修复6–9 ms、CPU等效几个百分点等数值不作为 Profile 输入。Mistral GQA 的完整 BF16 KV 几何确为每 token 128 KiB，512 tokens 约64 MiB；但实际准备可能只加载 boundary 之后的层，延迟必须实测。CPU、SSD staged、page-cache warm、真实冷读取分别报告。

### 5.4 成本表与负载支持

MeasurementKey 用执行形状与负载类别，PlannerSnapshot 验证当前请求/Source/generation。缺测量返回 UNSUPPORTED，不填0、不用J替代时间、不借高比例噪声单点外推低比例。

先保持 exact 单元；需要支持动态 mask 时，单独验证以 token counts/layout/boundary/active rows 为轴的 shape 类别与 joint anchors，再开放查询。样本最大值只称 empirical upper observation，不称风险已认证的 UCB。

后续并发测量采用 factorized cells + 少量 joint anchors；可含 active requests、copy-in-flight、staging backlog、bytes-in-flight、allocator reservations。GPU utilization 只能辅助，不能单独决定可用修复预算。当前未测并发类别不允许沿用单请求 Profile 声称支持。

## 6. 分层存储与两个独立决策

### 6.1 当前请求：选谁来服务

不固定采用 GPU>CPU>SSD 的硬排名。路径由质量资格与 joint cost 决定：GPU 副本未必质量合格；SSD 也可能在长请求中有足够 overlap。

foreground 完整 KV 传输只允许 frozen winner；它失败后不换 Source。已有 GPU-resident 非胜者无需删除，但不能为比较主动传完整 KV。

### 6.2 未来请求：是否晋升 SSD Variant

当前请求使用 CPU Source A，比较发现 SSD Source B 的状态更适合，可以产生 `PromotionCandidate(B)`；它不改变已冻结的 A。

但单次更低J不能证明B长期更有价值。晋升依据至少包括同content的近期使用机会、质量代理优势、预期warm访问收益、CPU空间、迁移/降级成本和冷却时间。首先作为可检验的缓存策略，不宣称长期最优。

首次实现采用**请求完成后的静止维护窗口**，与当前单请求 store 一致。真正异步重叠 worker 要等生命周期/带宽竞争实现并验证后再启用，不能在当前要求 quiescent 的 store 上直接开线程。

必须明确区分字节账本：

```text
foreground_nonwinner_full_kv_transfer_bytes = 0
background_promotion_full_kv_bytes >= 0
background_nonwinner_gpu_prefetch_bytes = 0
```

这修正了旧的全局 `nonwinner transfer=0` 口径：后台非胜者 SSD→CPU 是显式授权的存储维护，不是前台偷取 KV。它要有独立 quota、operation ID、租约、generation、完成校验和后续受益归因，且计入 trace 总成本。

已有已准入 execution/copy reservation 优先。后台无资源就延后；不阻塞本请求来完成“后台”任务。对于全部候选均在 SSD 的情况，正式 reuse 通过 FinalCommit 才复用，否则 dense，可另行登记晋升候选。

### 6.3 Exclusive backing + LRU 继续保留

- CPU满：最久未实际请求使用且未受保护的 backing 降到SSD。
- SSD满：删除最久未使用且未受保护的 backing；无副本后 tombstone。
- 迁移先写目标、校验、原子切换，再回收旧副本。允许事务期间双份，不允许丢失唯一健康 backing。
- promotion/demotion/comparison/snapshot 不更新 `last_request_use_epoch`；请求选择/绑定/实际使用沿用现有合同。
- probation有界grace、lease/copy/execution保护优先于LRU；VERIFIED不获得永久特权。
- 新晋升对象不得靠修改last-use变成“最近使用”；防反复晋升使用单独cooldown/维护预算，不改LRU事实。

### 6.4 Warm floor 是软目标

每content可有0/1/2/4的warm target，热度和对象大小决定是否值得分配；全局字节预算可迫使目标降为0。不是每content固定4份CPU KV，也不是给warm set不可淘汰特权。

首版保留普通LRU作为默认与对照，先评估warm target策略是否降低端到端总成本；不要同时改Source替换、物化、晋升三个策略后无法归因。

### 6.5 SelectionState 与完整 KV backing 解耦

新结构：

```text
SourceVariant
├─ canonical full KV Artifact：CPU或SSD backing
└─ exact BF16 pre-RoPE K SelectionStates：独立有预算的CPU backing
   └─ 有界GPU comparison workspace
```

Mistral 512tokens 单checkpoint K约1MiB，d1/d2×16Sources约32MiB；legacy五checkpoint×16约80MiB。相比完整16份KV小，但不可以忽略，更不能免费常驻整个语料池。

先为可比较的admitted候选保留CPU状态预算；不足则降低驻留集合/明确state unavailable，不能读取完整KV替代。不要把tensor塞进廉价metadata字段，也不默认所有状态永久驻GPU。压缩/anchor签名暂不做主线，要单独资格验证后才考虑。

## 7. 可实现的接口和总流程

建议复用现有类而非再建一套推理框架，增量增加：

```text
SourceComparisonObservation:
  source_id, completed_depth, state_digest, metric, rho_trim, score,
  comparison_scope, comparison_timing

SourceAccessPreview:
  source_id, artifact_id, replica_generation, reference_repair_policy,
  cost_status, estimated_request_cost, measurement_digest, snapshot

WinnerResidualCurve:
  winner_id, metric, observation_depth, ranking_digest,
  ratio_points, repair_counts, tail_means, quality_support_digest

RepairPlan:
  winner_id, boundary, initial_ratio, layer_support_digests,
  quality_evidence, cost_query, execution_objective, snapshot

PromotionCandidate:
  source_id, observed_request_id, target_tier, expected_future_benefit_basis,
  estimated_bytes, generation, quota, cooldown, reason
```

伪代码（第14节联合候选与固定参考比例基线使用显式dispatch隔离；CFO和剪枝不是必经步骤）：

```text
native_prefix_and_inventory(request)
sources = lookup_previously_published_exact_variants()
comparison_policy = frozen_policy_or_default_full_compare(sources)
winner_proposal = null
for depth in frozen_online_checkpoints:
    curves = observe_k_once_and_build_ratio_curves(sources, depth, comparison_policy)
    if dispatch == joint_source_ratio:
        qualified_pairs = frozen_ratio_specific_proxy_filter(curves)
    else:
        qualified_pairs = frozen_reference_policy_proxy_filter(curves)
    preview = supported_request_access_plans(qualified_pairs)
    proposal = choose_by_frozen_objective(preview)
    if legal_checkpoint_decision(proposal) and source_local_gate1(proposal):
        winner_proposal = proposal
        break  # pre-Lmax failure may continue probing; no Source is frozen yet

if winner_proposal is null:
    dense_and_optional_budgeted_exact_materialization()
else:
    proposal = winner_proposal
    winner = proposal.source
    atomic_freeze_and_logical_lease(winner)
    resource_admit_and_prepare_only_winner()
    current_winner_metric = full_repair_check_before_selective_layer()
    curve = sort_once_and_compute_ratio_tails(current_winner_metric)
    plans = same_source_quality_and_cost_supported_plans(curve, proposal)
    plan = choose_efficiency_or_quality_budget_objective(plans)
    if plan missing or fresh_final_commit(plan) fails:
        dense_without_reselecting_source()
    else:
        continue_existing_decoder_state(plan)  # no early-layer replay
    finalize_request_and_release_or_quarantine_resources()

record_request_quality_timing_and_coverage()
publish_only_budgeted_exact_dense_materialization_after_lookup()
enqueue_promotion_candidates_without_changing_foreground_decision()
run_budgeted_quiescent_maintenance_between_requests()
```

任何case的独立dense reference/oracle/canonical capture不隐藏在TTFT外后宣称免费：分别报告服务TTFT、总GPU工作与trace端到端总耗时。

## 8. 先验证什么：单 Segment 证据优先

### 8.1 最优先不是并发，而是 Source 价值矩阵

相同请求、相同exact C、不同真实历史上下文创建的最多16个canonical Sources；不人为加噪声KV制造差异。

对每个Source独立测固定repair网格，包含新5–30%候选与r=1；有需要保留50/75%离线诊断点。与exact dense比较生成答案、F1、teacher-forced分布和耗时。

首先保持所有Source相同residency、backend、boundary和repair规则，回答：

1. Source不同，真实QA和所需repair是否显著不同？
2. deep/full-candidate oracle比Latest/Random/CFO有多少可实现上限？
3. 早层Residual-K能否捕获这部分上限？
4. 节省的repair/回退成本能否覆盖真实selection开销？

各Source使用同一repair metric/比例，但根据其自身漂移产生mask，不能把Oracle Source的mask强行给所有Source。验证两个后端数值等价时才要求同Source、同mask。

固定ratio并不保证答案质量单调；r_min_oracle是离线网格中的最小合格点，属于hindsight，不是在线oracle-free结果。

### 8.2 严格匹配的对照

主比较：

```text
matched native Prefix + dense remainder
same repair backend + Latest Source
same repair backend + seeded Random Source
same repair backend + CFO Source choice
same repair backend + ProbeKV Source choice
quality-qualified Source/ratio Oracle (offline upper bound)
```

另外保留uncached dense参考。若目标C已完全为Prefix命中，则ProbeKV应跳过而不是声称再次加速它。Prefix长度/命中blocks在配对中一致。

`same backend+CFO selector` 不等于完整Cache-Craft系统，`matched patched loop` 不等于unmodified CacheBlend；图表必须准确命名。后续论文若做系统级SOTA对比，要分别完成官方实现适配与公平配置审计。

### 8.3 数据、统计和停止规则

- 只使用冻结development partitions；fit/validation按content group和文档来源隔离。相同C的变体和重复请求不能分到独立认证的两边。
- 最终validation只验证，失败不回头调同一份数据。
- 延续质量比较建议：整体平均F1下降不超过0.01、各数据集不超过0.02；case coverage/oracle使用预注册的case阈值。这是相对dense的质量合同，不等于dense答案是ground truth。
- r=1要求greedy token exact、至少32 teacher位置relative-L2<=1e-4、digest/mask/生命周期正确；QA失败与运行正确性失败分开。
- 用独立请求/content-group为重采样单位报告配对区间；重复timing、多个ratio和16个Source不是独立样本。
- 90个独立单位0失败的单侧95%Clopper–Pearson上界约3.27%，不能声称1%认证；后续统计认证另立合同。
- 不预设“必须2x成功”，也不因为Source差异不明显而筛掉失败case。

若oracle相对简单Source策略都没有稳定质量/成本优势，暂停更复杂的warm/promotion/concurrency机制，先重新审视多Variant收益假设。若oracle有优势而selector无收益，优先修选择；若selection有效但端到端不快，优先修overhead/I/O。不要把三种失败混为“kernel不够快”。

## 9. Coverage、存储和并发的后续验证

### 9.1 Coverage(K)是主增长指标

K={1,2,4,8,16}分别从相同初始条件、相同全局字节预算因果重放，禁止用最终K16池回看过去。

分别报告 compatible-proxy、selected、committed、QA-qualified且actual saving>0的Coverage；分母与Prefix-only/ineligible类别固定。报告总请求结果，不能仅对已成功commit请求求平均速度。

\[
\Delta Coverage(K_a\rightarrow K_b)=Coverage(K_b)-Coverage(K_a).
\]

真实受限LRU系统的曲线不保证单调；出现负边际收益必须保留，不通过平滑成饱和曲线掩盖。近饱和K是存储建议，不改变上限16。

### 9.2 存储价值单独测

分开比较GPU驻留、pinned CPU、SSD staged冷/暖页缓存。随机或轮换Source-tier映射，避免“最好Source恰好全在CPU”的混淆。

依次对比普通LRU、LRU+独立SelectionState、再加有界promotion/warm目标；记录未来request收益、维护bytes、staging时间、HBM/CPU占用、重复迁移、写放大和trace总时长。

CPU/SSD独占backing不代表OS page cache没有临时副本；冷热实验需记录实际IO与可控性，不能凭Python snapshot相同声称冷缓存相同。

### 9.3 并发最后开展

单Segment单请求闭环后，依次验证多Segment的union masks，再验证多个活跃请求。已commit路径不回滚，尚未决定的Segment按dense fallback纳入joint timeline。

多Segment统一ratio计划若不能同时满足各Segment质量floor和已有support上限，就不能静默降低某个Segment的质量要求；在commit前部分dense或采用原有受验证策略。A/C与legacy不因“很早开始复用”被删除，只有证据支持的简化dispatch才启用。

并发测试再加入排队、selection workspace、H2D、SSD staging、CPU pool、promotion竞争；分别报告SLO、TTFT、吞吐和公平性，不用高负载降低未经认证的repair比例。

## 10. 确认后的修改顺序与模块

### P0：统一合同和离线诊断接口，不切生产默认

- 更新独立policy版本/manifest，定义新ratio grid、proxy术语与两种目标，保留旧schema/Profile只读含义。
- 扩展 `v8_schema10_native_oracle.py`：多ratio、真实QA、teacher trace、matched backend、scope隔离。
- 完善数据group隔离、不可变task清单和原始证据校验。
- 新增 `SourceComparisonObservation`、`WinnerResidualCurve` 的边界测试。

### P1：将已优化后端接到live Source选择

- 修改 `cacheblend_continuation.py`、`v8_schema10_online_backend.py` 及native context适配：真实comparison→freeze→continuation，确保层只执行一次。
- 修改 `v8_schema10_selector.py`：隔离旧band policy与新qualified-cost policy，逐候选UNSUPPORTED处理；保留早停与legacy。
- 必须计入selection、projection、handoff、lease和FinalCommit，不能继续fixed-winner代替在线选择。
- 先resident零Prefix，随后native Prefix组合和CPU streaming分别过正确性门。

### P2：胜者repair曲线和最终准入

- 扩展 `v8_schema10_slack_repair.py`：真实quality-support验证、有限网格、效率/质量模式，不再接受只有任意digest的生产资格。
- 扩展 `v8_schema10_cost_provider.py`：mask改变重新查询，shape类别先验证再开放；actual/predicted/proxy分字段。
- 复用现有winner-specific repair和uniform/gradual模块；先fixed support，后no-reentry gradual。

### P3：独立SelectionState和受控存储维护

- 修改 `v8_schema10_storage.py` / pool：独立state预算与backing、请求间promotion queue、事务迁移、quiescent清理。
- 不改变exact canonical、LRU、grace、lease优先级；首次不做真正异步store mutation。
- 维护计费独立且可归因，失败不影响当前已完成请求的数值结果。

### P4：经单请求证据通过后再扩展

- 多Segment joint masks、并发reservation与load-aware cost cells。
- 真正异步promotion、stress/robustness与Qwen独立复验。
- 不为赶实验日期自动冻结未测Profile，不把缺表全部dense说成复用已通过。

### 无卡必测

1. rho_trim与repair positions类型隔离；一次排序得到的尾和/ceil/tie结果正确。
2. K/V metric不同不能复用阈值；N=1、NaN、空尾、r=1端点分流。
3. 未测ratio不可执行；更多ratio不默认quality合格。
4. 一个SSD候选缺成本，不连带否决已有合法CPU候选。
5. 新Source-cost目标与旧band策略结果可独立重放；freeze后失败不换Source。
6. early layers不重放；stale handoff、lease、snapshot失败不可commit。
7. 15% support缩小后不可重升30%；FinalCommit发生在selective前。
8. 总预算保留sunk；overlap不重复计费；错误dense分母不能通过。
9. SSD fullKV与CPU SelectionState可独立放置，字节预算真实。
10. background promotion不改Source、不刷新LRU、不传GPU非胜者；失败保留原backing。
11. Coverage无未来可见性、QA与timing完整、negative gain不被丢弃。
12. 旧配置/Profile/结果兼容；新模式不能借用旧资格。

运行compileall、全量unittest、合同validator、相关新旧配置和diff检查。所有本地测试只验证状态与接口，不认证GPU或QA。用户确认前不实施本节代码修改。

## 11. 有卡实验的阶段门与最终维护框架

本轮不租/连GPU。有卡前重新确认实例、价格、预算和任务清单；不沿用过期硬件/费用记录。

顺序：

1. Mistral同后端单Segmentcorrectness与真实Source矩阵，确认Source价值上限。
2. live selection接入continuation，对比Latest/Random/CFO/ProbeKV；同residency先排除I/O混淆。
3. winner repair网格与质量验证、实际效率/质量两模式、matched Prefix组合。
4. CPU逐层overlap，再SSD冷/暖及存储维护；先无promotion再有promotion。
5. Qwen独立tokenizer/Source/Profile重复关键结论。
6. 单Segment通过后多Segment，再并发。每步新失败保留证据，用新SHA与输出目录复验。

长期H1–H5不再反复换主问题：

| 阶段 | 固定科学问题 |
|---|---|
| H1 | 为什么需要多个历史Variant：Source-quality/repair差异、Coverage(K)、增长成本 |
| H2 | 无训练current-state selector能否以可接受开销接近质量合格Oracle |
| H3 | winner-specific repair、有限ratio选择与load/compute overlap能否改善质量—延迟曲线 |
| H4 | CPU/SSD、LRU、promotion及并发资源是否维持净收益和可控成本 |
| H5 | 全部冻结后的locked端到端验证，包括fallback、构建和维护成本 |

论文主结论至少同时需要：真实Source价值、可实现selector捕获价值、匹配质量下端到端净收益；只证明某个Source的J更小、某个copy被隐藏、或某个case速度更快均不足。

本轮结束状态：

```text
unified_plan_status = no_gpu_candidates_implemented_pending_real_evidence
runtime_code_modified_this_turn = true
gpu_started_this_turn = false
formal_profile_bundle_frozen = false
gpu_runtime_qualified = false
h1_h2_execution_allowed = false
paper_evidence = false
locked_test_accessed = false
```

## 12. 外部论文核查及采用边界

已解析4项相关工作，均找到原始发布页面；这是来源/论点核查，不是复现它们的性能。学术检索连接不可用、OpenAlex备用请求SSL失败后，使用官方arXiv/USENIX页面核对，不依据聊天中的二手数字冻结系统参数。

- [CacheBlend](https://arxiv.org/abs/2405.16444)：报告其测试条件下通过选择性重算与KV retrieval流水降低TTFT。可借鉴后端机制，不能把其倍数作为ProbeKV必须达到的跨配置目标，也不能推出任何15%修复都与dense相同质量。
- [ProphetKV](https://arxiv.org/abs/2602.02579)：提出query-irrelevant token挤占修复预算的问题。支持加入后置query敏感性测试；不据此在本阶段复制新的query-aware修复算法。
- [CacheSlide，USENIX FAST 2026](https://www.usenix.org/conference/fast26/presentation/liu-yang)：面向agent的相对位置复用并包含learned correction。与本方案的historical Source选择不是同一问题，不以其倍数推算本项目收益。
- [DAF](https://arxiv.org/abs/2607.21599)：页面注明EuroSys 2026 poster track；作为质量/后端设计的相关工作线索，不称为已经独立验证的完整系统基准结论。

建议今后维护这一份框架的版本与证据附录，而不是每次讨论都新增一组未经验证的在线机制。

## 13. 用户七项补充的收敛与无卡实施记录（2026-09-10）

本节更新前文细节，作为下一轮开发的直接依据；旧生产dispatch和已归档实验仍按原合同解释。

### 13.1 Source先使用固定参考比例，不根据胜者事后改分数

本小节保留为已实现的固定参考比例基线；第14节另增联合 Source×ratio 候选。旧的“只保留两个首轮参考候选”不限制新候选的九点曲线，但旧 Profile 仍只支持其原网格。

用户正确指出：J本来就是某个trim ratio下的Rremain，不能先含糊地说“选最好Source”，然后才定义如何评价Source。

明确顺序：

```text
先固定 rho_ref（本轮默认0.15）
→ 每个Source计算J_s(rho_ref)
→ 按已有质量代理/成本规则选Source
→ freeze
→ 胜者使用实际repair metric确定执行计划
```

只保留两个首轮参考候选：0.05与0.15。0.05是“更少trim、更重视大漂移”的对照，不因为它是最小执行比例就天然最好。0.15是与既有基线衔接的默认，不是数学最优常数。

排序可能交叉。例如20个token的Source A有2个极大漂移、其余接近0；Source B各token都有中等漂移。trim5%和trim15%可以得出相反Source排序，因此必须用同请求QA/repair成本判断哪种排序更有用，不能仅比较两种不同trim下J的绝对数值。

Profile选择时每个模型确定一个参考rho，跨数据集固定；阈值与rho、completed depth、metric、长度支持范围绑定。不能依据本请求最后用了哪个repair ratio，再回头改Source评分。更不能从“运行中最常见ratio”自动在线更新rho，让实验策略随时间漂移。

无卡新增 `residual_tail_curve`：一次排序和一次尾和扫描输出5–30%的score曲线，既可复用15%主分数，也可用于离线5%对照；不返回可直接进入runtime的repair mask。

### 13.2 语义切分采用确定性预处理，不做按当前耗时切换

新候选规则：

- 基准T={512,1024,2048}，窗口W={50,100}，单位均为模型token。
- 在[T-W,T+W]找段落末；不存在再找句末；不存在再找逗号/分号等分句末。
- 同类边界优先接近目标，soft block alignment只作更低优先级的选择依据。
- tokenizer使用整篇文本的offset mapping，不能用字符数代替token数，也不能逐prefix反复decode/tokenize。
- 语义模式要求文本重新编码得到的IDs与原manifest严格相同；不一致则拒绝，不能偷偷换token身份。
- 切分结果、规则版本、tokenizer、document revision和offset摘要进入manifest；重复请求直接使用同一规范边界。

有限窗口内可能根本没有标点，逗号本身也不等于整句结束。因此“固定长度上限+永不切长句+固定token回退”不能同时保证。本版选择有界长度并显式记录`forced_token_cut`，不声称绝对语义完整。若后续选择严格整句方案，超长Segment必须使用独立长度支持或dense路径，不能悄悄外推Profile。

TTFT优先的实现方式是**提前切好并缓存**，而不是每次先试语义切分、发现慢了再重切固定块。如果要比较固定切分，它使用独立`fixed_tokens_v1` provenance，并在数据准备/运行配置冻结时选定。昂贵的语义处理不进入普通请求关键路径。

首先只改retrieved document之后的reuse segmentation，保持检索文档和顺序不变；若以后连retrieval embedding的知识单位也一起改，须单独实验，不能把检索改善算成Source选择改善。

这里没有大模型训练。512/1024/2048可作为runtime测量anchors，但**不够认证浮动长度的QA和成本**。必须加入真实非整档长度、不同边界类别的held-out cases；未被支持的长度/shape继续UNSUPPORTED，不虚构插值。

实现已接入原有离线入口 `canonicalize_v8_profile_cases.py` 的显式`--chunker-policy`选项，默认仍为legacy。输出新增逐case切分audit；未改线上canonical边界。

### 13.3 d1保留50%、d2精筛：候选优化，不是默认保证

固定解释：d1=完成第1个Transformer block后观察第2层K；不是只有embedding信息的d0。

```text
d1比较当前可比较cohort
→ 若原策略已合法早停则不必继续
→ 否则按相同rho_ref保留低漂移的50%
→ 至少保留2个（真正唯一候选例外）
→ 截止位置并列或numeric slack内候选全部保留
→ d2仅比较保留cohort
→ 质量、成本、freeze规则照常执行
```

50%只表示名义筛选比例。d1全部接近时应全保留，不能任意删除8个。初始16个Source时，16+8对比16+16是总比较数量约减少25%，并不是整个selection/TTFT减少50%；真实时间还受transfer/microbatch和d1开销影响。

两阶段筛选不冻结Source、不加载完整KV。绑定request/segment、Source inventory和rho；请求、pool inventory或rho变化使计划失效。d2不能用另一批未声明候选冒充保留集合。

被d1淘汰的Source可能是d2真正winner，因此先shadow：额外计算完整d2 cohort用于审计，不参与线上选择、不回写winner。统计winner retention、absolute regret、matched-quality cost regret与总selection时间。若误淘汰损失大，保留全候选或退回legacy，不预设50%必须晋升。

原始correctness_eligible_k永远不因pruning变小。16→8后即使8个全失败，也不能声称完整池无兼容Source；只能依既有规则dense/有限exploration。CFO截断与d1截断分别记录。

`plan_depth2_shortlist`、绑定校验与`audit_depth2_pruning`已实现并有CPU测试；**尚未启用到生产live dispatcher**。新task spec明确`live_dispatch_integration_complete=false`。

### 13.4 CacheBlend对齐：不是从V-only改成K-only

论文核查结果：1篇原始论文已确认；“15%普遍最优”这一解读不成立；固定实现metric已用本地patch核查。

[CacheBlend v3 §5.1](https://arxiv.org/html/2405.16444v3#S5.SS1)将15%描述为其实验中质量下降很小的经验最低参考，同时讨论根据加载条件提高比例。这不是所有模型、数据或Source上15%最优的证明。

项目固定 `patches/cacheblend/0002-probekv-segment-repair-mask.patch` 的原实现上下文和本地固定xformers路径均使用：

\[
e_j^V=\sum_h\sum_k(V_{j,h,k}^{current}-V_{j,h,k}^{source})^2.
\]

因此当前匹配后端的V-only不是偏离CacheBlend；不能凭“论文说KV deviation”就直接改成K-only。

真正需要隔离的是：`winner_repair_drifts`以及部分历史aux measurement使用**归一化V差异**，这与固定patch的**未归一化V平方差**排名并不总相同。输入dtype的减法、平方、reduction行为也属于版本合同，不能忽略BF16/FP32差异。

新增 `cacheblend_pinned_value_scores` 精确镜像固定算子并测试其与normalized V的排名反例。新实验spec使用`value_squared_l2_pinned_dtype`；旧normalized metric保持原名和历史含义。未修改历史补丁，不将本地算子测试称为CUDA后端资格。GPU阶段要求同Source、同V输入、同绝对mask的matched算子/执行检查。

### 13.5 两种模式以效率优先为主

新development spec显式设置`default_execution_objective=efficiency_first`。

现有slack proposal新增两目标参数：效率模式最小化有支持的完整预测成本，完全相同成本时选更高ratio；质量模式最大化gamma预算内的合格ratio。历史shadow API默认保留原质量模式以免重释旧调用，新入口必须显式传效率模式。

两者仍仅为proposal，不因一个quality digest就获得生产资格；FinalCommit、真实质量支持与新鲜snapshot不能省略。新增网格须显式传入，旧Profile不自动获得5%、25%的支持。

### 13.6 本次已完成与未完成

已完成：

- Source参考比例的一次排序曲线、5%/15%排序反例测试。
- 两层50%候选计划、并列保护、min-K、stale binding、full-d2 shadow误淘汰审计。
- token-offset语义切分、句号/缩写/小数/中文标点处理、固定回退审计、旧manifest入口接入。
- pinned V算子对齐辅助函数、与normalized V排名差异测试。
- 两目标slack proposal、显式新ratio grid、成本缺失和无资格声明保护。
- 分解式候选spec生成器：4个reference/cascade组合，9个chunker组合；不做它们与所有Source/ratio/长度的完整笛卡尔积。

首轮无卡验收：当前全量测试为817项，1项历史可选依赖跳过；compileall、合同validator、22套本地配置和diff检查通过。生成 `artifacts/source-policy-development-20260910/candidate-spec.json`，其GPU授权保持false，模型/质量/成本Profile保持null。无GPU运行。后续观测重放入口的验证见13.8。

尚未完成，不得声称已上线：

- 新候选的真实模型/GPU continuation实测（两阶段cohort和独立native capture入口已完成CPU接口测试）。
- 新质量threshold与Source Oracle的真实测量。
- 非整档长度实际kernel/cost验证。
- Prefix+新路径、CPU streaming、SSD/promotion与并发资格。
- 候选级缺成本处理已实现并测试；GPU阶段仍需验证实测成本表命中与UNSUPPORTED分支。

下一步仍无卡：核验真实模型上下文的失败清理覆盖，补齐tokenizer资产与既有development partition的本地审计，冻结代码、patch和任务清单。独立native diagnostic入口已存在，但CPU接口测试不等于真实CUDA行为已验证。

### 13.7 GPU实验顺序更新

1. 同后端、单Segment、固定15%真实Source矩阵与QA，先证明选择价值。
2. 采集完整 Source×九点 repair 的开发矩阵，对比固定参考比例与第14节联合选择；拟合与验证按content group隔离，不在验证集反复改threshold/reference。
3. 在相同已固定的评分/质量规则下比较全量、d1剪枝、CFO+d1剪枝；先用完整观测审计损失，再实际运行剪枝路径计时。随后验证效率/质量模式和负载余量，不能拿shadow时长冒充生产时长。
4. 固定块与语义块比较，保持retrieval不变；包括真实浮动长度、forced-cut比例、预处理开销和Source命中稳定性。
5. Prefix组合与CPU逐层overlap；随后Qwen独立复验。
6. 单Segment结论成立后再多Segment、SSD维护和并发。

无效或负结果保留；根据实验晋升策略，而不是因为方案已写入文档就强制启用。未确认有效前保留fixed15、全候选d1/d2和legacy回退路径。

### 13.8 完整d1/d2观测与离线四策略重放（无卡续轮）

新增 `source_policy_replay.py` 和 `replay_source_policy_observations.py`，实现以下闭环：

```text
独立comparison K tensors
→ observe_depth_k：BF16输入、FP32归一化K drift、逐token原始值
→ build_observation：绑定request/segment/model/tokenizer/code/patch/config/partition
→ 完整d1及同cohort d2文件 + 文件SHA
→ 离线重放：rho={5%,15%} × d2 keep={100%,50%}
→ 排名、原始scope、winner retention、absolute residual regret
```

观测文件绑定绝对位置、K geometry、每层current K digest、各Source Selection K digest及完整correctness-eligible ID清单。深度固定为completed d1/d2，分别观察第2/3层K。只有d2覆盖整个原d1 cohort时才能计算剪枝审计；不能只给幸存候选的数据。

这里的“完整cohort”不等于“完整存储池”。若CFO先将16个缩为4个，观测的两层都必须覆盖这4个，但原始eligible仍为16，报告不能声称完整池无兼容Source。

安全与解释约束：

- d1剪枝不等于Source freeze；离线输出的`ranked_source_id`只是排名候选，未通过真实质量/成本准入。
- 真正唯一候选与预算仅够比较一个严格分开。后者`ranked_source_id=null`、margin为null，记录`INSUFFICIENT_RANKING_COVERAGE`。
- d1并列保护可能保留全部16个；第16个在d2变成最优但被剪掉时，审计必须报告失败，不补回它来伪造线上成功。
- 所有`qa_passed`、`matched_quality_ttft_ms`及剪枝执行耗时在本入口保持null。完整cohort shadow的运行时间不能当作半候选路径实测时间，更不能按候选数线性折算。
- 当前shadow固定为dense轨迹。它不能代表前面Segment已复用后的policy-conditioned状态；多Segment须另行采集。
- 小型Selection K的诊断hash/host extraction必须计入诊断工作，不能接到正式在线TTFT路径再隐藏开销。本接口不读取完整Artifact、不申请winner transfer、不改变Source池或lease。
- 输入文件须提供实际SHA；内部observation digest、depth/geometry/cohort/provenance再次校验。重复JSON key、坏摘要、缺层、缺Source、非有限值均拒绝。输出存在时拒绝覆盖。
- `native_hook_diagnostic`只是来源标签，不是CUDA正确性证明；即使带该标签，GPU qualification、生产准入和paper evidence仍为false。

入口用法（已存在的真实/CPU测试观测文件，不生成伪实测）：

```text
python scripts/replay_source_policy_observations.py
  --input <完整观测文件>
  --input-sha256 <该文件真实SHA256>
  --output <新的输出文件>
```

新增测试覆盖第16个Source成为winner、d1平局、剪枝损失、两类K=1、partial scope、输入篡改、off-by-one、geometry、非有限值、CLI保留旧输出，以及native capture计划和V-only指标。全量测试817项：816通过、1项历史可选依赖跳过；compileall、合同validator、22套本地配置和diff检查通过。

### 13.9 本地 CUDA 原语验证边界

新增 `scripts/run_local_cuda_primitives.py`，仅用于验证 Residual-K 批量/分批比较、固定 V-only 控制、pinned CPU→GPU 分层传输和 qualification digest 生命周期。该脚本不加载论文模型、不产生 QA 结果、不访问 locked test，输出中的 `gpu_runtime_qualified`、`paper_evidence` 始终为 false。当前系统 Python 的 Torch 2.4.1 不支持本机 RTX 5070 Ti 的 sm_120；隔离 CUDA 12.8 wheel 下载未形成可安装包，因此本地 CUDA 执行保持 pending，不能替代 A800 哨兵。

**代码与实测分开记录**：`run_source_policy_capture.py`已接通canonical构建、实际池发布、原生request接口、d1/d2观测和重放；尚缺本版本真实GPU资源释放、QA及时间关联证据。rho=15%的live cascade和原始V指标为显式opt-in；rho=5%仍要求新的Profile合同。既有生产默认、旧Profile、fixed15与legacy均未切换；本轮未连接服务器或启动GPU。

### 13.10 租卡前输入与代码身份加固（2026-09-11）

- Capture dry-run调用生产native attachment校验，模型资产/配置错误必须在加载模型前拒绝，而不是仅校验JSON摘要。
- 单Segment历史请求严格按epoch递增，同exact-content cohort必须归属同一content group；禁止注入teacher、强制repair或诊断绕过字段。
- Handoff将未跟踪的源码、配置、测试、补丁和文档计入dirty检查；历史Git bundle不计入。源文件摘要覆盖整个`src/probekv`，不只覆盖`v8_schema10_*.py`。
- 本轮825项测试：824通过、1项可选依赖跳过。无真实CUDA运行。
- 本地搜索未找到可直接使用的tokenizer文件或冻结development partition原始文件；历史model audit不等于这些资产本身。此项仍须解决，不能生成占位hash。
- 本机Torch wheel续传确认尚缺约558 MiB，下载45秒后超时并保留成功字节；本地CUDA原语仍pending，不把下载失败作为系统GPU正确性失败。

## 14. Source×repair 联合选择与比较链简化（2026-09-11，待实施）

本节将用户本轮建议纳入下一阶段方案。**当前交付是方案修订，不是运行时代码修改或新算法资格声明。**复用现有 identity、Pool、continuation、cost provider 和 FinalCommit，不另起一套大框架，不增加 schema。新增策略使用显式 policy/manifest 版本，保留旧配置、固定参考比例与 legacy 的原解释。

目标是假设并验证：更多历史 Source 能否提供更低的最低合格 repair ratio，且这部分节省足以覆盖比较、加载和维护成本。不能预先写成“多 Source 必然优于 CacheBlend、质量更高且 TTFT 更低”。

### 14.1 CFO 与 d1/d2：简单全量为基线，有净收益才剪枝

K 指原始 correctness-eligible Source 数，而不是剪枝后的数量。计划默认：

- K≤8：全量比较，不使用 CFO 或固定50%淘汰；已经满足合法早停规则时可以停止后续 checkpoint。
- 8<K≤16：仍以全量比较为基线，下面三条路径分别预注册并测量。
- 全量不等于无条件分配显存；SelectionState 能放下则 one-shot vectorized compare，否则使用同一 allocator 的有界 microbatch，不为比较加载完整 KV。

| 路径 | d1 输入 | d2 输入 | 用途 |
|---|---|---|---|
| `full_compare` | 全部 eligible 且 state 有效的候选 | 全部 d1 候选 | 默认和正确性/质量参考 |
| `d1_half` | 同上 | d1 排名较好的约50% | 只测试 d1 剪枝是否有收益 |
| `cfo_half_d1_half` | CFO 排名前约50% | 该 cohort 中 d1 排名较好的约50% | 更强的有损剪枝消融 |

第三条在 K=16 时名义上为16→8→4，而非“先比较一半、d2再比较另一半”。用户所说的 d2“剩下的”明确指**保留下来的优先候选**。比例用 ceil，至少保留2个；并列及冻结 numeric slack 范围内保留所有候选，因此实际数量可多于50%。真正仅有1个 correctness-eligible Source 仍走独立 single-candidate 路径；截断后只剩1个不得触发 margin early-exit。

首轮 d1 剪枝按冻结的参考 K 分数排序，作为有损启发式，不称为联合成本最优剪枝。不得仅因候选在5%处不合格，就将可能在15%或30%处合格的候选认定为全比例失败。用完整 d2×ratio 观测衡量误淘汰了多少合格/更快计划；未通过验证就保留全量比较，不继续堆新的排序器。

QCFuse 风格的 anchor/critical-state 只作为另一条独立的降成本候选：`anchor → d1 → d2`。本方案不把 `CFO → anchor → d1 → d2` 叠加为默认链；同一实验单元最多启用一种 shortlist 优化，另一路作为配对消融。否则候选误删、元数据读取和 anchor 提取开销无法归因。若全量比较与任一剪枝链的实际请求 TTFT 差异不超过预注册测量误差，默认采用全量比较并退役剪枝路径。

启用条件不是“少比了几个 Source”，而是同请求、同质量条件下有可重复的净时间收益：

```text
CFO metadata读取/排序 + d1 + 剪枝 + d2 + state搬运/同步
与 full_compare 对比
→ selection wall-clock、请求TTFT、QA、winner/feasible-plan recall、cost regret
```

CFO full-prefill metadata 生成和存储费用另计入 trace 总成本/摊销，不能隐藏。anchor 提取、Summary H2D、排序、同步、候选丢失和缓存命中也全部记账。全量 shadow 捕获不能用于声称剪枝后的真实时间；必须另跑真正减少传输/比较的执行路径。采用预注册重复数与配对区间；差异不足以超出测量不确定性时选 `full_compare`。CFO 不再是主路径必须执行的步骤；保留为可关闭消融和历史结果读取能力，通过实测后才可能启用。

当前借鉴状态必须单独记录：

| 借鉴方向 | 当前仓库状态 | 下一步边界 |
|---|---|---|
| SparseX-style sparse-Q / non-reuse query 选择 | **未接入 SparseX runtime**；现有 `d1_half` 只是基于 SelectionState 的候选剪枝，不能称作 SparseX | 仅在单 Segment full-compare 基线成立后，作为独立 repair/探测消融；不改变 canonical KV 或主 selector |
| QCFuse-style anchor / critical-layer probe | **已写入本节候选方案，尚无生产实现** | 与 CFO、d1 剪枝互斥；只测 anchor 提取、状态搬运和选择收益，不能与其他剪枝叠加后归因 |
| Cache-Craft CFO | 已有历史 schema/诊断代码和测试，但在本轮新主路径中不是必选 | 只有 end-to-end 实测低于全量比较才启用，否则保留为 baseline |

因此，当前不能在论文或 GPU handoff 中写成“已采用 SparseX/QCFuse”。只能写“受 SparseX/QCFuse 启发的候选路径已登记，待独立实测”。

### 14.2 一次 K 漂移、一次排序、九点 Source 曲线

新候选网格统一为：

\[
\mathcal R=\{0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30\}.
\]

每个 Source、每个实际执行 checkpoint 只计算一次 per-token normalized pre-RoPE K drift。按 drift 从小到大排序得到 a_(1)…a_(N)，以绝对 token position 打破平局，前缀和 P(t)=sum_(j≤t) a_(j)。候选质量代理定义为：

\[
m_{score}(r)=\min(N-1,\lceil rN\rceil),\qquad
R_s^{(d)}(r)=\frac{P_s^{(d)}(N-m_{score}(r))}{N-m_{score}(r)}.
\]

这里明确采用 `proxy_trim_ratio(r)=r` 的新候选映射；实际 repair 仍用 winner-specific 固定后端 metric。此映射必须以真实 QA 验证，不能自动继承固定 rho=0.15 的阈值。

单层漂移/排序约 O(KN log N)，排序后取九点约 O(K|R|)。这是复用统计，不是九次模型 forward，也不是已经免费得到九条真实 repair 执行结果；GPU排序、reduction、读取标量和同步仍计时间。d1 与 d2 状态不同，不能声称一次 d1 漂移也免除了 d2 计算。

“比例包含”指剔除集合嵌套：Top5%⊆Top7.5%⊆…⊆Top30%；准备复用的集合方向相反。不同 Source 的曲线可能交叉，5%或15%排名不一定等于其他比例排名。

必须区分两个对象：

- `SourceResidualCurve`：所有被比较 Source 的 K 质量代理曲线，不返回 runtime repair mask。
- `WinnerRepairPlan`：只在胜者上，用固定 CacheBlend patch 的实际 V drift/算子生成修复位置和后续 support。

K 剔除的 token 与 V-repair 的 token 可能不同，因此 K tail **不是实际 V-repair 后的残余误差**。阈值需绑定这种“由K预测V-repair质量”的具体组合。先验证其代理有效性；若不成立，回到 fixed15 基线，不能悄悄读取全部候选 V/full KV 来掩盖失败。r=0/1 只作独立端点，N=1、不支持长度和非有限值保持明确 dense/失败分流。

### 14.2a CacheBlend 对齐的 KV deviation 修复指标

当前运行时仍有 V-only 历史路径，是因为固定补丁的已验证算子只计算 V 差异；这解释了“代码为什么还在 V-only”，但不是把它当作论文最终方法的理由。下一轮新增 `kv_deviation` 候选，使用同一 winner、同一绝对位置和同一 repair-check 状态，同时观测 K 与 V：

\[
\delta^K_j=\frac{\lVert K^{cur}_j-K^{src}_j\rVert_2}{\max(\lVert K^{cur}_j\rVert_2,\epsilon)},\qquad
\delta^V_j=\frac{\lVert V^{cur}_j-V^{src}_j\rVert_2}{\max(\lVert V^{cur}_j\rVert_2,\epsilon)},
\]

\[
\delta^{KV}_j=\sqrt{(\delta^K_j)^2+(\delta^V_j)^2}.
\]

如果固定 CacheBlend patch 要求未归一化平方差，则另外记录其 `raw_operator_deviation`；不能用归一化分数冒充 patch 的实际 Top-K。主候选是否采用归一化 KV、raw KV 或保留 V-only，必须由同一模型、同一 Source、同一 mask 的 QA 与端到端 TTFT 配对结果决定。

实际执行链为：

```text
winner freeze
→ repair-check 层完整得到 current K/V
→ 按冻结 KV deviation 生成 repair candidates
→ 运行真实 repair + 后续 decoder
→ 检查 token/logit/QA 与 matched dense
```

K/V deviation 只用于胜者的 repair 候选，不能让16个候选 Source 都加载 V。K SelectionState 仍是 Source 选择的轻量数据面。若 KV 候选增加了 visible compute、破坏 load/compute overlap或质量无改善，则保留 V-only 为兼容基线；若它在固定质量门下减少 TTFT，才可晋升为新主策略。所有结论均须标注 metric，不能把不同 metric 的结果合并成单一“CacheBlend repair”。

### 14.3 每比例质量阈值：离线固定，线上查表

阈值记为：

\[
\theta_{model,d,r,metric,repair\_policy,length\_support}.
\]

`metric` 同时描述 K proxy 与实际 repair metric，`repair_policy` 覆盖 boundary/support schedule。为每个 Source 建立：

\[
\mathcal Q_s=\{r\in\mathcal R:
R_s^{(d)}(r)\le\theta_{model,d,r,\ldots}
\land quality\_support(s,r,d,\ldots)\}.
\]

`quality_support` 是冻结 Profile 支持的输入范围/策略证据，不是当前请求已知的答案标签。线上先查询已冻结规则，离线才执行真实 QA 标注。

- 各点可以有不同阈值，不是把“5%”直接当成一个通用误差百分比。阈值不得根据当前请求临时放宽。
- 可以一次计算全部曲线，再从低到高查找最小合格点；无需逐点重跑模型。
- 5%失败不能推出7.5%及更高比例失败；30%失败也不能在未验证单调资格规则时反推所有低点失败。必须检查全部受支持点。
- drift tail 本身单调不增，不代表 QA 或跨点阈值合格性单调；每个实际采用的点都必须有质量支持。
- 缺失阈值/长度/repair-policy 支持为 `UNSUPPORTED`，不能记作“Source很差”，更不能由此宣称池 mismatch。

开发时先采集完整 Source×ratio 的真实结果，按 content group 隔离 fit/validation。fit 上冻结表与策略，validation 只验证。d1/d2 和 legacy 各自可执行深度必须有对应记录。此处仍是经验质量规则，不重新引入 learned r_safe、不声称确定性的 QA 保证或已完成1%风险认证。

### 14.4 联合决定 Source 与最低合格 repair，而非硬选最热或最小 J

每个 Source 的最小质量代理合格比例：

\[
r_{min,proxy}(s)=\min\mathcal Q_s.
\]

仅在集合非空时定义这个最小值；空集合记录为null及对应原因，不能默认成0%或30%。这是候选起点，不是“整条请求一定最快”的结论。在有合法质量、执行形状成本和资源支持的有限组合上选择：

\[
\mathcal F=\{(s,r):s\in\mathcal S_{compared},\ r\in\mathcal Q_s,
cost\_supported(s,r),\ resource\_preview\_feasible(s,r)\},
\]

\[
(s^*,r_{plan})=\arg\min_{(s,r)\in\mathcal F}
\left[T_{elapsed,q}^{actual}+\widehat F_{joint,q}(s,r)\right].
\]

最多16×9个轻量计划查询，不是16×9次在线 full-KV执行。复用共享 shape查询、一次排名和简洁查表；所有成本查询必须受支持。多Segment阶段不枚举各Segment Source组合的笛卡尔积，继续使用请求级 joint planner，本轮先只做单Segment。

可行集合为空时不计算argmin：未到冻结最大选择深度且策略允许时继续probe；选择已结束则dense。early-exit的合法性也必须针对新联合目标重新验证，不继承仅按固定15% residual margin得到的资格。生产冻结前仍使用已验证基线，不把更复杂候选自动设置为默认。

效率模式在成本可区分时选最快计划；仅在**预先冻结的测量误差容忍带**内允许优先更高合格比例，之后按稳定Source ID等规则打破平局。不能用很宽的“差不多”掩盖真实TTFT损失。质量模式仍为独立策略：在最终gamma预算内优先更高有支持比例，并报告其代价。

GPU/CPU/SSD 不作为硬优先级：

- GPU/CPU Source 在质量合格且预计总成本更低时胜出，即使其 J 不是最小。
- SSD Source 的较低 repair 成本或更好质量如果足以抵消实际加载，可以胜出。
- 只比较同一冻结 metric、比例和深度的 residual；不能把不同 r 的原始 J 大小直接当统一效用。
- 所有比较仍只读取小型 SelectionState；full KV 仅在最终 winner freeze、lease和资源准入后准备。

Source身份只freeze一次。winner实际repair-check/资源/FinalCommit失败时dense，保留所选Source审计；不得偷偷切成第二名。Gate1使用候选source-local futility成本，gamma=1.0；不新增每个ratio一道runtime barrier。FinalCommit仍在首个不可逆reuse层之前验证完整请求gamma=0.8，不允许后置到已发生selective执行之后。

### 14.5 I/O余量与repair：纠正不等式，优先完整TTFT

对确实能流水重叠的层，简化诊断模型是：

\[
\widehat T_l(r)=
\max(\widehat T_{aggregate-load,l+1},\widehat T_{union-repair,l}(r))
+\widehat T_{nonoverlap,l}.
\]

| 情况 | 正确动作 |
|---|---|
| load=8ms，最低合格repair=5ms | 若其他合格比例repair≤8ms且完整joint未来不变，可增加repair利用I/O余量 |
| load=3ms，最低合格repair=8ms | 已计算受限；提高repair通常更慢，优先较低合格比例/更经济Source，必要时dense |
| GPU resident，load接近0 | 没有可白用的load bubble；降低合格repair可能更有价值，仍须计attention/mandatory rows/setup |

因此用户设想中的“计算比加载慢时再增加repair”不能作为规则。**当加载比计算慢，才可能多修复而不增加预计关键路径。**降低repair在compute-bound场景最有潜在收益；多Source在load-bound场景仍可能通过减少加载、改善质量或命中率有价值，不作“一定没有价值”的断言。

不用最大化 overlap 百分比代替最小化TTFT。故意增加计算可能提高重叠率却不产生任何提速；首层裸露加载、末层收尾、staging/launch、完整attention及mandatory suffix均纳入joint timeline。用实测overlap与端到端配对时间验证简化模型，不把预测gamma通过写成实际20%提速保证。

当前无reentry：首个commit前可扩大初始support；后续只能 M_(l+1)⊆M_l。已缩到10%的状态不能因为后层I/O慢再凭空升到20%。先验证fixed-support九点路径，再加入经过完整schedule质量验证的gradual，quality floor与各点支持不得跳过。

### 14.6 Dense物化与SSD晋升，服务决策与维护分开

全部**已比较、已支持**Source×ratio都不合格时允许dense；但只有完整可判定存储池证据，才能写 `no_compatible_stored_variant_proven=true`，且证明范围仅是冻结proxy规则下的当前池，不是全局语义新颖性。CFO/d1剪枝、state缺失、阈值缺失或仅因成本失败都不构成完整 mismatch。

dense后可沿用已有 `CONTENT_MISS / COMPLETE_SCOPE_ABSOLUTE_MISMATCH / BUDGET_TRUNCATED_EXPLORATION` 等原因和配额决定物化。只允许实际无缓存exact dense full-prefill创建canonical；Prefix+dense remainder、selective-derived及r=1 reuse结果不直接入池。预算/容量/保护不允许时跳过，不能每次dense都无条件追加。当前请求不可见自己新建Variant。

选定SSD winner后可按实际路径读取到pinned CPU/staging再GPU；持久晋升只有在CPU容量、lease与LRU规则允许且目标写入校验完成后才切换exclusive backing。不等于永久存三份，也不等于当前所有SSD候选都晋升。

未胜出SSD Source若有未来请求价值，仅进入独立有预算的请求间maintenance候选；不能在当前selection关键路径偷偷搬完整KV。promotion收益与写放大单独对照，迁移不刷新last_request_use_epoch。首次验证关闭主动非胜者promotion，先验证当前请求选择本身的收益。

### 14.7 简化清单与不可删除边界

| 机制 | 下一轮处理 |
|---|---|
| 强制CFO、强制两层50%剪枝 | 去掉必选地位；仅在实际净收益和质量验证通过后启用 |
| 为每个ratio重复计算K drift/完整排序 | 合并为一次漂移、排序与累计和；按执行深度分别计算 |
| 固定参考ratio唯一决定Source | 保留为基线，增加联合Source×ratio候选，不先验指定5%/15%最优 |
| 在线学习r_safe、每层自由调阈值 | 不引入；冻结有限表、已支持范围与显式fallback |
| 为比较加载多个Source完整KV | 禁止；SelectionState与winner repair数据面分离 |
| 多套重复成本口径、重复Source freeze | 复用一份joint estimator及一次身份冻结；Gate1与FinalCommit职责不混用 |
| 5% selection硬时间门槛 | 不恢复为主路径；旧预算模式仅保留消融，成本计入完整TTFT |
| Sparse-Q/anchor等新repair或探测分支 | 暂不并入必选链；先验证现有K选择+匹配V repair，再决定独立消融价值 |

不可删：exact token/model identity、canonical来源约束、Prefix排除、Source freeze、物理/逻辑lease、HBM/staging上限、不可变digest合同、请求级FinalCommit、QA和原始事件、成本缺失dense以及历史legacy/fixed15回退。主路径精简不等于删除并发/内存正确性，也不依据“通过率高”提前移除安全检查。

### 14.8 下一轮无卡模块与必测项

以下为待实施变更，不计入第13节“已完成”清单：

| 模块 | 下一轮改动 |
|---|---|
| `source_policy_development.py` | 九点规范曲线/整数计数；三种比较策略spec；K≤8全量策略 |
| `source_policy_replay.py` | 全候选Source×ratio重放、各级剪枝scope与feasible-plan regret，保留旧四策略读取 |
| `v8_schema10_selector.py` / contracts | joint proposal与缺阈值/缺成本reason；显式区分score trim与runtime repair |
| `v8_schema10_slack_repair.py` | 胜者实际metric、九点quality support、效率/质量及no-reentry一致性 |
| `v8_schema10_cost_provider.py` / online backend | 批量shape查询、完整sunk/joint future、candidate独立UNSUPPORTED；不增加新的经济Gate |
| `v8_schema10_storage.py` / Pool | 复用原事务、exclusive backing与LRU；promotion独立开关/费用，不新造淘汰分数 |
| capture/Oracle/manifest/handoff入口 | 全候选观测与实际执行计时分离；九点真实QA任务；fit/validation与代码/数据/patch绑定 |

新增无卡测试：

1. 九点曲线等于逐点参考，ceil/并列正确；相同repair_count可去重计算但不合并Profile身份。
2. K曲线与V-repair token集合不同的反例，不允许score索引串入mask。
3. 5%失败但15%通过、比例资格不单调、曲线交叉与缺阈值分别处理。
4. K≤8全量；K16三条cohort路径、并列保留、截断K1与partial mismatch正确。
5. Source16可成为联合winner；较差J的热Source或较低repair的SSD Source均可在正确成本下胜出。
6. 缺某个候选/ratio成本不连带否定其他合法计划，不填0或外推。
7. load-bound可利用余量，compute-bound不通过增加repair伪造提速；成本平局规则固定。
8. freeze后不换Source、FinalCommit拒绝仍保留选择审计、support不可重入。
9. dense物化遵守canonical与预算；partial pool不声称完整mismatch；迁移不刷新LRU。
10. shadow不计作实际CFO剪枝计时；旧Profile不能自动授权新增比例与联合策略。

先完成这些CPU接口/状态机、manifest和回归检查，再冻结租卡前SHA与任务清单。**本地GPU实验不是必需前置**；真实模型CUDA、时间和质量验证留给A800，不以本地驱动/wheel问题无限阻塞无卡收尾。必要tokenizer/partition/patch资产缺失仍列pending，不造hash。

### 14.9 A800验证顺序与策略晋升

只规划，不在本次启动或租卡：

1. Mistral单Segment，同后端fixed15与r=1正确性、Prefix组合、mask/Source完整性先过门。
2. group-isolated开发集收集全Source×九比例真实QA/first-token timing及d1/d2 K曲线；r0/r1单列。冻结指标、阈值候选和误差容忍规则后采集，不能看结果再定义成功。
3. 固定cost测量支持和质量映射，对比fixed-rho5、fixed-rho15与joint Source×ratio；不全交叉chunker/并发/存储配置。fixed15不通过也不降低质量门。
4. 固定评分方案后测试三种候选链；报告CFO额外成本、winner/qualified-plan recall和完整TTFT。有损剪枝损失或成本不值得就选择全量，不因为写了代码强制启用。
5. GPU-resident→CPU streaming→SSD staged，实际验证最低合格ratio和I/O slack；记录首末层裸露I/O、overlap和预测超支。效率优先为主，质量模式单独报告。
6. 全部最终策略仅在未用于fit的validation trace上检验；各K独立因果重放，包含dense fallback、物化、staging与维护。Source-only Oracle只能从真实QA合格候选中取最优。
7. Qwen独立tokenizer/Source/threshold/成本复验；单Segment证据成立后再进入语义chunker对照、多Segment和并发，不提前扩大矩阵。

H1继续以Coverage(K)、MarginalGain和最低合格repair分布证明多Variant价值；H2增加比较链开销/剪枝损失与joint选择；H3覆盖九点质量—时间曲线与I/O余量；H4才验证promotion/层级存储/并发。H5仍需全部Profile与资格冻结后单独授权，不访问locked test。

所有最终判断基于matched native Prefix+dense remainder和同后端单Source/CacheBlend对照，计入全部selection与fallback。没有净收益时保留负结果并选择更简单基线；本次不承诺“保证TTFT至少提升20%”。当前计划状态继续为未正式冻结、未GPU资格验证、非论文证据。

### 14.10 本轮实验顺序、停机门与不合理路径剔除

为避免同时打开太多可能误删候选的机制，单 Segment 的实验顺序固定如下：

1. **执行与质量端点**：matched native Prefix+dense remainder、同后端 full dense、固定15% V-only、r=1 token/logit/digest/绝对位置正确性。r=1 是执行器端点，不计作性能胜利。
2. **Source 价值矩阵**：K=1/4/8/16，在同一 residency 下全量比较所有候选，输出每个 Source 的 K 曲线、KV/V repair 质量、QA、完整 TTFT 和质量合格的实际节省。若多Source没有降低最低合格 ratio或没有净收益，H1保留负结果，不启用剪枝。
3. **Source×ratio 联合候选**：固定九点网格，利用一次漂移排序得到候选，再按冻结阈值和成本选择最小合格比例。先完成 fit/validation 的阈值冻结，再运行 validation；任何缺 support 的点都 dense。该步骤不同时改变 chunker、Prefix策略或存储层。
4. **比较链消融**：只在 K>8 比较 `full_compare`、`d1_half`、`cfo_half_d1_half`；QCFuse anchor 另开互斥 arm。每个 arm 同时报告候选 recall/regret、QA和选择链完整 wall-clock。若优化没有超过全量比较的误差区间，退役所有剪枝作为冗余设计。
5. **I/O 与 repair**：先 GPU-resident，再 pinned CPU，再 SSD-staged；使用已选 Source 的 KV deviation/V-only 两个明确 metric，测 load/compute 的逐层 overlap。load-bound 才允许在质量支持内增加 ratio；compute-bound 选择更低合格 ratio。所有真实成本含 staging、首末层裸露、selection和fallback。
6. **动态池与维护**：只在单 Segment 正收益和正确性通过后，验证 exact dense 物化、当前请求不可见新Variant、CPU/SSD exclusive backing、promotion和 per-content LRU。迁移不刷新 `last_request_use_epoch`，非胜者不进入当前关键路径。
7. **Qwen 复验**：独立 tokenizer、Source池、阈值和成本表，重复步骤1–5；公共代码修复后用新SHA回归已通过的 Mistral端点。
8. **最后才进入 multi-Segment/并发/H1–H5**：本阶段不把一个 Segment 的局部正收益外推到联合请求，也不在 Source×ratio 尚未证明时扩大矩阵。

每一级的停止条件是：correctness failure 立即停止；质量下降超过预设门或完整请求 TTFT 无净收益则保留失败并退回更简单的上一条路径；成本缺测、候选截断或 unsupported 只计作 dense/abstain，不计作算法成功。任何“CFO+anchor+d1+d2”多重链只有在分拆实测仍有独立收益时才可进入后续消融，默认不启用。

“保证 TTFT 不增加、质量不降低”的工程含义也固定为两层：在运行时，只有包含 selection、staging、load、repair、overlap 和 fallback 的完整预测关键路径满足 FinalCommit 才允许 selective reuse；缺任何支持就走 dense。预测不能替代事实，因此在配对 validation 上若实际 TTFT 置信区间不优于 matched dense，或 QA/token/logit 超过门限，不能宣布该 Source×ratio 计划有效，必须退回 fixed15/V-only 或 dense。不存在用较快的 Source-only 时间掩盖请求端到端变慢的例外。
