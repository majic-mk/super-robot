# ProbeKV：单 Segment 闭环与 Dense Oracle 驱动的下一阶段计划

更新：2026-09-08。状态：**计划及研究候选，不是已实现或已通过的实验合同**。

审计代码基线：`5fc3197eaa8a7725598fabd7ec374646994bbf97`；分支：
`codex/probekv-schema10-evidence-integration`。

本补充整合用户附件的 Early State / K-V-KV-QK / Oracle / repair 下界建议，
并保留此前尚未完成的单 Segment overlap、执行器性能和真实选择验证。
附件 SHA256：`1f460e7b7254f33489af56f5df151f551e463b6561fccb28c43a870fcb11e6bf`。

附件中的“v9”暂作研究方向称呼。**本计划不把 protocol 8 / schema10 改为 protocol 9，
不重新解释旧 schema9，也不恢复历史学习型 selector 或 calibration 在线路径。**
本次只更新计划，不改运行代码、不连接服务器、不启动 GPU、不提交或推送。

## 1. 总体结论与不变边界

应采纳的是“先用真实 Oracle 证明信号有效，再决定是否改方法”，而不是现在同时增加
在线 `r_safe`、conformal、第二后端和新的淘汰算法。

保持以下边界：

- 单活跃请求、单个待复用 Segment；该 Segment 可有最多 16 个历史 Source Variant。
- Mistral 先完成闭环，Qwen 随后独立验证；多 Segment 和并发不在本阶段启动。
- CacheBlend 为主后端；保留 training-free Residual-K、d1/d2 候选及 legacy 回退。
- 新模式不恢复固定 5% selection-time 拒绝门；选择成本仍进入真实请求总成本。
- Gate1 保持当前 source-local futility `gamma=1.0`；资源准入和请求级
  FinalCommit `gamma=0.8` 保留。
- Source freeze 后不可换 Source；只允许 winner full-KV preparation，且必须有租约及 HBM reservation。
- canonical Source 仅来自 exact dense full prefill；`r=1` reuse 也不是 canonical 来源。
- 保留一份 BF16 Artifact、CPU/SSD exclusive backing、按需 GPU hot copy、grace 和 LRU。
- 未验证的新信号仅进入离线诊断；不得因离线分数改善直接替换生产 selector。
- 正式 Profile、qualification、完整 H1–H5、locked test 均不在本阶段执行。

## 2. 当前证据与未完成事项

历史 Mistral eager Prefix＋non-prefix `r=1` primitive 已有正确性证据，见
[GPU checkpoint](PROBEKV_SCHEMA10_MISTRAL_R1_GPU_CHECKPOINT_20260908.md)。
它不能替代新的 windowed 路径验证，也不是生产选择或 QA 收益证明。

本次源码核验发现：

| 对象 | 当前实现事实 | 下一步含义 |
| --- | --- | --- |
| 原生正确性入口 | `correctness_target["prefetch_window"] = 0` | 当前 r=1 检查实际走 eager，不能证明 window=1/2 正确 |
| hot Replica 注册 | 只查询已提交 `layer_events`，未排除 `pending_layers` | 可能将部分层完成误认为完整 Artifact 已 READY |
| all-ready 对照 | 只等待已有 events | windowed 模式下不是严格的全部层 ready 对照 |
| steady-state 预取 | 使用 `next_layer + 1` | 初始 window=2 不等于持续两层 lookahead |
| Source Oracle | 仅允许 fixed15/r1，拒绝 teacher tokens / capture logits | 已有 QA oracle 可复用，但分布 Oracle 尚需接通 |
| 当前 selector | 有 absolute threshold、margin、depth 与 scope 检查 | 不是随机选层，也不是完全没有质量代理 |
| 历史补丁 | resumable layer 调用含逐层 `end.synchronize()` | 需测量同步代价，不应先假定全部差额都来自 Python |

最近 fixed-Source 与 resumable-dense 的诊断差异，不等于相对 native Prefix 基线的收益；
也不证明 Residual-K 选对了 Source。当前尚不能宣称完整生产选择、准入和正收益已验证。

## 3. 附件建议的采纳裁决

| 建议 | 裁决 | 限定或修正 |
| --- | --- | --- |
| 离线 checkpoint profiling，在线 strong/stable early exit | 采纳 | 保留现有有限 checkpoint；最大深度不是必到深度 |
| Dense × all Sources Oracle | 优先采纳 | 顺序执行、同请求同 boundary/repair；不能用 residual 自己充当 ground truth |
| K/V/KV 比较 | 采纳为离线消融 | 保留 normalized、trimmed K 主基线；选择 trim 不等于 repair mask |
| 少量 Q/anchor 的 QK 信号 | 有条件采纳为离线候选 | 真实 current Q、位置重对齐、GQA/causal mask、完整成本核验后才讨论上线 |
| Oracle Safe Set | 修正后采纳 | 绝对质量合格集与相对 near-best 集分开，二者不能互换 |
| per-Source 最低 repair ratio | 采纳为离线标签 | 仅在测试网格、固定 backend/boundary/support 下成立，不直接预测上线 |
| calibrated abstention / online r_safe predictor | 暂缓 | 属于新统计/方法合同，不是恢复旧开关；先保留经验阈值和 dense abstention |
| 保留 CacheBlend | 采纳 | 先把当前后端证据补齐 |
| SparseX 第二后端 | 仅列后续可行性候选 | 不加入本轮必做项，不声称可直接替换或必需用于证明创新 |
| SpecCache 辅助小模型 | 不进入当前主线 | 理由是新增模型/数据面成本；使用预训练小模型本身不等于需要训练 |
| CFO 按成本决定 shortlist 是否值得 | 采纳 | 保留全比较、CFO 和简单排序对照；不恢复固定 5% 门 |
| diversity / k-medoids 淘汰 | 暂不采纳为在线策略 | 当前用户选择的是 LRU；先看 Coverage(K) 与 marginal gain |

## 4. 必须修正的数学与因果解释

### 4.1 深度、早停和质量不是同一件事

保留 completed-depth 定义：

\[
K_{obs}^{(d)}=W_K^{(d+1)}\operatorname{Norm}_{d+1}(h^{(d)}).
\]

左侧上标表示已完成的 Transformer blocks，右侧投影来自第 d+1 层。
`d=0` 仅为负对照，禁止 online lock；最早合法观察为 d=1。
观察 K 的投影不代表第 d+1 层 causal self-attention 已完成；实际 reuse boundary
还受 repair-check、Source ready 和 scheduler 约束，必须分别记录。

\[
M_d=\frac{J_{second}^{(d)}-J_{best}^{(d)}}{\max(J_{second}^{(d)},\epsilon)}.
\]

优势 margin 只能说明相对排序，不能证明误差有统计上界。现行 absolute threshold
继续作为经验资格规则，不能写成“95%安全”。d1 不能凭一个观察触发“连续两 checkpoint
稳定”，只能走合法的 strong 或真正单候选路径。

- 真正 `correctness_eligible_k=1`：合法观察后可走 single-candidate lock，margin=null。
- `correctness_eligible_k>1, compared_k=1`：不能走正常 margin early exit。
- shortlist 中第一名明显领先，不能排除未比较候选更好。
- Source freeze 后停止线上比较；wrong-early-lock 只从独立 shadow/oracle replay 得出。

### 4.2 Safe Set 不等于 Near-best Set

固定 request、Source snapshot、backend、boundary、repair/support 和数值策略，定义：

\[
E_s=\frac1T\sum_{t=1}^{T}KL(P_{dense,t}\Vert P_{s,t}).
\]

定义绝对误差合格集和相对近最优集：

\[
\mathcal S_{error}=\{s:E_s\le\epsilon_{KL}\},\qquad
\mathcal S_{near}=\{s:E_s\le(1+\tau)E_{min}+\epsilon_{numeric}\}.
\]

near-best 可能全部质量不合格；不得称为 safe。另记录 QA-qualified 集，
`QualityQualifiedSafeSet = S_error ∩ S_QA`。KL 接近 dense 也不等于任务答案正确。

`epsilon_KL`、near-best 容忍度、KL regret denominator floor 必须在 held-out validation
运行前由 fit-only 规则确定并绑定 manifest；未确定时只报告连续误差/排名，safe-hit=null。
禁止把 QA 的 F1 下降阈值 0.02 直接当 KL 阈值。

采用 tie-aware Top-1 / near-best hit，另报 absolute regret 和带预注册 denominator floor
的 normalized regret。全相同分数的 Spearman、空 safe set 条件指标为 null，不能填 0 或 1。

### 4.3 r=1 失败属于正确性问题

在本项目定义的 exact dense 前段、完整 repair、正确 Prefix/position/mask 和同数值栈下，
`r=1` 必须满足已冻结的 dense 等价合同。附件的“Source 即使 100% repair 仍不安全”
不能作为正常 Source-quality 标签；出现这种情况应先停止并检查执行器、数值路径或测量。

也不能用 r=1 的 Source 排名证明 backend independence：全修复时 Source 影响应消失，
排名通常退化为并列。r=0 是另一个可选诊断端点，不代表合理主 operating point。

### 4.4 最小 repair ratio 不等于最快 Source

更低的 repair 需求可能对应 SSD Source、更大 staging 或更晚 ready。仍应比较匹配
质量条件下的完整请求关键路径，而不是只比较 ratio 或把 load 与 compute 简单相加：

\[
T_{reuse}^{pred}=T_{sunk}^{actual}+\widehat T_{joint-future},\qquad
T_{reuse}^{pred}\le0.8T_{dense,matched}.
\]

此处仍是预测准入，不是实际 TTFT 永不超支的保证；实际超支单独报告。
当前三个角色不变：Source-local futility、准备资源约束、请求级 FinalCommit。

## 5. P0：先完成单 Segment windowed 正确性和 overlap 证据

先修复现有实现，不等待新算法实验：

1. 明确每层 pending/submitted/ready/consumed；完整 GPU hot Replica 注册必须覆盖全部层，
   pending 为空且完成事件通过。失败 copy 不得丢失原 Source 引用或被误记完成。
2. all-ready 对照必须提交并等待全部层；`copy_in_flight` 不得忽略尚未提交层。
3. 定义并实现持续的 eager/window1/window2 调度；无 Prefix 的 composite KV 也必须完整。
4. qualification 的 destination digest 等全部层完成后再算；不能强制切回 eager 来证明 windowed。
5. 分别验证三种调度的 greedy IDs、至少 32 个共同 teacher 位置 relative-L2<=1e-4，
   before/destination/after logical digest、Prefix/Suffix/absolute mask、异常资源释放。
6. 从全部 copy/compute 区间的交集与并集计算 overlap，并区分 host wait、GPU wait、
   GPU kernel busy 和包含 CPU 提交空隙的 CUDA Event envelope。
7. 缺少 event 的计时为 unavailable，不用 ticket 起点伪装具体层 copy 起点。
8. 用实际 GPU timeline 核验一组 anchor；仪表开/关做配对，量化 instrumentation 扰动。

验收不是“overlap 数值越大越好”，而是“正确依赖下，请求 TTFT 没有被等待/重复传输拖慢”。
故意推迟 copy 得到更多区间交集不能算性能提升。CPU 状态机测试不能代替真实 windowed r=1。

## 6. P1：执行器开销隔离与收益空间

同请求、同 Prefix 条件、同 sampling/数值策略/计时端点下测量四个 arm：

1. native Prefix＋dense remainder：主要性能基线；
2. resumable dense：定位暂停/逐层执行开销；
3. fixed Source＋fixed15：隔离 repair 执行与 loader；
4. 完整 live selector＋FinalCommit：生产系统路径。

从相同初始状态交错重复，报告 mean/P50/P95、波动和所有 dense fallback。
请求 TTFT、完整 trace（含物化/staging/写入/排队）总成本分别报告。

先 profile，再做不改算法的优化：

- 逐层 end.synchronize、CPU 等 event、scalar read/.cpu/.tolist 隐式同步；
- 实际 QKV、attention、MLP 输入行数和 kernel 时间，而不只读 active-row 日志；
- 同深度重复 QKV projection、mask/position 构造、scatter/composite KV 和 Prefix 传输；
- full digest、pinning、反复 snapshot/planner、同步日志对关键路径的影响。

统计读 event 尽量延后，数据依赖使用正确 stream/event；不能笼统删除所有同步。
保持 Source 只读、BF16 数值合同和异常清理。每个补丁单独留前后对照、新 SHA 与新结果目录。

若 fixed-Source 执行本身尚无 matched baseline 收益空间，应先修执行器，不把更复杂 selector
当作 TTFT 问题的默认解法；小型离线 Oracle 仍可在 correctness 后验证科学假设。

## 7. P2：Dense × All-Sources 与信号验证

### 7.1 统一 Oracle 数据合同

扩展现有 `v8_schema10_native_oracle.py`，不另建一套绕开真实 adapter 的模拟 Oracle。

- 使用冻结 development partition，按 content group 隔离 fit 和 validation；不访问 locked test。
- 每个 request 到达时冻结当前可见 Source 集合，最多 16 个；禁止当前/未来物化泄漏。
- no-cache exact dense 是质量 reference；native Prefix＋dense remainder 是主要性能 reference。
  两种 reference 不得混称，也必须核验比较对象采用的数值策略。
- 每个 Source 顺序运行，固定 boundary、repair、support 规则；不同时搬入 16 份 full KV。
- 所有 Source 使用相同 teacher sequence，采用完整词表 FP32 log-softmax，KL 为主、JS 为辅，
  另报 greedy token agreement、真实答案和版本化 QA scorer。不得只用 top-k 词表近似却称完整 KL。
- 保存原始 token/logit 证据、teacher identity、dtype、temperature、位置、model/patch/Source 摘要。
  长度不足、非有限值、缺文件须显式处理，不能默认为通过。
- Dense 答案不正确也保留；同时报告绝对 QA 和相对 dense 的变化，不能只保留漂亮案例。
- QA 合格约束下的最快 Source Oracle 与最小分布误差 Oracle 分开；其 source-only 时间
  不替代包含选择/调度/排队的 request E2E 时间。

固定15为首轮参考。先回答“同内容的历史 Source 是否有可重复、可预测的质量差异”，
再扩大 repair 网格。若 Source 差异极小，诚实报告，而不是为制造排名增加机制。

### 7.2 K/V/KV：先做无训练对照

保留当前每 token normalized drift 与整数 trim：

\[
m=\min(N-1,\lceil\rho N\rceil),\qquad
J_s=\operatorname{mean}_{j\notin Top_m}(\delta_{s,j}).
\]

首轮 rho=0.15，后续仅使用已有 `{0.10,0.15,0.20,0.25,0.30}` grid。
比较 normalized K、normalized V、等权归一化 KV；KV 的具体组合式须先固定，
不在结果出来后改权重。untrimmed 另作明确消融，不能替换主公式却沿用旧结果。

V/其他辅助观察可保存为独立诊断 SelectionState，但不是新 full-KV Artifact。
score trim indices 始终不得直接进入实际 repair mask。

### 7.3 QK：有界、因果一致的诊断候选

建议测试的 query-conditioned attention-logit drift 为：

\[
\Delta Z_{s,a,j,h}=
\frac{\langle Q^{rot}_{current,a,h},
K^{rot}_{current,j,g(h)}-K^{rot}_{source\rightarrow current,j,g(h)}\rangle}
{\sqrt{d_h}}.
\]

其中 g(h) 是真实 GQA head mapping；仅保留合法 causal edges。先在 query/head 轴上
按预注册 RMS 归约为每 token score，再使用相同整数 trim，从而能与 K/V/KV公平比较。

实施合同：

- Q 必须来自当前请求已执行 dense early state，所有 Source 共享同一 Q；不得借用未来
  deep Oracle Q、答案 token、候选特有 Q，或额外执行未记账的完整 forward。
- 若 user question 位于 RAG 后方，可使用在该 checkpoint 已实际计算的 question Q。
  token 文本已知不等于对应 hidden state 已可免费获得；缺合法 Q 则该信号 unavailable。
- Source 保存 pre-RoPE K，按当前绝对位置和同一 RoPE 配置旋转；禁止直接减历史位置的
  post-RoPE K，也禁止重复施加 RoPE。
- query/anchor 只从已合法计算的非缓存行选择；Prefix cached rows 不进入 ProbeKV 比较。
  读取 question/suffix 的 Q 不授权把这些行纳入 Source repair mask。
- 固定有界 query 数和确定性 anchor 规则；首轮一个配置，不扫描 heads×anchors×depth 的
  大网格。全 heads 可作为初始公平配置，selected heads 仅在另行预注册后消融。
- 实测 Q 提取、状态读取/传输、旋转、GEMM、reduction 和 scratch 峰值，不能只报 GEMM。
  one-shot-if-fit，否则 microbatch，均保留统一 HBM lease。
- 先用小张量 eager reference 测位置重对齐、GQA、scaling、mask 和有限值。

QK 是有动机的代理，不是 Dense error 上界：softmax 对常数平移不敏感，V、后续层传播和
query 抽样也会影响结果。因此只有在独立 validation 上相对 K 提供值得其成本的增益时，
才提出生产候选；不因“公式更接近 attention”自动升级。

## 8. P3：checkpoint 与 CFO 的验证

首先覆盖既有 checkpoints：Mistral `{1,2,4,5,8}`；Qwen `{1,2,4,5,7}`。
比较 d1_only、d1_d2_rescue、legacy。离线可以从一次 dense capture 读取多个 checkpoint，
但线上成本必须按真实 stop depth 测，不能把离线共享采集解释为免费 probe。

做两种独立实验：

1. 固定较晚公共 reuse boundary：隔离观测 depth 对 Source 排序的信息价值。
2. 各策略自然可行 boundary：比较端到端早停收益和实际 repair/ready 成本。

只有既有 checkpoints 呈现明确缺口才预注册少量补点。附件的 d1–d12 全扫列为可选离线
扩展，不立即加入在线，不静默突破历史最大深度或其 Profile namespace。

Primary 指标包括质量合格 safe-set hit、绝对/normalized regret、SelectionCoverage、
wrong early lock、实际 selection latency 和最终 matched-quality TTFT。
同时报告条件命中率与全体合格请求上的有效覆盖率，避免靠大量 abstain 抬高准确率。
统计单位是 request/content group，不将 16 Sources、多个 ratio 或计时重复当独立请求。

CFO 保持廉价排序/shortlist 身份：比较全 K 向量化、CFO shortlist、简单确定性 shortlist。
优先比较谁和最终选谁不能混同；report safe-set Recall@K、shortlist regret、实际总成本。
经验 Recall@K 不是每请求必不漏的保证，CFO 不能恢复为无审计的“安全剪枝”。
全 K 比较足够便宜时可选择全比较，但不在本次删除 CFO 或恢复固定 5% 时间门。

## 9. P4：per-Source repair 下界，仅作离线标签

固定 backend、first reuse boundary、repair metric、token/support 选择、数值策略及请求，
在已有 ratio grid 上测量：

`{0.10,0.12,0.15,0.20,0.30,0.50,0.75,1.00}`。

r=0 和 0.05 仅可作为另行预注册的诊断端点，不能静默删掉历史 0.12 或更换正式网格。

定义带测试范围的标签：

\[
r_{suffix-qualified}^{grid}(q,s,b,B,\pi)
=\min\{r\in\mathcal R:\forall r'\in\mathcal R,r'\ge r,
Pass(q,s,b,B,\pi,r')\}.
\]

其中 B 是 backend，pi 是完整 repair/support policy，Pass 同时记录分布误差、QA 与正确性。
该标签只说明已测试离散点的性质，不证明未测比例、未来请求或所有层都安全。
保留不单调结果和计时波动，不删失败点。缺点时标签为 incomplete/null；r=1 正确性失败则
整组进入 correctness failure，不能标成“Source 的最低安全比例大于100%”。

对 gradual policy，单个 scalar ratio 不能代表整个 layer schedule；先做 fixed-ratio
Source 研究，再单独研究 schedule。不能把离线每 Source 的 ratio vector 悄悄移入当前
request/layer-uniform 的在线 repair controller。

本阶段不实现在线预测器，不根据 Oracle 测得的“最佳 Source”强制生产 commit。

2026-09-09 后续采纳补充：优先评估“离线 model/scope 基础比例＋请求级 slack repair”，
详见 [Slack repair 候选合同](PROBEKV_SLACK_REPAIR_CANDIDATE.md)。因此 per-Source
最低比例曲线保留为机制诊断，不再作为必须建立在线预测器的前置标签工程。
该候选沿用 gamma=0.8，不能按附件的 gamma=0.95 执行；基础比例未验证前不启用。

## 10. 暂缓项与何时重新评估

### 10.1 在线 r_safe、conformal 与 calibrated abstention

保留质量代理不足、缺成本或资源不可行时 dense 的机制。在线预测误差上界属于新增方法，
本阶段不启用。恢复旧 `calibration.py` 不构成新路径的有效证明。

若未来 Oracle 显示固定阈值难以取得质量/覆盖平衡，再单独批准：

- 明确标签、非一致性分数和训练/校准/验证的 content-group 隔离；
- 审计动态池/时间序列是否满足所需交换性，必要时使用独立 trace 评估；
- 对 Source×checkpoint×ratio 的适应性搜索给出同时控制或相应校正，而非多个独立95%拼接；
- 区分 marginal coverage、每请求 conditional guarantee、以及被接受请求中的 failure rate；
- 报告分布外失效、abstention、coverage、selector 开销；不声称 margin 等于概率；
- 明确“无训练 Source 排名”与“数据拟合的 quality/repair policy”的不同，必要时修改论文定位。

普通 conformal 的边际覆盖不自动保证每个请求或已接受子集有相同安全率，参见
[Angelopoulos & Bates，第3节](https://arxiv.org/html/2107.07511v6)。

### 10.2 第二后端与 Source Pool diversity

SparseX 只做后续兼容性候选。先审计代码可获得性/许可、模型、vLLM/attention 栈、pre-RoPE
representation、Prefix/block 生命周期与 r=1；不能拿摘要声称即插即用。
跨后端验证报告每后端内“ProbeKV相对各自基线”的效果，排名一致是补充，不要求它必然成立。
不同 backend/repair 下质量排序不同不自动否定 multi-Source 方法。

保持 per-content LRU 和 CPU/SSD LRU，先因果重放 `Coverage(K)`，再报 marginal gain、
实际字节和写放大。diversity 指标可作相关性观察，k-medoids/farthest-point 仅列远期对照。
少存 Variant 并不保证总存储或总比较成本按同一比例下降；SelectionState、Prefix、元数据等
开销必须实测。不得用多样性替代用户已指定的未来请求覆盖率目标。

## 11. 模块、测试与分阶段产物

以下为后续实施映射，不代表本次已修改：

| 阶段 | 主要现有模块 | 产物/验收 |
| --- | --- | --- |
| P0 | cacheblend_v6_online_engine.py、v8_schema10_staging.py、v8_schema10_native_adapter.py、native_correctness 模块及脚本 | pending/ready 生命周期测试；三调度 r1；可信 overlap raw timeline |
| P1 | native adapter、CacheBlend 独立补丁、cost_collection / measured_costs、事件采集 | 四 arm 实测；实际 kernel shape；关键路径分解；不伪造 positive gain |
| P2 | v8_schema10_native_oracle.py、原生 capture、v8_selection_state_store.py、schema7 repair 的独立 trim 类型 | Dense×Source 的 token/logit/QA；K/V/KV/QK 参考测试及指标报告 |
| P3 | v8_schema10_selector.py、profile_analysis、development manifest/replay | depth/scope/early-lock/CFO 成本消融；候选策略仍未正式冻结 |
| P4 | Oracle runner、repair Profile/manifest 生成器 | 完整离散 repair 标签、缺点标记、r1 故障隔离、不单调结果保留 |
| P5 | 实际 online backend、snapshot/causal coverage、A/B、aggregate | 单 Segment 真实 commit＋matched QA/E2E；随后再决定多 Segment |

新增测试至少覆盖：windowed pending 不误注册、真实 all-ready、持续 window2、copy失败回滚、
无 Prefix 路径、instrumentation 缺测量、KL相同分布为0、共同teacher一致、QA与relative-set隔离、
near-best全部不合格、全并列相关系数null、单候选marginnull、shortlist漏赢家、QK当前位置/GQA/
causal reference、空合法query、trim与repair类型隔离、repair suffix-label缺点及r1失败。

本地验证沿用 compileall、全量 unittest、合同 validator、所有兼容配置、diff whitespace；
GPU结果另行产生。文档更新本身不新增 readiness 或通过结果。

按每阶段完成情况更新证据账本，至少包含：code/patch/model/tokenizer/config/manifest SHA、
数据分组、Source集合、teacher与logit摘要、实际dispatch/boundary/ratio/scope、timer类型、
失败/缺失原因、源文件摘要及原始输出目录。只允许恢复不可变成功任务，保留全部失败证据。

## 12. 顺序、停止点与旧任务保留

执行顺序：

`P0 windowed correctness → P1 单 Segment 性能分解 → P2 Dense Oracle/信号 → P3 depth/CFO → P4 repair 标签 → P5 完整生产闭环`。

P0失败阻止新信号的GPU研究；P2无有意义Source差异时不默认增加更复杂 predictor。
P1/P2之后小规模任务先评估 ETA，不生成 depth×Source×ratio×tier×dataset 的完整笛卡尔积。
运行任务前冻结具体 manifest、阈值/统计规则、数据预算和停止规则；未确定项保持 pending。
本计划不下单、不开新实验，也不重设此前已取消的总租卡费用上限。

此后仍需单独完成的既有任务：

1. 单 Segment 动态物化→发布→后续命中→生产 freeze/preparation/FinalCommit 的闭环；
2. 因果 K coverage、实际 Gate1 paired A/B 与成本误差验证；
3. request/layer-uniform I/O-aware repair 独立对照，fixed15 保留；
4. Mistral 完成后 Qwen 复验，模型 namespace 完全隔离；
5. 单 Segment 通过后再进入多 Segment、真实并发及完整 Profile/qualification；
6. H1 coverage/growth、H2 selection、H3 repair/I/O、H4 runtime、H5 locked 保留，不提前执行。

本轮计划更新不改变以下状态：

```text
windowed_r1_qualified = false
single_segment_production_gain_verified = false
online_r_safe_predictor_enabled = false
online_conformal_enabled = false
second_backend_enabled = false
formal_profile_bundle_frozen = false
gpu_runtime_qualified = false
h1_h2_execution_allowed = false
paper_evidence = false
locked_test_accessed = false
```

以上为计划状态说明，不替代机器 Gate，也不覆盖历史 primitive PASS。

## 13. 文献核对记录与适用范围

2026-09-08 核对附件五条引用：5条题名/标识符均匹配官方页面，未发现题名错配。
学术 MCP 未提供；按 citation-verification 工作流使用 arXiv/ACL 官方来源直接核对。
本次是摘要/机制定位核验，不是复现、全部公式审计或第三方代码兼容证明。

| 文献 | 核对结果及计划用途 |
| --- | --- |
| [CacheBlend，arXiv v3](https://arxiv.org/abs/2405.16444v3) | verified；非前缀融合、选择性重算与读取/计算流水化；继续主后端 |
| [QCFuse，arXiv v1](https://arxiv.org/abs/2606.05875v1) | verified；chunk-anchor query probing、critical-layer profiling，实现在 SGLang；借鉴信号和实验思路，不直接移植 |
| [ProphetKV，arXiv v3](https://arxiv.org/abs/2602.02579v3) | verified；query-driven token recomputation，属于 repair/token-selection 相关工作，不等于已解决本项目历史Variant选择 |
| [SpecCache，ACL 2026](https://aclanthology.org/2026.acl-long.859/) | verified；利用轻量 speculative model 的深层 hidden-state norm；当前不引入额外模型 |
| [SparseX，arXiv v2](https://arxiv.org/abs/2606.01751v2) | verified；segment sharing、Sparse-Q、full+sparse hybrid及其vLLM集成；后续兼容性候选 |

这些论文支持设计动机，不证明 ProbeKV 已优于它们。QCFuse/ProphetKV/SparseX 的上述记录
以 arXiv 版本核对，不把预印本记录当作已确认的会议录用。本文新增贡献是否成立仍取决于
Dense Oracle、独立validation和真实matched-quality收益。
