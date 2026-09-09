# ProbeKV：Offline Base + Latency-Slack Repair 候选

日期：2026-09-09。基于当前 protocol 8 / schema10 开发分支。

用户建议附件 SHA256：`79d3b627d2f89422b579501398f3300179b361add1c46d45d06fbb5c4199daf7`。

结论：采纳方向，但**不恢复 per-Source 在线 r_safe 预测器、不放宽 FinalCommit、
不声称更高 repair 必然更高质量**。本文件是研究候选合同，不是正式 Profile 或 GPU Gate。

## 1. 与当前系统的关系

现有 request/layer-uniform I/O controller 候选是在 load bubble 内保留更多 repair，
然后应用 floor 和 no-reentry。新建议更进一步：即使增加 repair 会延长 TTFT，也可在
仍满足既定加速目标的范围内接受该代价。两种目标不同，保留为独立对照：

- `fixed15`：当前真实执行参考，不宣称15%已是普遍安全值。
- I/O-balanced：优先利用能被传输隐藏的额外计算。
- slack-maximal 候选：在明确请求总时间上限内选择较高 repair。

新增候选暂不进入 schema10 已冻结 policy 枚举或生产 dispatcher。旧配置、Profile、
repair floor 解释和结果保持不变。不用修改旧 `certified_floor` 字段来偷偷表示新 `r_base`。

## 2. Source 选择不变

保留 exact eligibility、absolute residual threshold、scope/margin/depth 规则，
以及 Source freeze、租约、winner-only full-KV preparation。

`argmin residual` 只说明相对最好，不等于绝对质量合格；margin 不能替代 absolute threshold。
真正单候选 margin=null 的合法路径仍保留。当前 K 为主，QK 仍是未验证候选。

source residual trim ratio、基础 repair 比例、实际执行比例是三个独立变量：

`rho_trim != r_base != necessarily r_exec`。

本策略不更换已冻结 Source，不重新比较第二名，不把 Source-score trim indices 当 repair。

## 3. 基础比例是离线资格，不是在线质量预测

首版优先每模型/固定 selection-dispatch 一套基础比例，避免立刻扩展大量 length buckets。
若验证明确出现长度差异，再预注册少量 bucket；模型、selection metric/threshold、depth、
first reuse boundary、repair metric/support policy、长度范围及数值栈必须绑定。

基础比例 `r_base` 来自 Dense Oracle 和独立 content-group validation。必须评价真实 selector
选择出来的 Source，而不是让质量 Oracle 替 selector 挑选最好的 Source 后再测。

继续使用已有八点 grid：

`{0.10,0.12,0.15,0.20,0.30,0.50,0.75,1.00}`。

附件的0.25/0.35等不自动加入，Source trim 的0.25也不是 repair grid 点。
若希望新增比例，需先预注册并实测其支持。r=0/1保持诊断端点角色，r=1失败进入正确性诊断。

保留 raw per-ratio QA、KL/JS、样本数、非单调点和风险上界。98.8%观测通过率本身不能证明
15%已满足指定的风险合同。阈值未定、证据不足或 scope 不匹配，不得声明“安全基础比例”。
质量验证失败时 dense；fixed15 只是保留参考，不能用它强行绕过质量失败。

不假设所有 `r>=r_base` 自动合格。记录有证据支持的集合 `R_quality(scope)`，
动态选择过程本身也需验证，不能只验证各个固定 ratio 后就宣称自适应策略已通过。
本轮不冻结新的 RepairPolicyProfile，不新增 per-Source predictor。

## 4. 仍使用0.8，而不是0.95或1.0

对当前请求、已冻结 winner、候选 support/schedule 和 snapshot：

\[
T_{pred}(r)=T_{sunk,actual}+\widehat T_{joint-future}(r).
\]

\[
R_{feasible}=\{r\in R_{quality}\cap R_{cost}:r\ge r_{base},
T_{pred}(r)\le0.8T_{dense,matched},\ resource\ feasible\}.
\]

基础比例须有合法测量且满足准入；之后候选为：

\[
r_{exec}^{proposal}=\max R_{feasible}.
\]

没有支持或基础比例不经济则 dense。正式执行仍需资源 reservation 和 fresh FinalCommit，
不能把这个候选返回值当作新的 commit 权限。

例如 Dense=100ms，基础方案68ms，则可用的 gamma slack 是12ms，不是32ms：

| 候选总时间 | 当前合同 |
| --- | --- |
| 79ms | 可能合格，仍须质量、资源和snapshot检查 |
| 82ms | 拒绝，即使仍比Dense快 |
| 93ms | 拒绝，不能采用附件95ms上限 |

slack 是预估可支配预算，不是物理上“闲着不用的免费时间”。更高 repair 可以牺牲一部分
原有加速幅度，因此最终必须同时报告质量改善和额外耗时，不只报告仍然快于Dense。

## 5. critical path 与 support 合同

- 查询完整 boundary-to-first-token future，不直接用孤立 `T_repair(r)` 替代。
- load/compute overlap、staging、blocking、interference 和剩余层执行由联合时间线计价；
  已花费 probe/选择/准备时间必须保留，不能在Source READY后消失。
- 改 ratio、support、boundary 后重新生成 exact cost query；缺点不能填零或静默插值。
- planner 自己的开销不免费；FinalCommit 必须重新读取实际 sunk 和状态，而不是复用旧 slack。
- ratio 选择不保证实际 wall-clock 上界；实际 prediction overrun 单独报告，不能降低gamma。
- 首轮只在单 Segment 的**第一个 selective layer 之前**选一次较高 initial support，
  用同一 winner ranking 的 Top-ceil(r*N) 构造固定-ratio剩余执行候选。
- 比例之间支持集合嵌套、tie按绝对位置确定；这减少混杂，但不证明输出误差严格单调。
- 后续 gradual 可保持或缩小，不能把之前丢弃的token重新加入。不能先只算15%的行，
  到后层出现I/O空隙时再凭空恢复35%的正确current state。
- 若以后接入逐层 slack controller，必须满足 `M_(l+1) subset M_l`、上一比例和已执行路径
  约束，且测整个候选schedule，不把每层最大ratio独立相加。
- 多 Segment仍暂缓。未来依旧request/layer-uniform，不恢复各Source独立ratio预测。

## 6. 本次落地范围

新增 `src/probekv/v8_schema10_slack_repair.py` 的
`propose_single_segment_slack_repair`，复用 `ProfiledJointTimelineEstimator`：

1. 验证single-Segment、pre-commit、d+1、matching dense reference、snapshot与成本摘要。
2. 接受已冻结Source、独立winner repair ranking及调用方提供的development质量支持集合。
3. 每个合法ratio重建完整remaining masks并查询exact joint measurement。
4. 缺测量保留UNSUPPORTED；基础比例不支持/不经济则dense proposal。
5. 使用0.8预算，选择可承受的最高受支持ratio；保留Source和逐候选审计。
6. 查询前后检查snapshot，不修改原始shape、生产query audit、Source pool或allocator。
7. 结果固定 `production_admission_allowed=false`、`fresh_final_commit_required=true`、
   `gpu_runtime_qualified=false`、`paper_evidence=false`。

这是**development shadow planner**：质量摘要仅用于记录调用方证据来源，此接口不加载或
认证正式质量Profile。模拟成本仅允许使用原成本provider的显式CPU测试入口，不能生成GPU证书。
原生adapter目前只允许fixed15/r1；本次没有放开该限制，未接管GPU repair执行。

因此本次完成的是“候选决策算法＋已有实测查询接口＋测试”，并非已经完成生产集成。
没有可验证的质量Profile、ratio cost表及windowed正确性证据前，不能开启该策略。

## 7. 实验与实施顺序

继续以 [单 Segment 下一阶段计划](PROBEKV_SINGLE_SEGMENT_ORACLE_NEXT_PHASE_PLAN.md) 的P0为先。

1. 修完windowed r=1和overlap证据；不由本候选绕开。
2. 单Segment Dense Oracle验证现行Source selector；固定selection/absolute gate。
3. 对选中Source测基础repair grid，fit/validation隔离，决定合法quality-support范围。
4. 同shape采集实测joint成本，shadow比较fixed15 / I/O-balanced / slack-maximal。
5. 先实现并测试原生ratio-support消费接口、预提交snapshot和资源核验；再开放显式实验dispatch，
   不通过 `correctness_repair_ratio` 诊断字段伪装生产策略。
6. 独立验证自适应策略的QA、KL/JS、请求TTFT、实际overrun、HBM/复制成本及abstain率；
   不能仅凭“更多repair”晋升。
7. Mistral单Segment通过后再安排Qwen；正式Profile、H1–H5及多Segment仍不自动启动。

所需测试包含：0.8而非0.95、缺成本不插值、缺基础成本dense、绝对资格失败不升级、
quality grid holes、mask嵌套、Prefix/Suffix归属、score trim隔离、部分ranking拒绝、
stale snapshot、Source保持、规划开销记账、未验证返回值不能当production admission。

本次没有租卡、连接服务器、运行GPU、冻结Profile或启用在线预测器；用户已有未提交文档
与Git bundle均保留。新的主方法名称仅为候选，不在本次更改schema/论文贡献声明。

## 8. 本地验证记录（2026-09-09）

- 新增21项slack proposal测试通过。
- 全量718项：717通过，1项因缺少ijson跳过。
- compileall、合同validator、22套local_system配置、diff/新增文件空白检查通过。
- 全量回归暴露了旧collector测试对无卡＋空provenance的异常顺序冲突：现先验证provenance，
  再检查真实CUDA；CPU拒绝测试使用完整provenance，仍禁止fake timing。
- 本地使用`.venv-validation`；测试中的模拟cost rows只验证算法/接口，不能证明GPU收益。
- 没有修改原生fixed15/r1 ratio限制，没有运行GPU，没有提交或推送。
