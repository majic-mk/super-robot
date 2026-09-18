# 支持句组：固定15% K/V Source质量与完整深度诊断

## 固定条件与范围

- 执行SHA：`1e03b2ecf6a1a34ea197ee054cbcd625f99bac88`，server51405，Mistral A800。
- 新开发扩展来自原prepared_stream的未使用calibration组；旧90-case partition不改。
- 两个内容组、8个历史Source、34个目标请求；所有目标共享切片包含原始标注的完整支持句。
- 与旧MuSiQue、2Wiki和HotPot pilot的origin/group/共享token均无重合。
- 每组四个历史prefix不同；完整原始上下文无截断，最大prompt1739 tokens。
- 两个共享Segment分别112、200 tokens：是自然短文档，不是512-token性能样本。
- 固定repair=0.15、`normalized_kv_deviation`、first reuse layer=9；观察深度1/2/4/5/8。
- 全部validation-only，不训练或调整阈值。本文的d1、d2指各深度全候选argmin，
  不是已经验证了带early-exit和拒绝阈值的实际d1/d2 rescue dispatch。

## 完整性

8 Source × 5深度共40个canonical/live self-state检查全部exact digest相等、relative-L2=0。
目标与Source的token身份/位置对齐审计通过。34 × (dense+4 Source)=170次QA动作完成。
首轮矩阵wall-clock 107.732秒仅为开发作业时长，不是生产TTFT。

同一SHA/manifest在新进程独立重复完整矩阵：100.975秒。170次动作的Source身份、生成token和
F1全部一致；34份完整depth observations相同；重复的40项自对齐检查仍exact、relative-L2=0。
重复不是新的独立质量样本。作业结束GPU为0 MiB、0%，未自动启动后续实验。

## 结果

质量合格定义保持相对同请求dense answer-F1下降不超过0.02，不等于绝对答案正确。
Oracle是每个请求实际测得的QA最佳Source，不是深层Residual-K最小Source。

| 策略 | 组0：5目标平均F1 | 组1：29目标平均F1 | 组1：相对dense质量合格 |
|---|---:|---:|---:|
| dense | 0.7022 | 0.5114 | 29/29（参照） |
| earliest固定Source | 0.8800 | 0.4243 | 25/29 |
| latest固定Source | 0.7035 | 0.4387 | 24/29 |
| 事后最佳固定Source | 0.8800 | 0.4811 | 26/29 |
| 每请求QA Oracle | 0.8800 | 0.5156 | 27/29 |
| Residual-K d1 | 0.7011 | 0.4243 | 25/29 |
| Residual-K d2 | 0.7011 | 0.3985 | 24/29 |
| Residual-K d4 | 0.7011 | 0.4467 | 25/29 |
| Residual-K d5 | 0.8800 | 0.4208 | 24/29 |
| Residual-K d8 | 0.7011 | 0.4243 | 25/29 |

事后最佳固定Source与Oracle均是上界，不是可部署选择器。组0有固定Source达到所有请求的
最高F1，没有互补性；组1不存在这种固定Source。组1中9/29个请求不同Source的F1有差异，
其余并列，不能只用高tie-hit率声称选择准确。

组1的Oracle相对最佳固定Source仅增加1个合格请求（3.45个百分点），平均F1增加0.03448。
这是有限的机制机会，不是多Source净收益或总体工作负载必要性的证明。

## 可解释反例

Source编号按预注册历史顺序1—4。

- target13，问《Delightful Forest》导演国籍：dense及Source4回答Chinese；Source1回答Hong Kong，
  Source2/3拒答。d1/d2/d4/d5/d8均未选择Source4。
- target05，问《Life Gamble》导演去世日期：Source1/2/3给出22 June 2002；Source4说日期未知。
  与上一例共同证明不能始终使用Source4解决全部请求。
- target02，问《Four Riders》导演国籍：Source1拒答，其余三个回答Chinese；所有观察深度
  都选Source1。d1 margin约0.199、d2约0.282，说明较大Residual margin也不是QA正确性的保证。

另有日期答案冗长/短答的F1差异，必须保留原始答案区分格式和语义；不能把所有F1差异
都说成事实错误。上述国籍和日期未知的例子不是纯格式差异。

## 判定与下一步

1. **身份对齐问题**：本次自对齐与目标身份检查没有发现错误，不等于证明所有执行器场景正确。
2. **浅层信息不足**：单纯加深到d8没有修复上述错误，不能继续把所有失败归因于浅层太浅。
3. **评分代理**：当前Residual-K并不稳定排序实际QA质量；不依据这34个请求反复调阈值。
4. **多Source必要性**：一个新组存在小幅质量互补性，尚无可靠selector利用它，更没有含选择/存储
   成本的在线收益证明。先做评分代理的预注册诊断与独立验证，再考虑性能扩展。
5. 本次是支持句富集机制实验，29个请求共享同一文档，不当作29个独立内容组或质量尾部认证样本。

不启动SparseX/QCFuse、多Segment、Profile冻结或正式资格。下一轮应先确定：评分是否需要
考虑误差落在什么token及其下游影响；候选诊断先用已有记录，涉及新增在线信号时另行冻结成本与验证合同。

## 产物

服务器前缀：`/root/autodl-tmp/probekv_stage2/artifacts/`

- `support-freeze-1e03b2e-v1/{geometry.json,pilot.json}`
- `support-env-1e03b2e-v1/environment_lock.json`
- `qa-support-legacy-1e03b2e-server51405-v1/`
- `qa-support-legacy-1e03b2e-server51405-repeat1/`
- `dominance-support-1e03b2e-v1.json`

pilot文件SHA256：`4061fd5c0ffb9db6eb75196b24526e75b0796dba4424b9fddb567fdeefb6237f`。
原始token、答案、分数及其摘要保留。没有覆盖旧实验目录。

本地952项测试：951通过、1历史依赖跳过；contract validator通过。
`paper_evidence=false`、`gpu_runtime_qualified=false`、`locked_test_accessed=false`。
