# 支持句切片普查：未使用的开发内容组

## 目的与边界

此前30个QA目标中，24个共享文档是干扰文档。文档级支持标签也不能证明
canonical切片包含支持句。本次仅用CPU核验原始标注和精确token位置，不读取模型输出、
不调阈值、不启动GPU、不修改原90-case partition。

代码：`632b877`（初版`548dd5c`）。服务器：51405。
来源只取既有prepared_stream中calibration/corpus-repeat行，排除原development partition的全部group。
使用原canonical切分规则；offset tokenizer的完整parent token IDs必须逐项等于已存token IDs。
原始官方train文件SHA必须匹配；不读取locked test。

## 实际结果

| 数据集 | 未使用内容组/切片 | 完整支持句case-target对 | 干扰文档对 | 拒绝 |
|---|---:|---:|---:|---:|
| 2Wiki | 30/30 | 45 | 22 | 1 |
| HotPotQA | 25/25 | 6 | 10 | 0 |

以上计数不是独立质量样本数，也不是原始数据集流量分布。它们经过既有重复文档采样和
corpus-derived pseudotime历史约束；不能称生产请求时间序列。

2Wiki有8组至少一个支持句目标，其中两组分别有5和29个：

- `2WikiMultiHopQA:2c4d2edd4f5e68f97c99d235d365ca5c4b0332d79d44b6dc18d4444d11ad7df6`
- `2WikiMultiHopQA:3de89412564f872eebadbd9379c223efcf45e542047f72367aaee46c10a86c5cb`

HotPot的6个支持句目标分布于4组，没有一组达到5个。2Wiki一条目标的同名文档无法
唯一对应句级支持标注，保留`audit_rejected`和原因，不算正例或负例。

MuSiQue只有段落级支持标签，不混入“切片包含完整支持句”证据；本次未运行该数据集普查。

## 证据文件

服务器目录：`/root/autodl-tmp/probekv_stage2/artifacts/`

- `support-census-2wiki-632b877-v1.json`
  SHA256 `2621bae0bbf2e0cd76386cca8854c9be76c0bbe699c26ce0651a1a89edd3a00c`
- `support-census-hotpot-632b877-v1.json`
  SHA256 `f84c5ac0f09aae8162bfb024d36d0230c603fd272c59112d064095595a55f26f`

初版2Wiki在歧义标注处fail-closed，没有产生成功报告；初版HotPot输出保留。
第一次2Wiki命令中传入的SHA存在转录错误，文件校验拒绝；核对prepared_stream原审计与
实际文件SHA一致后更正命令，没有绕过校验或改变数据。

## 尚未授权执行的新样本

报告始终`gpu_execution_allowed=false`。下一步必须完成：

1. 四个历史Source与各目标的完整原始prompt、上下文差异、4096-token长度检查。
2. 与旧pilot的origin/content重合检查、跨组origin隔离及确定性预注册抽样。
3. 单独冻结新development扩展清单，不覆写旧90-case清单，不以QA结果选样本。
4. 在相同fixed15 K/V修复、boundary与引擎下比较dense和每个Source；固定Source、
   实际QA oracle、d1/d2/完整legacy逐层评分分别报告。

支持句存在不等于模型一定依赖该句，更不等于多Source已经具有互补收益。
只有QA矩阵揭示单固定Source无法覆盖的质量收益，且在线额外成本后仍有收益，才能推进多Source主线。

## 回归

951项测试通过运行，950通过、1项历史依赖跳过；contract validator通过。
新增token身份、支持句完整/部分/区间外、歧义拒绝和旧分组/train/test排除测试。
未修改运行时选择器、repair、CFO、经济Gate或模型执行路径。
