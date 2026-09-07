# schema10 单请求后端：无卡实施检查点

2026-09-06。基线 `6a8e6be`，开发分支
`codex/probekv-schema10-evidence-integration`。不合并 main，不改服务器，
不启动 GPU。没有增加 schema 或新选择算法。

## 已实现并通过 CPU 回归

| 模块 | 本轮修改 | 为什么修改 |
| --- | --- | --- |
| `v8_schema10_pool.py` | content-local replacement transaction，规划/提交使用同一个 LRU victim；Source freeze 与逻辑租约原子化 | 防止底层重新按价值选 victim、保护变化后误删、模型 purge 越过逻辑租约 |
| `v8_schema10_storage.py` | BF16 tensor/SSD 文件，独立 SelectionState 文件，exclusive CPU/SSD backing，真实目标写入校验后切换 | 避免只有逻辑 tier map、迁移失败丢原副本、半成品提前可见 |
| `v8_schema10_source_metadata.py` | 发布前检查 token/KV geometry、完整 CFO 层数与摘要 | 缺 metadata 的对象不能计入可见池 |
| `v8_schema10_online_backend.py` | 单活跃请求控制器、真实张量 Residual-K、完整 inventory、准备/最终准入、请求结束后物化 | 共享 Pool/selector/证据路径；不允许当前请求使用刚创建的 Variant |
| `v8_schema10_cost_provider.py` | 按完整 execution mask 查询实测 joint future，缺测量返回 UNSUPPORTED | 不相加多个 Segment TTFT，剪枝后重建 mask/query，不把未知成本填零 |
| `v8_schema10_measured_costs.py` | SHA/provenance 绑定的暂定成本表查询 | Gate1 marginal lower 与 Source future upper 分开，不能混写成同一个成本 |
| `v8_schema10_native_blocks.py` | vLLM 0.4.1 原生 allocate/block table/refcount/free 与 decode append 接口 | 禁止固定 block 区间；selective prefill 不能发布为 exact Prefix |
| `v8_schema10_event_log.py` | 追加 hash chain、原子结果、到达/首 token 时间重算、摘要及分组校验 | 原始事件缺失不能靠手填 passed 解锁；未知 QA/成本保持 null |
| `v8_schema10_single_request_sentinel.py` | FAST/legacy 各 10 个 trace cell，共 20 个；严格矩阵/版本/预算校验 | 不把部署接口缺失或少跑任务解释为通过 |

Source score 只读取 live current K 与独立 SelectionState，不读 fixture 的
current layers，也不为比较上传完整 KV。完整 winner KV 只在逻辑/物理租约后
读取，H2D 准备前必须取得 HBM reservation。

新模式仍使用 `end_to_end_aware`；历史 `legacy_fixed_fraction` 保留。
FinalCommit 仍为：

`actual arrival-to-boundary elapsed + supported joint future <= 0.8 * matched dense TTFT`

CPU 测试里的时间表显式标记 `cpu_test_only/fake_timing`，不能作为线上成本表。
真正的暂定哨兵表必须绑定 model/code/patch/GPU/config，并保留真实样本。
样本最大值只是暂定经验上界，不是正式校准后的统计保证。

### 存储和失败边界

- 比较、迁移和 snapshot 不刷新请求使用时间；selection/binding 才是使用。
- 对象在完整 KV、SelectionState、CFO metadata、Artifact/backing 都成功后才发布。
- 只有整个请求 exact dense full prefill 才能物化，局部 dense 或 r=1 reuse 不行。
- 普通请求只记录不保留实体的 snapshot descriptor；显式 A/B 才保留恢复副本。
  A/B 结束释放 snapshot，使旧 SSD 文件可回收。SSD 页缓存不可控必须报告。
- 缺 matched dense/Source-local/preparation/joint 成本是正常 dense 回退。
  freeze/lease 失败保留提议 Source 审计，不换 runner-up。
- 真正 copy/执行异常先 fence/清理资源，再标记后端 failed；不继续下一请求。
- 队列等待计入 TTFT，最后一个请求的物化时间也计入吞吐的服务结束端点。

## 尚未完成：不能宣称 GPU-ready

1. 将 `NativeBlockRequest` 接入真实 Mistral/Qwen FAST 请求上下文，绑定真实
   Prefix pre-RoPE shadow 与采样/working-KV 生命周期。目前它只是原生分配原语，
   不是完整部署 adapter。
2. 独立接通 legacy 调度 adapter，不能借
   `force_nonpaper_measurement_admission` 或 FAST 的 d1/d2 barrier 冒充。
3. 接通真实 GPU hot Replica 注册和 SSD→有界 pinned staging→GPU；
   目前 tensor/file store 已可实际保存/迁移，SSD 反序列化仍是 host tensor，
   不能将它宣称为已经完成的异步 staged GPU loader。
4. 接通实际成本采集、真实 answer/QA/Source Oracle adapter，以及实际 development
   trace/tokenizer/snapshot 审计。不能用这里的 CPU 小张量测试代替 Mistral sentinel。
5. 再确认实例、价格和预算，冻结完整 GPU manifest 后运行 Mistral 小哨兵。

后端故意不声明 `real_cuda_native_online_backend=true`，没有默认生产 factory。
哨兵入口因此会 fail closed，不偷偷使用旧 RuntimeFixture executor。

## 本地验证与交接

```powershell
$env:PYTHONPATH="$PWD/src"
.venv-validation\Scripts\python.exe -m compileall -q src scripts tests
.venv-validation\Scripts\python.exe -m unittest discover -s tests -v
.venv-validation\Scripts\python.exe scripts/validate_contract.py
git diff --check
```

提交到开发分支后生成精确 SHA 检查点：

```powershell
.venv-validation\Scripts\python.exe scripts/prepare_schema10_single_request_handoff.py --output artifacts/single-request-<exact-sha>
```

该脚本重跑全部本地 system 配置，保存日志与 SHA，不导入/启动 vLLM。
模型 tokenizer assets 未在本地审计时记 null，不虚构 tokenizer hash。
它输出的是**未完成 GPU 准备的工程交接文件**，不是实验资格门。

允许报告 `single_request_backend_local_tests_passed=true`；当前仍保持：

```text
artifact_preparation_ready = false
ready_for_single_request_gpu_sentinel = false
single_request_runtime_sentinel_passed = false
formal_profile_bundle_frozen = false
integrated_concurrency_qualified = false
gpu_runtime_qualified = false
h1_h2_execution_allowed = false
paper_evidence = false
locked_test_accessed = false
```

首次 GPU 计划仍为单张 Mistral A800，最多 4 小时、7.5 元/小时、30 元。
20 个 trace cell 只验证因果池/策略路径，不能代替 Prefix、r=1、CFO、成本、
真实 A/B/Oracle 正确性前置任务；本轮尚不生成这些 GPU 结果。
