# Member 4 训练与评估结果

## Member 2 正式统一评估

### 实验设置

- 数据集：`AI4DS/sql_generator_no_cot`
- 随机种子：42
- 数据划分：7,517 train / 940 validation / 940 test
- 输入：Member 2 原始完整 Prompt
- 模型：`T5ForConditionalGeneration`，60,506,624 参数
- 设备：CPU
- Batch size：4
- 解码：Greedy，`num_beams=1`，`max_length=256`，`do_sample=False`
- 推理：`model.eval()` 和 `torch.inference_mode()`
- 未发生 OOM 或 batch size 降级

### 正式测试结果

| 指标 | 结果 |
|---|---:|
| 测试样本数 | 940 |
| Raw Exact Match | 83/940（8.83%） |
| Normalized Exact Match | 88/940（9.36%） |
| 仅归一化后匹配 | 5/940（0.53%） |
| 空预测 | 0 |
| 重复预测 | 2 |
| SQLGlot Syntax-only Validity | 829/940（88.19%） |
| SQLGlot Parse Error | 111/940（11.81%） |

## Member 3 状态

| 指标 | 结果 |
|---|---:|
| 测试样本数 | 940 |
| Raw Exact Match | 118/940（12.55%） |
| Normalized Exact Match | 125/940（13.29%） |
| 仅归一化后匹配 | 4/940（0.42%） |
| 空预测 | 0 |
| SQLGlot Syntax-only Validity | 843/940（89.68%） |
| SQLGlot Parse Error | 97/940（10.32%） |



## 条件式 Schema-aware Validity

当前数据中的部分 reference/schema 本身无法通过 SQLite 检查，因此不能报告完整 940 条测试集的 Schema-aware Validity。

Reference 质量门结果：

- valid：781
- schema_error：106
- missing_table：30
- missing_column：22
- execution_error：1
- coverage：781/940（83.09%）

仅在 781 条 reference-valid 样本上，Conditional Schema-aware SQL Validity 为 328/781（42.00%）。该指标只适用于通过 reference 质量门的 781 条样本，不能描述为全测试集 SQL Validity，也不能把其余 159 条失败归因于模型。

## 主要错误分析

互斥主要错误类别：

| 类别 | 数量 |
|---|---:|
| exact_match | 88 |
| missing_column | 379 |
| syntax_error | 110 |
| aggregation_error | 92 |
| filter_error | 82 |
| join_error | 65 |
| wrong_table | 52 |
| structurally_different | 19 |
| ordering_limit_error | 18 |
| wrong_column | 13 |
| missing_table | 8 |
| unclassified | 8 |
| alias_error | 3 |
| grouping_error | 3 |

最大问题是缺失列或列选择错误，其次是语法、聚合、过滤和 Join 错误。模型能生成较高比例的可解析 SQL，但严格 Exact Match 仍较低。

Execution Accuracy 不可用，因为项目未提供填充后的数据库以及测试样本到数据库的映射。历史结果 103/940 仅是 Member 2 旧实验记录，未参与本次推理或指标计算，不能覆盖本次 88/940。



## 验证

- Member 4 测试：103 passed
- AST 解析通过
- 评估过程没有创建预测、日志或结果文件
- checkpoint 未修改
- 报告编辑前，工作树已有 `src/evaluate.py` 和 `tests/test_evaluate.py` 的未提交修改；本次没有覆盖、撤销或提交这些修改
