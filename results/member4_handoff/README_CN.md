# Member 3 V3 → Member 4 交接说明

## 交接目标

Member 3 已完成 V3：相关标识符优先的 Schema-aware Prompt、目标 SQL 规范化、完整验证集
Exact Match Checkpoint 选择，以及 Beam 8 + Schema 约束重排序。Member 4 可直接在完整
940 条测试预测上完成 Evaluation 和 Error Analysis。

## 主要文件

- `src/improvement.py`：Member 3 的完整改进方法与训练入口。
- `src/config.py`：模型名称、随机种子、序列长度和默认路径。
- `src/data_prepare.py`：数据清洗、80/10/10 划分、Tokenization 和 DataLoader。
- `src/baseline.py`：`improvement.py` 复用的训练与预测函数。
- `training_no_cot_dataset.csv`：本次实验使用的数据集。
- `results/improved_predictions.csv`：Beam 8 + Schema 重排序的 940 条最终测试预测。
- `results/improved_greedy_predictions.csv`：Greedy Search（`num_beams=1`）的 940 条测试预测。
- `results/improved_training_log.csv`：8 个 Epoch 的训练 Loss、验证 Loss和验证 Exact Match。
- `results/beam_search_comparison.json`：Greedy 与 Beam Search 的精确匹配对比。
- `results/baseline_v3_comparison.json`：Member 2 Baseline 与 V3 的940条配对比较。
- 项目根目录 `results/baseline_predictions.csv`：Member 2 的940条完整 Baseline 预测。
- `docs/experiments/figures/member3_training_trends.png`：Baseline/V3 Loss 与 V3 验证 EM 趋势图。
- `tests/test_improvement.py`：Member 3 方法的自动测试。

两个预测 CSV 都包含以下两列，并保持相同的测试样本顺序：

- `target_sql`：数据集中的参考 SQL；
- `predicted_sql`：模型生成的 SQL。

## 已记录结果

- Epoch 8 Validation Loss：`0.240114`
- Epoch 8 Validation Exact Match：`143/940`（`15.2128%`）
- Greedy Test Exact Match：`133/940`（`14.1489%`）
- Beam 8 + Schema 重排序 Test Exact Match：`166/940`（`17.6596%`）
- Beam 相比 Greedy 增加 33 条，绝对提升 `3.5106` 个百分点
- Member 2 Baseline Exact Match：`103/940`（`10.9574%`）
- V3 相比 Baseline 增加63条，绝对提升 `6.7021` 个百分点，相对提升约 `61.17%`
- Baseline/V3 配对：共同正确72条、仅 Baseline 正确31条、仅 V3 正确94条、共同错误743条

Exact Match 只是字符串诊断指标。Member 4 应进一步计算 SQL Validity、Execution
Accuracy，并对表、列、Join、聚合、过滤条件和嵌套查询等错误类型进行分析。

趋势图显示两组 Loss 均稳定下降，未观察到验证 Loss 回升。V3 的验证 Exact Match 在
Epoch 3→4 提升最大，到 Epoch 7→8 只增加0.21个百分点，说明后期进入收益递减阶段。

## Member 4 建议流程

1. 先使用两个预测 CSV 实现并验证 Evaluation 和 Error Analysis。
2. 使用数据库实例补充 Execution Accuracy。
3. 对表、列、Join、聚合、过滤、排序和嵌套查询进行错误分类。
4. 使用 `baseline_v3_comparison.json` 检查 Baseline/V3 配对结果；两个 Tokenizer 导致8行
   参考标签空格格式不同，最终统一评估应优先读取原始测试集参考 SQL。
5. 将 V3 与其他模型在同一 940 条测试顺序上比较。

完整训练命令：

```bash
python -m src.improvement \
  --dataset_path training_no_cot_dataset.csv \
  --model_name google/flan-t5-base \
  --epochs 8 \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_beams 8 \
  --schema_rerank_weight 0.75 \
  --early_stopping_patience 2 \
  --checkpoint_dir checkpoints/member3_v3 \
  --results_dir results/member3/v3_full_run \
  --no_fp16
```

该命令会根据完整验证集 Exact Match 选择最佳 Checkpoint。模型权重体积较大，不应提交到
GitHub；本交接包只包含 Member 4 进行评估所需的代码、数据和预测文件。
