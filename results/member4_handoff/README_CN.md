# Member 3 → Member 4 交接说明

## 交接目标

Member 3 已实现问题优先的 Schema-aware Prompt 和 Beam Search。Member 4 可在此基础上
完成 Training Pipeline、Checkpoint、Early Stopping、Evaluation 和 Error Analysis。

## 主要文件

- `src/improvement.py`：Member 3 的完整改进方法与训练入口。
- `src/config.py`：模型名称、随机种子、序列长度和默认路径。
- `src/data_prepare.py`：数据清洗、80/10/10 划分、Tokenization 和 DataLoader。
- `src/baseline.py`：`improvement.py` 复用的训练与预测函数。
- `training_no_cot_dataset.csv`：本次实验使用的数据集。
- `results/improved_predictions.csv`：Beam Search（`num_beams=4`）的 940 条测试预测。
- `results/improved_greedy_predictions.csv`：Greedy Search（`num_beams=1`）的 940 条测试预测。
- `results/improved_training_log.csv`：3 个 Epoch 的训练和验证 Loss。
- `results/beam_search_comparison.json`：Greedy 与 Beam Search 的精确匹配对比。
- `tests/test_improvement.py`：Member 3 方法的自动测试。

两个预测 CSV 都包含以下两列，并保持相同的测试样本顺序：

- `target_sql`：数据集中的参考 SQL；
- `predicted_sql`：模型生成的 SQL。

## 已记录结果

- Epoch 3 Validation Loss：`0.345968`
- Greedy Exact Match：`42/940`（`4.4681%`）
- Beam Search Exact Match：`46/940`（`4.8936%`）
- Beam Search 绝对提升：4 条样本，即 `0.4255` 个百分点

Exact Match 只是字符串诊断指标。Member 4 应进一步计算 SQL Validity、Execution
Accuracy，并对表、列、Join、聚合、过滤条件和嵌套查询等错误类型进行分析。

## Member 4 建议流程

1. 先使用两个预测 CSV 实现并验证 Evaluation 和 Error Analysis。
2. 将 `improvement.py` 接入统一 Training Pipeline。
3. 根据 Validation Loss 保存最佳 Checkpoint，并加入 Early Stopping。
4. 对 Baseline 与改进模型使用相同测试样本进行最终比较。

完整训练命令：

```bash
python -m src.improvement \
  --dataset_path training_no_cot_dataset.csv \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --num_beams 4 \
  --checkpoint_dir checkpoints/full_run \
  --results_dir results/full_run
```

该命令会将最佳模型写入
`checkpoints/full_run/improved_t5-small/best/`。正式 Checkpoint 的生成、保存和最终评估
属于 Member 4 的工作范围。
