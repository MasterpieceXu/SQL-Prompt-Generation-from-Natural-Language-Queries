# Member 3 正式结果（V3）

本目录保存 Member 3 V3 在 Kaggle Tesla T4 x2 上的正式运行结果。训练、验证和测试划分为
7,517 / 940 / 940，所有结果均来自完整数据，不是 smoke test。

## 核心结果

- 最佳验证结果：Epoch 8，143/940，Normalised Exact Match = 15.21%
- V3 Schema-aware Greedy：133/940，Normalised Exact Match = 14.15%
- V3 Beam 8 + Schema 重排序：166/940，Normalised Exact Match = 17.66%
- Beam 相比 Greedy：增加 33 条，绝对提升 3.51 个百分点
- V3 最终结果相比 V2 的 98/940：增加 68 条，绝对提升 7.23 个百分点
- Member 2 Baseline：103/940（10.96%）
- V3 最终结果相比 Baseline：增加63条，绝对提升6.70个百分点

## 文件说明

- `improved_training_log.csv`：8 个 Epoch 的训练 Loss、验证 Loss 和完整验证集 Exact Match
- `improved_greedy_predictions.csv`：完整 940 条 Greedy 预测
- `improved_predictions.csv`：完整 940 条 Beam 8 + Schema 重排序预测
- `beam_search_comparison.json`：两种解码方法的配对统计
- `baseline_v3_comparison.json`：Member 2 Baseline 与 V3 的完整940条配对统计

## 给 Member 4

请优先使用 `improved_predictions.csv` 进行 SQL Validity、Execution Accuracy 和错误类型
分析。两个预测 CSV 的行顺序和 `target_sql` 完全一致，可直接逐行比较。Normalised Exact
Match 是严格字符串指标，语义等价但写法不同的 SQL 仍可能被判错。与 Baseline 比较时，
两个 Tokenizer 造成8行参考标签空格格式不同，比较程序已验证其行对应关系并在 JSON 中记录。

完整模型权重不放入 GitHub；需要时从 Kaggle V9 Output 或小组约定的大文件渠道获取。
