# Member 3 正式结果索引（V2 与 V3）

本目录保存 Member 3 在相同 7,517 / 940 / 940 数据划分上的正式结果。

## 应使用哪个版本

- 根目录的四个旧结果文件是 V2，最终测试为 98/940（10.43%），保留用于历史对比。
- `v3_full_run/` 是当前最终版本 V3，最终测试为 166/940（17.66%）。
- Member 4 应优先使用 `v3_full_run/improved_predictions.csv`。

## V3 核心结果

- 完整验证集最佳结果：143/940（15.21%）
- Greedy 测试结果：133/940（14.15%）
- Beam 8 + Schema 重排序测试结果：166/940（17.66%）
- Beam 相比 Greedy：增加 33 条，绝对提升 3.51 个百分点
- V3 相比 V2：增加 68 条，绝对提升 7.23 个百分点，相对提升约 69.39%
- Member 2 Baseline：103/940（10.96%）
- V3 相比 Baseline：增加63条，绝对提升6.70个百分点，相对提升约61.17%

## V3 文件

- `v3_full_run/improved_training_log.csv`
- `v3_full_run/improved_greedy_predictions.csv`
- `v3_full_run/improved_predictions.csv`
- `v3_full_run/beam_search_comparison.json`
- `v3_full_run/baseline_v3_comparison.json`
- `v3_full_run/README_CN.md`
- `../../docs/experiments/figures/member3_training_trends.png`：Baseline/V3 Loss 与 V3 验证 EM 趋势图

## 趋势结论

- Baseline 和 V3 的训练/验证 Loss 均持续下降，没有出现验证 Loss 回升。
- V3 Validation Exact Match 从0.21%提高到15.21%。
- Epoch 3→4 的增幅最大（+4.79个百分点）；Epoch 7→8 仅增加0.21个百分点，显示后期收益递减。
- 不同模型的 Loss 绝对值不可直接横向比较，应结合 Exact Match 和 Member 4 的 Execution Accuracy。

Normalised Exact Match 是严格字符串指标。SQL Validity、Execution Accuracy 和错误类型分析
由 Member 4 使用统一数据库执行环境补充。Baseline 与 V3 的940行顺序一致；两个 Tokenizer
造成8行参考标签空格格式不同，详见比较 JSON。完整模型权重不放入 GitHub，可从 Kaggle
V9 Output 或小组约定的大文件渠道获取。
