# Member 3——改进方法与实验结果

本节介绍 Member 3 实现的模型改进方法以及在 Kaggle T4 GPU 上完成的实验。下文中的
贪心搜索（Greedy Search）与束搜索（Beam Search）结果来自同一个经过训练的
Schema-aware 模型，并使用顺序完全相同的 940 条测试样本。SQL 执行准确率以及 Member 2
原始 Baseline 的指标仍需由小组统一评估后补充。

## 1. 改进方法

本项目从两个方面改进 T5-small Baseline：第一，使用问题优先的 Schema-aware 输入格式；
第二，在生成 SQL 时使用 Beam Search。为了尽可能公平地衡量这些改动的效果，改进模型与
Baseline 保持相同的数据清洗规则、随机划分种子、最大序列长度、优化器、学习率、批量大小
和训练轮数。

### 1.1 问题优先的 Schema-aware 输入

数据集中的原始 Prompt 依次包含通用任务说明、数据库 Schema、自然语言问题、可选 Hint
以及输出格式说明。因此，Baseline 虽然能够看到 Schema 信息，但整体输入较为冗长，而且
最重要的自然语言问题通常位于 Prompt 后半部分。T5-small 最多接收 512 个输入 Token；当
Prompt 过长时，右侧内容会被截断，模型可能无法完整看到问题或 Hint。

改进后的格式化程序从原始 Prompt 中提取 Schema、问题和可选 Hint，删除重复的通用说明
和分隔符，然后重新组织为以下格式：

```text
translate natural language to SQLite:
question: <natural-language question>
hint: <optional hint>
schema:
<database DDL>
```

将问题放在最前面，可以降低其因右侧截断而丢失的风险；明确的 `question`、`hint` 和
`schema` 标签也能帮助模型区分用户要求与数据库结构。这里保留完整的 Schema DDL，而不只
保留表名和列名，因为列描述、主外键约束和表之间的关系都可能影响 SQL 的生成。

在完整的 9,399 条原始 CSV 数据上，所有 Prompt 均被成功解析。输入的平均字符数从
1,437.25 降低至 814.24，减少了 43.35%。当 T5 Tokenizer 可用时，程序还会自动统计并
记录 Token 层面的截断情况。

### 1.2 Beam Search 解码

Baseline 使用贪心解码（`num_beams=1`）：在每个生成步骤中直接选择当前概率最高的
Token。一旦某一步选择了局部最优但整体不理想的 Token，后续生成就可能得到较差的完整
SQL。这个问题在存在多个候选表名、别名、Join 路径或聚合结构时尤其明显。

改进系统默认使用 4 个 Beam（`num_beams=4`）。在每一步生成中，Beam Search 会保留
多个候选 Token 序列，继续扩展并比较它们的完整得分，最后返回总得分最高的 SQL。实验使用
1.0 的 Length Penalty，避免搜索过程额外偏好过短或过长的 SQL。Beam Search 不会改变
模型参数，只改变推理阶段的解码策略，因此其计算量也高于 Greedy Search。

## 2. 对照实验设计

实现沿用 Baseline 的数据清洗方法，并使用随机种子 42 进行确定性的 80%/10%/10% 训练、
验证和测试集划分。默认训练配置如下：

- 模型：T5-small
- 优化器：AdamW
- 学习率：`5e-5`
- Batch Size：4
- Epoch：3
- 最大输入长度：512 Token
- 最大目标长度：256 Token

程序根据验证集 Loss 选择最佳 Checkpoint，测试集不会参与模型选择。

在消融实验中，同一个通过验证集选出的 Schema-aware 模型分别使用 `num_beams=1` 和
`num_beams=4` 解码。这样可以在不改变模型参数的情况下，单独观察 Beam Search 带来的
影响。若要衡量 Prompt 重构的独立作用，则需要将 Member 2 原始 Baseline 的预测与
Schema-aware Greedy 预测放在相同且顺序一致的测试样本上比较。

Member 3 程序使用标准化精确匹配（Normalised Exact Match）作为诊断指标。比较前会将
SQL 转为小写、合并多余空格、删除末尾分号，并统一标点周围的空格。需要注意，语义等价的
SQL 仍可能拥有不同的文本形式，因此最终项目评估还应由 Member 4 补充 SQL 有效性和执行
准确率（Execution Accuracy）。

## 3. 实验结果与分析

模型在 Kaggle Tesla T4 GPU 上完成了 3 个 Epoch 的训练。训练 Loss 从 1.1003 下降至
0.4692，验证 Loss 从 0.5123 下降至 0.3460。验证 Loss 在三个 Epoch 中持续下降，说明
模型仍在学习；但如果要判断最佳训练轮数，还需要更多 Epoch 和 Early Stopping 实验。

| 系统 | 输入格式 | 解码方法 | Validation Loss | 标准化精确匹配 | Execution Accuracy |
|---|---|---|---:|---:|---:|
| Member 2 Baseline | 原始 Prompt | Greedy，beam 1 | 等待 Member 2 | 等待 Member 2 | 等待 Member 4 |
| Schema-aware Greedy | 问题优先的 Schema-aware 输入 | Greedy，beam 1 | 0.3460 | 4.47%（42/940） | 等待 Member 4 |
| 本文改进方法 | 问题优先的 Schema-aware 输入 | Beam Search，beam 4 | 0.3460 | 4.89%（46/940） | 等待 Member 4 |

Beam Search 在 940 条测试样本中得到 46 条精确匹配，而 Greedy Search 得到 42 条。
因此 Beam Search 多生成了 4 条正确结果，绝对提升为 0.4255 个百分点，相对于
Schema-aware Greedy 的分数提升了 9.52%。在成对比较中，有 6 条样本仅由 Beam Search
正确生成，另有 2 条仅由 Greedy Search 正确生成。

这一结果表明，对当前训练好的模型而言，Beam Search 带来了幅度较小但可测量的解码改进。
不过，由于本次运行时没有获得 Member 2 在相同测试样本上的预测文件，目前不能据此声称
本文方法已经优于 Member 2 的原始 Baseline。

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 1.1003 | 0.5123 |
| 2 | 0.5806 | 0.3949 |
| 3 | 0.4692 | 0.3460 |

本方法的主要目标是减少截断造成的重要问题信息丢失、更清楚地区分问题与 Schema 的作用，
并在解码过程中探索更多候选 SQL。小组最终的 Error Analysis 应重点比较表和列的选择、
别名、Join、聚合、过滤条件、排序以及嵌套查询等错误类型。

由于精确字符串匹配会把文本形式不同但语义等价的 SQL 判为错误，因此在 Member 4 的执行
准确率结果完成后，最终结论应优先参考 Execution Accuracy，而不能只依赖 Exact Match。

## 4. 复现实验

使用项目配置的 Hugging Face 数据集路径运行：

```bash
python -m src.improvement --epochs 3 --batch_size 4 --learning_rate 5e-5 --num_beams 4
```

使用本地 CSV 数据集运行：

```bash
python -m src.improvement \
  --dataset_path "/path/to/training_no_cot_dataset.csv" \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --num_beams 4
```

本次 T4 实验已保存以下 Member 3 结果文件：

- `results/full_run/improved_training_log.csv`：每个 Epoch 的训练和验证 Loss；
- `results/full_run/improved_greedy_predictions.csv`：Schema-aware Greedy 预测；
- `results/full_run/improved_predictions.csv`：Schema-aware Beam Search 预测；
- `results/full_run/beam_search_comparison.json`：Greedy 与 Beam Search 的消融比较；
- `results/prompt_format_comparison.json`：获得 Baseline 预测后，可生成原始 Prompt 与
  Schema-aware Prompt 的比较；
- `results/improvement_comparison.json`：获得 Baseline 预测后，可生成原始 Baseline 与组合
  改进方法的比较。

`improvement.py` 在训练时会将验证 Loss 最低的模型写入
`checkpoints/full_run/improved_t5-small/best/`。根据小组分工，正式 Training Pipeline、
Checkpoint 保存、Early Stopping 和最终 Evaluation 由 Member 4 负责，因此该目录是
Member 4 运行完整训练流程后产生的输出路径，不属于 Member 3 本地交付文件。

如果已有预测 CSV，可以在不重新训练的情况下进行比较：

```bash
python -m src.improvement --compare_only
```

## 5. 局限性

输入格式化程序依赖数据集中的章节标题；如果标题不存在，程序会安全地保留完整原始 Prompt。
Beam Search 会增加推理时间，并不能保证生成的 SQL 一定可执行或语义正确。当前字符串指标
也无法识别所有语义等价的查询。

此外，本项目目前采用随机行级划分，因此相同数据库 Schema 可能同时出现在训练集和测试集
中。在没有采用 Schema-disjoint Split（不同集合使用完全不同 Schema）之前，不应将当前
结果描述为对全新数据库结构的泛化能力。
