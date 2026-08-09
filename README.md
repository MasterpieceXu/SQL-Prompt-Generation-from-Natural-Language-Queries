# SQL Generator

COMP9444 小组项目：自然语言到 SQL 查询生成。

本项目目标是使用 `AI4DS/sql_generator_no_cot` 数据集，训练并评估能够将自然语言问题转换为有效 SQLite 查询语句的模型。

## 最终提交说明

最终分析以根目录下的单个 notebook 为主：

- `COMP9444_SQL_Prompt_Generation_Final(1).ipynb`：最终 notebook，默认使用已保存的结果文件进行离线验证，不要求提交数据集或模型权重。
- `COMP9444_Project_Report_TeamName.tex`：Summary Report 的 LaTeX 源文件。
- `COMP9444_Project_Report_TeamName.pdf`：编译后的报告 PDF。
- `final_page_numbers_fixed.pptx`：最终展示文件。

Notebook 中的训练、数据下载和 checkpoint 重放选项默认关闭。提交版本主要复核已经生成的 prediction、training log 和 evaluation artifact；因此在没有原始数据、数据库实例或模型权重时，仍可以检查报告中的核心数字。

如果本地结果文件完整，可以运行离线 demo：

```bash
python -m src.main demo
```

该命令复核 baseline 与改进系统的固定测试集比较。完整训练或重新下载数据则需要安装 `requirements.txt` 中的依赖，并准备相应数据和 checkpoint。

## 仓库结构

```text
SQL-Generator/
|-- src/
|   |-- data_prepare.py      # Member 1: 数据集、清洗、划分、tokenization、DataLoader
|   |-- baseline.py          # Member 2: T5-small baseline model
|   |-- improvement.py       # Member 3: 模型改进方法
|   |-- train.py             # Member 4: 训练流程、checkpoint、early stopping
|   |-- evaluate.py          # Member 4: evaluation、error analysis
|   |-- main.py              # Member 5: 代码整合、demo、复现入口
|   |-- config.py            # 共享配置：路径、模型名、random seed、参数
|   `-- __init__.py
|
|-- notebooks/               # 探索性 notebook 和 EDA
|-- docs/
|   |-- report/              # 报告草稿和最终报告内容
|   |-- experiments/         # 实验记录、结果表格、观察结论
|   `-- meeting_notes/       # 会议记录和任务追踪
|
|-- results/                 # 小型评估结果、预测样例
|-- checkpoints/             # 本地模型 checkpoint，不上传到 GitHub
|-- requirements.txt
|-- README.md
`-- .gitignore
```

正在编辑中的探索 notebook 可以暂时留在根目录。准备共享或提交时，建议移动到 `notebooks/`。

`src/` 里的部分 `.py` 文件仍然是 starter template。`src/data_prepare.py` 已经完成数据读取、清洗、划分、tokenization 和 DataLoader，可以直接供后续 baseline、training 和 evaluation 代码调用。

## 数据预处理使用说明

### 当前配置参数

共享参数统一放在 `src/config.py` 中，当前数据预处理使用以下设置：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `RANDOM_SEED` | `42` | 控制 train / validation / test 的可复现划分 |
| `MAX_INPUT_LENGTH` | `512` | prompt 的最大 token 长度，过长部分会被截断，基于T5-small，选择最大512输入|
| `MAX_TARGET_LENGTH` | `256` | SQL target 的最大 token 长度，过长部分会被截断 |
| `DATASET_PATH` | `hf://datasets/AI4DS/sql_generator_no_cot/training_no_cot_dataset.csv` | Hugging Face 数据集路径 |
| `BASELINE_MODEL_NAME` | `t5-small` | 用于加载 tokenizer 的 baseline 模型 |

这些参数集中在 `config.py`，后续成员调整模型、输入长度或输出长度时，不需要直接修改 `data_prepare.py`。

### 预处理流程

`src/data_prepare.py` 的处理流程如下：

1. `load_raw_dataset()` 从 `DATASET_PATH` 读取原始 CSV。
2. `inspect_dataset(df)` 检查数据规模、列名、缺失值、空字符串和重复行。
3. `clean_dataset(df)` 删除重复行，去除 prompt 两端空格，并从 Markdown SQL code block 中提取纯 SQL，输出 `prompt` 和 `sql` 两列。
4. `split_dataset(df)` 按照 80% / 10% / 10% 划分 train、validation 和 test，并使用 `RANDOM_SEED` 保证划分可复现。
5. `tokenize_dataset(...)` 使用 `t5-small` tokenizer 处理 prompt 和 SQL，生成 `input_ids`、`attention_mask` 和 `labels`。
6. `build_dataloaders(...)` 使用 `DataCollatorForSeq2Seq` 动态 padding，并返回 train、validation、test 三个 PyTorch DataLoader。

当前数据集经过清洗后有 9397 条样本，默认划分结果约为：train 7517 条、validation 940 条、test 940 条。

### 直接运行完整预处理检查

请在项目根目录运行，并使用一个已经安装项目依赖的 conda 环境。环境名称由每个人自行决定，下面的 `your_env_name` 需要替换成自己的环境名：

```bash
conda activate your_env_name
python -m src.data_prepare
```

该命令会读取数据集、加载 tokenizer、完成 tokenization，并打印 batch shape、tokenized dataset 和三个 DataLoader 的大小。数据和 tokenized dataset 当前不会自动保存到仓库；训练代码应在运行时调用这些函数。

### 在训练代码中调用

其他成员可以按下面的方式获取 DataLoader：

```python
from transformers import AutoTokenizer

from src.config import BASELINE_MODEL_NAME
from src.data_prepare import (
    build_dataloaders,
    clean_dataset,
    load_raw_dataset,
    split_dataset,
    tokenize_dataset,
)

df = clean_dataset(load_raw_dataset())
train_df, val_df, test_df = split_dataset(df)

tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)
tokenized_dataset = tokenize_dataset(
    train_df,
    val_df,
    test_df,
    tokenizer,
)

train_loader, val_loader, test_loader = build_dataloaders(
    tokenized_dataset,
    tokenizer,
    batch_size=8,
)
```

训练时使用 `train_loader`，验证时使用 `val_loader`，最终测试和 error analysis 使用 `test_loader`。DataLoader 返回的 batch 包含 `input_ids`、`attention_mask` 和 `labels`，可以直接传给后续的 T5 training loop。

## 小组分工

### Member 1: Dataset and Data Processing（约 20%）

主要代码：

- 下载并分析 SQL Generator No CoT 数据集。
- 分析数据集大小、列结构、`prompt` / `response` 格式、缺失值和重复值。
- 制定并实现数据清洗规则。
- 完成 train / validation / test split。
- 准备 tokenization 和 DataLoader。
- 主要文件：`src/data_prepare.py`

报告负责：

- Dataset
- Data Preprocessing

共同参与：

- Review 所有代码
- 帮助修改 Report
- 参与 PPT 和 presentation

### Member 2: Baseline Model（约 20%）

主要代码：

- 实现 T5-small baseline。
- 设置 loss function。
- 配置 optimizer 和基础 hyperparameters。
- 实现 baseline model saving。
- 主要文件：`src/baseline.py`

报告负责：

- Baseline
- Model Architecture

共同参与：

- 阅读并 review 其他成员代码
- 修改报告
- 参与 presentation

### Member 3: Model Improvement（约 20%）

主要代码：

- 实现模型改进方法，例如 schema-aware input、beam search、prompt design，或在可行时尝试 T5-base。
- 将改进方法与 baseline 进行对比。
- 主要文件：`src/improvement.py`

报告负责：

- Proposed Method
- Improvements

共同参与：

- Review baseline
- 帮助实验
- 参与 PPT

### Member 4: Training and Evaluation（约 20%）

主要代码：

- 搭建共享 training pipeline。
- 实现 checkpoint 保存和加载。
- 根据需要加入 early stopping。
- 实现 evaluation metrics 和 error analysis。
- 主要文件：`src/train.py`、`src/evaluate.py`

报告负责：

- Experimental Setup
- Results
- Discussion

共同参与：

- Review 全部代码
- 参与 PPT

### Member 5: Integration（约 20%）

主要代码：

- 合并并整理所有成员代码。
- 维护 GitHub workflow，处理 merge conflict。
- 修复 integration bugs。
- 准备最终 demo。
- 确保项目可复现，包括 random seed 和可运行命令。
- 主要文件：`src/main.py`、`src/config.py`

报告负责：

- Introduction
- Conclusion
- 整理格式
- References

共同参与：

- 修改所有章节
- PPT 和最终 presentation

## GitHub 协作流程

如果你第一次参与这个项目，请先把仓库下载到自己电脑：

```bash
git clone https://github.com/MasterpieceXu/SQL-Prompt-Generation-from-Natural-Language-Queries.git
cd SQL-Prompt-Generation-from-Natural-Language-Queries
```

安装依赖：

```bash
conda activate your_env_name
python -m pip install -r requirements.txt
```

每次开始写代码前，先拉取最新版本，避免和别人代码冲突：

```bash
git pull
```

修改完成后，先查看自己改了哪些文件：

```bash
git status
```

把需要提交的文件加入暂存区：

```bash
git add path/to/your_file.py
```

例如：

```bash
git add src/data_prepare.py
git add notebooks/data_analysis.ipynb
```

提交改动：

```bash
git commit -m "briefly describe your change"
```

例如：

```bash
git commit -m "add dataset cleaning analysis"
```

推送到 GitHub：

```bash
git push
```

日常流程可以简单记成：

```text
第一次：git clone
每次开始前：git pull
写完以后：git status -> git add -> git commit -> git push
```

## 协作规则

- 如果使用 AI 辅助写代码，请先让 AI 阅读 `README.md`，并明确告诉 AI 自己负责的 member 和文件范围。
- 使用 AI 时，默认只修改自己负责的文件；如果确实需要修改其他成员的文件，请先在群里说明。
- push 前一定要检查 `git status`，确认没有把本地探索 notebook、checkpoint、大文件或无关文件一起提交。
- 每个人优先修改自己负责的文件，减少 merge conflict。
- 如果需要修改别人负责的文件，请先在群里说明。
- 如果新增第三方 Python 库，请同时更新 `requirements.txt`。
- 不要提交大型模型文件、下载的数据集或生成的 checkpoint。`checkpoints/` 只用于本地训练输出。
- 可复用代码放在 `src/`。探索、分析和画图可以先放在 notebook。
- push 前请运行自己修改的部分，确认不会破坏当前流程。
- commit message 尽量清晰，例如：
  - `add dataset inspection notebook`
  - `implement baseline t5 training`
  - `add sql validity evaluation`

## 环境配置

安装依赖：

```bash
conda activate your_env_name
python -m pip install -r requirements.txt
```

如果运行时报缺少某个 package，请把它添加到 `requirements.txt`，并一起提交。
