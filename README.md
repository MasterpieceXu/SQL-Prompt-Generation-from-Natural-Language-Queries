# SQL Generator

COMP9444 小组项目：自然语言到 SQL 查询生成。

本项目目标是使用 `AI4DS/sql_generator_no_cot` 数据集，训练并评估能够将自然语言问题转换为有效 SQLite 查询语句的模型。

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

`src/` 里的 `.py` 文件目前是 starter template：里面只有每个成员负责的函数入口和 TODO，不是最终实现。大家可以在对应文件里继续补代码，也可以根据实际实验需要调整函数名称和结构。

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
pip install -r requirements.txt
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
pip install -r requirements.txt
```

如果运行时报缺少某个 package，请把它添加到 `requirements.txt`，并一起提交。
