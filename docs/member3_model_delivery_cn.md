# Member 3 V3 模型交付与复现说明

## 1. 交付目的

本交付包用于让 Member 4/5 **直接加载 Member 3 已训练完成的模型并生成预测**，不需要重新训练。

正式模型来自 Kaggle Notebook `COMP9444 Member3 T5 Improvement`，Script Version ID 为
`338566734`。正式测试结果为：

- 测试集：940 条；
- Schema-aware Greedy：133/940，Normalized Exact Match 为 14.15%；
- Beam 8 + Schema Reranking：166/940，Normalized Exact Match 为 17.66%。

## 2. 完整交付包内容

压缩包名称：`COMP9444_member3_v3_delivery.zip`

解压后的主要目录如下：

```text
COMP9444_member3_v3_delivery/
├── README_CN.md
├── checkpoint/
│   ├── best/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── training_config.json
│   ├── provenance_manifest.json
│   ├── requirements-reproduction.txt
│   ├── smoke_test_results.json
│   └── evidence/
├── code/
│   ├── src/improvement.py
│   ├── src/member3_checkpoint.py
│   └── tests/test_member3_checkpoint.py
└── data/
    └── training_no_cot_dataset.csv
```

旧的 `COMP9444_member3_results.zip` 只有预测结果和训练日志，不包含
`model.safetensors`，不能用于加载模型。

## 3. 文件完整性

正式模型权重：

```text
文件：checkpoint/best/model.safetensors
大小：990345064 bytes
SHA-256：7a68cf7c6fd4de08b2adf0d084c9341126e5f85c1f3635ee43a8674a70ccf212
```

正式数据集：

```text
文件：data/training_no_cot_dataset.csv
SHA-256：3cd686c5effc02764d073b9939ad91b79402e4a9f3e4539ad280e1c620857455
```

如果任一哈希不一致，不应使用该文件生成正式报告结果。

## 4. 环境安装

在项目根目录中安装依赖：

```bash
pip install -r requirements.txt
```

如需最大限度复现原始环境，参考交付包中的：

```text
checkpoint/requirements-reproduction.txt
```

加载器会使用本地文件，不会重新从Hugging Face下载基础模型。

## 5. 推荐使用方法：只预测，不训练

先将交付包中的两个源文件放入项目的 `src/` 目录：

```text
code/src/improvement.py          -> 项目/src/improvement.py
code/src/member3_checkpoint.py   -> 项目/src/member3_checkpoint.py
```

然后在项目根目录执行小规模测试：

```bash
python -m src.improvement \
  --predict_only \
  --checkpoint_path "/path/to/COMP9444_member3_v3_delivery/checkpoint/best" \
  --dataset_path "/path/to/COMP9444_member3_v3_delivery/data/training_no_cot_dataset.csv" \
  --results_dir "results/member3_smoke_reproduction" \
  --batch_size 1 \
  --num_beams 8 \
  --max_prediction_batches 1 \
  --expected_model_sha256 "7a68cf7c6fd4de08b2adf0d084c9341126e5f85c1f3635ee43a8674a70ccf212"
```

确认小规模测试正常后，删除 `--max_prediction_batches 1` 运行完整940条预测：

```bash
python -m src.improvement \
  --predict_only \
  --checkpoint_path "/path/to/COMP9444_member3_v3_delivery/checkpoint/best" \
  --dataset_path "/path/to/COMP9444_member3_v3_delivery/data/training_no_cot_dataset.csv" \
  --results_dir "results/member3_v3_reproduction" \
  --batch_size 2 \
  --num_beams 8 \
  --length_penalty 0.9 \
  --repetition_penalty 1.08 \
  --schema_rerank_weight 0.75 \
  --expected_model_sha256 "7a68cf7c6fd4de08b2adf0d084c9341126e5f85c1f3635ee43a8674a70ccf212"
```

`--predict_only`执行以下步骤：

1. 校验并加载本地checkpoint；
2. 使用原始随机种子42恢复相同的7,517/940/940数据划分；
3. 使用与V3训练一致的Schema-aware输入和SQL目标格式；
4. 生成Greedy预测；
5. 生成Beam 8候选并执行Schema Reranking；
6. 写出预测文件和运行manifest。

该模式不会创建optimizer、不会执行backward、不会修改checkpoint，也不会重新训练。

## 6. Python加载接口

如Member 4希望在自己的评估程序中直接取得模型对象，可以使用：

```python
from src.member3_checkpoint import load_verified_checkpoint

tokenizer, model = load_verified_checkpoint(
    "/path/to/checkpoint/best",
    device="cpu",
    expected_model_sha256=(
        "7a68cf7c6fd4de08b2adf0d084c9341126e5f85c1f3635ee43a8674a70ccf212"
    ),
)
```

不要在加载后调用 `model.tie_weights()`。正式V3 checkpoint保存了独立的
`shared.weight` 与 `lm_head.weight`；加载器会保留两者的训练结果。

仅加载模型对象不足以复现166/940。正式结果还依赖 `improvement.py` 中的：

- `build_schema_aware_input`；
- `canonicalize_target_sql`；
- `configure_generation_strategy`；
- `select_schema_valid_candidate`。

因此，推荐优先调用 `--predict_only`，再由Member 4读取生成的CSV进行统一评估。

## 7. 输出文件

预测完成后，`--results_dir` 中会生成：

```text
improved_greedy_predictions.csv   # Schema-aware Greedy预测
improved_predictions.csv          # Beam 8 + Schema Reranking预测
beam_search_comparison.json       # 两种解码方法的Normalized Exact Match比较
prediction_manifest.json          # checkpoint、数据、参数、行数和哈希记录
```

正式完整运行时，两个CSV都应包含表头加940条预测记录。

## 8. 测试与验收

项目代码测试：

```bash
python -m pytest -q tests/test_improvement.py tests/test_member3_checkpoint.py
```

验收要求：

- checkpoint SHA-256正确；
- checkpoint可以在不训练的情况下加载；
- `shared.weight`和`lm_head.weight`均被保留；
- `--predict_only`不会调用训练或optimizer；
- 小规模预测能够生成Greedy和Beam结果；
- 完整运行输出940条且测试顺序一致；
- 正式报告应使用完整940条结果，不应使用早期40条诊断结果。

## 9. GitHub交付边界

`model.safetensors`约944MB，超过普通GitHub单文件限制，不应提交到GitHub。

GitHub中只需提交：

- `src/improvement.py`；
- `src/member3_checkpoint.py`；
- `tests/test_member3_checkpoint.py`；
- 本交付文档；
- `requirements.txt`中的`safetensors`依赖。

完整模型压缩包应通过Kaggle Output、Google Drive、OneDrive或其他大文件渠道共享。
