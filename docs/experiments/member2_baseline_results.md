# Member 2 - T5-small Baseline Experiment Handoff

## Run Status

- Status: completed successfully on Kaggle
- Device: Tesla T4 (`cuda`)
- Dataset: `AI4DS/sql_generator_no_cot`
- Cleaned split sizes: 7,517 train / 940 validation / 940 test
- Approximate end-to-end runtime: 20 minutes
- Best checkpoint criterion: lowest validation loss
- Downloaded archive: `member2_baseline_output.zip`

## Reproducible Configuration

| Parameter | Value |
|---|---:|
| Model | `t5-small` |
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Batch size | `4` |
| Epochs | `3` |
| Random seed | `42` |
| Maximum input length | `512` |
| Maximum target length | `256` |
| Decoding | Greedy (`num_beams=1`) |

Equivalent command:

```bash
python -m src.baseline --epochs 3 --batch_size 4 --learning_rate 5e-5 --max_prediction_batches 10
```

## Loss History

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 1.0610 | 0.5069 |
| 2 | 0.5774 | 0.3919 |
| 3 | 0.4640 | 0.3432 |

Both losses decreased across all three epochs. Epoch 3 produced the lowest validation
loss and was therefore saved as the best checkpoint. These losses are training
diagnostics, not Text-to-SQL task accuracy.

## Prediction Sanity Check

The script generated 40 predictions from the first 10 test batches. All 40 outputs began
with `SELECT` and contained `FROM`, so fine-tuning taught the model the broad SQL output
format. None of the 40 predictions was a literal whitespace-normalised match to its
target. This small deterministic subset is only a diagnostic check: literal exact match
is strict and can reject semantically equivalent SQL, while the subset is not the full
test set.

Representative error categories observed in the sample:

1. **Table alias and schema-linking errors.** The model often selected a plausible table
   or column but attached it to the wrong alias after a join.
2. **Column hallucination or substitution.** Some predictions introduced a non-target
   column or placed a valid column under the wrong table.
3. **Aggregation and ordering errors.** A query requiring `SUM`, `GROUP BY`, `ORDER BY`,
   and `LIMIT` was simplified to an incorrect direct column selection.
4. **Filter errors.** Some conditions used the wrong table alias, threshold, or field even
   when the overall query structure was close to the target.

One near-structure example correctly generated a percentage calculation with a join and
date filter, but counted a different column. This illustrates why syntactically fluent
SQL is not sufficient for semantic correctness.

## Files in the Result Archive

- `results/baseline_training_log.csv`
- `results/baseline_predictions.csv`
- `checkpoints/baseline_t5_small/baseline_config.json`
- `checkpoints/baseline_t5_small/best/model.safetensors`
- tokenizer and model configuration files for the best checkpoint

The model weights should not be committed to GitHub. Share the archive through Kaggle,
OneDrive, Google Drive, or another large-file channel.

## Evaluation Tasks for Member 4

1. Load the `best` checkpoint and generate predictions for the complete 940-example test
   split.
2. Report the group's agreed task metrics, such as normalised exact match, SQL validity,
   and execution accuracy when executable databases are available.
3. Compare the baseline with the improved model using the same test split and decoding
   conditions.
4. Extend the error analysis using shared categories for schema linking, joins,
   aggregation, filtering, nesting, and invalid SQL.

Do not present the 40-row diagnostic literal-match count as the project's final test
accuracy.
