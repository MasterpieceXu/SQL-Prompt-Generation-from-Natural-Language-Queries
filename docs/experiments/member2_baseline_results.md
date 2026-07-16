# Member 2 - T5-small Baseline Experiment Handoff

## Run Status

- Status: completed successfully on Kaggle
- Device: Tesla T4 (`cuda`)
- Dataset: `AI4DS/sql_generator_no_cot`
- Cleaned split sizes: 7,517 train / 940 validation / 940 test
- Approximate end-to-end runtime: 20 minutes
- Best checkpoint criterion: lowest validation loss
- Downloaded archive: `member2_baseline_output.zip`

## Baseline Assumptions

1. The prompt contains sufficient schema and question information for SQL generation.
2. Truncating inputs to 512 tokens and targets to 256 tokens does not remove essential
   content for most examples.
3. Treating SQL as a left-to-right sequence is an acceptable simple baseline, even
   though semantically equivalent SQL can use different clause or condition orderings.
4. Greedy decoding is used to keep the baseline deterministic and to leave beam search
   as a separately measurable improvement.
5. Validation loss is used for checkpoint selection, but it is not treated as SQL task
   accuracy.

## Incremental Verification

The pipeline was tested before the full experiment rather than only at the end:

1. A local one-batch test verified dataset loading, tokenisation, forward propagation,
   backpropagation, validation, checkpoint writing, and prediction CSV generation.
2. A Kaggle smoke test used 5 training batches, 2 validation batches, and 1 prediction
   batch on a Tesla T4. It completed without CUDA, data, or checkpoint errors.
3. The full run completed all 3 epochs and all 235 validation batches per epoch.
4. The downloaded result ZIP passed an integrity check and contained the model weights,
   tokenizer, model configuration, run configuration, loss log, and predictions.
5. Held-out predictions were inspected to confirm that the model learned SQL structure
   and to identify schema-linking and semantic failure cases.

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

![T5-small baseline training and validation loss](figures/member2_baseline_loss.png)

**Figure 1.** Training and validation token-level cross-entropy loss across the three
baseline epochs. Both series decreased, with no loss-curve evidence of instability or
overfitting during this short run. Validation loss being lower than training loss is
plausible because dropout is active during training and disabled during evaluation.

No exploding loss, NaN loss, CUDA out-of-memory error, or increasing validation-loss
trend was observed. The three-epoch curve therefore provides no evidence of unstable
training or overfitting. However, semantic SQL errors in the diagnostic predictions
show that low token loss does not rule out task-level underperformance.

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

## Fair Comparison Protocol

Future models should be compared with this baseline using:

- the same cleaned 7,517 / 940 / 940 data split and random seed 42;
- the same preprocessing, tokenizer family, and maximum input/target lengths unless the
  changed input representation is the explicitly tested improvement;
- the same complete test set and the same normalisation and execution rules;
- the same task metrics, including normalised exact match, SQL validity, and execution
  accuracy when databases are available;
- a clearly documented single incremental change where possible, such as schema-aware
  formatting or beam search, so the cause of any improvement is interpretable.

The test set must not be used for hyperparameter selection. Validation loss or validation
task metrics should select settings and checkpoints; the test set should be used only for
the final baseline-versus-improved comparison.

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

Regenerate Figure 1 from the recorded values with:

```bash
python docs/experiments/plot_member2_loss.py
```
