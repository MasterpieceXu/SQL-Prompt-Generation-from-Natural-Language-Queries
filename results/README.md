# Baseline Result Files

- `baseline_training_log.csv`: train and validation token-level cross-entropy loss
  from the completed three-epoch T5-small Kaggle run.
- `baseline_predictions.csv`: predictions for the first 10 test batches (40 examples)
  generated as a diagnostic sample. This is not the complete 940-example test output
  and must not be reported as the final test-set score.
- `baseline_config.json`: hyperparameters and cleaned dataset split sizes used by the run.

For a fair baseline-versus-improved-model comparison, generate predictions for all 940
test examples from the saved best checkpoint, then evaluate both models on the same test
split with the same normalisation and metrics.
