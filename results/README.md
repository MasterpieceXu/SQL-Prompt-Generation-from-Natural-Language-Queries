# Baseline Result Files

- `baseline_training_log.csv`: train and validation token-level cross-entropy loss
  from the completed five-epoch T5-small Kaggle run.
- `baseline_predictions.csv`: predictions for all 940 examples in the held-out test
  split, generated with greedy decoding.
- `baseline_config.json`: hyperparameters and cleaned dataset split sizes used by the run.
- `baseline_full_summary.json`: selected checkpoint and complete-test normalized
  exact-match result.

Using the shared conservative normalization (lowercase, trim, collapse whitespace, and
remove one trailing semicolon), the baseline matched 71 of 940 targets: 7.5532%.
For a fair baseline-versus-improved-model comparison, evaluate both models on the same
test split with the same training budget, normalization, and metrics.
