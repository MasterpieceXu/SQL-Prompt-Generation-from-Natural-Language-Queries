# Member 3 T4 Experiment Handoff

This directory contains the completed three-epoch T5-small improvement experiment.

## Files

- `improved_greedy_predictions.csv`: schema-aware model decoded greedily (`num_beams=1`).
- `improved_predictions.csv`: the same model decoded with beam search (`num_beams=4`).
- `improved_training_log.csv`: training and validation loss for each epoch.
- `beam_search_comparison.json`: paired normalised exact-match comparison.

Both prediction files contain 940 ordered test examples with `target_sql` and
`predicted_sql` columns. They can be passed directly to the shared Member 4 evaluation
pipeline.

## Recorded result

- Best validation loss: `0.345968` (epoch 3)
- Greedy exact match: `42/940` (`4.4681%`)
- Beam-search exact match: `46/940` (`4.8936%`)
- Absolute beam-search improvement: `4` examples (`0.4255` percentage points)

Normalised exact match is a diagnostic string metric. Member 4 should report SQL
validity and execution accuracy before the group draws its final performance
conclusion.

The validation-selected model and tokenizer are preserved in Kaggle notebook Version
#1 under `member3_project/checkpoints/full_run/improved_t5-small/best`.
