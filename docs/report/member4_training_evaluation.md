# Member 4 — Training and Evaluation

## Scope and evidence

Member 4 implemented the shared PyTorch/Hugging Face-style sequence-to-sequence training and evaluation components in `src/train.py` and `src/evaluate.py`. No new model training or controlled baseline-versus-improved comparison was run for this report.

Verified locally on Windows, Python 3.9.13, pytest 8.4.2, and CPU:

| Evidence | Result |
|---|---|
| Full implementation test suite | **142 passed, 1 skipped** |
| Skipped test | CUDA-only; CUDA unavailable |

These are implementation-test results, not model-quality results.

## Training

`train_model` provides device-safe batch handling, training and validation loops, optional gradient clipping and scheduling, bounded smoke runs, history, and global-step tracking. Loss is weighted by valid target tokens when labels are available, with a safe fallback otherwise. Non-finite training loss is rejected.

`EarlyStopping` supports `min`/`max` modes, `patience`, and `min_delta`. It tracks serializable best-score, epoch, bad-epoch, and stopped state; non-finite validation scores cannot become the best score.

Checkpointing provides:

- atomic `best.pt` and `latest.pt` writes;
- model, optimizer, scheduler, history, run configuration, and early-stopping state;
- CPU-safe loading with structural validation;
- optimizer-state migration to the target device;
- resume continuity for epoch, global step, history, and historical best checkpoints.

Restricted `weights_only=True` loading is preferred. Any legacy compatibility fallback is only for trusted project-generated checkpoints.

## Evaluation

`evaluate_model` runs generation under `model.eval()` and `torch.no_grad()`, preserves input batches, decodes labels safely, checks prediction/target alignment, and returns records, timing, throughput, and normalized exact-match metrics. Optional bounded evaluation supports smoke testing.

Normalization is conservative: lowercase, trim, collapse whitespace, and remove one trailing semicolon. It does not establish semantic equivalence.

`check_sql_validity` uses an isolated in-memory SQLite database and `EXPLAIN`, with mutation and external-access operations blocked. Categories include `valid`, `empty_sql`, `schema_error`, `syntax_error`, `missing_table`, `missing_column`, and `execution_error`.

Validity is parsing/planning evidence only. Populated databases are unavailable, so result-set execution accuracy cannot be measured.

Error analysis emits deterministic, non-exclusive heuristic labels such as `wrong_table`, `wrong_column`, `join_error`, `aggregation_error`, and `filter_error`. These labels indicate structural differences, not proven semantic errors.

Prediction records can be written as UTF-8 JSONL. Output is validated and atomically replaced so failed serialization or replacement preserves an existing valid file.

## Limitations

- The 7,517/940/940 split is row-level, so schemas may occur across splits.
- Exact match may reject semantically equivalent SQL.
- SQL validity does not prove correctness.
- Error labels are heuristic and may overlap.
- Timing is hardware- and protocol-dependent.
- Smoke tests are not full experiments.

## Historical result boundary

Member 3's experiment summary reports:

| Decoding | Exact match |
|---|---:|
| Schema-aware greedy | **42/940 = 4.4681%** |
| Schema-aware beam-4 | **46/940 = 4.8936%** |

The complete prediction files were absent from the reviewed checkout, so Member 4 did not independently recompute these values. They are not a controlled comparison against Member 2 and are not execution accuracy.

## Integration requirements

The final pipeline must:

- initialize compared systems from the same exact pretrained checkpoint and tokenizer revision;
- use the same split, order, all 940 aligned test rows, training settings, checkpoint-selection rule, and evaluation code;
- retain sample IDs, targets, predictions, and schema DDL;
- select validation-chosen `best.pt`;
- reject incomplete or misaligned predictions, including count, order, target, or sample-ID mismatches;
- store separate baseline and improved artifacts without claiming execution accuracy.
