# Member 4 — Experimental Setup

## Status

The lightweight unified evaluator is prepared for a future controlled run. The two
trained checkpoint directories are not currently available, so no new model inference
or final comparison has been performed. The evaluator does not train, download, or
substitute a model and does not persist per-example predictions.

## Data

| Split | Rows |
|---|---:|
| Train | 7,517 |
| Validation | 940 |
| Test | 940 |
| Total | 9,397 |

Seed: **42**. Maximum input/target lengths: **512/256** tokens.

The split is row-level, not schema-disjoint; schema leakage across splits is possible. Save a dataset/split fingerprint containing source revision, preprocessing settings, stable row IDs or hashes, split manifest, and SHA-256 hash.

Record truncation counts/rates and token-length summaries separately for each input format and for targets.

## Controlled comparison

The command validates both local checkpoint directories before reading the dataset or
running inference. It constructs the cleaned, deterministic test split once and passes
the same ordered sample IDs and reference SQL to both systems. Member 2 receives the
baseline prompt; Member 3 receives its repository-defined schema-aware prompt. Each
checkpoint is loaded with its own saved tokenizer and evaluated separately so the first
model can be released before the second is loaded.

Both runs share the requested seed, batch size, device, beam count, maximum generated
token count, normalization, validity checker, and timing implementation. Cross-run
comparison is rejected unless prediction count, sample order, sample IDs, and reference
SQL match exactly. `--max-samples` is only for a bounded diagnostic run; results from a
subset must never be reported as the final 940-example comparison.

## Metrics

| Metric | Reporting rule |
|---|---|
| Raw exact match | Byte-for-byte prediction/reference count and percentage |
| Normalized exact match | Count/940 and percentage |
| SQL validity | Count/940, rate, and validity categories |
| Heuristic error analysis | Non-exclusive and primary-category counts |
| Inference time | Time spent in synchronized model generation calls |
| Average latency | Total inference time divided by evaluated examples |

`Difference` means **Improved − Baseline**. Accuracy and validity differences are reported in **percentage points**.

Because predictions are paired by test example, report a paired-bootstrap confidence interval for the exact-match difference; McNemar's test is optional. If only seed 42 is run, conclusions apply only to that run.

The evaluator uses `model.eval()` and `torch.inference_mode()`, measures generation with
`time.perf_counter()`, and synchronizes CUDA immediately around timed generation calls.

SQL validity must use the same schema DDL for both systems. It measures parsing/planning, not semantics. Populated databases are unavailable, so result-set execution accuracy cannot be reported.

## Required records and artifacts

- write-once run configuration and hash;
- exact software/hardware environment and resolved dependencies;
- dataset/split fingerprint and aligned test manifest;
- Git commit, branch, `git status --short`, and uncommitted patch if any;
- model/tokenizer revisions and all training, stopping, decoding, device, and precision settings;
- truncation statistics;
- epoch history and run logs;
- aggregate summaries and only a small number of representative errors in the report;
- timing protocol and measurements;
- external checkpoint directory paths plus checkpoint hashes and metadata.

Keep large checkpoints outside Git according to repository policy.

## Threats

- row-level schema leakage;
- exact match rejecting equivalent SQL;
- validity not proving semantic correctness;
- overlapping heuristic error labels;
- unequal prompt/target truncation;
- single-seed training variance;
- timing noise;
- test-driven selection;
- incomplete or misaligned artifacts.

## Results template — keep empty until the real run

### Future full-run command

Run from the repository root after both complete external checkpoint directories are
available:

```powershell
python -m src.evaluate `
  --member2-checkpoint "C:\path\to\member2\best" `
  --member3-checkpoint "C:\path\to\member3_v3\best" `
  --batch-size 4 `
  --seed 42 `
  --device auto `
  --num-beams 1 `
  --max-new-tokens 256
```

Omit `--max-samples` for the final evaluation. A diagnostic invocation may add
`--max-samples 5`, but its metrics must not replace full-run results. Output is printed
to the terminal; per-example predictions stay in memory. Execution Accuracy remains
unavailable because neither populated databases nor a test-example-to-database mapping
is present.

### Run record (complete only after the full run)

| Field | Value |
|---|---|
| Experiment ID/date | `<empty>` |
| Git revision/status | `<empty>` |
| Environment/hardware | `<empty>` |
| Dataset/split fingerprint | `<empty>` |
| All 940 rows aligned | `<empty>` |
| Timing protocol | `<empty>` |

### Configuration

| Setting | Baseline | Improved |
|---|---|---|
| Model/tokenizer revision | `<empty>` | `<empty>` |
| Seed | `<empty>` | `<empty>` |
| Optimizer / scheduler | `<empty>` | `<empty>` |
| Learning rate / batch size | `<empty>` | `<empty>` |
| Epoch limit / selected epoch | `<empty>` | `<empty>` |
| Early stopping | `<empty>` | `<empty>` |
| Input / target limits | `<empty>` | `<empty>` |
| Gradient clipping / precision | `<empty>` | `<empty>` |
| Device | `<empty>` | `<empty>` |
| Decoding / beam / length settings | `<empty>` | `<empty>` |
| Evaluated checkpoint/hash | `<empty>` | `<empty>` |

### Results

| Metric | Baseline | Improved | Improved − Baseline |
|---|---:|---:|---:|
| Raw exact match (count/940) | `<empty>` | `<empty>` | `<empty>` |
| Raw exact match (%) | `<empty>` | `<empty>` | `<empty> pp` |
| Normalized exact match (count/940) | `<empty>` | `<empty>` | `<empty>` |
| Normalized exact match (%) | `<empty>` | `<empty>` | `<empty> pp` |
| SQL valid (count/940) | `<empty>` | `<empty>` | `<empty>` |
| SQL validity (%) | `<empty>` | `<empty>` | `<empty> pp` |
| Inference time (s) | `<empty>` | `<empty>` | `<empty>` |
| Average latency (s/example) | `<empty>` | `<empty>` | `<empty>` |

### Paired exact match

| Outcome | Count |
|---|---:|
| Both correct | `<empty>` |
| Baseline only | `<empty>` |
| Improved only | `<empty>` |
| Both incorrect | `<empty>` |
| Paired-bootstrap 95% CI | `<empty>` |
| McNemar p-value, if used | `<empty>` |

### Interpretation

- Controlled-comparison conclusion: `<empty>`
- Heuristic error observations: `<empty>`
- Timing interpretation: `<empty>`
- Deviations and limitations: `<empty>`
