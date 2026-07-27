# Member 4 — Experimental Setup

## Status

This is the protocol for a future controlled experiment. No new training or model evaluation was run.

Verified implementation-test environment:

| OS | Python | pytest | Device | Result |
|---|---|---|---|---|
| Windows | 3.9.13 | 8.4.2 | CPU | **142 passed, 1 skipped** |

The skipped test was CUDA-only. The formal run must capture resolved package versions, Python/PyTorch/CUDA details, hardware, and peak memory.

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

Baseline and improved runs must use:

- the same exact pretrained checkpoint and tokenizer revision;
- the same 7,517/940/940 split, order, and all 940 aligned test rows;
- identical optimizer, learning rate, batch size, epoch limit, sequence limits, early stopping, and validation-based checkpoint selection;
- identical evaluation code and normalization;
- greedy-versus-greedy when isolating prompt design;
- the same selected model checkpoint when comparing greedy with beam-4;
- the same hardware and timing protocol for performance comparisons.

Reject incomplete or mismatched prediction files; counts, sample IDs, targets, and order must align exactly. Never compare only a shared prefix.

Record failures, interruptions, resumes, bounded smoke runs, exact environment, Git commit/branch/status, and any uncommitted diff used by the run.

Member 3's summary reports greedy **42/940 = 4.4681%** and beam-4 **46/940 = 4.8936%**. Complete prediction files were absent, so Member 4 did not independently recompute these historical values. They are neither a controlled comparison against Member 2 nor execution accuracy.

## Metrics

| Metric | Reporting rule |
|---|---|
| Normalized exact match | Count/940 and percentage |
| SQL validity | Count/940, rate, and validity categories |
| Heuristic error analysis | Non-exclusive and primary-category counts |
| Inference time | Fixed scope, warm-up, repetitions, device, batch, decoding |
| Throughput | Examples/second under the same timing protocol |

`Difference` means **Improved − Baseline**. Accuracy and validity differences are reported in **percentage points**.

Because predictions are paired by test example, report a paired-bootstrap confidence interval for the exact-match difference; McNemar's test is optional. If only seed 42 is run, conclusions apply only to that run.

The evaluator measures one invocation with `time.perf_counter()`. The runner must perform warm-up and repeated measurements; synchronize CUDA around timed regions when applicable.

SQL validity must use the same schema DDL for both systems. It measures parsing/planning, not semantics. Populated databases are unavailable, so result-set execution accuracy cannot be reported.

## Required records and artifacts

- write-once run configuration and hash;
- exact software/hardware environment and resolved dependencies;
- dataset/split fingerprint and aligned test manifest;
- Git commit, branch, `git status --short`, and uncommitted patch if any;
- model/tokenizer revisions and all training, stopping, decoding, device, and precision settings;
- truncation statistics;
- epoch history and run logs;
- complete per-condition JSONL predictions and aggregate summaries;
- timing protocol and measurements;
- `best.pt` and `latest.pt` plus hashes and metadata.

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

### Run

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
| Exact match (count/940) | `<empty>` | `<empty>` | `<empty>` |
| Exact match (%) | `<empty>` | `<empty>` | `<empty> pp` |
| SQL valid (count/940) | `<empty>` | `<empty>` | `<empty>` |
| SQL validity (%) | `<empty>` | `<empty>` | `<empty> pp` |
| Inference time (s) | `<empty>` | `<empty>` | `<empty>` |
| Throughput (examples/s) | `<empty>` | `<empty>` | `<empty>` |

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
