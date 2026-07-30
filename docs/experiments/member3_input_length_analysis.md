# Member 3 - Input Token Length Analysis

## Purpose

This analysis measures whether the question-first Schema-aware formatter reduces the
amount of input truncated by T5-small's 512-token limit. It is an input-efficiency and
information-retention experiment, not a task-accuracy measurement.

## Method

- Dataset: `training_no_cot_dataset.csv`
- Cleaning: shared `src.data_prepare.clean_dataset`
- Cleaned rows: 9,397
- Tokenizer: `t5-small`
- Special tokens: included
- Truncation during measurement: disabled
- Original input: cleaned dataset `prompt`
- Improved input: `src.improvement.build_schema_aware_input(prompt)`
- Truncation threshold: more than 512 tokens

## Results

| Input | Mean tokens | Median | 95th percentile | Maximum | More than 512 |
|---|---:|---:|---:|---:|---:|
| Original prompt | 371.64 | 359 | 541 | 1,313 | 690/9,397 (7.34%) |
| Question-first Schema-aware prompt | 241.64 | 229 | 411 | 1,183 | 138/9,397 (1.47%) |

The formatter reduced the mean input length by exactly 130 tokens, or 34.98%. It moved
552 examples from above the 512-token limit to within the limit. The number of inputs
requiring truncation fell from 690 to 138, an 80.0% relative reduction. No example that
was originally within the limit became over-length after formatting. The remaining 138
examples are still truncated and may lose part of their schema.

## Interpretation

The analysis confirms that the formatter has a measurable input-retention benefit: it
substantially reduces how often T5-small truncates the prompt. This does not by itself
show better SQL accuracy. The final V3 system also changes the model, target formatting,
checkpoint selection, and decoding strategy, so its accuracy gain cannot be attributed
to input shortening alone. Token retention and task performance must therefore be
reported as separate outcomes.

## Complete baseline comparison

Member 2 subsequently supplied all 940 Baseline predictions. On the shared test order,
the ten-epoch `t5-small` Baseline achieved 103/940 (10.96%) Normalised Exact Match, while
Member 3 V3 achieved 166/940 (17.66%). V3 therefore added 63 exact matches, an absolute
gain of 6.70 percentage points. The paired comparison is recorded in
`results/member3/v3_full_run/baseline_v3_comparison.json`.
