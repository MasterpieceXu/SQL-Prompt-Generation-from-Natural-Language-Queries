# Member 2 - Baseline Model and Architecture

This file contains report-ready material for the **Baseline** and **Model Architecture**
parts of the group summary report. Experimental metrics and handoff notes are recorded
separately in `docs/experiments/member2_baseline_results.md`.

## Baseline Model

We selected `t5-small` as the trainable baseline for natural-language-to-SQL
generation. T5 formulates every task in a text-to-text format [1], so the project can
represent the database schema and natural-language question as an input string and the
target SQL query as an output string. This gives the project a simple, reproducible
sequence-to-sequence baseline before introducing task-specific improvements such as
schema-aware input formatting or beam search.

The baseline was deliberately kept generic. It uses the schema text already present in
each prompt but does not add an explicit schema graph, constrained SQL decoder, or SQL
grammar. This design provides a clear comparison point for later methods. It is also
motivated by earlier Text-to-SQL research: SQLNet shows that plain sequence generation
can be sensitive to equivalent condition orderings [2], while RAT-SQL demonstrates the
importance of explicit schema linking and relation modelling [3].

## Model Architecture

`t5-small` is an encoder-decoder Transformer. The encoder reads up to 512 input tokens
containing the schema and question. Its hidden representation is attended to by the
decoder, which autoregressively generates up to 256 SQL tokens. The downloaded model
configuration uses a hidden size of 512, a feed-forward size of 2048, six encoder
layers, six decoder layers, eight attention heads, and dropout of 0.1.

During training, the cleaned SQL query is supplied as the decoder label sequence. T5
computes token-level cross-entropy loss, while padded label positions are represented by
`-100` and ignored by the loss. Parameters are updated with AdamW using a learning rate
of `5e-5`. The experiment uses batch size 4, three epochs, and random seed 42. The model
with the lowest validation loss is saved as the best checkpoint. For the baseline,
inference uses greedy decoding (`num_beams=1`) so that beam search remains a separate,
measurable improvement rather than part of the comparison model.

The baseline validation loss decreased from 0.5069 after epoch 1 to 0.3432 after epoch
3, showing stable learning. However, a diagnostic sample of 40 generated queries also
showed the expected limitations of a plain sequence model: outputs were SQL-like, but
errors remained in table aliases, column selection, joins, filters, and aggregation.
These observations motivate the schema-aware and decoding improvements evaluated later
in the project. Final task-level exact-match and execution metrics should be reported by
the shared evaluation pipeline rather than inferred from validation loss alone.

The corresponding training and validation loss curve is available at
`docs/experiments/figures/member2_baseline_loss.png`.

## References for the Group Bibliography

[1] C. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer," *Journal of Machine Learning Research*, vol. 21, no. 140,
pp. 1-67, 2020.

[2] X. Xu, C. Liu, and D. Song, "SQLNet: Generating Structured Queries from Natural
Language without Reinforcement Learning," arXiv:1711.04436, 2017.

[3] B. Wang et al., "RAT-SQL: Relation-Aware Schema Encoding and Linking for
Text-to-SQL Parsers," in *Proceedings of ACL*, 2020.

## Short Presentation Script

My contribution was the T5-small baseline. I formulated Text-to-SQL as a text-to-text
task: the encoder reads the schema and natural-language question, and the decoder
generates SQL token by token. I implemented model loading, AdamW optimisation,
cross-entropy training, checkpoint selection, and prediction generation. Across three
epochs, validation loss decreased from 0.5069 to 0.3432. The model learned to produce
SQL-shaped queries, but schema-linking and alias errors remained, which provides a clear
motivation for the group's improved method.
