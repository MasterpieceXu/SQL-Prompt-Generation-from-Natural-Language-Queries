"""Member 3: schema-aware prompting and schema-constrained generation.

The baseline receives a long, generic instruction followed by the database schema and
question.  This module makes the task-specific information more explicit and places the
question before a compact schema so that important information is not lost when a long
input is truncated.  During generation it reranks beam candidates with deterministic
schema checks, which discourages nonexistent tables/columns and broken aliases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config import (
    BASELINE_MODEL_NAME,
    CHECKPOINT_DIR,
    DATASET_PATH,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
)
from checkpoints.member3_v3_checkpoint_corrected.load_verified_checkpoint import load_verified_checkpoint


_SCHEMA_HEADING = re.compile(r"(?im)^\s*Database\s+Schema\s*$")
_QUESTION_HEADING = re.compile(r"(?im)^\s*Question\s*:\s*")
_HINT_HEADING = re.compile(r"(?im)^\s*Hint\s*:\s*")
_RESPONSE_INSTRUCTION = re.compile(r"(?im)^\s*Please\s+respond\b")
_CREATE_TABLE = re.compile(
    r"(?i)\bCREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+([`\"\[]?[\w.]+[`\"\]]?)"
)
_FOREIGN_KEY = re.compile(
    r"(?is)foreign\s+key\s*\(\s*([`\"\[]?\w+[`\"\]]?)\s*\)\s*"
    r"references\s+([`\"\[]?[\w.]+[`\"\]]?)\s*\(\s*([`\"\[]?\w+[`\"\]]?)\s*\)"
)
_SQL_TABLE_REFERENCE = re.compile(
    r"(?i)\b(?:from|join)\s+([`\"\[]?[\w.]+[`\"\]]?)"
    r"(?:\s+(?:as\s+)?([A-Za-z_]\w*))?"
)
_QUALIFIED_COLUMN = re.compile(
    r"(?i)(?<![\w.])([`\"\[]?[A-Za-z_]\w*[`\"\]]?)\."
    r"([`\"\[]?[A-Za-z_]\w*[`\"\]]?)"
)
_SQL_RESERVED_WORDS = {
    "cross",
    "full",
    "group",
    "having",
    "inner",
    "join",
    "left",
    "limit",
    "on",
    "order",
    "outer",
    "right",
    "union",
    "where",
}
_SQL_CANONICAL_KEYWORDS = {
    "all", "and", "as", "asc", "avg", "between", "by", "case", "cast",
    "count", "cross", "desc", "distinct", "else", "end", "exists", "from",
    "full", "group", "having", "in", "inner", "is", "join", "left", "like",
    "limit", "max", "min", "not", "null", "offset", "on", "or", "order",
    "outer", "right", "select", "sum", "then", "union", "when", "where", "with",
}


def _normalise_identifier(identifier: str) -> str:
    """Return a case-insensitive SQL identifier without quoting characters."""
    return str(identifier).strip().strip("`\"[]").split(".")[-1].lower()


def _compact_text(text: str) -> str:
    """Collapse whitespace in a question or hint without changing its words."""
    return re.sub(r"\s+", " ", text).strip()


def _clean_schema(schema: str) -> str:
    """Remove dataset separators and repeated blank lines from schema DDL."""
    lines = [line.rstrip() for line in schema.strip().splitlines()]
    lines = [line for line in lines if line.strip() != "###"]

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank
    return "\n".join(cleaned_lines).strip()


def parse_schema_catalog(schema: str) -> dict[str, set[str]]:
    """Parse CREATE TABLE statements into a case-insensitive table/column map.

    The dataset DDL is intentionally parsed with a small line-oriented parser instead
    of a SQL dependency.  This keeps Kaggle setup simple and is sufficient for the
    CREATE TABLE format used by the assignment dataset.
    """
    catalog: dict[str, set[str]] = {}
    current_table: str | None = None
    parenthesis_depth = 0

    for original_line in str(schema).splitlines():
        line = original_line.strip()
        table_match = _CREATE_TABLE.search(line)
        if table_match is not None:
            current_table = _normalise_identifier(table_match.group(1))
            catalog.setdefault(current_table, set())
            parenthesis_depth = line.count("(") - line.count(")")
            continue

        if current_table is None:
            continue

        code = line.split("--", 1)[0].strip().rstrip(",").strip()
        lowered = code.lower()
        is_constraint = lowered.startswith(
            ("primary key", "foreign key", "constraint", "unique", "check")
        )
        if code and not code.startswith(")") and not is_constraint:
            column_match = re.match(r"([`\"\[]?[A-Za-z_]\w*[`\"\]]?)\s+", code)
            if column_match is not None:
                catalog[current_table].add(_normalise_identifier(column_match.group(1)))

        parenthesis_depth += line.count("(") - line.count(")")
        if parenthesis_depth <= 0:
            current_table = None

    return catalog


def _identifier_terms(identifier: str) -> set[str]:
    """Split a schema identifier into conservative terms used for relevance ranking."""
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", identifier).lower())
    terms = set(words)
    for word in tuple(words):
        if len(word) > 3 and word.endswith("s"):
            terms.add(word[:-1])
    return terms


def _question_terms(question: str, hint: str = "") -> set[str]:
    return _identifier_terms(f"{question} {hint}")


def compact_schema(schema: str, question: str = "", hint: str = "") -> str:
    """Linearise DDL and rank relevant tables/columns first without dropping schema."""
    cleaned_schema = _clean_schema(schema)
    catalog = parse_schema_catalog(cleaned_schema)
    if not catalog:
        return cleaned_schema

    query_terms = _question_terms(question, hint)
    original_table_order = {table: index for index, table in enumerate(catalog)}

    def identifier_score(identifier: str) -> int:
        return len(_identifier_terms(identifier).intersection(query_terms))

    def table_score(table: str) -> int:
        # Explicit table mentions are stronger than column-word overlap.
        return 4 * identifier_score(table) + sum(
            identifier_score(column) for column in catalog[table]
        )

    ranked_tables = sorted(
        catalog,
        key=lambda table: (-table_score(table), original_table_order[table]),
    )
    table_parts = []
    for table in ranked_tables:
        ranked_columns = sorted(
            catalog[table],
            key=lambda column: (-identifier_score(column), column),
        )
        table_parts.append(f"{table}({', '.join(ranked_columns)})")

    foreign_keys: list[str] = []
    current_table = ""
    for line in cleaned_schema.splitlines():
        table_match = _CREATE_TABLE.search(line)
        if table_match is not None:
            current_table = _normalise_identifier(table_match.group(1))
        for match in _FOREIGN_KEY.finditer(line):
            source_column = _normalise_identifier(match.group(1))
            target_table = _normalise_identifier(match.group(2))
            target_column = _normalise_identifier(match.group(3))
            foreign_keys.append(
                f"{current_table}.{source_column}->{target_table}.{target_column}"
            )

        # Some rows use an inline form such as ``work_id INTEGER references works``.
        inline_match = re.search(
            r"(?i)^\s*([`\"\[]?\w+[`\"\]]?)\s+.+?\breferences\s+"
            r"([`\"\[]?[\w.]+[`\"\]]?)(?:\s*\(\s*([`\"\[]?\w+[`\"\]]?)\s*\))?",
            line.split("--", 1)[0],
        )
        if inline_match is not None and not line.lstrip().lower().startswith("foreign key"):
            source_column = _normalise_identifier(inline_match.group(1))
            target_table = _normalise_identifier(inline_match.group(2))
            target_column = (
                _normalise_identifier(inline_match.group(3))
                if inline_match.group(3)
                else "primary_key"
            )
            foreign_keys.append(
                f"{current_table}.{source_column}->{target_table}.{target_column}"
            )

    compact = "tables: " + "; ".join(table_parts)
    if foreign_keys:
        compact += "\nforeign keys: " + "; ".join(dict.fromkeys(foreign_keys))
    return compact


def extract_prompt_sections(prompt: str) -> dict[str, str]:
    """Extract schema, question, and optional hint from a dataset prompt.

    The public dataset uses explicit ``Database Schema``, ``Question:``, and
    sometimes ``Hint:`` headings.  An empty dictionary is returned when the required
    schema/question headings are absent so callers can safely fall back to the original
    prompt rather than silently dropping information.
    """
    prompt = str(prompt).replace("\r\n", "\n").replace("\r", "\n")
    schema_match = _SCHEMA_HEADING.search(prompt)
    question_match = _QUESTION_HEADING.search(prompt)
    if schema_match is None or question_match is None or schema_match.end() >= question_match.start():
        return {}

    hint_match = _HINT_HEADING.search(prompt, question_match.end())
    response_match = _RESPONSE_INSTRUCTION.search(prompt, question_match.end())

    question_end = len(prompt)
    if hint_match is not None:
        question_end = hint_match.start()
    elif response_match is not None:
        question_end = response_match.start()

    hint = ""
    if hint_match is not None:
        hint_end = response_match.start() if response_match is not None else len(prompt)
        hint = _compact_text(prompt[hint_match.end() : hint_end])

    return {
        "schema": _clean_schema(prompt[schema_match.end() : question_match.start()]),
        "question": _compact_text(prompt[question_match.end() : question_end]),
        "hint": hint,
    }


def build_schema_aware_input(example: Mapping[str, Any] | str) -> str:
    """Build a concise question-first input with a compact relational schema.

    ``example`` may be a dataset row containing ``prompt`` or a prompt string.  The
    question is deliberately placed first because T5 truncates over-length inputs; this
    layout protects the user request while retaining the schema DDL needed for linking
    table and column names.
    """
    prompt = str(example.get("prompt", "")) if isinstance(example, Mapping) else str(example)
    sections = extract_prompt_sections(prompt)
    if not sections or not sections["schema"] or not sections["question"]:
        return f"translate natural language to SQLite:\ninput: {_compact_text(prompt)}"

    parts = [
        "translate to SQLite. Use only the listed tables and columns. Return only SQL:",
        f"question: {sections['question']}",
    ]
    if sections["hint"]:
        parts.append(f"hint: {sections['hint']}")
    parts.extend(
        [
            "schema: relevant identifiers first",
            compact_schema(
                sections["schema"],
                question=sections["question"],
                hint=sections["hint"],
            ),
        ]
    )
    return "\n".join(parts)


def canonicalize_target_sql(sql: str) -> str:
    """Return a stable training target while preserving quoted literal contents.

    The transformation deliberately avoids alias renaming or condition reordering,
    because those operations require a full SQL parser and can silently change meaning.
    """
    text = str(sql).replace("```sql", "").replace("```", "").strip()
    literals: list[str] = []

    def protect_literal(match: re.Match[str]) -> str:
        literals.append(match.group(0))
        return f"__SQL_LITERAL_{len(literals) - 1}__"

    text = re.sub(r"'(?:''|[^'])*'", protect_literal, text)
    text = re.sub(r";\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for keyword in sorted(_SQL_CANONICAL_KEYWORDS, key=len, reverse=True):
        text = re.sub(rf"\b{keyword}\b", keyword.upper(), text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s*\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*(<>|!=|<=|>=|=|<|>)\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for index, literal in enumerate(literals):
        text = text.replace(f"__SQL_LITERAL_{index}__", literal)
    return text


def configure_generation_strategy(
    num_beams: int = 4,
    max_length: int = MAX_TARGET_LENGTH,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.05,
) -> dict[str, Any]:
    """Return deterministic generation arguments for the improved model."""
    if num_beams < 1:
        raise ValueError("num_beams must be at least 1")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than 0")
    config = {
        "max_length": max_length,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
    }
    if num_beams > 1:
        config.update(
            {
                "length_penalty": length_penalty,
                "early_stopping": True,
            }
        )
    return config


def schema_violation_penalty(sql: str, schema: str | Mapping[str, set[str]]) -> float:
    """Return a deterministic penalty for clear schema and SQL-structure mistakes.

    This is deliberately conservative: it checks only references that can be verified
    reliably without executing SQL.  It does not penalise an unfamiliar unqualified
    word, because that word could be a function, keyword, value, or output alias.
    """
    catalog = parse_schema_catalog(schema) if isinstance(schema, str) else dict(schema)
    sql_text = str(sql).strip()
    lowered_sql = sql_text.lower()
    penalty = 0.0

    if not re.match(r"(?is)^\s*(?:with\b.+?\bselect\b|select\b)", sql_text):
        penalty += 4.0
    if sql_text.count("(") != sql_text.count(")"):
        penalty += 2.0

    aliases: dict[str, str] = {}
    for match in _SQL_TABLE_REFERENCE.finditer(sql_text):
        table = _normalise_identifier(match.group(1))
        alias = _normalise_identifier(match.group(2) or table)
        if alias in _SQL_RESERVED_WORDS:
            alias = table
        aliases[alias] = table
        if table not in catalog:
            penalty += 2.0

    for match in _QUALIFIED_COLUMN.finditer(sql_text):
        alias = _normalise_identifier(match.group(1))
        column = _normalise_identifier(match.group(2))
        table = aliases.get(alias)
        if table is None:
            # Do not punish schema-qualified table names that are directly valid.
            if alias not in catalog:
                penalty += 1.0
            continue
        if table in catalog and column not in catalog[table]:
            penalty += 1.0

    # A generated join such as T3.ActorID = T3.ActorID is syntactically valid but
    # cannot connect two tables and occurred repeatedly in the first experiment.
    tautologies = re.findall(
        r"(?i)\b([A-Za-z_]\w*\.[A-Za-z_]\w*)\s*=\s*\1\b",
        sql_text,
    )
    penalty += 1.5 * len(tautologies)

    # Penalise obvious decoder loops while still allowing legitimate repeated columns.
    tokens = re.findall(r"[A-Za-z_]\w*|\d+|[(),.=<>*/+-]", lowered_sql)
    if len(tokens) >= 16:
        four_grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
        penalty += 0.25 * max(0, len(four_grams) - len(set(four_grams)))
    return penalty


def select_schema_valid_candidate(
    candidates: Sequence[str],
    model_scores: Sequence[float],
    schema: str | Mapping[str, set[str]],
    schema_rerank_weight: float = 0.75,
) -> tuple[str, int, list[float]]:
    """Select an n-best candidate using model score minus schema penalty."""
    if not candidates or len(candidates) != len(model_scores):
        raise ValueError("candidates and model_scores must have the same non-zero length")
    if schema_rerank_weight < 0:
        raise ValueError("schema_rerank_weight cannot be negative")

    combined_scores = [
        float(model_score)
        - schema_rerank_weight * schema_violation_penalty(candidate, schema)
        for candidate, model_score in zip(candidates, model_scores)
    ]
    best_index = max(range(len(candidates)), key=combined_scores.__getitem__)
    return candidates[best_index], best_index, combined_scores


def load_improved_model(model_name: str = BASELINE_MODEL_NAME):
    """Load a T5-family tokenizer and model for the controlled comparison."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Missing dependency: transformers. Install requirements.txt before training."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def _apply_schema_aware_format(dataframe):
    """Return a copy with the baseline prompt replaced by the improved prompt."""
    improved = dataframe.copy()
    improved["prompt"] = improved["prompt"].map(build_schema_aware_input)
    return improved


def _token_length_summary(prompts: Sequence[str], tokenizer) -> dict[str, float | int]:
    """Summarise input-token lengths and the number affected by truncation."""
    lengths: list[int] = []
    chunk_size = 256
    for start in range(0, len(prompts), chunk_size):
        encoded = tokenizer(
            list(prompts[start : start + chunk_size]),
            add_special_tokens=True,
            truncation=False,
        )
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])

    if not lengths:
        return {"count": 0, "mean": 0.0, "max": 0, "over_max_input_length": 0}
    return {
        "count": len(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
        "max": max(lengths),
        "over_max_input_length": sum(length > MAX_INPUT_LENGTH for length in lengths),
    }


def prepare_improved_dataloaders(
    tokenizer,
    batch_size: int,
    dataset_path: str = DATASET_PATH,
    canonicalize_targets: bool = True,
):
    """Create improved DataLoaders using exactly the baseline cleaning and split."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("Missing dependency: pandas. Install requirements.txt.") from exc

    from src.data_prepare import build_dataloaders, clean_dataset, split_dataset, tokenize_dataset

    clean_df = clean_dataset(pd.read_csv(dataset_path))
    train_df, val_df, test_df = split_dataset(clean_df)

    if canonicalize_targets:
        for dataframe in (train_df, val_df, test_df):
            dataframe["sql"] = dataframe["sql"].map(canonicalize_target_sql)

    raw_prompts = clean_df["prompt"].tolist()
    improved_train = _apply_schema_aware_format(train_df)
    improved_val = _apply_schema_aware_format(val_df)
    improved_test = _apply_schema_aware_format(test_df)
    improved_prompts = (
        improved_train["prompt"].tolist()
        + improved_val["prompt"].tolist()
        + improved_test["prompt"].tolist()
    )

    tokenized_dataset = tokenize_dataset(
        improved_train,
        improved_val,
        improved_test,
        tokenizer,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenized_dataset,
        tokenizer,
        batch_size=batch_size,
    )
    metadata = {
        "split_sizes": {
            "train": len(improved_train),
            "validation": len(improved_val),
            "test": len(improved_test),
        },
        "raw_prompt_tokens": _token_length_summary(raw_prompts, tokenizer),
        "improved_prompt_tokens": _token_length_summary(improved_prompts, tokenizer),
        "target_canonicalization": canonicalize_targets,
        # Used only while predicting; removed before the JSON run config is written.
        "_test_schemas": [
            extract_prompt_sections(prompt).get("schema", "")
            for prompt in test_df["prompt"].tolist()
        ],
    }
    return train_loader, val_loader, test_loader, metadata


def _decode_labels(labels, tokenizer) -> list[str]:
    """Decode padded target labels back into SQL strings."""
    labels = labels.detach().cpu().clone()
    labels[labels == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


def write_improved_predictions(
    model,
    tokenizer,
    test_loader,
    device,
    output_path: Path,
    generation_args: Mapping[str, Any],
    max_prediction_batches: int | None = None,
    schemas: Sequence[str] | None = None,
    schema_rerank_weight: float = 0.75,
) -> int:
    """Generate SQL, optionally rerank n-best beams, and return rows written."""
    import torch
    from tqdm import tqdm

    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with torch.no_grad(), output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["target_sql", "predicted_sql"])
        writer.writeheader()

        for step, batch in enumerate(tqdm(test_loader, desc="Improved predict"), start=1):
            if max_prediction_batches is not None and step > max_prediction_batches:
                break

            labels = batch.pop("labels")
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            generation_config = dict(generation_args)
            beam_count = int(generation_config.get("num_beams", 1))
            use_schema_reranking = (
                schemas is not None and beam_count > 1 and schema_rerank_weight > 0
            )
            if use_schema_reranking:
                generation_config.update(
                    {
                        "num_return_sequences": beam_count,
                        "return_dict_in_generate": True,
                        "output_scores": True,
                    }
                )

            generated_output = model.generate(**model_inputs, **generation_config)
            if use_schema_reranking:
                decoded_candidates = tokenizer.batch_decode(
                    generated_output.sequences,
                    skip_special_tokens=True,
                )
                sequence_scores = generated_output.sequences_scores.detach().cpu().tolist()
                predictions = []
                batch_size = labels.shape[0]
                for row_index in range(batch_size):
                    start = row_index * beam_count
                    end = start + beam_count
                    schema_index = rows_written + row_index
                    schema = (
                         schemas[schema_index]
                         if schemas is not None and schema_index < len(schemas)
                         else ""
                    )
                    candidates = decoded_candidates[start:end]
                    scores = sequence_scores[start:end]
                    if schema:
                        selected, _, _ = select_schema_valid_candidate(
                            candidates,
                            scores,
                            schema,
                            schema_rerank_weight,
                        )
                    else:
                        selected = candidates[0]
                    predictions.append(selected)
            else:
                generated_ids = (
                    generated_output.sequences
                    if hasattr(generated_output, "sequences")
                    else generated_output
                )
                predictions = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            targets = _decode_labels(labels, tokenizer)

            for target_sql, predicted_sql in zip(targets, predictions):
                writer.writerow({"target_sql": target_sql, "predicted_sql": predicted_sql})
                rows_written += 1
    return rows_written


def evaluate_generation_exact_match(
    model,
    tokenizer,
    data_loader,
    device,
    generation_args: Mapping[str, Any],
    max_batches: int | None = None,
) -> dict[str, int | float]:
    """Generate validation SQL and return normalised exact-match statistics."""
    import torch
    from tqdm import tqdm

    model.eval()
    correct = 0
    total = 0
    previous_use_cache = getattr(model.config, "use_cache", None)
    if previous_use_cache is not None:
        model.config.use_cache = True
    try:
        with torch.no_grad():
            for step, batch in enumerate(
                tqdm(data_loader, desc="Validation exact match"),
                start=1,
            ):
                if max_batches is not None and step > max_batches:
                    break
                labels = batch["labels"]
                model_inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key != "labels"
                }
                generated_ids = model.generate(**model_inputs, **dict(generation_args))
                predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                targets = _decode_labels(labels, tokenizer)
                correct += sum(
                    normalize_sql(prediction) == normalize_sql(target)
                    for target, prediction in zip(targets, predictions)
                )
                total += len(targets)
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def checkpoint_is_better(
    val_exact_match: float,
    val_loss: float,
    best_exact_match: float,
    best_val_loss: float,
) -> bool:
    """Prefer validation exact match, using loss only to break exact-score ties."""
    return val_exact_match > best_exact_match or (
        math.isclose(val_exact_match, best_exact_match, abs_tol=1e-12)
        and val_loss < best_val_loss
    )


def write_improved_training_log(history: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write V3 loss and validation generation metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_exact_match",
        "val_exact_correct",
        "val_exact_total",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def normalize_sql(sql: str) -> str:
    """Apply conservative formatting normalisation for exact-match comparison."""
    sql = str(sql).replace("```sql", "").replace("```", "").strip().lower()
    sql = re.sub(r";\s*$", "", sql)
    sql = re.sub(r"\s+", " ", sql)
    # Match compound comparison operators before their one-character components.
    # This makes formatting variants such as ``medal_id!=4`` and
    # ``medal_id != 4`` comparable while keeping the normalisation deterministic.
    sql = re.sub(r"\s*(<>|!=|<=|>=|[(),=<>+*/-])\s*", r"\1", sql)
    return sql.strip()


def _target_alignment_key(sql: str) -> str:
    """Return a tokenizer-tolerant key used only to verify prediction row alignment.

    T5-family tokenizers can decode label whitespace differently, including whitespace
    immediately inside a quoted literal.  Removing whitespace is too permissive for an
    accuracy metric, but it is useful as a secondary safeguard after both files have
    already been generated from the same deterministic test split.  Exact-match scores
    continue to use :func:`normalize_sql` and each file's recorded target.
    """
    return re.sub(r"\s+", "", normalize_sql(sql))


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"target_sql", "predicted_sql"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain target_sql and predicted_sql columns")
    return rows


def compare_with_baseline(
    baseline_path: str | Path = Path(RESULTS_DIR) / "baseline_predictions.csv",
    improved_path: str | Path = Path(RESULTS_DIR) / "improved_predictions.csv",
    output_path: str | Path | None = Path(RESULTS_DIR) / "improvement_comparison.json",
    baseline_label: str = "baseline",
    improved_label: str = "improved",
) -> dict[str, Any]:
    """Compare aligned baseline/improved predictions using normalised exact match.

    If one CSV contains more predictions, only the shared leading subset is compared.
    The target SQL is checked row by row to prevent a misleading comparison caused by
    different test ordering.
    """
    baseline_rows = _read_prediction_rows(Path(baseline_path))
    improved_rows = _read_prediction_rows(Path(improved_path))
    paired_count = min(len(baseline_rows), len(improved_rows))
    if paired_count == 0:
        raise ValueError("Both prediction files must contain at least one prediction")

    baseline_correct = 0
    improved_correct = 0
    baseline_only_correct = 0
    improved_only_correct = 0
    target_format_variants = 0

    for index in range(paired_count):
        baseline_target_raw = baseline_rows[index]["target_sql"]
        improved_target_raw = improved_rows[index]["target_sql"]
        baseline_target = normalize_sql(baseline_target_raw)
        improved_target = normalize_sql(improved_target_raw)
        if baseline_target != improved_target:
            if _target_alignment_key(baseline_target_raw) != _target_alignment_key(
                improved_target_raw
            ):
                raise ValueError(f"Target mismatch at prediction row {index + 1}")
            target_format_variants += 1

        baseline_match = normalize_sql(baseline_rows[index]["predicted_sql"]) == baseline_target
        improved_match = normalize_sql(improved_rows[index]["predicted_sql"]) == improved_target
        baseline_correct += int(baseline_match)
        improved_correct += int(improved_match)
        baseline_only_correct += int(baseline_match and not improved_match)
        improved_only_correct += int(improved_match and not baseline_match)

    baseline_score = baseline_correct / paired_count
    improved_score = improved_correct / paired_count
    summary = {
        "metric": "normalized_exact_match",
        "comparison": f"{baseline_label}_vs_{improved_label}",
        "paired_predictions": paired_count,
        "baseline_file_rows": len(baseline_rows),
        "improved_file_rows": len(improved_rows),
        "baseline_correct": baseline_correct,
        "improved_correct": improved_correct,
        "baseline_score": round(baseline_score, 6),
        "improved_score": round(improved_score, 6),
        "absolute_improvement": round(improved_score - baseline_score, 6),
        "percentage_point_improvement": round((improved_score - baseline_score) * 100, 4),
        "baseline_only_correct": baseline_only_correct,
        "improved_only_correct": improved_only_correct,
        "both_correct": baseline_correct - baseline_only_correct,
        "neither_correct": paired_count
        - baseline_only_correct
        - improved_only_correct
        - (baseline_correct - baseline_only_correct),
        "target_format_variants": target_format_variants,
        "target_alignment": "row order plus tokenizer-tolerant whitespace check",
        "note": "String exact match is diagnostic; execution accuracy should be added by Member 4.",
    }

    if output_path is not None:
        comparison_path = Path(output_path)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_one_epoch_improved(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_fp16: bool = True,
    max_train_batches: int | None = None,
) -> float:
    """Train one epoch with AMP, gradient accumulation, clipping, and scheduling."""
    import torch
    from tqdm import tqdm

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    model.train()
    amp_enabled = bool(use_fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    usable_steps = min(
        len(train_loader),
        max_train_batches if max_train_batches is not None else len(train_loader),
    )
    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(train_loader, total=usable_steps, desc=f"Improved epoch {epoch} train")
    for step, batch in enumerate(progress, start=1):
        if step > usable_steps:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Training loss became non-finite. T5/FLAN-T5 can be unstable "
                    "with FP16 on T4 GPUs; rerun without --fp16."
                )
            scaled_loss = loss / gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_update = step % gradient_accumulation_steps == 0 or step == usable_steps
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total_steps, 1)


def build_improved_optimizer_and_scheduler(model, train_loader, args):
    """Build AdamW and a warm-up/linear-decay schedule for the stronger run."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    usable_batches = min(
        len(train_loader),
        args.max_train_batches if args.max_train_batches is not None else len(train_loader),
    )
    updates_per_epoch = math.ceil(usable_batches / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    return optimizer, scheduler, total_updates, warmup_steps


def train_improved_model(args: argparse.Namespace) -> dict[str, Any]:
    """Fine-tune the schema-aware model, predict with beams, and save results."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("Missing dependency: torch. Install requirements.txt.") from exc

    from src.baseline import evaluate_baseline, save_baseline_model, set_seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    tokenizer, model = load_improved_model(args.model_name)
    model.to(device)
    
    if args.predict_only:
        checkpoint_path = Path("checkpoints/member3_v3_checkpoint_corrected/best")
        """
        tokenizer, model = load_improved_model(str(checkpoint_path))
        """
        tokenizer, model = load_verified_checkpoint(checkpoint_path)
    else:
        tokenizer, model = load_improved_model(args.model_name)
    model.to(device)

    train_loader, val_loader, test_loader, metadata = prepare_improved_dataloaders(
        tokenizer,
        args.batch_size,
        args.dataset_path,
        canonicalize_targets=args.canonicalize_targets,
    )
    test_schemas = metadata.pop("_test_schemas")
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    optimizer, scheduler, total_updates, warmup_steps = build_improved_optimizer_and_scheduler(
        model,
        train_loader,
        args,
    )

    model_slug = args.model_name.replace("/", "_").replace("\\", "_")
    run_dir = Path(args.checkpoint_dir) / f"improved_{model_slug}"
    best_dir = run_dir / "best"
    results_dir = Path(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    beam_generation_args = configure_generation_strategy(
        num_beams=args.num_beams,
        max_length=MAX_TARGET_LENGTH,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
    )
    greedy_generation_args = configure_generation_strategy(
        num_beams=1,
        max_length=MAX_TARGET_LENGTH,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
    )
    config = {
        "method": (
            "V3 relevance-ranked compact schema, canonical SQL targets, validation exact-match "
            "checkpointing, early stopping, and schema-constrained beam reranking"
        ),
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_optimizer_updates": total_updates,
        "max_grad_norm": args.max_grad_norm,
        "fp16": bool(args.fp16 and device.type == "cuda"),
        "gradient_checkpointing": args.gradient_checkpointing,
        "schema_rerank_weight": args.schema_rerank_weight,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_selection": "validation_normalized_exact_match_then_validation_loss",
        "seed": args.seed,
        "device": str(device),
        "dataset_path": args.dataset_path,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "greedy_generation": greedy_generation_args,
        "beam_generation": beam_generation_args,
        **metadata,
    }
    (run_dir / "improved_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
   

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_val_exact_match = -1.0
    best_epoch = 0
    epochs_without_exact_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_improved(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_fp16=args.fp16,
            max_train_batches=args.max_train_batches,
        )
        val_loss = evaluate_baseline(
            model,
            val_loader,
            device,
            max_val_batches=args.max_val_batches,
        )
        val_exact = evaluate_generation_exact_match(
            model,
            tokenizer,
            val_loader,
            device,
            greedy_generation_args,
            max_batches=args.max_val_prediction_batches,
        )
        val_exact_match = float(val_exact["accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_exact_match": val_exact_match,
                "val_exact_correct": int(val_exact["correct"]),
                "val_exact_total": int(val_exact["total"]),
            }
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_exact_match={val_exact_match:.4%} "
            f"({val_exact['correct']}/{val_exact['total']})"
        )
        previous_best_exact = best_val_exact_match
        if checkpoint_is_better(
            val_exact_match,
            val_loss,
            best_val_exact_match,
            best_val_loss,
        ):
            best_val_exact_match = val_exact_match
            best_val_loss = val_loss
            best_epoch = epoch
            save_baseline_model(model, tokenizer, best_dir)

        if val_exact_match > previous_best_exact:
            epochs_without_exact_improvement = 0
        else:
            epochs_without_exact_improvement += 1
        if (
            args.early_stopping_patience > 0
            and epochs_without_exact_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping: validation exact match did not improve for "
                f"{args.early_stopping_patience} epoch(s)."
            )
            break

    write_improved_training_log(history, results_dir / "improved_training_log.csv")
    config.update(
        {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_exact_match": best_val_exact_match,
            "best_validation_loss": best_val_loss,
        } 
    )
    (run_dir / "improved_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    if args.predict_only:
        best_dir = Path("checkpoints/member3_v3_checkpoint_corrected/best")
    

    # Reload the validation-selected checkpoint rather than assuming the final epoch is best.
    #best_tokenizer, best_model = load_improved_model(str(best_dir))
    if args.predict_only:
        best_tokenizer, best_model = tokenizer, model
    else:
        best_tokenizer, best_model = load_improved_model(str(best_dir))

    best_model.to(device)
    best_model.config.use_cache = True
    greedy_prediction_path = results_dir / "improved_greedy_predictions.csv"
    greedy_prediction_count = write_improved_predictions(
        best_model,
        best_tokenizer,
        test_loader,
        device,
        greedy_prediction_path,
        greedy_generation_args,
        max_prediction_batches=args.max_prediction_batches,
    )
    print(f"Wrote {greedy_prediction_count} greedy predictions to {greedy_prediction_path}")

    prediction_path = results_dir / "improved_predictions.csv"
    if args.num_beams == 1:
        # Generate the configured output separately to keep filenames stable for comparison.
        prediction_count = write_improved_predictions(
            best_model,
            best_tokenizer,
            test_loader,
            device,
            prediction_path,
            greedy_generation_args,
            max_prediction_batches=args.max_prediction_batches,
        )
    else:
        prediction_count = write_improved_predictions(
            best_model,
            best_tokenizer,
            test_loader,
            device,
            prediction_path,
            beam_generation_args,
            max_prediction_batches=args.max_prediction_batches,
            schemas=test_schemas,
            schema_rerank_weight=args.schema_rerank_weight,
        )
    print(f"Improved run finished. Best checkpoint: {best_dir}")
    print(f"Wrote {prediction_count} predictions to {prediction_path}")

    beam_comparison = compare_with_baseline(
        greedy_prediction_path,
        prediction_path,
        results_dir / "beam_search_comparison.json",
        baseline_label="schema_aware_greedy",
        improved_label=f"schema_aware_beam_{args.num_beams}",
    )

    baseline_path = Path(args.baseline_predictions)
    if baseline_path.exists():
        prompt_comparison = compare_with_baseline(
            baseline_path,
            greedy_prediction_path,
            results_dir / "prompt_format_comparison.json",
            baseline_label="original_prompt_greedy",
            improved_label="schema_aware_prompt_greedy",
        )
        combined_comparison = compare_with_baseline(
            baseline_path,
            prediction_path,
            results_dir / "improvement_comparison.json",
            baseline_label="original_prompt_greedy",
            improved_label=f"schema_aware_prompt_beam_{args.num_beams}",
        )
        comparisons = {
            "prompt_format": prompt_comparison,
            "beam_search": beam_comparison,
            "combined_method": combined_comparison,
        }
        print(json.dumps(comparisons, indent=2))
        return comparisons

    print(f"Baseline predictions not found at {baseline_path}; comparison was skipped.")
    return {
        "prediction_count": prediction_count,
        "best_validation_loss": best_val_loss,
        "best_validation_exact_match": best_val_exact_match,
        "best_epoch": best_epoch,
        "beam_search": beam_comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare Member 3 improvements.")
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--length_penalty", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--schema_rerank_weight", type=float, default=0.75)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable FP16 explicitly. Disabled by default because T5 can produce NaN on T4.",
    )
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_val_prediction_batches", type=int, default=None)
    parser.add_argument("--max_prediction_batches", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument(
        "--no_target_canonicalization",
        action="store_false",
        dest="canonicalize_targets",
        help="Disable conservative canonical formatting of SQL training targets.",
    )
    parser.set_defaults(canonicalize_targets=True)
    parser.add_argument("--baseline_predictions", default=str(Path(RESULTS_DIR) / "baseline_predictions.csv"))
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Compare existing baseline/improved CSV files without training.",
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Load the existing Member 3 checkpoint and generate predictions without training.",
    )
    parser.add_argument(
        "--improved_predictions",
        default=str(Path(RESULTS_DIR) / "improved_predictions.csv"),
        help="Used with --compare_only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare_only:
        summary = compare_with_baseline(
            args.baseline_predictions,
            args.improved_predictions,
            Path(args.results_dir) / "improvement_comparison.json",
        )
        print(json.dumps(summary, indent=2))
        return    

    if args.predict_only:
        args.epochs = 0

    train_improved_model(args)


if __name__ == "__main__":
    main()
