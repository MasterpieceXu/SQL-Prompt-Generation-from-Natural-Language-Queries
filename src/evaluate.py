"""Member 4: evaluation and error analysis."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Mapping

import torch


def raw_exact_match(target_sql: str | None, predicted_sql: str | None) -> bool:
    return target_sql == predicted_sql


def normalize_sql(sql: str | None) -> str:
    """Normalize SQL conservatively for exact string comparison."""
    if sql is None:
        return ""

    normalized = re.sub(r"\s+", " ", str(sql).strip().lower())
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def normalized_exact_match(target_sql: str | None, predicted_sql: str | None) -> bool:
    """Return whether one prediction exactly matches its target after normalization."""
    return normalize_sql(target_sql) == normalize_sql(predicted_sql)


def evaluate_normalized_exact_match(
    targets: Sequence[str | None],
    predictions: Sequence[str | None],
) -> dict[str, int | float]:
    """Evaluate normalized exact match over two aligned SQL sequences."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")

    correct = sum(
        normalized_exact_match(target, prediction)
        for target, prediction in zip(targets, predictions)
    )
    total = len(targets)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def exact_match_score(predictions, references):
    """Calculate normalized exact match for aligned predictions and references."""
    return evaluate_normalized_exact_match(references, predictions)


_SQLITE_WRITE_ACTIONS = {
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ANALYZE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_VTABLE,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_UPDATE,
}


def _validity_result(
    valid: bool,
    category: str,
    error_message: str | None = None,
) -> dict[str, bool | str | None]:
    return {
        "valid": valid,
        "category": category,
        "error_message": error_message,
    }


def _sqlite_error_category(message: str) -> str:
    lowered = message.lower()
    if "one statement at a time" in lowered:
        return "syntax_error"
    if "no such table" in lowered:
        return "missing_table"
    if "no such column" in lowered:
        return "missing_column"
    if "syntax error" in lowered or "incomplete input" in lowered:
        return "syntax_error"
    return "execution_error"


_CREATE_TABLE_START = re.compile(
    r"\bCREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+"
    r"(?:[A-Za-z_]\w*|\"(?:[^\"]|\"\")+\"|`[^`]+`|\[[^\]]+\])"
    r"(?:\s*\.\s*(?:[A-Za-z_]\w*|\"(?:[^\"]|\"\")+\"|`[^`]+`|\[[^\]]+\]))?"
    r"\s*\(",
    re.IGNORECASE,
)


def _sqlite_compatible_schema(schema_sql: str) -> str:
    code_positions = [False] * len(schema_sql)
    index = 0
    state = "code"
    while index < len(schema_sql):
        character = schema_sql[index]
        following = schema_sql[index + 1] if index + 1 < len(schema_sql) else ""

        if state == "line_comment":
            if character in "\r\n":
                state = "code"
        elif state == "block_comment":
            if character == "*" and following == "/":
                state = "code"
                index += 2
                continue
        elif state in {"single_quote", "double_quote", "backtick"}:
            delimiter = {
                "single_quote": "'",
                "double_quote": '"',
                "backtick": "`",
            }[state]
            if character == delimiter:
                if following == delimiter:
                    index += 2
                    continue
                state = "code"
        elif state == "bracket":
            if character == "]":
                state = "code"
        elif character == "-" and following == "-":
            state = "line_comment"
            index += 2
            continue
        elif character == "/" and following == "*":
            state = "block_comment"
            index += 2
            continue
        elif character in "'\"`[":
            state = {
                "'": "single_quote",
                '"': "double_quote",
                "`": "backtick",
                "[": "bracket",
            }[character]
        else:
            code_positions[index] = True
        index += 1

    if state not in {"code", "line_comment"}:
        return schema_sql

    removals: list[int] = []
    for match in _CREATE_TABLE_START.finditer(schema_sql):
        opening_index = match.end() - 1
        if not code_positions[match.start()] or not code_positions[opening_index]:
            continue

        depth = 0
        last_top_level_code: int | None = None
        closing_index: int | None = None
        for cursor in range(opening_index, len(schema_sql)):
            if not code_positions[cursor]:
                continue
            character = schema_sql[cursor]
            if character == "(":
                if depth == 1:
                    last_top_level_code = cursor
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    closing_index = cursor
                    break
                if depth == 1:
                    last_top_level_code = cursor
            elif depth == 1 and not character.isspace():
                last_top_level_code = cursor

        if closing_index is None:
            return schema_sql
        if last_top_level_code is not None and schema_sql[last_top_level_code] == ",":
            removals.append(last_top_level_code)

    if not removals:
        return schema_sql
    removal_set = set(removals)
    return "".join(
        character
        for position, character in enumerate(schema_sql)
        if position not in removal_set
    )


def check_sql_validity(
    predicted_sql: str | None,
    schema_sql: str | None = None,
    *,
    timeout_seconds: float = 1.0,
    max_sql_length: int = 1_000_000,
) -> dict[str, bool | str | None]:
    """Conservatively check whether SQLite can parse and plan one read query.

    This checks SQL validity only. It does not execute the predicted query and
    does not measure whether the query has the intended semantics.
    """
    if predicted_sql is None or not str(predicted_sql).strip():
        return _validity_result(False, "empty_sql", "SQL is empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if max_sql_length < 1:
        raise ValueError("max_sql_length must be at least 1")

    sql = str(predicted_sql)
    if len(sql) > max_sql_length:
        return _validity_result(
            False,
            "execution_error",
            f"SQL exceeds the maximum length of {max_sql_length} characters",
        )

    connection = sqlite3.connect(":memory:")
    deadline = time.monotonic() + timeout_seconds

    def stop_after_deadline() -> int:
        return int(time.monotonic() >= deadline)

    try:
        connection.set_progress_handler(stop_after_deadline, 1000)

        if schema_sql is not None:
            try:
                # Schema setup happens only in the isolated in-memory database.
                # ATTACH/DETACH are denied to prevent access to external files.
                connection.set_authorizer(
                    lambda action, _arg1, _arg2, _db, _source: (
                        sqlite3.SQLITE_DENY
                        if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}
                        else sqlite3.SQLITE_OK
                    )
                )
                connection.executescript(_sqlite_compatible_schema(str(schema_sql)))
            except sqlite3.Error as error:
                return _validity_result(False, "schema_error", str(error))

        def read_only_authorizer(
            action: int,
            _arg1: str | None,
            _arg2: str | None,
            _database: str | None,
            _source: str | None,
        ) -> int:
            if action in _SQLITE_WRITE_ACTIONS:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(read_only_authorizer)
        try:
            # EXPLAIN makes SQLite parse, resolve names, and build a plan
            # without running the predicted statement.
            connection.execute(f"EXPLAIN {sql}").fetchall()
        except (sqlite3.Error, sqlite3.Warning) as error:
            message = str(error)
            return _validity_result(
                False,
                _sqlite_error_category(message),
                message,
            )
        return _validity_result(True, "valid")
    finally:
        connection.close()


def sql_validity_check(predictions):
    """Compatibility wrapper for checking one SQL string or a sequence."""
    if isinstance(predictions, (str, type(None))):
        return check_sql_validity(predictions)
    return [check_sql_validity(prediction) for prediction in predictions]


_ERROR_CATEGORY_ORDER = (
    "exact_match",
    "empty_output",
    "syntax_error",
    "missing_table",
    "missing_column",
    "wrong_table",
    "wrong_column",
    "join_error",
    "aggregation_error",
    "filter_error",
    "grouping_error",
    "ordering_limit_error",
    "nested_query_error",
    "alias_error",
    "structurally_different",
    "unclassified",
)
_SQL_KEYWORDS = {
    "all", "and", "as", "asc", "between", "by", "case", "desc", "distinct",
    "else", "end", "exists", "from", "full", "group", "having", "in", "inner",
    "is", "join", "left", "like", "limit", "not", "null", "offset", "on", "or",
    "order", "outer", "right", "select", "then", "union", "when", "where", "with",
}
_AGGREGATE_FUNCTIONS = {"avg", "count", "group_concat", "max", "min", "sum", "total"}
_CLAUSE_BOUNDARIES = {
    "where", "group", "having", "order", "limit", "offset", "union",
}


def _sql_tokens(sql: str | None) -> list[str]:
    """Tokenize enough SQL structure for conservative deterministic comparison."""
    text = "" if sql is None else str(sql)
    # String literals are values rather than SQL structure. Mask their contents,
    # including doubled SQL quote escapes, before token inspection.
    text = re.sub(r"'(?:''|[^'])*'", " ? ", text)
    return re.findall(
        r'"(?:""|[^"])*"|`[^`]*`|\[[^\]]*\]|'
        r'[a-zA-Z_][a-zA-Z0-9_$]*|\d+(?:\.\d+)?|'
        r'<>|!=|<=|>=|[(),.*=<>+\-/]',
        text.lower(),
    )


def _identifier(token: str) -> str:
    if token[:1] == token[-1:] and token[:1] in {'"', "`"}:
        return token[1:-1].replace(token[:1] * 2, token[:1])
    if token.startswith("[") and token.endswith("]"):
        return token[1:-1]
    return token


def _table_names(tokens: Sequence[str]) -> list[str]:
    tables: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token not in {"from", "join"}:
            continue
        candidate = tokens[index + 1]
        if candidate != "(" and candidate not in _SQL_KEYWORDS:
            tables.append(_identifier(candidate))
    return tables


def _select_columns(tokens: Sequence[str]) -> set[str] | None:
    """Return simple projected identifiers, or None for a complex projection."""
    try:
        start = tokens.index("select") + 1
        end = tokens.index("from", start)
    except ValueError:
        return None
    projection = tokens[start:end]
    if any(token in {"(", ")", "*"} for token in projection):
        return None
    columns = {
        _identifier(token)
        for token in projection
        if (
            re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_$]*|".*"|`.*`|\[.*\]', token)
            and token not in _SQL_KEYWORDS
        )
    }
    # Dotted qualifiers and explicit aliases make attribution ambiguous.
    if "." in projection or "as" in projection:
        return None
    return columns


def _clause(tokens: Sequence[str], start_words: tuple[str, ...]) -> tuple[str, ...] | None:
    width = len(start_words)
    start = next(
        (
            index + width
            for index in range(len(tokens) - width + 1)
            if tuple(tokens[index:index + width]) == start_words
        ),
        None,
    )
    if start is None:
        return None
    end = len(tokens)
    for index in range(start, len(tokens)):
        if tokens[index] in _CLAUSE_BOUNDARIES:
            if index == start and tokens[index] == start_words[-1]:
                continue
            end = index
            break
    return tuple(tokens[start:end])


def _aliases(tokens: Sequence[str]) -> set[tuple[str, str]]:
    aliases: set[tuple[str, str]] = set()
    for index, token in enumerate(tokens[:-1]):
        if token == "as" and index > 0:
            alias = tokens[index + 1]
            if alias not in _SQL_KEYWORDS:
                aliases.add((_identifier(tokens[index - 1]), _identifier(alias)))
        elif token in {"from", "join"} and index + 2 < len(tokens):
            source, alias = tokens[index + 1:index + 3]
            if (
                source != "("
                and alias not in _SQL_KEYWORDS
                and alias not in {",", ")", "(", "."}
            ):
                aliases.add((_identifier(source), _identifier(alias)))
    return aliases


def analyze_prediction_error(
    target_sql: str | None,
    predicted_sql: str | None,
    validity_result: Mapping[str, Any] | None = None,
    prompt: str | None = None,
    sample_id: Any = None,
) -> dict[str, Any]:
    """Classify one SQL prediction using deterministic structural evidence.

    The result describes textual/structural differences only; it does not claim
    that two differently written queries are or are not semantically equivalent.
    """
    del prompt  # Accepted for record-oriented callers; deliberately not interpreted.
    if validity_result is not None and not isinstance(validity_result, Mapping):
        raise ValueError("validity_result must be a mapping or None")

    validity = dict(validity_result) if validity_result is not None else None
    validity_category = validity.get("category") if validity is not None else None
    error_message = validity.get("error_message") if validity is not None else None
    exact_match = normalized_exact_match(target_sql, predicted_sql)
    categories: set[str] = set()

    if exact_match:
        categories.add("exact_match")
    elif not normalize_sql(predicted_sql):
        categories.add("empty_output")
    elif validity_category in {
        "empty_sql", "syntax_error", "missing_table", "missing_column",
    }:
        categories.add(
            {
                "empty_sql": "empty_output",
                "syntax_error": "syntax_error",
                "missing_table": "missing_table",
                "missing_column": "missing_column",
            }[validity_category]
        )
    else:
        target_tokens = _sql_tokens(target_sql)
        predicted_tokens = _sql_tokens(predicted_sql)
        target_tables = _table_names(target_tokens)
        predicted_tables = _table_names(predicted_tokens)

        if (
            target_tables
            and predicted_tables
            and set(target_tables) != set(predicted_tables)
        ):
            categories.add("wrong_table")

        target_columns = _select_columns(target_tokens)
        predicted_columns = _select_columns(predicted_tokens)
        if (
            set(target_tables) == set(predicted_tables)
            and target_columns is not None
            and predicted_columns is not None
            and target_columns != predicted_columns
        ):
            categories.add("wrong_column")

        if (
            target_tokens.count("join") != predicted_tokens.count("join")
            or (
                "join" in target_tokens
                and "join" in predicted_tokens
                and _clause(target_tokens, ("on",))
                != _clause(predicted_tokens, ("on",))
            )
        ):
            categories.add("join_error")

        target_aggregates = [
            token for index, token in enumerate(target_tokens[:-1])
            if token in _AGGREGATE_FUNCTIONS and target_tokens[index + 1] == "("
        ]
        predicted_aggregates = [
            token for index, token in enumerate(predicted_tokens[:-1])
            if token in _AGGREGATE_FUNCTIONS and predicted_tokens[index + 1] == "("
        ]
        if (
            target_aggregates != predicted_aggregates
            or target_tokens.count("distinct") != predicted_tokens.count("distinct")
        ):
            categories.add("aggregation_error")

        if _clause(target_tokens, ("where",)) != _clause(
            predicted_tokens, ("where",)
        ):
            categories.add("filter_error")
        if (
            _clause(target_tokens, ("group", "by"))
            != _clause(predicted_tokens, ("group", "by"))
            or _clause(target_tokens, ("having",))
            != _clause(predicted_tokens, ("having",))
        ):
            categories.add("grouping_error")
        if any(
            _clause(target_tokens, words) != _clause(predicted_tokens, words)
            for words in (("order", "by"), ("limit",), ("offset",))
        ):
            categories.add("ordering_limit_error")
        if target_tokens.count("select") != predicted_tokens.count("select"):
            categories.add("nested_query_error")
        if _aliases(target_tokens) != _aliases(predicted_tokens):
            categories.add("alias_error")

        if not categories:
            categories.add(
                "structurally_different"
                if target_tokens != predicted_tokens
                else "unclassified"
            )

    ordered_categories = [
        category for category in _ERROR_CATEGORY_ORDER if category in categories
    ]
    return {
        "sample_id": sample_id,
        "exact_match": exact_match,
        "categories": ordered_categories,
        "primary_category": ordered_categories[0],
        "target_sql": target_sql,
        "predicted_sql": predicted_sql,
        "validity": validity,
        "error_message": error_message,
    }


def summarize_error_analysis(
    records: Sequence[Mapping[str, Any]],
    max_examples_per_category: int = 3,
) -> dict[str, Any]:
    """Aggregate prediction or analysis records into a deterministic summary."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of mappings")
    if not isinstance(max_examples_per_category, int) or max_examples_per_category < 0:
        raise ValueError("max_examples_per_category must be a non-negative integer")

    analyses: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record at index {index} must be a mapping")
        if "error_analysis" in record:
            record = record["error_analysis"]
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"error_analysis at record index {index} must be a mapping"
                )
        if "categories" in record:
            required = {
                "sample_id", "exact_match", "categories", "primary_category",
                "target_sql", "predicted_sql", "validity", "error_message",
            }
            missing = sorted(required.difference(record))
            if missing:
                raise ValueError(
                    f"analysis record at index {index} is missing: {', '.join(missing)}"
                )
            categories = record["categories"]
            if (
                isinstance(categories, (str, bytes))
                or not isinstance(categories, Sequence)
                or not categories
                or any(category not in _ERROR_CATEGORY_ORDER for category in categories)
            ):
                raise ValueError(f"analysis record at index {index} has invalid categories")
            if record["primary_category"] not in categories:
                raise ValueError(
                    f"analysis record at index {index} has invalid primary_category"
                )
            analyses.append(dict(record))
        else:
            missing = [
                field for field in ("target_sql", "predicted_sql")
                if field not in record
            ]
            if missing:
                raise ValueError(
                    f"prediction record at index {index} is missing: {', '.join(missing)}"
                )
            validity_result = record.get("validity_result", record.get("validity"))
            analyses.append(
                analyze_prediction_error(
                    record["target_sql"],
                    record["predicted_sql"],
                    validity_result=validity_result,
                    prompt=record.get("prompt"),
                    sample_id=record.get("sample_id", record.get("index")),
                )
            )

    total = len(analyses)
    category_counts = {category: 0 for category in _ERROR_CATEGORY_ORDER}
    primary_counts = {category: 0 for category in _ERROR_CATEGORY_ORDER}
    examples = {category: [] for category in _ERROR_CATEGORY_ORDER}
    for analysis in analyses:
        primary_counts[analysis["primary_category"]] += 1
        for category in analysis["categories"]:
            category_counts[category] += 1
            if len(examples[category]) < max_examples_per_category:
                examples[category].append(
                    {
                        "sample_id": analysis["sample_id"],
                        "target_sql": analysis["target_sql"],
                        "predicted_sql": analysis["predicted_sql"],
                        "error_message": analysis["error_message"],
                    }
                )

    exact_matches = sum(bool(analysis["exact_match"]) for analysis in analyses)
    return {
        "total": total,
        "exact_matches": exact_matches,
        "errors": total - exact_matches,
        "category_counts": category_counts,
        "category_rates": {
            category: count / total if total else 0.0
            for category, count in category_counts.items()
        },
        "primary_category_counts": primary_counts,
        "representative_examples": examples,
    }


def _batch_values(value: Any, count: int, field_name: str) -> list[Any]:
    """Return a batch metadata value as a list aligned to generated examples."""
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
        if value.ndim == 0:
            values = [values]
    elif isinstance(value, (str, bytes)) or value is None:
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]

    if len(values) != count:
        raise ValueError(
            f"{field_name} count ({len(values)}) does not match "
            f"prediction count ({count})"
        )
    return values


def _schema_values(batch: Mapping[str, Any], count: int) -> list[str | None]:
    """Return shared or per-example schema SQL, preferring ``schema_sql``."""
    values: list[Any] = [None] * count
    for field_name in ("schema", "schema_sql"):
        if field_name not in batch:
            continue
        value = batch[field_name]
        if isinstance(value, (str, bytes)) or value is None:
            field_values = [value] * count
        else:
            field_values = _batch_values(value, count, field_name)
        values = [
            fallback if preferred is None else preferred
            for fallback, preferred in zip(values, field_values)
        ]
    return [
        None if value is None else str(value)
        for value in values
    ]


def _decode_labels(labels: Any, tokenizer) -> list[str]:
    """Decode labels without changing the tensor held by the input batch."""
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    safe_labels = labels.detach().cpu().clone()
    safe_labels[safe_labels == -100] = tokenizer.pad_token_id
    return list(tokenizer.batch_decode(safe_labels, skip_special_tokens=True))


def _validate_existing_jsonl(path: Path) -> None:
    """Refuse to overwrite an existing file that is not valid object JSONL."""
    if not path.exists():
        return
    if not path.is_file():
        raise ValueError(f"Output path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    raise ValueError(f"blank line at line {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSON at line {line_number}")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"Refusing to overwrite malformed JSONL output: {path}"
        ) from error


def evaluate_model(
    model,
    data_loader,
    tokenizer,
    device,
    generation_kwargs: Mapping[str, Any] | None = None,
    max_batches: int | None = None,
    output_path: str | Path | None = None,
    include_validity: bool = False,
    include_error_analysis: bool = False,
) -> dict[str, Any]:
    """Generate SQL and optionally add validity and error-analysis details."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1")

    generation_options = dict(generation_kwargs or {})
    records: list[dict[str, Any]] = []
    batch_count = 0
    example_index = 0
    inference_seconds = 0.0
    metadata_fields = {"labels", "sample_id", "prompt", "target_sql"}
    if include_validity or include_error_analysis:
        metadata_fields.update({"schema_sql", "schema"})
    started_at = time.perf_counter()

    device = torch.device(device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch_number, batch in enumerate(data_loader):
            if max_batches is not None and batch_number >= max_batches:
                break
            batch_count += 1

            if "labels" not in batch and "target_sql" not in batch:
                raise ValueError("Each batch must contain labels or target_sql")

            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor) and key not in metadata_fields
            }
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started_at = time.perf_counter()
            generated_ids = model.generate(
                **model_inputs,
                **generation_options,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_started_at
            decoded_ids = (
                generated_ids.detach().cpu()
                if isinstance(generated_ids, torch.Tensor)
                else generated_ids
            )
            predictions = list(
                tokenizer.batch_decode(
                    decoded_ids,
                    skip_special_tokens=True,
                )
            )
            del decoded_ids, generated_ids, model_inputs

            if "target_sql" in batch:
                targets = _batch_values(
                    batch["target_sql"],
                    len(predictions),
                    "target_sql",
                )
                targets = ["" if target is None else str(target) for target in targets]
            else:
                targets = _decode_labels(batch["labels"], tokenizer)
                if len(targets) != len(predictions):
                    raise ValueError(
                        f"target count ({len(targets)}) does not match "
                        f"prediction count ({len(predictions)})"
                    )

            metadata: dict[str, list[Any]] = {}
            for field_name in ("sample_id", "prompt"):
                if field_name in batch:
                    metadata[field_name] = _batch_values(
                        batch[field_name],
                        len(predictions),
                        field_name,
                    )
            schemas = (
                _schema_values(batch, len(predictions))
                if include_validity or include_error_analysis
                else [None] * len(predictions)
            )

            for offset, (target_sql, predicted_sql) in enumerate(
                zip(targets, predictions)
            ):
                normalized_target = normalize_sql(target_sql)
                normalized_prediction = normalize_sql(predicted_sql)
                record: dict[str, Any] = {
                    "index": example_index,
                    "target_sql": target_sql,
                    "predicted_sql": predicted_sql,
                    "normalized_target_sql": normalized_target,
                    "normalized_predicted_sql": normalized_prediction,
                    "raw_exact_match": raw_exact_match(target_sql, predicted_sql),
                    "exact_match": normalized_target == normalized_prediction,
                    "normalized_exact_match": normalized_target == normalized_prediction,
                }
                for field_name, values in metadata.items():
                    record[field_name] = values[offset]

                validity_result = None
                if include_validity or include_error_analysis:
                    validity_result = check_sql_validity(
                        predicted_sql,
                        schemas[offset],
                    )
                if include_validity:
                    record["validity_result"] = validity_result
                    record["sql_valid"] = validity_result["valid"]
                    record["validity_category"] = validity_result["category"]
                    record["validity_error_message"] = validity_result["error_message"]
                if include_error_analysis:
                    analysis = analyze_prediction_error(
                        target_sql,
                        predicted_sql,
                        validity_result=validity_result,
                        prompt=record.get("prompt"),
                        sample_id=record.get("sample_id", example_index),
                    )
                    record["error_categories"] = analysis["categories"]
                    record["primary_error_category"] = analysis["primary_category"]
                    record["error_analysis"] = analysis
                records.append(record)
                example_index += 1

    if batch_count == 0:
        raise ValueError("data_loader produced no batches")

    elapsed_seconds = time.perf_counter() - started_at
    correct = sum(record["exact_match"] for record in records)
    raw_correct = sum(record["raw_exact_match"] for record in records)
    total = len(records)
    result = {
        "records": records,
        "raw_correct": raw_correct,
        "raw_accuracy": raw_correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "normalized_correct": correct,
        "normalized_accuracy": correct / total if total else 0.0,
        "inference_seconds": inference_seconds,
        "average_latency_seconds": inference_seconds / total if total else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": total / elapsed_seconds if elapsed_seconds else 0.0,
        "generation_kwargs": generation_options,
    }
    if include_validity:
        valid_count = sum(bool(record["sql_valid"]) for record in records)
        validity_categories: dict[str, int] = {}
        for record in records:
            category = str(record["validity_category"])
            validity_categories[category] = validity_categories.get(category, 0) + 1
        result.update(
            {
                "sql_valid_count": valid_count,
                "sql_validity_rate": valid_count / total if total else 0.0,
                "validity_category_counts": validity_categories,
            }
        )
    if include_error_analysis:
        result["error_summary"] = summarize_error_analysis(records)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _validate_existing_jsonl(path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                for record in records:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    return result


def error_analysis(predictions, references):
    """Analyze common generation errors."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    return summarize_error_analysis(
        [
            {"target_sql": target, "predicted_sql": prediction}
            for prediction, target in zip(predictions, references)
        ]
    )


def validate_checkpoint_directory(path: str | Path, label: str) -> Path:
    checkpoint = Path(path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {checkpoint}")
    if not checkpoint.is_dir():
        raise ValueError(f"{label} checkpoint is not a directory: {checkpoint}")

    missing: list[str] = []
    if not (checkpoint / "config.json").is_file():
        missing.append("config.json")
    weight_files = (
        list(checkpoint.glob("*.safetensors"))
        + list(checkpoint.glob("pytorch_model*.bin"))
    )
    if not weight_files:
        missing.append("model weights (*.safetensors or pytorch_model*.bin)")
    tokenizer_files = [
        checkpoint / "tokenizer.json",
        checkpoint / "spiece.model",
        checkpoint / "sentencepiece.bpe.model",
    ]
    if not (checkpoint / "tokenizer_config.json").is_file() or not any(
        candidate.is_file() for candidate in tokenizer_files
    ):
        missing.append("local tokenizer files")
    if missing:
        raise ValueError(
            f"{label} checkpoint is incomplete ({checkpoint}): "
            + ", ".join(missing)
        )
    return checkpoint.resolve()


def prepare_ordered_test_examples(max_samples: int | None = None) -> list[dict[str, Any]]:
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if max_samples is not None and max_samples > 5:
        raise ValueError("max_samples is smoke-test-only and cannot exceed 5")

    from src.data_prepare import clean_dataset, load_raw_dataset, split_dataset
    from src.improvement import extract_prompt_sections

    clean = clean_dataset(load_raw_dataset())
    _train, _validation, test = split_dataset(clean)
    if max_samples is not None:
        test = test.iloc[:max_samples]

    examples: list[dict[str, Any]] = []
    for sample_id, row in enumerate(test.itertuples(index=False)):
        prompt = str(row.prompt)
        examples.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "target_sql": str(row.sql),
                "schema_sql": extract_prompt_sections(prompt).get("schema", ""),
            }
        )
    if not examples:
        raise ValueError("The unified test split contains no examples")
    return examples


def _iter_evaluation_batches(
    examples: Sequence[Mapping[str, Any]],
    tokenizer,
    batch_size: int,
    *,
    member3_inputs: bool = False,
) -> Iterator[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    from src.config import MAX_INPUT_LENGTH

    if member3_inputs:
        from src.improvement import build_schema_aware_input

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [str(example["prompt"]) for example in chunk]
        if member3_inputs:
            prompts = [build_schema_aware_input(prompt) for prompt in prompts]
        encoded = tokenizer(
            prompts,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        batch = dict(encoded)
        batch.update(
            {
                "sample_id": [example["sample_id"] for example in chunk],
                "target_sql": [str(example["target_sql"]) for example in chunk],
                "schema_sql": [str(example.get("schema_sql", "")) for example in chunk],
            }
        )
        yield batch


def build_evaluation_batches(
    examples: Sequence[Mapping[str, Any]],
    tokenizer,
    batch_size: int,
    *,
    member3_inputs: bool = False,
) -> list[dict[str, Any]]:
    return list(
        _iter_evaluation_batches(
            examples,
            tokenizer,
            batch_size,
            member3_inputs=member3_inputs,
        )
    )


def verify_aligned_results(
    member2_result: Mapping[str, Any],
    member3_result: Mapping[str, Any],
) -> None:
    left = member2_result.get("records", [])
    right = member3_result.get("records", [])
    if len(left) != len(right):
        raise ValueError(
            "Member 2 and Member 3 prediction counts differ: "
            f"{len(left)} != {len(right)}"
        )
    for index, (member2_record, member3_record) in enumerate(zip(left, right)):
        for field in ("sample_id", "target_sql"):
            if member2_record.get(field) != member3_record.get(field):
                raise ValueError(
                    f"Evaluation alignment failure at index {index}: {field} differs"
                )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def _evaluate_local_checkpoint(
    checkpoint: Path,
    examples: Sequence[Mapping[str, Any]],
    *,
    member3: bool,
    batch_size: int,
    device: torch.device,
    generation_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if member3:
        from src.improvement import load_improved_model

        tokenizer, model = load_improved_model(str(checkpoint))
    else:
        from src.baseline import load_baseline_model

        tokenizer, model = load_baseline_model(str(checkpoint))

    try:
        batches = _iter_evaluation_batches(
            examples,
            tokenizer,
            batch_size,
            member3_inputs=member3,
        )
        return evaluate_model(
            model,
            batches,
            tokenizer,
            device,
            generation_kwargs=generation_kwargs,
            include_validity=True,
            include_error_analysis=True,
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _print_side_by_side(
    member2: Mapping[str, Any],
    member3: Mapping[str, Any],
) -> None:
    print("\nUnified Member 2 vs Member 3 evaluation")
    print(f"{'Metric':<28} {'Member 2':>16} {'Member 3':>16}")
    print("-" * 62)
    rows = (
        ("Examples", member2["total"], member3["total"]),
        ("Raw Exact Match", f"{member2['raw_correct']}/{member2['total']} ({member2['raw_accuracy']:.4%})", f"{member3['raw_correct']}/{member3['total']} ({member3['raw_accuracy']:.4%})"),
        ("Normalized Exact Match", f"{member2['normalized_correct']}/{member2['total']} ({member2['normalized_accuracy']:.4%})", f"{member3['normalized_correct']}/{member3['total']} ({member3['normalized_accuracy']:.4%})"),
        ("SQL Validity Rate", f"{member2['sql_valid_count']}/{member2['total']} ({member2['sql_validity_rate']:.4%})", f"{member3['sql_valid_count']}/{member3['total']} ({member3['sql_validity_rate']:.4%})"),
        ("Inference time (s)", f"{member2['inference_seconds']:.4f}", f"{member3['inference_seconds']:.4f}"),
        ("Average latency (s)", f"{member2['average_latency_seconds']:.6f}", f"{member3['average_latency_seconds']:.6f}"),
    )
    for name, left, right in rows:
        print(f"{name:<28} {str(left):>16} {str(right):>16}")
    print("Execution Accuracy: unavailable (no populated databases or test-to-database mapping).")
    for label, result in (("Member 2", member2), ("Member 3", member3)):
        compact_errors = {
            category: count
            for category, count in result["error_summary"]["primary_category_counts"].items()
            if count and category != "exact_match"
        }
        print(f"{label} primary error counts: {compact_errors}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the aligned, in-memory Member 2 versus Member 3 evaluation."
    )
    parser.add_argument("--member2-checkpoint", required=True)
    parser.add_argument("--member3-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def run_unified_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if args.max_samples is not None and args.max_samples > 5:
        raise ValueError("max_samples is smoke-test-only and cannot exceed 5")
    if args.num_beams < 1:
        raise ValueError("num_beams must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    member2_checkpoint = validate_checkpoint_directory(
        args.member2_checkpoint, "Member 2"
    )
    member3_checkpoint = validate_checkpoint_directory(
        args.member3_checkpoint, "Member 3"
    )
    device = _resolve_device(args.device)

    from src.baseline import set_seed

    set_seed(args.seed)
    examples = prepare_ordered_test_examples(args.max_samples)
    print(f"Member 2 checkpoint: {member2_checkpoint}")
    print(f"Member 3 checkpoint: {member3_checkpoint}")
    print(
        "Evaluation configuration: "
        f"examples={len(examples)}, batch_size={args.batch_size}, seed={args.seed}, "
        f"device={device}, num_beams={args.num_beams}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    generation_kwargs = {
        "do_sample": False,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
    }
    member2 = _evaluate_local_checkpoint(
        member2_checkpoint,
        examples,
        member3=False,
        batch_size=args.batch_size,
        device=device,
        generation_kwargs=generation_kwargs,
    )
    member3 = _evaluate_local_checkpoint(
        member3_checkpoint,
        examples,
        member3=True,
        batch_size=args.batch_size,
        device=device,
        generation_kwargs=generation_kwargs,
    )
    verify_aligned_results(member2, member3)
    _print_side_by_side(member2, member3)
    return {
        "member2": member2,
        "member3": member3,
        "execution_accuracy": None,
        "execution_accuracy_unavailable_reason": (
            "Populated databases and a test-example-to-database mapping are unavailable."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_unified_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
