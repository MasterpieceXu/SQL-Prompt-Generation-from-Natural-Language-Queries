"""Member 3 SQLGlot-based syntax validity metric."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlglot
from sqlglot import expressions as sqlglot_exp
from sqlglot.errors import ErrorLevel, ParseError, TokenError


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


def check_sqlglot_validity(
    predicted_sql: str | None,
    *,
    dialect: str = "sqlite",
) -> dict[str, bool | str | None]:
    """Check whether one prediction is a single SQLGlot-parseable query.

    This is a Member 3 syntax/structure metric only. It does not verify that
    referenced tables or columns exist and does not execute the query.
    """
    if predicted_sql is None or not str(predicted_sql).strip():
        return _validity_result(False, "empty_sql", "SQL is empty")

    try:
        statements = sqlglot.parse(
            str(predicted_sql),
            read=dialect,
            error_level=ErrorLevel.RAISE,
        )
    except (ParseError, TokenError, ValueError) as error:
        return _validity_result(False, "parse_error", str(error))

    expressions = [statement for statement in statements if statement is not None]
    if len(expressions) != 1:
        return _validity_result(
            False,
            "multiple_statements",
            f"Expected one SQL statement, parsed {len(expressions)}",
        )

    expression = expressions[0]
    if not isinstance(expression, sqlglot_exp.Query):
        return _validity_result(
            False,
            "non_query_statement",
            f"Expected a query, parsed {type(expression).__name__}",
        )

    return _validity_result(True, "valid")


def evaluate_sqlglot_validity_rate(
    predictions: Sequence[str | None],
    *,
    dialect: str = "sqlite",
) -> dict[str, Any]:
    """Return SQLGlot validity counts, rate, and invalid-category totals."""
    results = [
        check_sqlglot_validity(prediction, dialect=dialect)
        for prediction in predictions
    ]
    total = len(results)
    valid = sum(bool(result["valid"]) for result in results)
    category_counts: dict[str, int] = {}
    for result in results:
        category = str(result["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "metric": "sqlglot_validity_rate",
        "dialect": dialect,
        "valid": valid,
        "invalid": total - valid,
        "total": total,
        "rate": valid / total if total else 0.0,
        "category_counts": category_counts,
        "results": results,
    }
