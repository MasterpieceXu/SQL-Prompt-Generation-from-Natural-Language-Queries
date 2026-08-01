"""Tests for Member 3's independent SQLGlot validity metric."""

import pytest

from src.member3_sql_validity import (
    check_sqlglot_validity,
    evaluate_sqlglot_validity_rate,
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, name FROM users",
        "WITH named AS (SELECT name FROM users) SELECT name FROM named",
        "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)",
    ],
)
def test_accepts_single_sqlite_query(sql):
    assert check_sqlglot_validity(sql) == {
        "valid": True,
        "category": "valid",
        "error_message": None,
    }


@pytest.mark.parametrize(
    ("sql", "category"),
    [
        ("", "empty_sql"),
        ("SELECT FROM", "parse_error"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("DROP TABLE users", "non_query_statement"),
    ],
)
def test_rejects_invalid_or_non_query_output(sql, category):
    result = check_sqlglot_validity(sql)

    assert result["valid"] is False
    assert result["category"] == category
    assert result["error_message"]


def test_validity_rate_summary():
    result = evaluate_sqlglot_validity_rate(
        ["SELECT 1", "SELECT FROM", "DROP TABLE users", ""],
    )

    assert result["metric"] == "sqlglot_validity_rate"
    assert result["dialect"] == "sqlite"
    assert result["valid"] == 1
    assert result["invalid"] == 3
    assert result["total"] == 4
    assert result["rate"] == pytest.approx(0.25)
    assert result["category_counts"] == {
        "valid": 1,
        "parse_error": 1,
        "non_query_statement": 1,
        "empty_sql": 1,
    }
