"""Unit tests for Member 3 prompt formatting and comparison utilities."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.improvement import (
    build_schema_aware_input,
    canonicalize_target_sql,
    checkpoint_is_better,
    compact_schema,
    compare_with_baseline,
    configure_generation_strategy,
    extract_prompt_sections,
    normalize_sql,
    parse_schema_catalog,
    schema_violation_penalty,
    select_schema_valid_candidate,
)


SAMPLE_PROMPT = """You are a data science expert.
Below, you are presented with a database schema and a question.

Database Schema
###
CREATE TABLE Customers
(
    ID INTEGER PRIMARY KEY,
    age INTEGER
);
###
Question:
What is the total number of customers below 30?

Hint:
below 30 means age < 30;

Please respond with a SQL query between ```sql and ```.
"""


class PromptFormattingTests(unittest.TestCase):
    def test_extract_prompt_sections(self):
        sections = extract_prompt_sections(SAMPLE_PROMPT)
        self.assertIn("CREATE TABLE Customers", sections["schema"])
        self.assertNotIn("###", sections["schema"])
        self.assertEqual(
            sections["question"],
            "What is the total number of customers below 30?",
        )
        self.assertEqual(sections["hint"], "below 30 means age < 30;")

    def test_schema_aware_prompt_places_question_before_schema(self):
        improved = build_schema_aware_input({"prompt": SAMPLE_PROMPT})
        self.assertLess(improved.index("question:"), improved.index("schema:"))
        self.assertNotIn("You are a data science expert", improved)
        self.assertIn("customers(age, id)", improved)
        self.assertNotIn("CREATE TABLE", improved)

    def test_compact_schema_preserves_tables_columns_and_foreign_keys(self):
        schema = """CREATE TABLE parent
(
 id INTEGER PRIMARY KEY,
 name TEXT
);
CREATE TABLE child
(
 id INTEGER PRIMARY KEY,
 parent_id INTEGER,
 foreign key (parent_id) references parent(id)
);"""
        catalog = parse_schema_catalog(schema)
        self.assertEqual(catalog["parent"], {"id", "name"})
        self.assertEqual(catalog["child"], {"id", "parent_id"})
        compact = compact_schema(schema)
        self.assertIn("parent(id, name)", compact)
        self.assertIn("child.parent_id->parent.id", compact)

    def test_compact_schema_ranks_question_identifiers_without_dropping_tables(self):
        schema = """CREATE TABLE unrelated
(
 id INTEGER PRIMARY KEY,
 note TEXT
);
CREATE TABLE sales
(
 id INTEGER PRIMARY KEY,
 total_amount REAL
);"""
        compact = compact_schema(schema, question="What is the total amount of sales?")
        self.assertLess(compact.index("sales("), compact.index("unrelated("))
        self.assertIn("unrelated(id, note)", compact)

    def test_unknown_format_falls_back_without_dropping_text(self):
        original = "Use table A and answer how many rows it contains."
        improved = build_schema_aware_input(original)
        self.assertIn(original, improved)

    def test_generation_configuration(self):
        config = configure_generation_strategy(num_beams=4, max_length=128)
        self.assertEqual(config["num_beams"], 4)
        self.assertTrue(config["early_stopping"])
        with self.assertRaises(ValueError):
            configure_generation_strategy(num_beams=0)

    def test_target_canonicalization_preserves_literal_contents(self):
        sql = " select count ( id ) from Customers where Name = 'Van  Halen' ; "
        canonical = canonicalize_target_sql(sql)
        self.assertEqual(
            canonical,
            "SELECT COUNT(id) FROM Customers WHERE Name = 'Van  Halen'",
        )

    def test_checkpoint_selection_prefers_exact_match_then_loss(self):
        self.assertTrue(checkpoint_is_better(0.10, 0.8, 0.09, 0.2))
        self.assertTrue(checkpoint_is_better(0.10, 0.4, 0.10, 0.5))
        self.assertFalse(checkpoint_is_better(0.09, 0.1, 0.10, 0.5))

    def test_schema_reranker_rejects_invalid_column(self):
        schema = """CREATE TABLE Customers
(
 ID INTEGER PRIMARY KEY,
 age INTEGER
);"""
        valid = "SELECT T1.age FROM Customers AS T1"
        invalid = "SELECT T1.name FROM Customers AS T1"
        self.assertEqual(schema_violation_penalty(valid, schema), 0.0)
        self.assertGreater(schema_violation_penalty(invalid, schema), 0.0)
        selected, index, _ = select_schema_valid_candidate(
            [invalid, valid],
            [-0.1, -0.2],
            schema,
            schema_rerank_weight=0.75,
        )
        self.assertEqual(index, 1)
        self.assertEqual(selected, valid)


class ComparisonTests(unittest.TestCase):
    def test_normalize_sql_ignores_case_spacing_and_final_semicolon(self):
        first = " SELECT COUNT ( ID ) FROM Customers; "
        second = "select count(id) from customers"
        self.assertEqual(normalize_sql(first), normalize_sql(second))

    def test_normalize_sql_handles_compound_comparison_operator_spacing(self):
        pairs = [
            ("medal_id!=4", "medal_id != 4"),
            ("score<=10", "score <= 10"),
            ("score>=5", "score >= 5"),
            ("status<>0", "status <> 0"),
        ]
        for compact, spaced in pairs:
            with self.subTest(operator=compact):
                self.assertEqual(normalize_sql(compact), normalize_sql(spaced))

    def test_compare_with_baseline_accepts_equivalent_target_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            baseline_path = directory_path / "baseline.csv"
            improved_path = directory_path / "improved.csv"

            self._write_predictions(
                baseline_path,
                [("SELECT * FROM results WHERE medal_id!=4", "SELECT id FROM results")],
            )
            self._write_predictions(
                improved_path,
                [("SELECT * FROM results WHERE medal_id != 4", "SELECT * FROM results WHERE medal_id !=4")],
            )

            summary = compare_with_baseline(baseline_path, improved_path, output_path=None)
            self.assertEqual(summary["paired_predictions"], 1)
            self.assertEqual(summary["baseline_correct"], 0)
            self.assertEqual(summary["improved_correct"], 1)

    def test_compare_tracks_tokenizer_target_format_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            baseline_path = directory_path / "baseline.csv"
            improved_path = directory_path / "improved.csv"

            self._write_predictions(
                baseline_path,
                [("SELECT * FROM areas WHERE side = 'West'", "SELECT id FROM areas")],
            )
            self._write_predictions(
                improved_path,
                [("SELECT * FROM areas WHERE side = 'West '", "SELECT id FROM areas")],
            )

            summary = compare_with_baseline(baseline_path, improved_path, output_path=None)
            self.assertEqual(summary["paired_predictions"], 1)
            self.assertEqual(summary["target_format_variants"], 1)

    def test_compare_rejects_genuinely_different_target_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            baseline_path = directory_path / "baseline.csv"
            improved_path = directory_path / "improved.csv"

            self._write_predictions(
                baseline_path,
                [("SELECT id FROM Customers", "SELECT id FROM Customers")],
            )
            self._write_predictions(
                improved_path,
                [("SELECT name FROM Customers", "SELECT name FROM Customers")],
            )

            with self.assertRaisesRegex(ValueError, "prediction row 1"):
                compare_with_baseline(baseline_path, improved_path, output_path=None)

    def test_compare_with_baseline_reports_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            baseline_path = directory_path / "baseline.csv"
            improved_path = directory_path / "improved.csv"

            self._write_predictions(
                baseline_path,
                [("SELECT COUNT(ID) FROM Customers;", "SELECT ID FROM Customers;")],
            )
            self._write_predictions(
                improved_path,
                [("SELECT COUNT(ID) FROM Customers;", "select count(id) from customers")],
            )

            summary = compare_with_baseline(baseline_path, improved_path, output_path=None)
            self.assertEqual(summary["paired_predictions"], 1)
            self.assertEqual(summary["baseline_score"], 0.0)
            self.assertEqual(summary["improved_score"], 1.0)
            self.assertEqual(summary["percentage_point_improvement"], 100.0)

    @staticmethod
    def _write_predictions(path: Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["target_sql", "predicted_sql"])
            writer.writeheader()
            for target_sql, predicted_sql in rows:
                writer.writerow({"target_sql": target_sql, "predicted_sql": predicted_sql})


if __name__ == "__main__":
    unittest.main()
