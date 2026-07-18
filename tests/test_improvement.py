"""Unit tests for Member 3 prompt formatting and comparison utilities."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.improvement import (
    build_schema_aware_input,
    compare_with_baseline,
    configure_generation_strategy,
    extract_prompt_sections,
    normalize_sql,
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
        self.assertIn("CREATE TABLE Customers", improved)

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


class ComparisonTests(unittest.TestCase):
    def test_normalize_sql_ignores_case_spacing_and_final_semicolon(self):
        first = " SELECT COUNT ( ID ) FROM Customers; "
        second = "select count(id) from customers"
        self.assertEqual(normalize_sql(first), normalize_sql(second))

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
