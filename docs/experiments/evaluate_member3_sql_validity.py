"""Compute reproducible SQLGlot validity metrics for the formal predictions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.member3_sql_validity import evaluate_sqlglot_validity_rate  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "results" / "member3" / "v3_full_run"
PREDICTION_FILES = {
    "member2_baseline_greedy": REPO_ROOT / "results" / "baseline_predictions.csv",
    "member3_v2_beam_reranked": REPO_ROOT
    / "results"
    / "member3"
    / "improved_predictions.csv",
    "member3_v3_greedy": OUTPUT_DIR / "improved_greedy_predictions.csv",
    "member3_v3_beam_reranked": OUTPUT_DIR / "improved_predictions.csv",
}


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or "predicted_sql" not in rows[0]:
        raise ValueError(f"Prediction CSV lacks predicted_sql rows: {path}")
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison: dict[str, object] = {
        "metric": "sqlglot_validity_rate",
        "definition": (
            "A prediction is valid when it is non-empty, SQLGlot parses it as "
            "exactly one SQLite query, and the top-level expression is a query. "
            "The metric does not check schema names or execute SQL."
        ),
        "dialect": "sqlite",
        "systems": {},
    }

    final_rows: list[dict[str, str]] | None = None
    final_results: list[dict[str, bool | str | None]] | None = None
    systems = comparison["systems"]
    assert isinstance(systems, dict)
    for system, path in PREDICTION_FILES.items():
        rows = read_prediction_rows(path)
        result = evaluate_sqlglot_validity_rate(
            [row["predicted_sql"] for row in rows],
            dialect="sqlite",
        )
        systems[system] = {
            "file": str(path.relative_to(REPO_ROOT)),
            "valid": result["valid"],
            "invalid": result["invalid"],
            "total": result["total"],
            "rate": result["rate"],
            "percentage": 100 * float(result["rate"]),
            "category_counts": result["category_counts"],
        }
        if system == "member3_v3_beam_reranked":
            final_rows = rows
            final_results = result["results"]

    comparison_path = OUTPUT_DIR / "sqlglot_validity_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if final_rows is None or final_results is None:
        raise RuntimeError("Final V3 predictions were not evaluated")
    annotated_path = OUTPUT_DIR / "v3_sqlglot_validity.csv"
    with annotated_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "target_sql",
            "predicted_sql",
            "sqlglot_valid",
            "sqlglot_category",
            "sqlglot_error_message",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, validity in zip(final_rows, final_results):
            writer.writerow(
                {
                    "target_sql": row.get("target_sql", ""),
                    "predicted_sql": row["predicted_sql"],
                    "sqlglot_valid": validity["valid"],
                    "sqlglot_category": validity["category"],
                    "sqlglot_error_message": validity["error_message"] or "",
                }
            )

    for system, summary in systems.items():
        print(
            f"{system}: {summary['valid']}/{summary['total']} "
            f"({summary['percentage']:.2f}%)"
        )
    print(f"Saved {comparison_path.relative_to(REPO_ROOT)}")
    print(f"Saved {annotated_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
