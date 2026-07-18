"""Member 3: schema-aware prompting and beam-search improvements.

The baseline receives a long, generic instruction followed by the database schema and
question.  This module makes the task-specific information more explicit and places the
question before the schema so that it is less likely to disappear when a long input is
truncated.  It keeps the same cleaned data split, T5 family, optimiser, and training
loop as the baseline, then uses beam search during generation.
"""

from __future__ import annotations

import argparse
import csv
import json
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


_SCHEMA_HEADING = re.compile(r"(?im)^\s*Database\s+Schema\s*$")
_QUESTION_HEADING = re.compile(r"(?im)^\s*Question\s*:\s*")
_HINT_HEADING = re.compile(r"(?im)^\s*Hint\s*:\s*")
_RESPONSE_INSTRUCTION = re.compile(r"(?im)^\s*Please\s+respond\b")


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
    """Build a concise question-first input while preserving schema information.

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
        "translate natural language to SQLite:",
        f"question: {sections['question']}",
    ]
    if sections["hint"]:
        parts.append(f"hint: {sections['hint']}")
    parts.extend(["schema:", sections["schema"]])
    return "\n".join(parts)


def configure_generation_strategy(
    num_beams: int = 4,
    max_length: int = MAX_TARGET_LENGTH,
    length_penalty: float = 1.0,
) -> dict[str, Any]:
    """Return deterministic generation arguments for the improved model."""
    if num_beams < 1:
        raise ValueError("num_beams must be at least 1")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    return {
        "max_length": max_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "early_stopping": num_beams > 1,
    }


def load_improved_model(model_name: str = BASELINE_MODEL_NAME):
    """Load a T5-family tokenizer and model for the controlled comparison."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Missing dependency: transformers. Install requirements.txt before training."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
):
    """Create improved DataLoaders using exactly the baseline cleaning and split."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("Missing dependency: pandas. Install requirements.txt.") from exc

    from src.data_prepare import build_dataloaders, clean_dataset, split_dataset, tokenize_dataset

    clean_df = clean_dataset(pd.read_csv(dataset_path))
    train_df, val_df, test_df = split_dataset(clean_df)

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
) -> int:
    """Generate improved SQL predictions and return the number written."""
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
            generated_ids = model.generate(**model_inputs, **dict(generation_args))
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = _decode_labels(labels, tokenizer)

            for target_sql, predicted_sql in zip(targets, predictions):
                writer.writerow({"target_sql": target_sql, "predicted_sql": predicted_sql})
                rows_written += 1
    return rows_written


def normalize_sql(sql: str) -> str:
    """Apply conservative formatting normalisation for exact-match comparison."""
    sql = str(sql).replace("```sql", "").replace("```", "").strip().lower()
    sql = re.sub(r";\s*$", "", sql)
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([(),=<>+*/-])\s*", r"\1", sql)
    return sql.strip()


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

    for index in range(paired_count):
        baseline_target = normalize_sql(baseline_rows[index]["target_sql"])
        improved_target = normalize_sql(improved_rows[index]["target_sql"])
        if baseline_target != improved_target:
            raise ValueError(f"Target mismatch at prediction row {index + 1}")

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
        "note": "String exact match is diagnostic; execution accuracy should be added by Member 4.",
    }

    if output_path is not None:
        comparison_path = Path(output_path)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_improved_model(args: argparse.Namespace) -> dict[str, Any]:
    """Fine-tune the schema-aware model, predict with beams, and save results."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("Missing dependency: torch. Install requirements.txt.") from exc

    from src.baseline import (
        configure_baseline_training,
        evaluate_baseline,
        save_baseline_model,
        set_seed,
        train_one_epoch,
        write_training_log,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tokenizer, model = load_improved_model(args.model_name)
    model.to(device)
    optimizer = configure_baseline_training(model, args.learning_rate)
    train_loader, val_loader, test_loader, metadata = prepare_improved_dataloaders(
        tokenizer,
        args.batch_size,
        args.dataset_path,
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
    )
    greedy_generation_args = configure_generation_strategy(
        num_beams=1,
        max_length=MAX_TARGET_LENGTH,
        length_penalty=args.length_penalty,
    )
    config = {
        "method": "question-first schema-aware prompt plus beam search",
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
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
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            max_train_batches=args.max_train_batches,
        )
        val_loss = evaluate_baseline(
            model,
            val_loader,
            device,
            max_val_batches=args.max_val_batches,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_baseline_model(model, tokenizer, best_dir)

    write_training_log(history, results_dir / "improved_training_log.csv")

    # Reload the validation-selected checkpoint rather than assuming the final epoch is best.
    best_tokenizer, best_model = load_improved_model(str(best_dir))
    best_model.to(device)
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
        "beam_search": beam_comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare Member 3 improvements.")
    parser.add_argument("--model_name", default=BASELINE_MODEL_NAME)
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_prediction_batches", type=int, default=None)
    parser.add_argument("--baseline_predictions", default=str(Path(RESULTS_DIR) / "baseline_predictions.csv"))
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="Compare existing baseline/improved CSV files without training.",
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
    train_improved_model(args)


if __name__ == "__main__":
    main()
