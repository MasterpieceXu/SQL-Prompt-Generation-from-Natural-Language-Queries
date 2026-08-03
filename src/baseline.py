"""Member 2: T5-small baseline text-to-SQL model.

This module implements the first trainable baseline for the project. It uses the
shared data preparation code from ``src.data_prepare`` and trains a pretrained
T5-small model to generate SQL from the prompt text.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - helpful message for local setup
    raise SystemExit(
        "Missing dependency: transformers. Run `python -m pip install -r requirements.txt` "
        "from the project root before running the baseline."
    ) from exc

from src.config import (
    BASELINE_MODEL_NAME,
    CHECKPOINT_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data_prepare import (
    build_dataloaders,
    clean_dataset,
    load_raw_dataset,
    split_dataset,
    tokenize_dataset,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Make the baseline run reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_baseline_model(model_name: str = BASELINE_MODEL_NAME):
    """Load the baseline tokenizer and pretrained sequence-to-sequence model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def configure_baseline_training(model, learning_rate: float):
    """Define the optimizer used by the baseline model."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def prepare_baseline_dataloaders(tokenizer, batch_size: int):
    """Load, clean, split, tokenize, and build DataLoaders for baseline training."""
    raw_df = load_raw_dataset()
    clean_df = clean_dataset(raw_df)
    train_df, val_df, test_df = split_dataset(clean_df)
    tokenized_dataset = tokenize_dataset(train_df, val_df, test_df, tokenizer)
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenized_dataset,
        tokenizer,
        batch_size=batch_size,
    )
    split_sizes = {
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
    }
    return train_loader, val_loader, test_loader, split_sizes


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    """Move a tokenizer/DataLoader batch onto CPU or GPU."""
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device: torch.device,
    epoch: int,
    max_train_batches: int | None = None,
) -> float:
    """Train the baseline for one epoch and return average training loss."""
    model.train()
    total_loss = 0.0
    total_steps = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch} train")
    for step, batch in enumerate(progress, start=1):
        if max_train_batches is not None and step > max_train_batches:
            break

        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_baseline(
    model,
    val_loader,
    device: torch.device,
    max_val_batches: int | None = None,
) -> float:
    """Evaluate the baseline and return average validation loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    progress = tqdm(val_loader, desc="Validation")
    for step, batch in enumerate(progress, start=1):
        if max_val_batches is not None and step > max_val_batches:
            break

        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_steps += 1
        progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

    return total_loss / max(total_steps, 1)


def save_baseline_model(model, tokenizer, output_dir: Path) -> None:
    """Save model and tokenizer outputs into checkpoints/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def decode_labels(labels: torch.Tensor, tokenizer) -> list[str]:
    """Decode label tensors, replacing ignored loss positions with padding."""
    labels = labels.detach().cpu().clone()
    labels[labels == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


@torch.no_grad()
def write_baseline_predictions(
    model,
    tokenizer,
    test_loader,
    device: torch.device,
    output_path: Path,
    max_prediction_batches: int,
    num_beams: int,
) -> None:
    """Generate sample SQL predictions for Member 4 evaluation/error analysis."""
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["target_sql", "predicted_sql"])
        writer.writeheader()

        for step, batch in enumerate(tqdm(test_loader, desc="Predict"), start=1):
            if step > max_prediction_batches:
                break

            labels = batch["labels"]
            batch = move_batch_to_device(batch, device)
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=MAX_TARGET_LENGTH,
                num_beams=num_beams,
            )
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = decode_labels(labels, tokenizer)

            for target_sql, predicted_sql in zip(targets, predictions):
                writer.writerow(
                    {
                        "target_sql": target_sql,
                        "predicted_sql": predicted_sql,
                    }
                )


def write_training_log(history: Iterable[dict], output_path: Path) -> None:
    """Write train/validation loss history to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def train_baseline(args: argparse.Namespace) -> None:
    """Train the T5-small baseline model."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    tokenizer, model = load_baseline_model(args.model_name)
    model.to(device)
    optimizer = configure_baseline_training(model, args.learning_rate)
    train_loader, val_loader, test_loader, split_sizes = prepare_baseline_dataloaders(
        tokenizer,
        args.batch_size,
    )

    run_dir = Path(args.checkpoint_dir) / "baseline_t5_small"
    results_dir = Path(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "split_sizes": split_sizes,
        "num_beams": args.num_beams,
    }
    (run_dir / "baseline_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    history = []
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

        save_baseline_model(model, tokenizer, run_dir / f"epoch_{epoch}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_baseline_model(model, tokenizer, run_dir / "best")

    write_training_log(history, results_dir / "baseline_training_log.csv")

    # Evaluate the checkpoint selected by validation loss, not merely the final epoch.
    best_checkpoint = run_dir / "best"
    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tokenizer, model = load_baseline_model(str(best_checkpoint))
    model.to(device)

    write_baseline_predictions(
        model,
        tokenizer,
        test_loader,
        device,
        output_path=results_dir / "baseline_predictions.csv",
        max_prediction_batches=args.max_prediction_batches,
        num_beams=args.num_beams,
    )
    print(f"Baseline finished. Checkpoints: {run_dir}")
    print(f"Results: {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Member 2 T5-small baseline.")
    parser.add_argument("--model_name", default=BASELINE_MODEL_NAME)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--num_beams", type=int, default=1, help="1 means greedy decoding.")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_prediction_batches", type=int, default=10)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_baseline(parse_args())
