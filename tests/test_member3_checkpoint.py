"""Tests for Member 3 checkpoint delivery and inference-only interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from src import improvement
from src.member3_checkpoint import (
    REQUIRED_CHECKPOINT_FILES,
    load_verified_checkpoint,
    sha256_file,
    validate_checkpoint_dir,
)


def _write_minimal_checkpoint_files(checkpoint: Path) -> None:
    checkpoint.mkdir(parents=True)
    for filename in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint / filename
        if filename == "model.safetensors":
            save_file(
                {
                    "shared.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
                    "lm_head.weight": torch.arange(6, 12, dtype=torch.float32).reshape(3, 2),
                },
                path,
            )
        else:
            path.write_text("{}", encoding="utf-8")


class _TinyT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Embedding(3, 2)
        self.encoder = SimpleNamespace(embed_tokens=self.shared)
        self.decoder = SimpleNamespace(embed_tokens=self.shared)
        self.lm_head = torch.nn.Linear(2, 3, bias=False)
        # Simulate a Transformers version that tied the output head while loading.
        self.lm_head.weight = self.shared.weight
        self.config = SimpleNamespace(d_model=2, vocab_size=3, use_cache=False)


def test_validate_checkpoint_dir_reports_missing_files(tmp_path):
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        validate_checkpoint_dir(checkpoint)


def test_sha256_file_streams_expected_digest(tmp_path):
    payload = b"member3-checkpoint"
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_verified_loader_restores_distinct_shared_and_lm_head_weights(tmp_path):
    checkpoint = tmp_path / "best"
    _write_minimal_checkpoint_files(checkpoint)
    model = _TinyT5()
    tokenizer = object()

    with (
        patch("src.member3_checkpoint._load_tokenizer", return_value=tokenizer),
        patch(
            "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            return_value=model,
        ) as from_pretrained,
    ):
        loaded_tokenizer, loaded_model = load_verified_checkpoint(checkpoint)

    from_pretrained.assert_called_once_with(checkpoint.resolve(), local_files_only=True)
    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert loaded_model.shared.weight.data_ptr() != loaded_model.lm_head.weight.data_ptr()
    assert torch.equal(
        loaded_model.shared.weight,
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )
    assert torch.equal(
        loaded_model.lm_head.weight,
        torch.arange(6, 12, dtype=torch.float32).reshape(3, 2),
    )
    assert loaded_model.training is False


def test_predict_from_checkpoint_never_builds_optimizer_or_trains(tmp_path):
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"verified-weights")
    results_dir = tmp_path / "results"
    model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
    args = argparse.Namespace(
        checkpoint_path=str(checkpoint),
        expected_model_sha256=None,
        seed=42,
        no_cuda=True,
        batch_size=2,
        dataset_path="test-dataset.csv",
        canonicalize_targets=True,
        length_penalty=0.9,
        repetition_penalty=1.08,
        num_beams=8,
        schema_rerank_weight=0.75,
        max_prediction_batches=1,
        results_dir=str(results_dir),
    )

    with (
        patch(
            "src.improvement.load_verified_checkpoint",
            return_value=(object(), model),
        ) as load_checkpoint,
        patch(
            "src.improvement.prepare_improved_test_dataloader",
            return_value=(object(), ["CREATE TABLE users(id INTEGER);"], {"test_rows": 2}),
        ),
        patch("src.improvement.write_improved_predictions", side_effect=[2, 2]) as write,
        patch(
            "src.improvement.compare_with_baseline",
            return_value={"paired_predictions": 2},
        ),
        patch("src.improvement.build_improved_optimizer_and_scheduler") as build_optimizer,
        patch("src.improvement.train_one_epoch_improved") as train_epoch,
    ):
        summary = improvement.predict_from_checkpoint(args)

    load_checkpoint.assert_called_once()
    assert write.call_count == 2
    build_optimizer.assert_not_called()
    train_epoch.assert_not_called()
    assert summary["mode"] == "predict_only"
    assert summary["training_performed"] is False
    assert summary["prediction_count"] == 2
    manifest = json.loads(
        (results_dir / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["training_performed"] is False


def test_main_dispatches_predict_only_without_training():
    args = argparse.Namespace(compare_only=False, predict_only=True)
    with (
        patch("src.improvement.parse_args", return_value=args),
        patch("src.improvement.predict_from_checkpoint") as predict,
        patch("src.improvement.train_improved_model") as train,
    ):
        improvement.main()

    predict.assert_called_once_with(args)
    train.assert_not_called()
