"""End-to-end integration smoke tests for Member 4's local CPU workflow."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from src.evaluate import evaluate_model
from src.train import load_checkpoint, train_model


SCHEMA_SQL = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"


class TinyLocalModel(torch.nn.Module):
    """A trainable local model with the small interface Member 4 requires."""

    def __init__(self, initial_weight=0.25):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(initial_weight)))
        self.forward_grad_modes = []
        self.generation_calls = 0

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.forward_grad_modes.append(torch.is_grad_enabled())
        prediction = input_ids.float() * self.weight
        loss = torch.nn.functional.mse_loss(prediction, labels.float())
        return SimpleNamespace(loss=loss)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.generation_calls += 1
        # Generation stays deterministic so this smoke test tests integration,
        # rather than depending on convergence of a tiny synthetic model.
        return input_ids[:, :1].detach().clone()


class TinyLocalTokenizer:
    """Minimal tokenizer stub; it performs no downloads or network access."""

    sql_by_token = {
        0: "",
        1: "SELECT 1",
        2: "SELECT id FROM users",
        3: "SELECT FROM users",
    }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        rows = torch.as_tensor(token_ids).detach().cpu().tolist()
        return [self.sql_by_token[int(row[0])] for row in rows]


def _training_batches(count):
    return [
        {
            "input_ids": torch.tensor([[1.0], [2.0]]),
            "attention_mask": torch.ones((2, 1), dtype=torch.long),
            "labels": torch.tensor([[1.0], [2.0]]),
        }
        for _ in range(count)
    ]


def _evaluation_batches(count=2):
    tokens = (1, 3, 2)
    batches = []
    for index in range(count):
        token = tokens[index]
        batches.append(
            {
                "input_ids": torch.tensor([[token]], dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "target_sql": [
                    "SELECT 1" if index == 0 else "SELECT id FROM users"
                ],
                "sample_id": [f"sample-{index}"],
                "schema_sql": [SCHEMA_SQL],
            }
        )
    return batches


def _assert_batches_unchanged(batches, snapshots):
    assert len(batches) == len(snapshots)
    for batch, snapshot in zip(batches, snapshots):
        assert batch.keys() == snapshot.keys()
        for key, original_value in snapshot.items():
            if torch.is_tensor(original_value):
                assert torch.equal(batch[key], original_value)
            else:
                assert batch[key] == original_value


def test_member4_complete_cpu_training_checkpoint_and_evaluation(tmp_path):
    torch.manual_seed(4)
    checkpoint_dir = tmp_path / "checkpoints"
    output_path = tmp_path / "results" / "predictions.jsonl"
    train_batches = _training_batches(2)
    validation_batches = _training_batches(1)
    evaluation_batches = _evaluation_batches()
    train_snapshot = deepcopy(train_batches)
    validation_snapshot = deepcopy(validation_batches)
    evaluation_snapshot = deepcopy(evaluation_batches)

    model = TinyLocalModel()
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training = train_model(
        model=model,
        train_loader=train_batches,
        valid_loader=validation_batches,
        optimizer=optimizer,
        num_epochs=2,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        run_config={"workflow": "member4-integration", "device": "cpu"},
    )

    assert len(training["history"]) == 2
    assert training["global_step"] == 4
    assert any(
        not torch.equal(parameter, initial_parameters[name])
        for name, parameter in model.named_parameters()
    )
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    assert best_path.is_file()
    assert latest_path.is_file()

    fresh_model = TinyLocalModel(initial_weight=-7.0)
    metadata = load_checkpoint(fresh_model, latest_path, map_location="cpu")

    assert metadata["epoch"] == 1
    assert metadata["global_step"] == training["global_step"]
    assert metadata["best_score"] == pytest.approx(training["best_score"])
    assert metadata["history"] == training["history"]
    assert metadata["run_config"] == {
        "workflow": "member4-integration",
        "device": "cpu",
    }
    for saved_parameter, loaded_parameter in zip(
        model.parameters(), fresh_model.parameters()
    ):
        assert torch.equal(saved_parameter, loaded_parameter)

    evaluation = evaluate_model(
        model=fresh_model,
        data_loader=evaluation_batches,
        tokenizer=TinyLocalTokenizer(),
        device="cpu",
        include_validity=True,
        include_error_analysis=True,
        output_path=output_path,
    )

    assert evaluation["records"]
    assert evaluation["correct"] == 1
    assert evaluation["total"] == 2
    assert evaluation["accuracy"] == pytest.approx(0.5)
    assert "error_summary" in evaluation
    assert all("validity_result" in record for record in evaluation["records"])
    assert all("sql_valid" in record for record in evaluation["records"])
    assert output_path.is_file()
    jsonl_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert jsonl_rows == evaluation["records"]

    _assert_batches_unchanged(train_batches, train_snapshot)
    _assert_batches_unchanged(validation_batches, validation_snapshot)
    _assert_batches_unchanged(evaluation_batches, evaluation_snapshot)


def test_member4_resume_continues_epoch_history_and_global_step(tmp_path):
    first_checkpoint_dir = tmp_path / "first-run"
    resumed_checkpoint_dir = tmp_path / "resumed-run"
    train_batches = _training_batches(2)
    validation_batches = _training_batches(1)

    first_model = TinyLocalModel()
    first_optimizer = torch.optim.SGD(first_model.parameters(), lr=0.1)
    first_run = train_model(
        model=first_model,
        train_loader=train_batches,
        valid_loader=validation_batches,
        optimizer=first_optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=first_checkpoint_dir,
    )
    assert [entry["epoch"] for entry in first_run["history"]] == [0]
    assert first_run["global_step"] == 2

    resumed_model = TinyLocalModel(initial_weight=-3.0)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    metadata = load_checkpoint(
        resumed_model,
        first_checkpoint_dir / "latest.pt",
        optimizer=resumed_optimizer,
        map_location="cpu",
    )
    resumed_run = train_model(
        model=resumed_model,
        train_loader=train_batches,
        valid_loader=validation_batches,
        optimizer=resumed_optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=resumed_checkpoint_dir,
        starting_epoch=metadata["epoch"] + 1,
        starting_global_step=metadata["global_step"],
        history=metadata["history"],
        historical_best_checkpoint=first_checkpoint_dir / "best.pt",
    )

    assert [entry["epoch"] for entry in resumed_run["history"]] == [0, 1]
    assert [entry["global_step"] for entry in resumed_run["history"]] == [2, 4]
    assert resumed_run["global_step"] == 4
    assert resumed_run["best_checkpoint_path"] == (
        resumed_checkpoint_dir / "best.pt"
    )
    assert resumed_run["best_checkpoint_available"] is True
    assert resumed_run["best_checkpoint_path"].is_file()
    assert len(metadata["history"]) == 1
    assert (resumed_checkpoint_dir / "latest.pt").is_file()


def test_member4_bounded_smoke_test_limits_all_pipeline_stages(tmp_path):
    model = TinyLocalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training = train_model(
        model=model,
        train_loader=_training_batches(3),
        valid_loader=_training_batches(3),
        optimizer=optimizer,
        num_epochs=2,
        device="cpu",
        checkpoint_dir=tmp_path / "bounded-checkpoints",
        max_train_batches=1,
        max_validation_batches=1,
    )

    assert training["global_step"] == 2
    assert model.forward_grad_modes == [True, False, True, False]

    evaluation = evaluate_model(
        model=model,
        data_loader=_evaluation_batches(3),
        tokenizer=TinyLocalTokenizer(),
        device="cpu",
        max_batches=1,
        output_path=tmp_path / "bounded-results.jsonl",
    )

    assert evaluation["total"] == 1
    assert len(evaluation["records"]) == 1
    assert model.generation_calls == 1
