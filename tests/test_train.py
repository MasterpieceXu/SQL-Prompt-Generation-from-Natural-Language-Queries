import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from src.train import (
    EarlyStopping,
    early_stopping,
    load_checkpoint,
    move_optimizer_state_to_device,
    save_checkpoint,
    train_model,
)


class TinySeq2Seq(torch.nn.Module):
    def __init__(self, non_finite=False):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.non_finite = non_finite
        self.grad_modes = []
        self.seen_markers = []

    def forward(
        self, input_ids, attention_mask=None, labels=None, marker=None, **kwargs
    ):
        self.grad_modes.append(torch.is_grad_enabled())
        if marker is not None:
            self.seen_markers.extend(marker.detach().cpu().tolist())
        predictions = self.projection(input_ids.float())
        loss = torch.nn.functional.mse_loss(predictions, labels.float())
        if self.non_finite:
            loss = loss * torch.tensor(float("nan"), device=loss.device)
        return SimpleNamespace(loss=loss)


def _batches(count=2, batch_size=2):
    examples = [
        {
            "input_ids": torch.full((2,), float(index + 1)),
            "attention_mask": torch.ones(2),
            "labels": torch.zeros(2),
            "marker": torch.tensor(index),
        }
        for index in range(count)
        for _ in range(batch_size)
    ]
    return DataLoader(examples, batch_size=batch_size, shuffle=False)


def _run(
    *,
    model=None,
    train_loader=None,
    valid_loader=None,
    num_epochs=1,
    **kwargs,
):
    model = model or TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    result = train_model(
        model=model,
        train_loader=train_loader if train_loader is not None else _batches(),
        valid_loader=valid_loader if valid_loader is not None else _batches(1),
        optimizer=optimizer,
        num_epochs=num_epochs,
        device="cpu",
        **kwargs,
    )
    return model, optimizer, result


def test_one_complete_training_epoch_and_parameters_change():
    model = TinySeq2Seq()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    _, _, result = _run(model=model)

    assert len(result["history"]) == 1
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )


def test_validation_uses_no_gradients():
    model, _, _ = _run(train_loader=_batches(1), valid_loader=_batches(2))

    assert model.grad_modes == [True, False, False]


def test_scheduler_steps_once_per_training_batch():
    class CountingScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    scheduler = CountingScheduler()

    _run(train_loader=_batches(3), scheduler=scheduler)

    assert scheduler.steps == 3


def test_gradient_clipping_path(monkeypatch):
    calls = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm):
        calls.append(max_norm)
        return real_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    _run(train_loader=_batches(2), gradient_clip_value=0.25)

    assert calls == [0.25, 0.25]


def test_global_step_and_maximum_train_batch_limit():
    model, _, result = _run(
        train_loader=_batches(4),
        num_epochs=2,
        max_train_batches=2,
        starting_global_step=5,
    )

    assert result["global_step"] == 9
    assert [entry["global_step"] for entry in result["history"]] == [7, 9]
    assert model.seen_markers[:4] == [0, 0, 1, 1]


def test_maximum_validation_batch_limit():
    model, _, _ = _run(
        train_loader=_batches(1),
        valid_loader=_batches(4),
        max_validation_batches=2,
    )

    assert model.grad_modes == [True, False, False]
    assert model.seen_markers[-4:] == [0, 0, 1, 1]


def test_empty_train_loader_rejected():
    with pytest.raises(ValueError, match="train_loader produced no batches"):
        _run(train_loader=_batches(0))


def test_empty_validation_loader_rejected():
    with pytest.raises(ValueError, match="valid_loader produced no batches"):
        _run(valid_loader=_batches(0))


def test_history_structure():
    _, _, result = _run()
    entry = result["history"][0]

    assert set(entry) == {
        "epoch",
        "train_loss",
        "validation_loss",
        "global_step",
        "improved",
        "stopped",
        "elapsed_seconds",
    }
    assert entry["epoch"] == 0
    assert entry["elapsed_seconds"] >= 0


def test_latest_and_best_checkpoints_are_created(tmp_path):
    _, _, result = _run(checkpoint_dir=tmp_path)

    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert result["best_epoch"] == 0


def test_early_stopping_integration():
    stopper = EarlyStopping(patience=1)
    model = TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    result = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=5,
        device="cpu",
        early_stopping=stopper,
    )

    assert len(result["history"]) == 2
    assert result["history"][-1]["stopped"] is True
    assert result["stopped_early"] is True
    assert result["stopping_reason"] == "early_stopping"


def test_resume_with_starting_epoch_global_step_and_history():
    existing = [
        {
            "epoch": 2,
            "train_loss": 3.0,
            "validation_loss": 2.0,
            "global_step": 7,
            "improved": True,
            "stopped": False,
            "elapsed_seconds": 0.1,
        }
    ]

    _, _, result = _run(
        train_loader=_batches(1),
        starting_epoch=3,
        starting_global_step=7,
        history=existing,
    )

    assert [entry["epoch"] for entry in result["history"]] == [2, 3]
    assert result["global_step"] == 8
    assert existing == [result["history"][0]]


def test_resume_preserves_historical_best_in_same_directory(tmp_path):
    model = TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    first_run = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    best_path = tmp_path / "best.pt"
    original_best = best_path.read_bytes()

    resumed = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=tmp_path,
        starting_epoch=1,
        starting_global_step=first_run["global_step"],
        history=first_run["history"],
    )

    assert best_path.read_bytes() == original_best
    assert resumed["best_checkpoint_path"] == best_path
    assert resumed["best_checkpoint_available"] is True
    assert (tmp_path / "latest.pt").is_file()


def test_resume_copies_historical_best_into_new_directory(tmp_path):
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    model = TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    first_run = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=source_dir,
    )
    source_best = source_dir / "best.pt"

    resumed = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=destination_dir,
        starting_epoch=1,
        starting_global_step=first_run["global_step"],
        history=first_run["history"],
        historical_best_checkpoint=source_best,
    )

    destination_best = destination_dir / "best.pt"
    assert destination_best.read_bytes() == source_best.read_bytes()
    assert resumed["best_checkpoint_path"] == destination_best
    assert resumed["best_checkpoint_available"] is True
    assert (destination_dir / "latest.pt").is_file()


def test_resume_without_historical_checkpoint_reports_best_unavailable(tmp_path):
    history = [
        {
            "epoch": 0,
            "train_loss": 0.0,
            "validation_loss": -1.0,
            "global_step": 1,
            "improved": True,
            "stopped": False,
            "elapsed_seconds": 0.0,
        }
    ]
    model = TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    resumed = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=tmp_path,
        starting_epoch=1,
        starting_global_step=1,
        history=history,
    )

    assert resumed["best_score"] == -1.0
    assert resumed["best_epoch"] == 0
    assert resumed["best_checkpoint_path"] is None
    assert resumed["best_checkpoint_available"] is False
    assert not (tmp_path / "best.pt").exists()
    assert (tmp_path / "latest.pt").is_file()


def test_invalid_historical_best_checkpoint_is_rejected(tmp_path):
    history = [
        {
            "epoch": 0,
            "validation_loss": 1.0,
        }
    ]
    model = TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    missing_path = tmp_path / "missing-best.pt"

    with pytest.raises(
        FileNotFoundError,
        match="Historical best checkpoint does not exist",
    ):
        train_model(
            model,
            _batches(1),
            _batches(1),
            optimizer,
            num_epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path / "destination",
            starting_epoch=1,
            history=history,
            historical_best_checkpoint=missing_path,
        )


def test_new_improvement_replaces_materialized_historical_best(tmp_path):
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    model = TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    first_run = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=source_dir,
    )
    historical = [dict(first_run["history"][0], validation_loss=1_000_000.0)]

    resumed = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=destination_dir,
        starting_epoch=1,
        starting_global_step=first_run["global_step"],
        history=historical,
        historical_best_checkpoint=source_dir / "best.pt",
    )

    destination_best = destination_dir / "best.pt"
    metadata = load_checkpoint(model, destination_best)
    assert metadata["epoch"] == 1
    assert resumed["best_epoch"] == 1
    assert resumed["best_checkpoint_path"] == destination_best
    assert resumed["best_checkpoint_available"] is True
    assert (destination_dir / "latest.pt").is_file()


def test_original_batches_are_not_mutated():
    original_batch = next(iter(_batches(1)))
    originals = {
        key: value.clone() for key, value in original_batch.items()
    }

    _run(train_loader=[original_batch], valid_loader=[original_batch])

    assert original_batch.keys() == originals.keys()
    for key in original_batch:
        assert torch.equal(original_batch[key], originals[key])


class ReportingLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, batch_loss, labels=None):
        del input_ids, labels
        loss = batch_loss.float().mean() + self.weight * 0
        return SimpleNamespace(loss=loss)


def test_sequence_losses_are_weighted_by_valid_target_tokens():
    batches = [
        {
            "input_ids": torch.tensor([[1]]),
            "labels": torch.tensor([[1, -100, -100]]),
            "batch_loss": torch.tensor([2.0]),
        },
        {
            "input_ids": torch.tensor([[2]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "batch_loss": torch.tensor([4.0]),
        },
    ]
    model = ReportingLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    result = train_model(
        model,
        batches,
        batches,
        optimizer,
        num_epochs=1,
        device="cpu",
    )

    assert result["history"][0]["train_loss"] == pytest.approx(3.5)
    assert result["history"][0]["validation_loss"] == pytest.approx(3.5)


def test_sequence_loss_reporting_falls_back_to_example_weighting_without_labels():
    batches = [
        {
            "input_ids": torch.tensor([[1], [2]]),
            "batch_loss": torch.tensor([2.0, 2.0]),
        },
        {
            "input_ids": torch.tensor([[3]]),
            "batch_loss": torch.tensor([5.0]),
        },
    ]
    model = ReportingLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    result = train_model(
        model,
        batches,
        batches,
        optimizer,
        num_epochs=1,
        device="cpu",
    )

    assert result["history"][0]["train_loss"] == pytest.approx(3.0)
    assert result["history"][0]["validation_loss"] == pytest.approx(3.0)


def test_non_finite_training_loss_rejected():
    with pytest.raises(ValueError, match="Non-finite training loss"):
        _run(model=TinySeq2Seq(non_finite=True))


def test_first_score_improves():
    early_stopping = EarlyStopping()
    result = early_stopping.update(1.0, epoch=3)

    assert result.improved is True
    assert result.should_stop is False
    assert early_stopping.best_score == 1.0
    assert early_stopping.best_epoch == 3
    assert early_stopping.bad_epoch_count == 0


def test_improvement_resets_bad_epoch_count():
    early_stopping = EarlyStopping(mode="min", patience=3)
    early_stopping.update(1.0, epoch=0)
    early_stopping.update(1.1, epoch=1)
    result = early_stopping.update(0.9, epoch=2)

    assert result.improved is True
    assert early_stopping.bad_epoch_count == 0


def test_plateau_counts_as_non_improvement():
    early_stopping = EarlyStopping(patience=2)
    early_stopping.update(1.0, epoch=0)
    result = early_stopping.update(1.0, epoch=1)

    assert result.improved is False
    assert result.should_stop is False
    assert early_stopping.bad_epoch_count == 1


@pytest.mark.parametrize(
    ("mode", "initial", "boundary"),
    [("min", 1.0, 0.9), ("max", 1.0, 1.1)],
)
def test_min_delta_boundary_is_not_an_improvement(mode, initial, boundary):
    early_stopping = EarlyStopping(mode=mode, min_delta=0.1, patience=2)
    early_stopping.update(initial, epoch=0)
    result = early_stopping.update(boundary, epoch=1)

    assert result.improved is False
    assert early_stopping.best_score == initial
    assert early_stopping.bad_epoch_count == 1


def test_stops_exactly_at_patience():
    early_stopping = EarlyStopping(patience=2)
    early_stopping.update(1.0, epoch=0)
    before_patience = early_stopping.update(1.1, epoch=1)
    at_patience = early_stopping.update(1.2, epoch=2)

    assert before_patience.should_stop is False
    assert at_patience.should_stop is True
    assert early_stopping.bad_epoch_count == 2
    assert early_stopping.stopped is True


def test_mode_min_accepts_lower_score():
    early_stopping = EarlyStopping(mode="min")
    early_stopping.update(2.0, epoch=0)
    result = early_stopping.update(1.0, epoch=1)

    assert result.improved is True
    assert early_stopping.best_score == 1.0


def test_mode_max_accepts_higher_score():
    early_stopping = EarlyStopping(mode="max")
    early_stopping.update(1.0, epoch=0)
    result = early_stopping.update(2.0, epoch=1)

    assert result.improved is True
    assert early_stopping.best_score == 2.0


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(mode="median")


def test_invalid_patience_is_rejected():
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=0)


def test_invalid_min_delta_is_rejected():
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(min_delta=-0.1)


def test_nan_counts_as_non_improving_and_never_becomes_best():
    early_stopping = EarlyStopping(patience=2)
    result = early_stopping.update(math.nan, epoch=0)

    assert result.improved is False
    assert result.should_stop is False
    assert early_stopping.best_score is None
    assert early_stopping.best_epoch is None
    assert early_stopping.bad_epoch_count == 1


@pytest.mark.parametrize("score", [math.inf, -math.inf])
def test_infinity_counts_as_non_improving_and_never_becomes_best(score):
    early_stopping = EarlyStopping(patience=2)
    result = early_stopping.update(score, epoch=0)

    assert result.improved is False
    assert result.should_stop is False
    assert early_stopping.best_score is None
    assert early_stopping.best_epoch is None
    assert early_stopping.bad_epoch_count == 1


def test_state_dict_round_trip():
    original = EarlyStopping(mode="max", patience=3, min_delta=0.25)
    original.update(1.0, epoch=4)
    original.update(1.1, epoch=5)
    restored = EarlyStopping()

    restored.load_state_dict(original.state_dict())

    assert restored.state_dict() == original.state_dict()
    result = restored.update(1.3, epoch=6)
    assert result.improved is True
    assert restored.best_score == 1.3
    assert restored.best_epoch == 6


def test_early_stopping_compatibility_factory_returns_configured_instance():
    stopper = early_stopping(mode="max", patience=4, min_delta=0.25)

    assert isinstance(stopper, EarlyStopping)
    assert stopper is not None
    assert stopper.mode == "max"
    assert stopper.patience == 4
    assert stopper.min_delta == pytest.approx(0.25)


def test_already_stopped_early_stopping_is_rejected_before_iteration():
    iterated = False

    def batches():
        nonlocal iterated
        iterated = True
        yield next(iter(_batches(1)))

    model = TinySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    stopper = EarlyStopping(patience=1)
    stopper.update(1.0, epoch=0)
    stopper.update(1.0, epoch=1)

    with pytest.raises(ValueError, match="already stopped.*reset.*new"):
        train_model(
            model,
            batches(),
            _batches(1),
            optimizer,
            num_epochs=1,
            device="cpu",
            early_stopping=stopper,
        )

    assert iterated is False


def _nested_tensor_devices(value):
    if torch.is_tensor(value):
        return [value.device]
    if isinstance(value, dict):
        return [
            device
            for nested_value in value.values()
            for device in _nested_tensor_devices(nested_value)
        ]
    if isinstance(value, (list, tuple)):
        return [
            device
            for nested_value in value
            for device in _nested_tensor_devices(nested_value)
        ]
    return []


def test_move_optimizer_state_to_device_handles_nested_values():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameter = next(model.parameters())
    marker = object()
    optimizer.state[parameter] = {
        "tensor": torch.tensor(1.0),
        "dict": {"tensor": torch.tensor([2.0]), "marker": marker},
        "list": [torch.tensor(3.0), "unchanged"],
        "tuple": (torch.tensor(4.0), 5),
    }
    parameter_groups_before = list(optimizer.param_groups)

    move_optimizer_state_to_device(optimizer, "meta")

    assert all(
        device.type == "meta"
        for device in _nested_tensor_devices(optimizer.state[parameter])
    )
    assert optimizer.state[parameter]["dict"]["marker"] is marker
    assert optimizer.state[parameter]["list"][1] == "unchanged"
    assert optimizer.state[parameter]["tuple"][1] == 5
    assert len(optimizer.param_groups) == len(parameter_groups_before)
    assert all(
        after is before
        for after, before in zip(optimizer.param_groups, parameter_groups_before)
    )


def test_train_model_aligns_optimizer_state_with_model_device():
    model = TinySeq2Seq()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
    )

    model_device = next(model.parameters()).device
    state_devices = [
        device
        for state in optimizer.state.values()
        for device in _nested_tensor_devices(state)
    ]
    assert state_devices
    assert all(device == model_device for device in state_devices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_checkpoint_optimizer_state_is_migrated_when_resuming_on_cuda(tmp_path):
    model = TinySeq2Seq()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    first_run = train_model(
        model,
        _batches(1),
        _batches(1),
        optimizer,
        num_epochs=1,
        device="cpu",
        checkpoint_dir=tmp_path,
    )

    resumed_model = TinySeq2Seq()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    metadata = load_checkpoint(
        resumed_model,
        tmp_path / "latest.pt",
        optimizer=resumed_optimizer,
        map_location="cpu",
    )
    assert all(
        device.type == "cpu"
        for state in resumed_optimizer.state.values()
        for device in _nested_tensor_devices(state)
    )

    train_model(
        resumed_model,
        _batches(1),
        _batches(1),
        resumed_optimizer,
        num_epochs=1,
        device="cuda",
        starting_epoch=metadata["epoch"] + 1,
        starting_global_step=first_run["global_step"],
        history=metadata["history"],
    )

    assert next(resumed_model.parameters()).device.type == "cuda"
    assert all(
        device.type == "cuda"
        for state in resumed_optimizer.state.values()
        for device in _nested_tensor_devices(state)
    )


def _trained_components():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss = model(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_checkpoint_directory_creation_and_model_round_trip(tmp_path):
    model, _, _ = _trained_components()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "nested" / "checkpoints" / "model.pt"

    save_checkpoint(model, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    load_checkpoint(model, path)

    assert path.is_file()
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_optimizer_and_scheduler_state_round_trip(tmp_path):
    model, optimizer, scheduler = _trained_components()
    path = tmp_path / "state.pt"
    save_checkpoint(model, path, optimizer=optimizer, scheduler=scheduler)
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1.0)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=3, gamma=0.1
    )

    load_checkpoint(
        restored_model,
        path,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert restored_optimizer.state_dict()["state"]
    assert restored_optimizer.state_dict()["param_groups"] == (
        optimizer.state_dict()["param_groups"]
    )
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_metadata_and_early_stopping_round_trip(tmp_path):
    model = torch.nn.Linear(1, 1)
    stopper = EarlyStopping(mode="max", patience=4, min_delta=0.2)
    stopper.update(0.8, epoch=2)
    stopper.update(0.7, epoch=3)
    history = {"train_loss": [2.0, 1.0], "valid_score": [0.5, 0.8]}
    run_config = {"seed": 17, "tags": ["local", "test"]}
    path = tmp_path / "metadata.pt"
    save_checkpoint(
        model,
        path,
        epoch=3,
        global_step=42,
        best_score=0.8,
        history=history,
        early_stopping=stopper,
        run_config=run_config,
    )
    restored_stopper = EarlyStopping()

    metadata = load_checkpoint(
        model, path, early_stopping=restored_stopper
    )

    assert metadata == {
        "epoch": 3,
        "global_step": 42,
        "best_score": 0.8,
        "history": history,
        "run_config": run_config,
    }
    assert restored_stopper.state_dict() == stopper.state_dict()


def test_load_checkpoint_uses_cpu_map_location(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "cpu.pt"
    save_checkpoint(model, path)
    real_load = torch.load
    observed = {}

    def recording_load(*args, **kwargs):
        observed["map_location"] = kwargs.get("map_location")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    load_checkpoint(model, path, map_location="cpu")

    assert observed["map_location"] == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_load_checkpoint_uses_weights_only_safe_path(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "safe.pt"
    save_checkpoint(model, path)
    real_load = torch.load
    observed = {}

    def recording_load(*args, **kwargs):
        observed["weights_only"] = kwargs.get("weights_only")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    load_checkpoint(model, path)

    assert observed["weights_only"] is True


def test_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        load_checkpoint(torch.nn.Linear(1, 1), tmp_path / "missing.pt")


def test_malformed_checkpoint_raises_clear_error(tmp_path):
    path = tmp_path / "malformed.pt"
    torch.save({"epoch": 1}, path)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(torch.nn.Linear(1, 1), path)


def test_checkpoint_rejects_non_dictionary_top_level_object(tmp_path):
    path = tmp_path / "list.pt"
    torch.save([], path)

    with pytest.raises(ValueError, match="top-level object must be a dictionary"):
        load_checkpoint(torch.nn.Linear(1, 1), path)


def test_checkpoint_rejects_malformed_model_state_dict(tmp_path):
    path = tmp_path / "bad-model-state.pt"
    torch.save({"model_state_dict": "not-a-dictionary"}, path)

    with pytest.raises(ValueError, match="'model_state_dict' must be a dictionary"):
        load_checkpoint(torch.nn.Linear(1, 1), path)


@pytest.mark.parametrize(
    "state_key",
    [
        "optimizer_state_dict",
        "scheduler_state_dict",
        "early_stopping_state_dict",
    ],
)
def test_checkpoint_rejects_malformed_optional_state_dict(tmp_path, state_key):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / f"bad-{state_key}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            state_key: "not-a-dictionary",
        },
        path,
    )

    with pytest.raises(
        ValueError,
        match=rf"'{state_key}' must be a dictionary or None",
    ):
        load_checkpoint(model, path)


def test_checkpoint_operates_without_optional_state(tmp_path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "minimal.pt"

    save_checkpoint(model, path, epoch=1, global_step=2)
    metadata = load_checkpoint(torch.nn.Linear(1, 1), path)

    assert metadata["epoch"] == 1
    assert metadata["global_step"] == 2
