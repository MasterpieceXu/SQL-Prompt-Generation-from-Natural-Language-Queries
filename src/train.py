"""Member 4: shared training pipeline."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, Optional
import warnings

import torch


@dataclass(frozen=True)
class EarlyStoppingResult:
    """Result returned after observing one score."""

    improved: bool
    should_stop: bool


class EarlyStopping:
    """Track validation scores and signal when training should stop."""

    def __init__(
        self, mode: str = "min", patience: int = 1, min_delta: float = 0.0
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.bad_epoch_count = 0
        self.stopped = False

    def update(self, score: float, epoch: int) -> EarlyStoppingResult:
        """Observe a score and return whether it improved or should stop."""
        finite_score = math.isfinite(score)
        improved = finite_score and (
            self.best_score is None
            or (
                score < self.best_score - self.min_delta
                if self.mode == "min"
                else score > self.best_score + self.min_delta
            )
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epoch_count = 0
        else:
            self.bad_epoch_count += 1
            if self.bad_epoch_count >= self.patience:
                self.stopped = True

        return EarlyStoppingResult(improved=improved, should_stop=self.stopped)

    def state_dict(self) -> Dict[str, Any]:
        """Return all configuration and tracking state."""
        return {
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "bad_epoch_count": self.bad_epoch_count,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore configuration and tracking state."""
        restored = EarlyStopping(
            mode=state["mode"],
            patience=state["patience"],
            min_delta=state["min_delta"],
        )
        restored.best_score = state["best_score"]
        restored.best_epoch = state["best_epoch"]
        restored.bad_epoch_count = state["bad_epoch_count"]
        restored.stopped = state["stopped"]

        self.mode = restored.mode
        self.patience = restored.patience
        self.min_delta = restored.min_delta
        self.best_score = restored.best_score
        self.best_epoch = restored.best_epoch
        self.bad_epoch_count = restored.bad_epoch_count
        self.stopped = restored.stopped


def _move_nested_tensors_to_device(value, device):
    """Move tensors in a nested optimizer-state value to ``device``."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        for key, nested_value in value.items():
            value[key] = _move_nested_tensors_to_device(nested_value, device)
        return value
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            value[index] = _move_nested_tensors_to_device(nested_value, device)
        return value
    if isinstance(value, tuple):
        return tuple(
            _move_nested_tensors_to_device(nested_value, device)
            for nested_value in value
        )
    return value


def move_optimizer_state_to_device(optimizer, device):
    """Move all optimizer-state tensors recursively without changing groups."""
    target_device = torch.device(device)
    for state in optimizer.state.values():
        _move_nested_tensors_to_device(state, target_device)


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    num_epochs,
    device,
    scheduler=None,
    early_stopping=None,
    checkpoint_dir=None,
    gradient_clip_value=None,
    max_train_batches=None,
    max_validation_batches=None,
    starting_epoch=0,
    starting_global_step=0,
    history=None,
    run_config=None,
    historical_best_checkpoint=None,
):
    """Train and validate a sequence-to-sequence model.

    The model is expected to accept a batch as keyword arguments and return an
    object with a scalar ``loss`` attribute, as Hugging Face models do.
    """
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if max_train_batches is not None and max_train_batches < 1:
        raise ValueError("max_train_batches must be at least 1")
    if max_validation_batches is not None and max_validation_batches < 1:
        raise ValueError("max_validation_batches must be at least 1")
    if gradient_clip_value is not None and gradient_clip_value < 0:
        raise ValueError("gradient_clip_value must be non-negative")
    if early_stopping is not None and early_stopping.stopped:
        raise ValueError(
            "early_stopping is already stopped; reset it or create a new "
            "EarlyStopping instance before training"
        )
    try:
        if len(train_loader) == 0:
            raise ValueError("train_loader produced no batches")
    except TypeError:
        pass
    try:
        if len(valid_loader) == 0:
            raise ValueError("valid_loader produced no batches")
    except TypeError:
        pass

    device = torch.device(device)
    model.to(device)
    move_optimizer_state_to_device(optimizer, device)
    recorded_history = list(history) if history is not None else []
    global_step = starting_global_step
    stopped_early = False
    stopping_reason = "completed"

    if early_stopping is not None:
        best_score = early_stopping.best_score
        best_epoch = early_stopping.best_epoch
    else:
        finite_entries = [
            entry
            for entry in recorded_history
            if math.isfinite(float(entry["validation_loss"]))
        ]
        best_entry = (
            min(finite_entries, key=lambda entry: entry["validation_loss"])
            if finite_entries
            else None
        )
        best_score = (
            float(best_entry["validation_loss"])
            if best_entry is not None
            else None
        )
        best_epoch = best_entry["epoch"] if best_entry is not None else None

    checkpoint_directory = (
        Path(checkpoint_dir) if checkpoint_dir is not None else None
    )
    supplied_historical_best = (
        Path(historical_best_checkpoint)
        if historical_best_checkpoint is not None
        else None
    )
    if (
        supplied_historical_best is not None
        and not supplied_historical_best.is_file()
    ):
        raise FileNotFoundError(
            "Historical best checkpoint does not exist: "
            f"{supplied_historical_best}"
        )

    best_checkpoint_path = None
    historical_best_exists = best_score is not None and best_epoch is not None
    if checkpoint_directory is not None and historical_best_exists:
        candidate_best_path = checkpoint_directory / "best.pt"
        if candidate_best_path.is_file():
            best_checkpoint_path = candidate_best_path
        elif supplied_historical_best is not None:
            candidate_best_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=candidate_best_path.parent,
                    prefix=f".{candidate_best_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                shutil.copy2(supplied_historical_best, temporary_path)
                os.replace(temporary_path, candidate_best_path)
                temporary_path = None
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
            best_checkpoint_path = candidate_best_path

    def move_batch(batch):
        if not hasattr(batch, "items"):
            raise TypeError("Each batch must be a mapping of model arguments")
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def batch_size(batch):
        for preferred_key in ("labels", "input_ids", "attention_mask"):
            value = batch.get(preferred_key)
            if torch.is_tensor(value) and value.ndim > 0:
                return value.shape[0]
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return value.shape[0]
        return None

    def loss_weight(batch):
        """Use valid target tokens, falling back safely to batch size."""
        labels = batch.get("labels")
        if torch.is_tensor(labels):
            valid_target_tokens = int((labels != -100).sum().item())
            if valid_target_tokens > 0:
                return valid_target_tokens
        return batch_size(batch)

    def averaged_loss(
        weighted_loss,
        example_count,
        batch_loss,
        batch_count,
        all_batch_sizes_known,
    ):
        if batch_count == 0:
            return None
        if all_batch_sizes_known and example_count > 0:
            return weighted_loss / example_count
        return batch_loss / batch_count

    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_weighted_loss = 0.0
        train_example_count = 0
        train_batch_loss = 0.0
        train_batch_count = 0
        train_batch_sizes_known = True

        for batch_index, original_batch in enumerate(train_loader):
            if (
                max_train_batches is not None
                and batch_index >= max_train_batches
            ):
                break
            batch = move_batch(original_batch)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss).all().item():
                raise ValueError(
                    f"Non-finite training loss at epoch {epoch}, "
                    f"batch {batch_index}"
                )
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_value
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            loss_value = float(loss.detach().item())
            size = loss_weight(batch)
            if size is not None:
                train_weighted_loss += loss_value * size
                train_example_count += size
            else:
                train_batch_sizes_known = False
            train_batch_loss += loss_value
            train_batch_count += 1

        train_loss = averaged_loss(
            train_weighted_loss,
            train_example_count,
            train_batch_loss,
            train_batch_count,
            train_batch_sizes_known,
        )
        if train_loss is None:
            raise ValueError("train_loader produced no batches")

        model.eval()
        validation_weighted_loss = 0.0
        validation_example_count = 0
        validation_batch_loss = 0.0
        validation_batch_count = 0
        validation_batch_sizes_known = True
        with torch.no_grad():
            for batch_index, original_batch in enumerate(valid_loader):
                if (
                    max_validation_batches is not None
                    and batch_index >= max_validation_batches
                ):
                    break
                batch = move_batch(original_batch)
                loss = model(**batch).loss
                loss_value = float(loss.detach().item())
                size = loss_weight(batch)
                if size is not None:
                    validation_weighted_loss += loss_value * size
                    validation_example_count += size
                else:
                    validation_batch_sizes_known = False
                validation_batch_loss += loss_value
                validation_batch_count += 1

        validation_loss = averaged_loss(
            validation_weighted_loss,
            validation_example_count,
            validation_batch_loss,
            validation_batch_count,
            validation_batch_sizes_known,
        )
        if validation_loss is None:
            raise ValueError("valid_loader produced no batches")

        if early_stopping is not None:
            stopping_result = early_stopping.update(validation_loss, epoch)
            improved = stopping_result.improved
            stopped = stopping_result.should_stop
            best_score = early_stopping.best_score
            best_epoch = early_stopping.best_epoch
        else:
            improved = math.isfinite(validation_loss) and (
                best_score is None or validation_loss < best_score
            )
            if improved:
                best_score = validation_loss
                best_epoch = epoch
            stopped = False

        epoch_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "global_step": global_step,
            "improved": improved,
            "stopped": stopped,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        recorded_history.append(epoch_entry)

        if checkpoint_directory is not None:
            if improved:
                best_checkpoint_path = checkpoint_directory / "best.pt"
                save_checkpoint(
                    model,
                    best_checkpoint_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_score=best_score,
                    history=recorded_history,
                    early_stopping=early_stopping,
                    run_config=run_config,
                )
            save_checkpoint(
                model,
                checkpoint_directory / "latest.pt",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                history=recorded_history,
                early_stopping=early_stopping,
                run_config=run_config,
            )

        if stopped:
            stopped_early = True
            stopping_reason = "early_stopping"
            break

    return {
        "model": model,
        "history": recorded_history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint_path": best_checkpoint_path,
        "best_checkpoint_available": (
            best_checkpoint_path is not None
            and best_checkpoint_path.is_file()
        ),
        "global_step": global_step,
        "stopped_early": stopped_early,
        "stopping_reason": stopping_reason,
    }


def save_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
    epoch=0,
    global_step=0,
    best_score=None,
    history=None,
    early_stopping=None,
    run_config=None,
):
    """Atomically save model and training state to ``path``."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "history": history,
        "early_stopping_state_dict": (
            early_stopping.state_dict() if early_stopping is not None else None
        ),
        "run_config": run_config,
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(checkpoint, temporary_file)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
    early_stopping=None,
    map_location="cpu",
):
    """Load validated training state from ``path`` and return its metadata.

    Modern PyTorch versions use restricted ``weights_only`` loading. If an
    installed legacy PyTorch does not support that option, compatibility
    loading emits a warning and must only be used with trusted project output.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        warnings.warn(
            "This PyTorch version does not support restricted weights_only "
            "checkpoint loading. Only load trusted project output.",
            RuntimeWarning,
            stacklevel=2,
        )
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Malformed checkpoint: top-level object must be a dictionary"
        )

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(
            "Malformed checkpoint: 'model_state_dict' must be a dictionary"
        )

    optional_state_keys = (
        "optimizer_state_dict",
        "scheduler_state_dict",
        "early_stopping_state_dict",
    )
    for state_key in optional_state_keys:
        state = checkpoint.get(state_key)
        if state is not None and not isinstance(state, dict):
            raise ValueError(
                f"Malformed checkpoint: '{state_key}' must be a dictionary or None"
            )

    model.load_state_dict(model_state)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    early_stopping_state = checkpoint.get("early_stopping_state_dict")
    if early_stopping is not None and early_stopping_state is not None:
        early_stopping.load_state_dict(early_stopping_state)

    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_score": checkpoint.get("best_score"),
        "history": checkpoint.get("history"),
        "run_config": checkpoint.get("run_config"),
    }


def early_stopping(mode="min", patience=1, min_delta=0.0):
    """Return an :class:`EarlyStopping` compatibility instance."""
    return EarlyStopping(
        mode=mode,
        patience=patience,
        min_delta=min_delta,
    )
