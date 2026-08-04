"""Portable loader for the verified Member 3 Hugging Face checkpoint.

The formal V3 checkpoint stores ``shared.weight`` and ``lm_head.weight`` as two
distinct trained matrices.  Some Transformers versions automatically tie these
parameters while loading a T5 checkpoint.  This module restores the tensors exactly
as stored so inference remains compatible across the versions used by team members.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    file_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    """Resolve a checkpoint directory and fail clearly when files are missing."""
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint} is incomplete; missing: "
            + ", ".join(missing)
        )
    return checkpoint


def _load_tokenizer(checkpoint: Path):
    """Load locally, with a fallback for a tokenizer saved by Transformers 5.x."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    except (AttributeError, TypeError, ValueError):
        from transformers import T5TokenizerFast

        return T5TokenizerFast(
            tokenizer_file=str(checkpoint / "tokenizer.json"),
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=100,
            model_max_length=512,
        )


def load_verified_checkpoint(
    checkpoint_dir: str | Path,
    device: str | Any = "cpu",
    expected_model_sha256: str | None = None,
):
    """Load tokenizer/model locally while preserving the checkpoint's output head.

    No optimiser or training loop is created.  When ``expected_model_sha256`` is
    supplied, the weight file is verified before the model is loaded.
    """
    checkpoint = validate_checkpoint_dir(checkpoint_dir)
    model_path = checkpoint / "model.safetensors"

    if expected_model_sha256:
        actual_sha256 = sha256_file(model_path)
        if actual_sha256.lower() != expected_model_sha256.lower():
            raise RuntimeError(
                "Checkpoint SHA-256 mismatch: "
                f"expected {expected_model_sha256}, got {actual_sha256}"
            )

    try:
        import torch
        from safetensors import safe_open
        from transformers import AutoModelForSeq2SeqLM
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Missing checkpoint dependencies. Install requirements.txt before inference."
        ) from exc

    tokenizer = _load_tokenizer(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        local_files_only=True,
    )

    # Restore both raw matrices after AutoModel loading.  This is essential when an
    # older Transformers version silently tied the output head to shared embeddings.
    with safe_open(str(model_path), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        if {"shared.weight", "lm_head.weight"}.issubset(keys):
            shared_weight = weights.get_tensor("shared.weight")
            lm_head_weight = weights.get_tensor("lm_head.weight")
        else:
            shared_weight = None
            lm_head_weight = None

    if shared_weight is not None and lm_head_weight is not None:
        if shared_weight.shape != lm_head_weight.shape:
            raise RuntimeError(
                "Incompatible checkpoint matrices: shared.weight and lm_head.weight "
                "must have the same shape"
            )

        with torch.no_grad():
            model.shared.weight.copy_(shared_weight)
        model.encoder.embed_tokens = model.shared
        model.decoder.embed_tokens = model.shared

        if model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr():
            model.lm_head = torch.nn.Linear(
                model.config.d_model,
                model.config.vocab_size,
                bias=False,
            )
        with torch.no_grad():
            model.lm_head.weight.copy_(lm_head_weight)

        if model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr():
            raise RuntimeError("Checkpoint output head was unexpectedly tied after loading")
        if not torch.equal(model.shared.weight.detach().cpu(), shared_weight):
            raise RuntimeError("shared.weight was not restored exactly")
        if not torch.equal(model.lm_head.weight.detach().cpu(), lm_head_weight):
            raise RuntimeError("lm_head.weight was not restored exactly")

    model.to(device)
    model.eval()
    return tokenizer, model
