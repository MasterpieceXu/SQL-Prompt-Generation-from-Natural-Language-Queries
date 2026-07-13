"""Member 4: shared training pipeline."""


def train_model(model, train_loader, valid_loader):
    """Train a model using the shared training loop."""
    # TODO(Member 4): implement training loop, validation, logging, and saving.
    pass


def save_checkpoint(model, path):
    """Save a model checkpoint."""
    # TODO(Member 4): save model state or Hugging Face model files.
    pass


def load_checkpoint(model, path):
    """Load a model checkpoint."""
    # TODO(Member 4): restore model state for evaluation or continued training.
    pass


def early_stopping():
    """Implement early stopping if needed."""
    # TODO(Member 4): stop training when validation performance stops improving.
    pass
