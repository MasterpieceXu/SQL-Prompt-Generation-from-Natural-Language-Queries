"""Member 1: dataset loading, analysis, cleaning, split, and dataloader code."""


DATASET_NAME = "AI4DS/sql_generator_no_cot"


def load_raw_dataset():
    """Load the original SQL Generator No CoT dataset."""
    # TODO(Member 1): load the dataset from Hugging Face or local cache.
    pass


def inspect_dataset(df):
    """Inspect dataset size, columns, missing values, duplicates, and samples."""
    # TODO(Member 1): add EDA checks after notebook exploration is stable.
    pass


def clean_dataset(df):
    """Clean prompt/response pairs and return the cleaned dataset."""
    # TODO(Member 1): remove invalid rows, normalize SQL format, handle duplicates.
    pass


def split_dataset(df):
    """Create train, validation, and test splits."""
    # TODO(Member 1): implement reproducible train/validation/test split.
    pass


def tokenize_dataset(dataset, tokenizer):
    """Tokenize prompt and response fields for model training."""
    # TODO(Member 1): prepare model inputs and labels.
    pass


def build_dataloaders(tokenized_dataset):
    """Build train, validation, and test DataLoaders."""
    # TODO(Member 1): create PyTorch DataLoader objects.
    pass
