"""Shared project configuration."""


RANDOM_SEED = 42
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
DATASET_NAME = "AI4DS/sql_generator_no_cot"
DATASET_PATH = "hf://datasets/AI4DS/sql_generator_no_cot/training_no_cot_dataset.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

BASELINE_MODEL_NAME = "t5-small"

# TODO(Member 5/shared): add shared paths and hyperparameters as they become stable.
