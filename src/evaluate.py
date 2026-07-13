"""Member 4: evaluation and error analysis."""


def exact_match_score(predictions, references):
    """Calculate exact match between generated SQL and reference SQL."""
    # TODO(Member 4): normalize SQL strings and compute exact match.
    pass


def sql_validity_check(predictions):
    """Check whether generated SQL strings are syntactically valid."""
    # TODO(Member 4): use sqlparse or SQLite-based checks if appropriate.
    pass


def evaluate_model(model, test_loader):
    """Evaluate a trained model on the test set."""
    # TODO(Member 4): generate SQL predictions and compute metrics.
    pass


def error_analysis(predictions, references):
    """Analyze common generation errors."""
    # TODO(Member 4): categorize errors for Results and Discussion sections.
    pass
