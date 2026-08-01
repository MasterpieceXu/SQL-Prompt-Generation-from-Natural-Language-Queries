import json
import sqlite3

import pytest
import torch

import src.evaluate as evaluate_module
from src.evaluate import (
    analyze_prediction_error,
    build_evaluation_batches,
    check_sql_validity,
    evaluate_model,
    evaluate_normalized_exact_match,
    exact_match_score,
    normalize_sql,
    normalized_exact_match,
    parse_args,
    raw_exact_match,
    summarize_error_analysis,
    validate_checkpoint_directory,
    verify_aligned_results,
)


SCHEMA_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER
);
"""


def test_sql_validity_valid_select_against_schema():
    result = check_sql_validity("SELECT id, name FROM users", SCHEMA_SQL)

    assert result == {
        "valid": True,
        "category": "valid",
        "error_message": None,
    }


def test_sql_validity_valid_with_query():
    result = check_sql_validity(
        "WITH named AS (SELECT name FROM users) SELECT name FROM named",
        SCHEMA_SQL,
    )

    assert result["valid"] is True
    assert result["category"] == "valid"


@pytest.mark.parametrize(
    ("sql", "category"),
    [
        ("SELECT FROM users", "syntax_error"),
        ("SELECT * FROM absent", "missing_table"),
        ("SELECT absent_column FROM users", "missing_column"),
        ("", "empty_sql"),
        ("   ", "empty_sql"),
        (None, "empty_sql"),
    ],
)
def test_sql_validity_invalid_cases(sql, category):
    result = check_sql_validity(sql, SCHEMA_SQL)

    assert result["valid"] is False
    assert result["category"] == category
    assert result["error_message"]


def test_sql_validity_invalid_schema():
    result = check_sql_validity("SELECT 1", "CREATE TABLE broken (")

    assert result["valid"] is False
    assert result["category"] == "schema_error"
    assert result["error_message"]


def test_sql_validity_rejects_multiple_statements():
    result = check_sql_validity("SELECT 1; SELECT 2")

    assert result["valid"] is False
    assert result["category"] == "syntax_error"
    assert "one statement at a time" in result["error_message"].lower()


def test_sql_validity_rejects_attach(tmp_path):
    database_path = tmp_path / "attached.sqlite"
    result = check_sql_validity(f"ATTACH DATABASE '{database_path}' AS external")

    assert result["valid"] is False
    assert result["category"] == "execution_error"
    assert not database_path.exists()


def test_sql_validity_rejects_destructive_pragma():
    result = check_sql_validity("PRAGMA journal_mode = WAL")

    assert result["valid"] is False
    assert result["category"] == "execution_error"


def test_sql_validity_does_not_create_external_file(tmp_path):
    database_path = tmp_path / "must_not_exist.sqlite"
    schema = f"""
    ATTACH DATABASE '{database_path}' AS external;
    CREATE TABLE external.items (id INTEGER);
    """

    result = check_sql_validity("SELECT 1", schema)

    assert result["valid"] is False
    assert result["category"] == "schema_error"
    assert not database_path.exists()


def test_sql_validity_has_structured_result_keys():
    result = check_sql_validity("SELECT 1")

    assert set(result) == {"valid", "category", "error_message"}


def test_sql_validity_without_schema_supports_syntax_only_cases():
    assert check_sql_validity("SELECT 1")["valid"] is True
    result = check_sql_validity("SELECT (")
    assert result["valid"] is False
    assert result["category"] == "syntax_error"


def test_sql_validity_uses_standard_library_sqlite():
    assert sqlite3.sqlite_version


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("  SELECT * FROM Users  ", "select * from users"),
        ("SELECT  id\n\nFROM\tusers", "select id from users"),
        ("SELECT id FROM users;", "select id from users"),
        ("", ""),
        ("   ", ""),
        (None, ""),
    ],
)
def test_normalize_sql(sql, expected):
    assert normalize_sql(sql) == expected


def test_normalize_sql_does_not_rewrite_sql_syntax():
    assert normalize_sql("SELECT * FROM users WHERE id = 1") == (
        "select * from users where id = 1"
    )


def test_normalized_exact_match_correct():
    assert normalized_exact_match(
        "SELECT id FROM users",
        "  select  id\nFROM users; ",
    )


def test_normalized_exact_match_incorrect():
    assert not normalized_exact_match(
        "SELECT id FROM users",
        "SELECT name FROM users",
    )


def test_normalized_exact_match_handles_none_and_empty_string():
    assert normalized_exact_match(None, "")


def test_raw_exact_match_does_not_normalize():
    assert raw_exact_match("SELECT 1", "SELECT 1")
    assert not raw_exact_match("SELECT 1", "select 1;")


def test_evaluate_normalized_exact_match_lists():
    result = evaluate_normalized_exact_match(
        ["SELECT 1", "SELECT id FROM users", None],
        ["select 1;", "SELECT name FROM users", ""],
    )

    assert result["correct"] == 2
    assert result["total"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_exact_match_score_preserves_predictions_first_api():
    assert exact_match_score(
        ["select 1;", "select wrong"],
        ["SELECT 1", "SELECT 2"],
    ) == {"correct": 1, "total": 2, "accuracy": 0.5}


def test_evaluate_normalized_exact_match_empty_lists():
    assert evaluate_normalized_exact_match([], []) == {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
    }


def test_evaluate_normalized_exact_match_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate_normalized_exact_match(["SELECT 1"], [])


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.decoded_inputs = []

    def batch_decode(self, token_ids, skip_special_tokens=True):
        tensor = torch.as_tensor(token_ids).detach().cpu()
        self.decoded_inputs.append(tensor.clone())
        texts = []
        for row in tensor.tolist():
            tokens = [str(token) for token in row if token != self.pad_token_id]
            texts.append(" ".join(tokens))
        return texts


class TinyGenerator(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = [torch.tensor(output) for output in outputs]
        self.eval_calls = 0
        self.grad_modes = []
        self.devices = []
        self.generation_calls = []
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append(torch.device(args[0] if args else kwargs["device"]))
        return super().to(*args, **kwargs)

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def generate(self, input_ids, attention_mask=None, extra=None, **kwargs):
        self.grad_modes.append(torch.is_grad_enabled())
        self.devices.append(input_ids.device.type)
        self.generation_calls.append(
            {
                "input_ids": input_ids.detach().cpu().clone(),
                "attention_mask": attention_mask.detach().cpu().clone()
                if attention_mask is not None
                else None,
                "extra": extra.detach().cpu().clone() if extra is not None else None,
                "kwargs": kwargs,
            }
        )
        return self.outputs[len(self.generation_calls) - 1]


class SqlTokenizer(TinyTokenizer):
    sql_by_token = {
        1: "SELECT 1",
        2: "SELECT 2",
        3: "SELECT FROM users",
    }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        tensor = torch.as_tensor(token_ids).detach().cpu()
        self.decoded_inputs.append(tensor.clone())
        return [self.sql_by_token[row[0]] for row in tensor.tolist()]


def _batch(
    labels=None,
    *,
    target_sql=None,
    sample_id=None,
    prompt=None,
    extra=None,
    schema_sql=None,
    schema=None,
):
    labels = labels if labels is not None else [[1, 0], [2, 0]]
    batch = {
        "input_ids": torch.tensor([[10, 11], [12, 13]]),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "labels": torch.tensor(labels),
    }
    if target_sql is not None:
        batch["target_sql"] = target_sql
    if sample_id is not None:
        batch["sample_id"] = sample_id
    if prompt is not None:
        batch["prompt"] = prompt
    if extra is not None:
        batch["extra"] = extra
    if schema_sql is not None:
        batch["schema_sql"] = schema_sql
    if schema is not None:
        batch["schema"] = schema
    return batch


def test_one_complete_evaluation_batch_and_exact_match_counts():
    model = TinyGenerator([[[1, 0], [9, 0]]])
    tokenizer = TinyTokenizer()

    result = evaluate_model(model, [_batch()], tokenizer, "cpu")

    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["accuracy"] == pytest.approx(0.5)
    assert [record["index"] for record in result["records"]] == [0, 1]
    assert result["records"][0] == {
        "index": 0,
        "target_sql": "1",
        "predicted_sql": "1",
        "normalized_target_sql": "1",
        "normalized_predicted_sql": "1",
        "raw_exact_match": True,
        "exact_match": True,
        "normalized_exact_match": True,
    }


def test_multiple_batches_eval_no_grad_cpu_and_generation_kwargs():
    model = TinyGenerator([[[1, 0], [2, 0]], [[3, 0], [4, 0]]])
    tokenizer = TinyTokenizer()
    batches = [_batch(), _batch(labels=[[3, 0], [4, 0]])]

    result = evaluate_model(
        model,
        batches,
        tokenizer,
        torch.device("cpu"),
        generation_kwargs={"num_beams": 2},
    )

    assert result["total"] == 4
    assert model.eval_calls == 1
    assert model.grad_modes == [False, False]
    assert model.devices == ["cpu", "cpu"]
    assert [call["kwargs"] for call in model.generation_calls] == [
        {"num_beams": 2},
        {"num_beams": 2},
    ]
    assert result["generation_kwargs"] == {"num_beams": 2}


def test_evaluate_model_moves_model_to_requested_device():
    model = TinyGenerator([[[1, 0], [2, 0]]])

    evaluate_model(model, [_batch()], TinyTokenizer(), torch.device("cpu"))

    assert model.to_calls == [torch.device("cpu")]


def test_original_batch_not_mutated_and_other_tensor_field_is_used():
    batch = _batch(labels=[[1, -100], [2, 0]], extra=torch.tensor([7, 8]))
    originals = {key: value.clone() for key, value in batch.items()}
    model = TinyGenerator([[[1, 0], [2, 0]]])

    evaluate_model(model, [batch], TinyTokenizer(), "cpu")

    assert batch.keys() == originals.keys()
    assert all(torch.equal(batch[key], original) for key, original in originals.items())
    assert torch.equal(model.generation_calls[0]["extra"], torch.tensor([7, 8]))


def test_labels_with_ignore_index_are_safely_decoded():
    batch = _batch(labels=[[1, -100], [2, -100]])
    tokenizer = TinyTokenizer()

    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [batch],
        tokenizer,
        "cpu",
    )

    assert torch.equal(batch["labels"], torch.tensor([[1, -100], [2, -100]]))
    assert tokenizer.decoded_inputs[-1].tolist() == [[1, 0], [2, 0]]
    assert [record["target_sql"] for record in result["records"]] == ["1", "2"]


def test_batch_target_sql_takes_precedence_and_metadata_is_recorded():
    batch = _batch(
        labels=[[99, 0], [98, 0]],
        target_sql=["SELECT 1;", "SELECT 2"],
        sample_id=torch.tensor([41, 42]),
        prompt=["first prompt", "second prompt"],
    )

    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [batch],
        TinyTokenizer(),
        "cpu",
    )

    assert [record["target_sql"] for record in result["records"]] == [
        "SELECT 1;",
        "SELECT 2",
    ]
    assert result["records"][0]["sample_id"] == 41
    assert result["records"][1]["prompt"] == "second prompt"
    assert result["correct"] == 0


def test_max_batches_limits_evaluation():
    model = TinyGenerator([[[1, 0], [2, 0]], [[3, 0], [4, 0]]])

    result = evaluate_model(
        model,
        [_batch(), _batch(labels=[[3, 0], [4, 0]])],
        TinyTokenizer(),
        "cpu",
        max_batches=1,
    )

    assert result["total"] == 2
    assert len(model.generation_calls) == 1


def test_empty_loader_is_rejected():
    with pytest.raises(ValueError, match="data_loader produced no batches"):
        evaluate_model(TinyGenerator([]), [], TinyTokenizer(), "cpu")


def test_prediction_target_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="target count .* prediction count"):
        evaluate_model(
            TinyGenerator([[[1, 0]]]),
            [_batch()],
            TinyTokenizer(),
            "cpu",
        )


def test_jsonl_output_and_parent_directory_creation(tmp_path):
    output_path = tmp_path / "nested" / "results" / "predictions.jsonl"
    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch(sample_id=["α", "β"], prompt=["one", "two"])],
        TinyTokenizer(),
        "cpu",
        output_path=output_path,
    )

    assert output_path.is_file()
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == result["records"]
    assert rows[0]["sample_id"] == "α"
    assert rows[0]["predicted_sql"] == "1"
    assert rows[0]["exact_match"] is True


def test_jsonl_output_atomically_replaces_existing_file(tmp_path, monkeypatch):
    output_path = tmp_path / "predictions.jsonl"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    real_replace = evaluate_module.os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append((source, destination))
        assert source.parent == output_path.parent
        real_replace(source, destination)

    monkeypatch.setattr(evaluate_module.os, "replace", recording_replace)

    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch()],
        TinyTokenizer(),
        "cpu",
        output_path=output_path,
    )

    assert len(replacements) == 1
    assert replacements[0][1] == output_path
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == result["records"]
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_jsonl_serialization_failure_preserves_existing_file(tmp_path, monkeypatch):
    output_path = tmp_path / "predictions.jsonl"
    original = '{"old": true}\n'
    output_path.write_text(original, encoding="utf-8")

    def failing_dumps(*args, **kwargs):
        del args, kwargs
        raise TypeError("cannot serialize")

    monkeypatch.setattr(evaluate_module.json, "dumps", failing_dumps)

    with pytest.raises(TypeError, match="cannot serialize"):
        evaluate_model(
            TinyGenerator([[[1, 0], [2, 0]]]),
            [_batch()],
            TinyTokenizer(),
            "cpu",
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_jsonl_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    output_path = tmp_path / "predictions.jsonl"
    original = '{"old": true}\n'
    output_path.write_text(original, encoding="utf-8")

    def failing_replace(source, destination):
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(evaluate_module.os, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        evaluate_model(
            TinyGenerator([[[1, 0], [2, 0]]]),
            [_batch()],
            TinyTokenizer(),
            "cpu",
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_malformed_existing_jsonl_is_not_overwritten(tmp_path):
    output_path = tmp_path / "predictions.jsonl"
    output_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to overwrite malformed JSONL"):
        evaluate_model(
            TinyGenerator([[[1, 0], [2, 0]]]),
            [_batch()],
            TinyTokenizer(),
            "cpu",
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "not-json\n"


def test_elapsed_time_and_throughput_fields():
    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch()],
        TinyTokenizer(),
        "cpu",
    )

    assert result["elapsed_seconds"] >= 0
    assert result["examples_per_second"] >= 0


def test_evaluate_model_default_output_remains_unchanged():
    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch(schema_sql=SCHEMA_SQL)],
        TinyTokenizer(),
        "cpu",
    )

    assert set(result) == {
        "records", "raw_correct", "raw_accuracy", "correct", "total",
        "accuracy", "normalized_correct", "normalized_accuracy",
        "inference_seconds", "average_latency_seconds", "elapsed_seconds",
        "examples_per_second", "generation_kwargs",
    }
    assert set(result["records"][0]) == {
        "index", "target_sql", "predicted_sql", "normalized_target_sql",
        "normalized_predicted_sql", "raw_exact_match", "exact_match",
        "normalized_exact_match",
    }


def test_evaluate_model_validity_integration_enabled():
    batch = _batch(
        target_sql=["SELECT 1", "SELECT id FROM users"],
        schema_sql=SCHEMA_SQL,
    )
    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [batch],
        SqlTokenizer(),
        "cpu",
        include_validity=True,
    )

    record = result["records"][0]
    assert record["validity_result"] == {
        "valid": True, "category": "valid", "error_message": None,
    }
    assert record["sql_valid"] is True
    assert record["validity_category"] == "valid"
    assert record["validity_error_message"] is None


def test_evaluate_model_validity_integration_disabled(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("validity should not be checked")

    monkeypatch.setattr(evaluate_module, "check_sql_validity", unexpected_call)
    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch()],
        TinyTokenizer(),
        "cpu",
    )

    assert "validity_result" not in result["records"][0]


def test_evaluate_model_error_analysis_and_summary_enabled():
    result = evaluate_model(
        TinyGenerator([[[1, 0], [3, 0]]]),
        [_batch(target_sql=["SELECT 1", "SELECT 2"])],
        SqlTokenizer(),
        "cpu",
        include_error_analysis=True,
    )

    exact_record, invalid_record = result["records"]
    assert exact_record["error_categories"] == ["exact_match"]
    assert exact_record["primary_error_category"] == "exact_match"
    assert exact_record["error_analysis"]["exact_match"] is True
    assert invalid_record["error_categories"] == ["syntax_error"]
    assert invalid_record["primary_error_category"] == "syntax_error"
    assert result["error_summary"]["total"] == 2
    assert result["error_summary"]["category_counts"]["exact_match"] == 1
    assert result["error_summary"]["category_counts"]["syntax_error"] == 1


def test_evaluate_model_error_summary_returned_only_when_requested():
    ordinary = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch()],
        TinyTokenizer(),
        "cpu",
        include_validity=True,
    )
    analyzed = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch()],
        TinyTokenizer(),
        "cpu",
        include_error_analysis=True,
    )

    assert "error_summary" not in ordinary
    assert "error_summary" in analyzed


@pytest.mark.parametrize("schema_field", ["schema_sql", "schema"])
def test_evaluate_model_uses_schema_metadata(monkeypatch, schema_field):
    seen_schemas = []

    def recording_check(predicted_sql, schema_sql=None):
        seen_schemas.append(schema_sql)
        return {"valid": True, "category": "valid", "error_message": None}

    monkeypatch.setattr(evaluate_module, "check_sql_validity", recording_check)
    batch = _batch()
    batch[schema_field] = [SCHEMA_SQL, "CREATE TABLE items (id INTEGER);"]

    evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [batch],
        TinyTokenizer(),
        "cpu",
        include_validity=True,
    )

    assert seen_schemas == [SCHEMA_SQL, "CREATE TABLE items (id INTEGER);"]


def test_evaluate_model_missing_schema_is_handled_safely():
    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch()],
        SqlTokenizer(),
        "cpu",
        include_validity=True,
    )

    assert [record["sql_valid"] for record in result["records"]] == [True, True]


def test_evaluate_model_invalid_sql_is_classified_correctly():
    result = evaluate_model(
        TinyGenerator([[[3, 0], [2, 0]]]),
        [_batch(target_sql=["SELECT 1", "SELECT 2"])],
        SqlTokenizer(),
        "cpu",
        include_validity=True,
        include_error_analysis=True,
    )

    record = result["records"][0]
    assert record["sql_valid"] is False
    assert record["validity_category"] == "syntax_error"
    assert record["error_categories"] == ["syntax_error"]


def test_evaluate_model_avoids_duplicate_validity_calls(monkeypatch):
    calls = []

    def recording_check(predicted_sql, schema_sql=None):
        calls.append((predicted_sql, schema_sql))
        return {"valid": True, "category": "valid", "error_message": None}

    monkeypatch.setattr(evaluate_module, "check_sql_validity", recording_check)
    evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch(schema_sql=SCHEMA_SQL)],
        SqlTokenizer(),
        "cpu",
        include_validity=True,
        include_error_analysis=True,
    )

    assert calls == [("SELECT 1", SCHEMA_SQL), ("SELECT 2", SCHEMA_SQL)]


def test_evaluate_model_integrated_jsonl_fields_are_serializable(tmp_path):
    output_path = tmp_path / "integrated.jsonl"
    result = evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [_batch(sample_id=torch.tensor([7, 8]))],
        TinyTokenizer(),
        "cpu",
        include_validity=True,
        include_error_analysis=True,
        output_path=output_path,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == result["records"]
    assert "validity_result" in rows[0]
    assert "error_analysis" in rows[0]
    json.dumps(result)


def test_evaluate_model_integration_does_not_mutate_original_batch():
    batch = _batch(
        target_sql=["1", "2"],
        sample_id=torch.tensor([11, 12]),
        schema_sql=[SCHEMA_SQL, SCHEMA_SQL],
    )
    original_keys = set(batch)
    original_tensors = {
        key: value.clone()
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    original_metadata = {
        key: list(value)
        for key, value in batch.items()
        if isinstance(value, list)
    }

    evaluate_model(
        TinyGenerator([[[1, 0], [2, 0]]]),
        [batch],
        TinyTokenizer(),
        "cpu",
        include_validity=True,
        include_error_analysis=True,
    )

    assert set(batch) == original_keys
    assert all(torch.equal(batch[key], value) for key, value in original_tensors.items())
    assert all(batch[key] == value for key, value in original_metadata.items())


def _category(target, predicted, validity=None):
    return analyze_prediction_error(
        target,
        predicted,
        validity_result=validity,
        sample_id="sample",
    )


def test_error_analysis_exact_match():
    result = _category("SELECT id FROM users", " select id FROM users; ")

    assert result["exact_match"] is True
    assert result["categories"] == ["exact_match"]
    assert result["primary_category"] == "exact_match"


def test_error_analysis_empty_output():
    assert _category("SELECT id FROM users", "  ")["categories"] == ["empty_output"]


def test_error_analysis_syntax_error_from_validity_result():
    result = _category(
        "SELECT id FROM users",
        "SELECT FROM users",
        {"valid": False, "category": "syntax_error", "error_message": "near FROM"},
    )

    assert result["categories"] == ["syntax_error"]
    assert result["error_message"] == "near FROM"


@pytest.mark.parametrize(
    ("validity_category", "expected"),
    [("missing_table", "missing_table"), ("missing_column", "missing_column")],
)
def test_error_analysis_missing_schema_object(validity_category, expected):
    result = _category(
        "SELECT id FROM users",
        "SELECT id FROM absent",
        {"valid": False, "category": validity_category, "error_message": "missing"},
    )

    assert result["categories"] == [expected]


def test_error_analysis_wrong_table_detection():
    assert "wrong_table" in _category(
        "SELECT id FROM users", "SELECT id FROM customers"
    )["categories"]


def test_error_analysis_wrong_column_detection():
    assert "wrong_column" in _category(
        "SELECT id FROM users", "SELECT name FROM users"
    )["categories"]


def test_error_analysis_join_error():
    assert "join_error" in _category(
        "SELECT users.id FROM users JOIN orders ON users.id = orders.user_id",
        "SELECT users.id FROM users",
    )["categories"]


def test_error_analysis_aggregation_error():
    assert "aggregation_error" in _category(
        "SELECT COUNT(id) FROM users", "SELECT id FROM users"
    )["categories"]


def test_error_analysis_filter_error():
    assert "filter_error" in _category(
        "SELECT id FROM users WHERE id = 1", "SELECT id FROM users"
    )["categories"]


def test_error_analysis_grouping_error():
    assert "grouping_error" in _category(
        "SELECT name FROM users GROUP BY name", "SELECT name FROM users"
    )["categories"]


def test_error_analysis_ordering_or_limit_error():
    result = _category(
        "SELECT id FROM users ORDER BY id LIMIT 5",
        "SELECT id FROM users ORDER BY id DESC LIMIT 10",
    )

    assert "ordering_limit_error" in result["categories"]


def test_error_analysis_nested_query_error():
    assert "nested_query_error" in _category(
        "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders)",
        "SELECT id FROM users",
    )["categories"]


def test_error_analysis_alias_error():
    assert "alias_error" in _category(
        "SELECT u.id FROM users AS u", "SELECT users.id FROM users"
    )["categories"]


def test_error_analysis_structurally_different_fallback():
    result = _category("SELECT id + 1 FROM users", "SELECT id - 1 FROM users")

    assert result["categories"] == ["structurally_different"]


def test_error_analysis_categories_are_non_exclusive():
    result = _category(
        "SELECT COUNT(id) FROM users WHERE id > 1",
        "SELECT id FROM customers",
    )

    assert {"wrong_table", "aggregation_error", "filter_error"}.issubset(
        result["categories"]
    )


def test_error_analysis_primary_category_is_deterministic():
    first = _category(
        "SELECT COUNT(id) FROM users WHERE id > 1",
        "SELECT id FROM customers",
    )
    second = _category(
        "SELECT COUNT(id) FROM users WHERE id > 1",
        "SELECT id FROM customers",
    )

    assert first["categories"] == second["categories"]
    assert first["primary_category"] == second["primary_category"] == "wrong_table"


def test_error_summary_counts_and_rates():
    summary = summarize_error_analysis(
        [
            {"sample_id": 1, "target_sql": "SELECT id FROM users",
             "predicted_sql": "SELECT id FROM users"},
            {"sample_id": 2, "target_sql": "SELECT id FROM users",
             "predicted_sql": "SELECT name FROM users"},
            {"sample_id": 3, "target_sql": "SELECT id FROM users",
             "predicted_sql": ""},
        ]
    )

    assert summary["total"] == 3
    assert summary["exact_matches"] == 1
    assert summary["errors"] == 2
    assert summary["category_counts"]["wrong_column"] == 1
    assert summary["category_counts"]["empty_output"] == 1
    assert summary["category_rates"]["exact_match"] == pytest.approx(1 / 3)
    assert summary["category_rates"]["wrong_column"] == pytest.approx(1 / 3)
    assert summary["primary_category_counts"]["empty_output"] == 1


def test_error_summary_representative_example_limit_is_deterministic():
    records = [
        {
            "sample_id": sample_id,
            "target_sql": "SELECT id FROM users",
            "predicted_sql": f"SELECT column_{sample_id} FROM users",
        }
        for sample_id in range(4)
    ]

    summary = summarize_error_analysis(records, max_examples_per_category=2)

    examples = summary["representative_examples"]["wrong_column"]
    assert [example["sample_id"] for example in examples] == [0, 1]
    assert set(examples[0]) == {
        "sample_id", "target_sql", "predicted_sql", "error_message",
    }


def test_error_summary_empty_input():
    summary = summarize_error_analysis([])

    assert summary["total"] == 0
    assert summary["exact_matches"] == 0
    assert summary["errors"] == 0
    assert all(count == 0 for count in summary["category_counts"].values())
    assert all(rate == 0.0 for rate in summary["category_rates"].values())
    assert all(not examples for examples in summary["representative_examples"].values())


@pytest.mark.parametrize(
    "records",
    [
        [None],
        [{"target_sql": "SELECT 1"}],
        [{
            "sample_id": 1,
            "exact_match": False,
            "categories": ["not_a_category"],
            "primary_category": "not_a_category",
            "target_sql": "SELECT 1",
            "predicted_sql": "SELECT 2",
            "validity": None,
            "error_message": None,
        }],
    ],
)
def test_error_summary_rejects_malformed_records(records):
    with pytest.raises(ValueError, match="record"):
        summarize_error_analysis(records)


def test_evaluate_model_aggregates_unified_metrics():
    result = evaluate_model(
        TinyGenerator([[[1, 0], [3, 0]]]),
        [_batch(target_sql=["select 1;", "SELECT id FROM users"])],
        SqlTokenizer(),
        "cpu",
        include_validity=True,
        include_error_analysis=True,
    )

    assert result["raw_correct"] == 0
    assert result["raw_accuracy"] == 0.0
    assert result["normalized_correct"] == 1
    assert result["normalized_accuracy"] == pytest.approx(0.5)
    assert result["sql_valid_count"] == 1
    assert result["sql_validity_rate"] == pytest.approx(0.5)
    assert result["validity_category_counts"] == {
        "valid": 1,
        "syntax_error": 1,
    }
    assert result["inference_seconds"] >= 0.0
    assert result["average_latency_seconds"] == pytest.approx(
        result["inference_seconds"] / 2
    )


class RecordingBatchTokenizer:
    def __init__(self):
        self.prompt_batches = []

    def __call__(self, prompts, **kwargs):
        self.prompt_batches.append((list(prompts), dict(kwargs)))
        return {
            "input_ids": torch.arange(len(prompts)).reshape(-1, 1),
            "attention_mask": torch.ones((len(prompts), 1), dtype=torch.long),
        }


def test_build_evaluation_batches_preserves_example_order():
    examples = [
        {
            "sample_id": sample_id,
            "prompt": f"prompt-{sample_id}",
            "target_sql": f"SELECT {sample_id}",
            "schema_sql": "",
        }
        for sample_id in (4, 2, 9)
    ]
    tokenizer = RecordingBatchTokenizer()

    batches = build_evaluation_batches(examples, tokenizer, batch_size=2)

    assert [sample for batch in batches for sample in batch["sample_id"]] == [4, 2, 9]
    assert [target for batch in batches for target in batch["target_sql"]] == [
        "SELECT 4", "SELECT 2", "SELECT 9",
    ]
    assert [prompts for prompts, _kwargs in tokenizer.prompt_batches] == [
        ["prompt-4", "prompt-2"],
        ["prompt-9"],
    ]


def test_unified_batch_iterator_tokenizes_lazily_one_batch_at_a_time():
    examples = [
        {
            "sample_id": sample_id,
            "prompt": f"prompt-{sample_id}",
            "target_sql": f"SELECT {sample_id}",
            "schema_sql": "",
        }
        for sample_id in (0, 1, 2)
    ]
    tokenizer = RecordingBatchTokenizer()

    batches = evaluate_module._iter_evaluation_batches(
        examples,
        tokenizer,
        batch_size=2,
    )

    assert tokenizer.prompt_batches == []
    first = next(batches)
    assert first["sample_id"] == [0, 1]
    assert [prompts for prompts, _kwargs in tokenizer.prompt_batches] == [
        ["prompt-0", "prompt-1"],
    ]
    second = next(batches)
    assert second["sample_id"] == [2]
    assert [prompts for prompts, _kwargs in tokenizer.prompt_batches] == [
        ["prompt-0", "prompt-1"],
        ["prompt-2"],
    ]
    with pytest.raises(StopIteration):
        next(batches)


def _aligned_result(sample_ids=(0, 1), targets=("SELECT 1", "SELECT 2")):
    return {
        "records": [
            {"sample_id": sample_id, "target_sql": target}
            for sample_id, target in zip(sample_ids, targets)
        ]
    }


def test_verify_aligned_results_accepts_identical_order():
    verify_aligned_results(_aligned_result(), _aligned_result())


@pytest.mark.parametrize(
    "right",
    [
        _aligned_result(sample_ids=(1, 0)),
        _aligned_result(targets=("SELECT 2", "SELECT 1")),
        {"records": [{"sample_id": 0, "target_sql": "SELECT 1"}]},
    ],
)
def test_verify_aligned_results_rejects_misalignment(right):
    with pytest.raises(ValueError, match="differ|alignment"):
        verify_aligned_results(_aligned_result(), right)


def test_validate_checkpoint_directory_reports_missing_path(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Member 2 checkpoint does not exist"):
        validate_checkpoint_directory(missing, "Member 2")


def test_validate_checkpoint_directory_reports_incomplete_path(tmp_path):
    checkpoint = tmp_path / "incomplete"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        validate_checkpoint_directory(checkpoint, "Member 3")


def test_cli_argument_parsing():
    args = parse_args(
        [
            "--member2-checkpoint", "external/member2",
            "--member3-checkpoint", "external/member3",
            "--batch-size", "2",
            "--max-samples", "5",
            "--seed", "7",
            "--device", "cpu",
            "--num-beams", "4",
            "--max-new-tokens", "128",
        ]
    )

    assert args.member2_checkpoint == "external/member2"
    assert args.member3_checkpoint == "external/member3"
    assert args.batch_size == 2
    assert args.max_samples == 5
    assert args.seed == 7
    assert args.device == "cpu"
    assert args.num_beams == 4
    assert args.max_new_tokens == 128
