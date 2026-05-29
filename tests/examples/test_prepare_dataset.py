from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import examples.run_prepare_dataset as prepare_dataset
from ml_pipes import Pipeline
from ml_pipes.__main__ import _build_parser, cmd_run


def _submission(label: str, msg: str, **content_fields: str) -> dict[str, object]:
    content = {"msg": msg, **content_fields}
    return {"label": label, "content": content}


def _write_dataset(path: Path, submissions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"submissions": submissions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepared(label: str, msg: str, sort_key: str) -> prepare_dataset.PreparedSubmission:
    return prepare_dataset.PreparedSubmission(
        label=label,
        sort_key=sort_key,
        output_submission={"label": label, "content": {"msg": msg}},
    )


def _result(
    records: list[prepare_dataset.PreparedSubmission],
) -> list[prepare_dataset.PreparedSubmission]:
    return records


def _prepare_run(
    submissions: list[dict[str, object]],
    *,
    where: str = "",
    label_limits: str = "",
    dedupe_key: str = "normalized",
    message_format: str = "raw",
    min_length: int | str = 2,
    is_jp: bool | str = True,
) -> list[prepare_dataset.PreparedSubmission]:
    dedupe_selector = "cleaned_text" if dedupe_key == "cleaned" else "normalized_text"
    output_message_selector = {
        "raw": "submission.content.msg",
        "cleaned": "cleaned_text",
        "normalized": "normalized_text",
    }[message_format]
    normalize_training_text = partial(
        prepare_dataset.normalize_text_for_training,
        is_jp=prepare_dataset.parse_bool(is_jp),
        already_prepared=True,
    )
    pipeline = Pipeline(
        [
            prepare_dataset.ForEachItem(),
            prepare_dataset.WrapMappingInObject(
                as_="submission",
                state_factory=prepare_dataset.MessageState,
            ),
            prepare_dataset.FilterByExpression(src="submission", expression=where),
            prepare_dataset.RequireTextValue(src="submission.label", strip=True, allow_blank=False),
            prepare_dataset.FilterByAllowedValues(
                src="submission.label",
                allowed_values=prepare_dataset.parse_label_limits(label_limits),
            ),
            prepare_dataset.RequireTextValue(src="submission.content.msg"),
            prepare_dataset.MapValue(
                src="submission.content.msg",
                fn=prepare_dataset.prepare_text_content,
                as_="cleaned_text",
            ),
            prepare_dataset.RequireMinimumLength(src="cleaned_text", min_length=min_length),
            prepare_dataset.MapValue(
                src="cleaned_text",
                fn=normalize_training_text,
                as_="normalized_text",
            ),
            prepare_dataset.RequireMinimumLength(src=dedupe_selector, min_length=min_length),
            prepare_dataset.EndForEachItem(),
            prepare_dataset.Distinct(src=dedupe_selector),
            prepare_dataset.LimitByValue(src="submission.label", limits=label_limits),
            prepare_dataset.BuildPreparedRecords(
                submission="submission",
                label="submission.label",
                text=output_message_selector,
                sort_text=dedupe_selector,
            ),
        ]
    )
    return pipeline(submissions)


@dataclass
class CustomSubmissionCarrier:
    payload: dict[str, object] | None = None
    grouped_text: str | None = None
    training_text: str | None = None


def test_resolve_input_files_expands_directory_in_sorted_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "collected"
    _write_dataset(input_dir / "b.json", [])
    _write_dataset(input_dir / "a.json", [])

    resolved = prepare_dataset.ResolveInputFiles()(input_dir)

    assert [path.name for path in resolved] == ["a.json", "b.json"]


def test_load_submission_collection_merges_files_in_sorted_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "collected"
    _write_dataset(input_dir / "b.json", [_submission("spam", "second")])
    _write_dataset(input_dir / "a.json", [_submission("ham", "first")])

    input_files = prepare_dataset.ResolveInputFiles()(input_dir)
    loaded = prepare_dataset.LoadSubmissionCollection()(input_files)

    assert [item["content"]["msg"] for item in loaded if isinstance(item, dict)] == ["first", "second"]


def test_resolve_input_path_checks_cwd_before_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = tmp_path / "cwd"
    project = tmp_path / "project"
    cwd.mkdir()
    project.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(prepare_dataset, "PROJECT_ROOT", project)

    local_file = cwd / "sample.json"
    project_file = project / "sample.json"
    _write_dataset(local_file, [])
    _write_dataset(project_file, [])

    assert prepare_dataset.resolve_input_path("sample.json") == local_file.resolve()


def test_resolve_input_path_falls_back_to_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = tmp_path / "cwd"
    project = tmp_path / "project"
    cwd.mkdir()
    project.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(prepare_dataset, "PROJECT_ROOT", project)

    project_file = project / "dataset" / "sample.json"
    _write_dataset(project_file, [])

    assert prepare_dataset.resolve_input_path("dataset/sample.json") == project_file.resolve()


def test_prepare_submissions_applies_filter_cleanup_and_message_format() -> None:
    submissions = [
        _submission("spam", "  Visit https://example.com \n", gw="rakuten"),
        _submission("spam", "Ignore me", gw="other"),
    ]

    result = _prepare_run(
        submissions,
        where='content.gw == "rakuten" and label == "spam"',
        message_format="normalized",
    )

    assert len(result) == 1
    assert result[0].output_submission["content"]["msg"] == "visit <URL>"
    assert result[0].output_submission["content"]["gw"] == "rakuten"


def test_prepare_submissions_dedupe_respects_cleaned_vs_normalized() -> None:
    submissions = [
        _submission("spam", "Visit https://one.example"),
        _submission("spam", "Visit https://two.example"),
    ]

    cleaned_result = _prepare_run(submissions, dedupe_key="cleaned")
    normalized_result = _prepare_run(submissions, dedupe_key="normalized")

    assert len(cleaned_result) == 2
    assert len(normalized_result) == 1


def test_prepare_submissions_enforces_label_limits_and_stops_early() -> None:
    submissions = [
        _submission("ham", "first ham"),
        _submission("spam", "first spam"),
        _submission("ham", "second ham"),
    ]

    result = _prepare_run(submissions, label_limits="ham=1,spam=1")

    assert [record.output_submission["content"]["msg"] for record in result] == [
        "first ham",
        "first spam",
    ]
    assert Counter(record.label for record in result) == Counter({"ham": 1, "spam": 1})


def test_prepare_submissions_raises_when_label_limits_cannot_be_satisfied() -> None:
    submissions = [_submission("ham", "only one ham")]

    with pytest.raises(ValueError, match="Missing counts"):
        _prepare_run(submissions, label_limits="ham=2")


def test_prepare_submissions_accepts_custom_object_layout() -> None:
    pipeline = Pipeline(
        [
            prepare_dataset.ForEachItem(),
            prepare_dataset.WrapMappingInObject(
                as_="payload",
                state_factory=CustomSubmissionCarrier,
            ),
            prepare_dataset.FilterByExpression(
                src="payload",
                expression='content.gw == "rakuten"',
            ),
            prepare_dataset.RequireTextValue(src="payload.label", strip=True, allow_blank=False),
            prepare_dataset.FilterByAllowedValues(src="payload.label", allowed_values={"spam"}),
            prepare_dataset.RequireTextValue(src="payload.content.msg"),
            prepare_dataset.MapValue(
                src="payload.content.msg",
                fn=prepare_dataset.prepare_text_content,
                as_="grouped_text",
            ),
            prepare_dataset.RequireMinimumLength(src="grouped_text", min_length=2),
            prepare_dataset.MapValue(
                src="grouped_text",
                fn=partial(
                    prepare_dataset.normalize_text_for_training,
                    is_jp=True,
                    already_prepared=True,
                ),
                as_="training_text",
            ),
            prepare_dataset.RequireMinimumLength(src="training_text", min_length=2),
            prepare_dataset.EndForEachItem(),
            prepare_dataset.Distinct(src="training_text"),
            prepare_dataset.LimitByValue(src="payload.label", limits="spam=1"),
            prepare_dataset.BuildPreparedRecords(
                submission="payload",
                label="payload.label",
                text="training_text",
                sort_text="training_text",
            ),
        ]
    )

    result = pipeline(
        [
            _submission("spam", " Visit https://one.example ", gw="rakuten"),
            _submission("spam", "ignored", gw="other"),
        ]
    )

    assert len(result) == 1
    assert result[0].label == "spam"
    assert result[0].output_submission["content"]["msg"] == "visit <URL>"


def test_finalize_ordering_rejects_sort_labels_without_shuffle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sort_labels requires shuffle"):
        prepare_dataset.FinalizeOrdering(sort_labels=True)


def test_finalize_ordering_preserves_order_when_shuffle_is_disabled(tmp_path: Path) -> None:
    result = _result(
        [
            _prepared("spam", "msg-1", "spam\tmsg-1\t0001"),
            _prepared("ham", "msg-2", "ham\tmsg-2\t0002"),
        ],
    )

    ordered = prepare_dataset.FinalizeOrdering()(result)

    assert [record.output_submission["content"]["msg"] for record in ordered] == ["msg-1", "msg-2"]


def test_finalize_ordering_is_deterministic_with_seeded_shuffle(tmp_path: Path) -> None:
    records = [
        _prepared("spam", "msg-1", "spam\tmsg-1\t0001"),
        _prepared("ham", "msg-2", "ham\tmsg-2\t0002"),
        _prepared("spam", "msg-3", "spam\tmsg-3\t0003"),
    ]

    first = prepare_dataset.FinalizeOrdering(shuffle=42)(_result(list(records)))
    second = prepare_dataset.FinalizeOrdering(shuffle=42)(_result(list(records)))

    assert [record.output_submission["content"]["msg"] for record in first] == [
        record.output_submission["content"]["msg"] for record in second
    ]


def test_finalize_ordering_sort_labels_ignores_collected_order(tmp_path: Path) -> None:
    records_a = [
        _prepared("spam", "spam-b", "spam\tb\t0002"),
        _prepared("ham", "ham-a", "ham\ta\t0001"),
        _prepared("spam", "spam-a", "spam\ta\t0003"),
    ]
    records_b = list(reversed(records_a))

    ordered_a = prepare_dataset.FinalizeOrdering(shuffle=42, sort_labels=True)(_result(records_a))
    ordered_b = prepare_dataset.FinalizeOrdering(shuffle=42, sort_labels=True)(_result(records_b))

    assert [record.output_submission["content"]["msg"] for record in ordered_a] == [
        record.output_submission["content"]["msg"] for record in ordered_b
    ]


def test_build_prepare_dataset_pipeline_writes_wrapper_and_respects_overwrite(tmp_path: Path) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "prepared.json"
    _write_dataset(
        input_file,
        [
            _submission("spam", "  Hello \n"),
            _submission("ham", "second message"),
        ],
    )

    pipeline = prepare_dataset.build_prepare_dataset_pipeline(
        {"output_path": output_file, "message_format": "cleaned"}
    )

    result = pipeline(str(input_file))
    payload = json.loads(output_file.read_text(encoding="utf-8"))

    assert isinstance(result, list)
    assert len(result) == 2
    assert list(payload) == ["submissions"]
    assert payload["submissions"][0]["content"]["msg"] == "Hello"
    assert output_file.read_text(encoding="utf-8").startswith('{\n  "submissions": [')

    with pytest.raises(FileExistsError, match="Output already exists"):
        pipeline(str(input_file))


def test_prepare_dataset_pipeline_validates() -> None:
    contract = prepare_dataset.build_prepare_dataset_pipeline({"output_path": "prepared.json"}).validate()

    assert contract is not None


def test_build_prepare_dataset_collection_pipeline_matches_streaming_pipeline(tmp_path: Path) -> None:
    input_file = tmp_path / "input.json"
    stream_output = tmp_path / "prepared_stream.json"
    collection_output = tmp_path / "prepared_collection.json"
    _write_dataset(
        input_file,
        [
            _submission("spam", "Visit https://one.example"),
            _submission("ham", "Hello"),
            _submission("spam", "Visit https://two.example"),
            _submission("spam", "Unique offer", gw="rakuten"),
        ],
    )

    stream_pipeline = prepare_dataset.build_prepare_dataset_pipeline(
        {
            "output_path": stream_output,
            "message_format": "normalized",
            "overwrite": "true",
            "shuffle": 42,
            "sort_labels": "true",
        }
    )
    collection_pipeline = prepare_dataset.build_prepare_dataset_collection_pipeline(
        output_path=collection_output,
        message_format="normalized",
        overwrite="true",
        shuffle=42,
        sort_labels="true",
    )

    stream_result = stream_pipeline(str(input_file))
    collection_result = collection_pipeline(str(input_file))

    assert json.loads(stream_output.read_text(encoding="utf-8")) == json.loads(
        collection_output.read_text(encoding="utf-8")
    )
    assert stream_result == collection_result


def test_prepare_dataset_pipeline_inspect_captures_streaming_steps(tmp_path: Path) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "prepared.json"
    _write_dataset(
        input_file,
        [
            _submission("spam", "Visit https://one.example"),
            _submission("ham", "Hello"),
        ],
    )

    pipeline = prepare_dataset.build_prepare_dataset_pipeline(
        {"output_path": output_file, "message_format": "normalized", "overwrite": "true"}
    )

    result = pipeline.inspect(str(input_file))

    assert len(result.spans) == 8
    assert [span.label for span in result.spans] == [
        "0:ResolveInputFiles",
        "1:StreamSubmissions",
        "2:ForEachItem",
        "13:Distinct",
        "14:LimitByValue",
        "15:BuildPreparedRecords",
        "16:FinalizeOrdering",
        "17:WritePreparedDataset",
    ]
    assert not any(span.error for span in result.spans)
    assert isinstance(result.spans[1].output_value, str)
    assert "generator object" in result.spans[1].output_value
    assert result.spans[2].child_trace is not None
    assert [span.label for span in result.spans[2].child_trace.spans] == [
        "3:WrapMappingInObject",
        "4:FilterByExpression",
        "5:RequireTextValue",
        "6:FilterByAllowedValues",
        "7:RequireTextValue",
        "8:MapValue",
        "9:RequireMinimumLength",
        "10:MapValue",
        "11:RequireMinimumLength",
    ]
    assert isinstance(result.spans[3].output_value, list)
    assert len(result.spans[3].output_value) == 2
    assert output_file.exists()


def test_prepare_dataset_collection_pipeline_inspect_captures_collection_steps(tmp_path: Path) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "prepared.json"
    _write_dataset(
        input_file,
        [
            _submission("spam", "Visit https://one.example"),
            _submission("ham", "Hello"),
        ],
    )

    pipeline = prepare_dataset.build_prepare_dataset_collection_pipeline(
        output_path=output_file,
        message_format="normalized",
        overwrite="true",
    )

    result = pipeline.inspect(str(input_file))

    assert len(result.spans) == 8
    assert [span.label for span in result.spans] == [
        "0:ResolveInputFiles",
        "1:LoadSubmissionCollection",
        "2:ForEachItem",
        "13:Distinct",
        "14:LimitByValue",
        "15:BuildPreparedRecords",
        "16:FinalizeOrdering",
        "17:WritePreparedDataset",
    ]
    assert not any(span.error for span in result.spans)
    assert isinstance(result.spans[1].output_value, list)
    assert len(result.spans[1].output_value) == 2
    assert result.spans[2].child_trace is not None
    assert [span.label for span in result.spans[2].child_trace.spans] == [
        "3:WrapMappingInObject",
        "4:FilterByExpression",
        "5:RequireTextValue",
        "6:FilterByAllowedValues",
        "7:RequireTextValue",
        "8:MapValue",
        "9:RequireMinimumLength",
        "10:MapValue",
        "11:RequireMinimumLength",
    ]
    assert isinstance(result.spans[3].output_value, list)
    assert len(result.spans[3].output_value) == 2
    assert output_file.exists()


def test_cmd_run_prepare_dataset_example(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "prepared.json"
    _write_dataset(
        input_file,
        [
            _submission("spam", "Visit https://one.example"),
            _submission("ham", "Hello"),
            _submission("spam", "Visit https://two.example"),
        ],
    )

    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "examples.run_prepare_dataset",
            "--data-arg",
            f"input_path={input_file}",
            "--arg",
            f"output_path={output_file}",
            "--arg",
            "message_format=normalized",
            "--arg",
            "label_limits=ham=1,spam=1",
            "--arg",
            "shuffle=42",
            "--arg",
            "sort_labels=true",
            "--arg",
            "overwrite=true",
        ]
    )

    code = cmd_run(args)
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out

    assert code == 0
    assert len(payload["submissions"]) == 2
    assert sorted(item["label"] for item in payload["submissions"]) == ["ham", "spam"]
    assert "Final summary" in stdout


def test_cmd_run_prepare_dataset_collection_example(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "prepared.json"
    _write_dataset(
        input_file,
        [
            _submission("spam", "Visit https://one.example"),
            _submission("ham", "Hello"),
            _submission("spam", "Visit https://two.example"),
        ],
    )

    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "examples.run_prepare_dataset:build_prepare_dataset_collection_pipeline",
            "--data-arg",
            f"input_path={input_file}",
            "--arg",
            f"output_path={output_file}",
            "--arg",
            "message_format=normalized",
            "--arg",
            "label_limits=ham=1,spam=1",
            "--arg",
            "shuffle=42",
            "--arg",
            "sort_labels=true",
            "--arg",
            "overwrite=true",
        ]
    )

    code = cmd_run(args)
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out

    assert code == 0
    assert len(payload["submissions"]) == 2
    assert sorted(item["label"] for item in payload["submissions"]) == ["ham", "spam"]
    assert "Final summary" in stdout
