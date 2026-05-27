from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import examples.run_prepare_dataset_tracing as trace_script


def _submission(label: str, msg: str, **content_fields: str) -> dict[str, object]:
    content = {"msg": msg, **content_fields}
    return {"label": label, "content": content}


def _write_dataset(path: Path, submissions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"submissions": submissions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_run_trace_stream_pipeline_prints_trace_and_writes_output(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "prepared.json"
    _write_dataset(
        input_file,
        [
            _submission("spam", "Visit https://one.example"),
            _submission("ham", "Hello"),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_prepare_dataset_tracing.py",
            str(input_file),
            "--output-path",
            str(output_file),
            "--message-format",
            "normalized",
            "--overwrite",
        ],
    )
    exit_code = trace_script.main()
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "0:ResolveInputFiles" in stdout
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(payload["submissions"]) == 2


def test_save_inspection_report_collection_pipeline_saves_html(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "prepared.json"
    inspection_file = tmp_path / "prepared_trace.html"
    _write_dataset(
        input_file,
        [
            _submission("spam", "Visit https://one.example"),
            _submission("ham", "Hello"),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_prepare_dataset_tracing.py",
            "--pipeline",
            "collection",
            "--mode",
            "inspect",
            str(input_file),
            "--output-path",
            str(output_file),
            "--inspection-path",
            str(inspection_file),
            "--message-format",
            "normalized",
            "--overwrite",
        ],
    )
    exit_code = trace_script.main()
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert inspection_file.exists()
    assert output_file.exists()
    assert "0:ResolveInputFiles" in stdout
    assert "1:LoadSubmissionCollection" in stdout
    assert "Pipeline inspection" in inspection_file.read_text(encoding="utf-8")
