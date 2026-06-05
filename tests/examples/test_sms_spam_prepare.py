from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_sms_spam_prepare import (
    SMS_SPAM_MEMBER_NAME,
    build_sms_spam_prepare_pipeline,
    build_sms_spam_collection_input,
    prepare_sms_spam_dataset,
    read_sms_spam_rows,
    write_prepared_sms_dataset,
)

import pytest


def _write_sms_dataset(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for label, text in rows:
            handle.write(f"{label}\t{text}\n")


def test_sms_spam_collection_input_uses_cached_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / SMS_SPAM_MEMBER_NAME
    _write_sms_dataset(
        dataset_path,
        [
            ("ham", "See you at 6"),
            ("spam", "Win cash now"),
        ],
    )

    input_fn = build_sms_spam_collection_input(assets_dir=tmp_path)
    input_id, rows, tag, metadata = input_fn()

    assert input_id == "sms-spam-collection"
    assert tag is None
    assert metadata == {"dataset_path": str(dataset_path)}
    assert rows == [
        {"id": "sms-00000", "label": "ham", "text": "See you at 6"},
        {"id": "sms-00001", "label": "spam", "text": "Win cash now"},
    ]


@pytest.mark.parametrize("lazy", [False, True])
def test_sms_spam_prepare_pipeline_validates(lazy: bool) -> None:
    pipeline = build_sms_spam_prepare_pipeline(lazy=lazy)

    contract = pipeline.validate()

    assert contract is not None


@pytest.mark.parametrize("lazy", [False, True])
def test_sms_spam_prepare_pipeline_normalizes_filters_and_dedupes(tmp_path: Path, lazy: bool) -> None:
    dataset_path = tmp_path / SMS_SPAM_MEMBER_NAME
    _write_sms_dataset(
        dataset_path,
        [
            ("spam", "WIN cash now!!! http://promo.test/123"),
            ("spam", "WIN cash now!!! https://promo.test/456"),
            ("ham", "Call me at +1 555 123 4567 after 6"),
            ("ham", "Ok"),
        ],
    )
    rows = read_sms_spam_rows(dataset_path)

    pipeline = build_sms_spam_prepare_pipeline(
        min_chars=5,
        min_tokens=2,
        validation_ratio=0.2,
        test_ratio=0.2,
        dedupe=True,
        lazy=lazy,
    )
    prepared = pipeline(rows)

    assert len(prepared) == 2

    spam_record = next(record for record in prepared if record.label_name == "spam")
    ham_record = next(record for record in prepared if record.label_name == "ham")

    assert spam_record.text == "win [currency] now! [url]"
    assert spam_record.char_count == len("win [currency] now! [url]")
    assert spam_record.token_count == 4
    assert spam_record.dedupe_key == spam_record.text
    assert spam_record.split in {"train", "validation", "test"}

    assert ham_record.text == "call me at [phone] after [number]"
    assert ham_record.token_count == 6
    assert ham_record.split in {"train", "validation", "test"}

    summary = write_prepared_sms_dataset(
        prepared,
        raw_count=len(rows),
        before_dedupe_count=3,
        dataset_path=dataset_path,
        output_dir=tmp_path / "prepared",
        min_chars=5,
        min_tokens=2,
        validation_ratio=0.2,
        test_ratio=0.2,
        lazy=lazy,
    )

    assert summary["raw_records"] == 4
    assert summary["prepared_records_before_dedupe"] == 3
    assert summary["prepared_records"] == 2
    assert summary["duplicates_removed"] == 1
    assert summary["execution_mode"] == ("lazy" if lazy else "eager")
    assert (tmp_path / "prepared" / "summary.json").exists()


def test_prepare_sms_spam_dataset_lazy_matches_eager(tmp_path: Path) -> None:
    dataset_path = tmp_path / SMS_SPAM_MEMBER_NAME
    _write_sms_dataset(
        dataset_path,
        [
            ("spam", "WIN cash now!!! http://promo.test/123"),
            ("spam", "WIN cash now!!! https://promo.test/456"),
            ("ham", "Call me at +1 555 123 4567 after 6"),
            ("ham", "Ok"),
        ],
    )

    eager = prepare_sms_spam_dataset(
        assets_dir=tmp_path,
        output_dir=tmp_path / "prepared_eager",
        min_chars=5,
        min_tokens=2,
        validation_ratio=0.2,
        test_ratio=0.2,
        lazy=False,
    )
    lazy = prepare_sms_spam_dataset(
        assets_dir=tmp_path,
        output_dir=tmp_path / "prepared_lazy",
        min_chars=5,
        min_tokens=2,
        validation_ratio=0.2,
        test_ratio=0.2,
        lazy=True,
    )

    eager_dataset_path, eager_deduped, eager_summary = eager
    lazy_dataset_path, lazy_deduped, lazy_summary = lazy

    assert eager_dataset_path == lazy_dataset_path == dataset_path
    assert lazy_deduped == eager_deduped
    assert eager_summary["raw_records"] == 4
    assert eager_summary["prepared_records_before_dedupe"] == 3
    assert eager_summary["prepared_records"] == 2
    assert eager_summary["duplicates_removed"] == 1
    assert lazy_summary["raw_records"] == 4
    assert lazy_summary["prepared_records_before_dedupe"] == 3
    assert lazy_summary["prepared_records"] == 2
    assert lazy_summary["duplicates_removed"] == 1
    assert eager_summary["execution_mode"] == "eager"
    assert lazy_summary["execution_mode"] == "lazy"
