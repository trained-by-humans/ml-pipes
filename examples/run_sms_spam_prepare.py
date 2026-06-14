"""
Prepare the UCI SMS Spam Collection for spam-detection training.

The script downloads the dataset on first run, normalizes the text with a
`data_ops` pipeline, removes duplicates by normalized text, assigns
deterministic train/validation/test splits, and writes JSONL files ready for
model training.

Usage:
    python examples/run_sms_spam_prepare.py

    python examples/run_sms_spam_prepare.py \
        --lazy \
        --min-chars 8 \
        --min-tokens 3 \
        --inspect-html examples/.example_assets/sms_spam_prepare.html
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "examples"

from examples.common import ASSETS_DIR, add_assets_dir_arg
from ml_pipes import (
    CaptureCollector,
    CollectItems,
    Distinct,
    Filter,
    InputFn,
    LazyPerItem,
    MapValue,
    Pipeline,
    PipelineInspector,
    PerItem,
    StreamItems,
    WrapMappingInObject,
    data_factory,
    pipeline_factory,
)


SMS_SPAM_ARCHIVE_URLS = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
)
SMS_SPAM_ARCHIVE_NAME = "sms_spam_collection.zip"
SMS_SPAM_MEMBER_NAME = "SMSSpamCollection"
DEFAULT_SMS_SPAM_DIR = ASSETS_DIR / "sms_spam_collection"
DEFAULT_OUTPUT_DIR = ASSETS_DIR / "sms_spam_prepared"

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)")
_CURRENCY_RE = re.compile(r"(?i)(?:[$€£¥₹]+|\b(?:usd|eur|gbp|inr|cash)\b)")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PreparedSmsExample:
    raw: dict[str, str] = field(default_factory=dict)
    label: int = 0
    label_name: str = ""
    text: str = ""
    dedupe_key: str = ""
    split: str = ""
    char_count: int = 0
    token_count: int = 0


@dataclass(frozen=True)
class SmsSpamLineageReport:
    kept_records: list[dict[str, Any]]
    dropped_records: list[dict[str, Any]]
    duplicate_records: list[dict[str, Any]]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_lineage_row(row: dict[str, str], source_index: int) -> dict[str, Any]:
    return {
        "source_id": row["id"],
        "source_index": source_index,
        "source_label": row["label"],
        "source_text": row["text"],
        "source_text_sha256": _sha256_text(row["text"]),
    }


def _analyze_sms_spam_lineage(
    rows: list[dict[str, str]],
    *,
    min_chars: int,
    min_tokens: int,
    validation_ratio: float,
    test_ratio: float,
) -> SmsSpamLineageReport:
    splitter = _make_splitter(validation_ratio, test_ratio)
    kept_by_dedupe_key: dict[str, dict[str, Any]] = {}
    kept_records: list[dict[str, Any]] = []
    dropped_records: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []

    for source_index, row in enumerate(rows):
        base = _source_lineage_row(row, source_index)

        if not _has_known_label(row["label"]):
            dropped_records.append(
                {
                    **base,
                    "stage": "label",
                    "reason": "unknown_label",
                }
            )
            continue

        label = _normalize_label(row["label"])
        label_name = _label_name(label)
        normalized_text = _normalize_sms_text_or_none(row["text"])
        if normalized_text is None:
            dropped_records.append(
                {
                    **base,
                    "label": label,
                    "label_name": label_name,
                    "stage": "normalize",
                    "reason": "empty_after_normalization",
                }
            )
            continue

        char_count = _text_char_count(normalized_text)
        if char_count < min_chars:
            dropped_records.append(
                {
                    **base,
                    "label": label,
                    "label_name": label_name,
                    "normalized_text": normalized_text,
                    "char_count": char_count,
                    "min_chars": min_chars,
                    "stage": "min_chars",
                    "reason": "char_count_below_minimum",
                }
            )
            continue

        token_count = _text_token_count(normalized_text)
        if token_count < min_tokens:
            dropped_records.append(
                {
                    **base,
                    "label": label,
                    "label_name": label_name,
                    "normalized_text": normalized_text,
                    "char_count": char_count,
                    "token_count": token_count,
                    "min_tokens": min_tokens,
                    "stage": "min_tokens",
                    "reason": "token_count_below_minimum",
                }
            )
            continue

        dedupe_key = _dedupe_key(normalized_text)
        split = splitter(dedupe_key)
        lineage_row = {
            **base,
            "label": label,
            "label_name": label_name,
            "normalized_text": normalized_text,
            "dedupe_key": dedupe_key,
            "char_count": char_count,
            "token_count": token_count,
            "split": split,
        }
        kept = kept_by_dedupe_key.get(dedupe_key)
        if kept is None:
            kept_by_dedupe_key[dedupe_key] = lineage_row
            kept_records.append(lineage_row)
            continue

        duplicate_records.append(
            {
                **lineage_row,
                "stage": "dedupe",
                "reason": "duplicate_normalized_text",
                "kept_source_id": kept["source_id"],
                "kept_source_index": kept["source_index"],
                "kept_split": kept["split"],
            }
        )

    return SmsSpamLineageReport(
        kept_records=kept_records,
        dropped_records=dropped_records,
        duplicate_records=duplicate_records,
    )


def _validate_lineage_against_records(
    lineage_report: SmsSpamLineageReport,
    records: list[PreparedSmsExample],
) -> None:
    actual = [
        {
            "source_id": record.raw["id"],
            "label": record.label,
            "label_name": record.label_name,
            "normalized_text": record.text,
            "dedupe_key": record.dedupe_key,
            "char_count": record.char_count,
            "token_count": record.token_count,
            "split": record.split,
        }
        for record in records
    ]
    expected = [
        {
            "source_id": row["source_id"],
            "label": row["label"],
            "label_name": row["label_name"],
            "normalized_text": row["normalized_text"],
            "dedupe_key": row["dedupe_key"],
            "char_count": row["char_count"],
            "token_count": row["token_count"],
            "split": row["split"],
        }
        for row in lineage_report.kept_records
    ]
    if expected != actual:
        raise RuntimeError("SMS spam lineage analysis does not match prepared pipeline output.")


def _write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_prepare_counts(trace: InvocationTrace) -> tuple[int, int, int]:
    for span in trace.spans:
        if span.operator_type in {PerItem, LazyPerItem}:
            seen = span.attributes.get("seen")
            emitted = span.attributes.get("emitted")
            dropped = span.attributes.get("dropped")
            if isinstance(seen, int) and isinstance(emitted, int) and isinstance(dropped, int):
                return seen, emitted, dropped
            break
    raise RuntimeError("SMS spam prepare trace is missing PerItem/LazyPerItem counts.")


def _download_with_fallback(urls: tuple[str, ...], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as target:
                target.write(response.read())
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download SMS Spam Collection to {destination}") from last_error


def ensure_sms_spam_collection(
    assets_dir: Path | str = DEFAULT_SMS_SPAM_DIR,
    *,
    force_download: bool = False,
) -> Path:
    assets_path = Path(assets_dir)
    archive_path = assets_path / SMS_SPAM_ARCHIVE_NAME
    dataset_path = assets_path / SMS_SPAM_MEMBER_NAME

    if force_download:
        if archive_path.exists():
            archive_path.unlink()
        if dataset_path.exists():
            dataset_path.unlink()

    if dataset_path.exists():
        return dataset_path

    if not archive_path.exists():
        _download_with_fallback(SMS_SPAM_ARCHIVE_URLS, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extract(SMS_SPAM_MEMBER_NAME, assets_path)
    return dataset_path


def read_sms_spam_rows(dataset_path: Path | str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw_line = line.rstrip("\n")
            if not raw_line:
                continue
            try:
                label, text = raw_line.split("\t", 1)
            except ValueError:
                continue
            rows.append(
                {
                    "id": f"sms-{index:05d}",
                    "label": label,
                    "text": text,
                }
            )
    return rows


def _has_known_label(label: str) -> bool:
    return label.strip().lower() in {"ham", "spam"}


def _normalize_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "ham":
        return 0
    if normalized == "spam":
        return 1
    raise ValueError(f"Unsupported SMS spam label: {label!r}")


def _label_name(label: int) -> str:
    return "spam" if label == 1 else "ham"


def _normalize_sms_text_or_none(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", html.unescape(text))
    normalized = normalized.replace("\u00a0", " ").replace("\ufeff", " ")
    normalized = _CONTROL_RE.sub(" ", normalized)
    normalized = _URL_RE.sub(" [url] ", normalized)
    normalized = _EMAIL_RE.sub(" [email] ", normalized)
    normalized = _PHONE_RE.sub(" [phone] ", normalized)
    normalized = _CURRENCY_RE.sub(" [currency] ", normalized)
    normalized = _NUMBER_RE.sub(" [number] ", normalized)
    normalized = _REPEATED_PUNCT_RE.sub(r"\1", normalized)
    normalized = normalized.lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    if not normalized:
        return None
    return normalized


def _has_normalized_sms_text(text: str) -> bool:
    return _normalize_sms_text_or_none(text) is not None


def _normalize_sms_text(text: str) -> str:
    normalized = _normalize_sms_text_or_none(text)
    if normalized is None:
        raise ValueError("SMS text became empty after normalization.")
    return normalized


def _text_char_count(text: str) -> int:
    return len(text)


def _text_token_count(text: str) -> int:
    return len(text.split())


def _dedupe_key(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Dedupe key cannot be empty.")
    return cleaned


def _minimum_value_filter(minimum: int):
    def keep(value: int) -> bool:
        return value >= minimum

    return keep


def _validate_split_ratios(validation_ratio: float, test_ratio: float) -> None:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in [0, 1), got {validation_ratio}")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError(
            "validation_ratio + test_ratio must be < 1.0 "
            f"(got {validation_ratio + test_ratio:.3f})"
        )


def _make_splitter(validation_ratio: float, test_ratio: float):
    _validate_split_ratios(validation_ratio, test_ratio)
    test_cutoff = test_ratio
    validation_cutoff = test_ratio + validation_ratio

    def assign_split(text_key: str) -> str:
        digest = hashlib.sha1(text_key.encode("utf-8")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        if score < test_cutoff:
            return "test"
        if score < validation_cutoff:
            return "validation"
        return "train"

    return assign_split


@pipeline_factory
def sms_spam_prepare_pipeline(
    min_chars: int = 5,
    min_tokens: int = 2,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    lazy: bool = False,
) -> Pipeline[list[dict[str, str]], list[PreparedSmsExample]]:
    if min_chars < 0:
        raise ValueError(f"min_chars must be >= 0, got {min_chars}")
    if min_tokens < 0:
        raise ValueError(f"min_tokens must be >= 0, got {min_tokens}")

    splitter = _make_splitter(validation_ratio, test_ratio)
    region_open = LazyPerItem() if lazy else PerItem()
    region_close = StreamItems() if lazy else CollectItems()
    operators: list[Any] = [
        region_open,
        WrapMappingInObject(target="raw", state_factory=PreparedSmsExample),
        Filter(_has_known_label, source="raw.label"),
        MapValue(_normalize_label, source="raw.label", target="label"),
        MapValue(_label_name, source="label", target="label_name"),
        Filter(_has_normalized_sms_text, source="raw.text"),
        MapValue(_normalize_sms_text, source="raw.text", target="text"),
        MapValue(_text_char_count, source="text", target="char_count"),
        Filter(_minimum_value_filter(min_chars), source="char_count"),
        MapValue(_text_token_count, source="text", target="token_count"),
        Filter(_minimum_value_filter(min_tokens), source="token_count"),
        MapValue(_dedupe_key, source="text", target="dedupe_key"),
        MapValue(splitter, source="dedupe_key", target="split"),
        region_close,
        Distinct(source="dedupe_key"),
    ]

    pipeline = Pipeline(operators)
    pipeline.validate()
    return pipeline


@data_factory
def sms_spam_collection_input(
    assets_dir: str | Path = DEFAULT_SMS_SPAM_DIR,
    force_download: int = 0,
    sample_count: int = 0,
) -> InputFn:
    if sample_count < 0:
        raise ValueError(f"sample_count must be >= 0, got {sample_count}")

    dataset_path = ensure_sms_spam_collection(assets_dir, force_download=bool(force_download))
    rows = read_sms_spam_rows(dataset_path)
    if sample_count > 0:
        rows = rows[:sample_count]

    def fn() -> tuple[str, Any, str | None, dict | None]:
        return ("sms-spam-collection", rows, None, {"dataset_path": str(dataset_path)})

    return fn


def _serialize_record(record: PreparedSmsExample) -> dict[str, Any]:
    return {
        "id": record.raw["id"],
        "label": record.label,
        "label_name": record.label_name,
        "text": record.text,
        "char_count": record.char_count,
        "token_count": record.token_count,
        "split": record.split,
    }


def _materialize_records(records: Iterable[PreparedSmsExample]) -> list[PreparedSmsExample]:
    if isinstance(records, list):
        return records
    return list(records)


def _split_records(records: list[PreparedSmsExample]) -> dict[str, list[PreparedSmsExample]]:
    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        if record.split not in splits:
            raise ValueError(f"Unexpected split value: {record.split!r}")
        splits[record.split].append(record)
    return splits


def _label_counts(records: list[PreparedSmsExample]) -> dict[str, int]:
    counts = Counter(record.label_name for record in records)
    return {"ham": counts.get("ham", 0), "spam": counts.get("spam", 0)}


def _output_locations(records: list[PreparedSmsExample]) -> dict[str, dict[str, Any]]:
    locations: dict[str, dict[str, Any]] = {}
    for split_name, split_items in _split_records(records).items():
        output_file = f"{split_name}.jsonl"
        for output_index, record in enumerate(split_items):
            locations[record.raw["id"]] = {
                "output_file": output_file,
                "output_index": output_index,
            }
    return locations


def write_prepared_sms_dataset(
    records: list[PreparedSmsExample],
    *,
    raw_count: int,
    before_dedupe_count: int,
    dataset_path: Path,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    min_chars: int = 5,
    min_tokens: int = 2,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    lazy: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_records = _split_records(records)

    for split_name, split_items in split_records.items():
        jsonl_path = output_path / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in split_items:
                handle.write(json.dumps(_serialize_record(record), ensure_ascii=False) + "\n")

    summary = {
        "dataset_path": str(dataset_path),
        "raw_records": raw_count,
        "prepared_records_before_dedupe": before_dedupe_count,
        "prepared_records": len(records),
        "duplicates_removed": before_dedupe_count - len(records),
        "execution_mode": "lazy" if lazy else "eager",
        "min_chars": min_chars,
        "min_tokens": min_tokens,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "split_counts": {name: len(items) for name, items in split_records.items()},
        "label_counts": {
            "all": _label_counts(records),
            **{name: _label_counts(items) for name, items in split_records.items()},
        },
    }

    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _write_sms_spam_lineage_artifacts(
    *,
    records: list[PreparedSmsExample],
    lineage_report: SmsSpamLineageReport,
    dataset_path: Path,
    output_dir: Path | str,
    pipeline_description: str,
    run_args: dict[str, Any],
    input_selection: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_locations = _output_locations(records)

    kept_lineage = [
        {
            **row,
            **output_locations[row["source_id"]],
        }
        for row in lineage_report.kept_records
    ]
    duplicate_lineage = [
        {
            **row,
            **{
                "kept_output_file": output_locations[row["kept_source_id"]]["output_file"],
                "kept_output_index": output_locations[row["kept_source_id"]]["output_index"],
            },
        }
        for row in lineage_report.duplicate_records
    ]
    dropped_lineage = list(lineage_report.dropped_records)

    kept_lineage_path = output_path / "kept_lineage.jsonl"
    dropped_lineage_path = output_path / "dropped_lineage.jsonl"
    duplicate_lineage_path = output_path / "duplicate_lineage.jsonl"
    _write_jsonl(kept_lineage_path, kept_lineage)
    _write_jsonl(dropped_lineage_path, dropped_lineage)
    _write_jsonl(duplicate_lineage_path, duplicate_lineage)

    artifact_paths = {
        "train.jsonl": output_path / "train.jsonl",
        "validation.jsonl": output_path / "validation.jsonl",
        "test.jsonl": output_path / "test.jsonl",
        "summary.json": output_path / "summary.json",
        "kept_lineage.jsonl": kept_lineage_path,
        "dropped_lineage.jsonl": dropped_lineage_path,
        "duplicate_lineage.jsonl": duplicate_lineage_path,
    }
    manifest = {
        "version": 1,
        "pipeline": {
            "description": pipeline_description,
        },
        "run": {
            "args": run_args,
        },
        "input": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "selection": input_selection,
        },
        "output": {
            "path": str(output_path),
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in artifact_paths.items()
            },
        },
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the prepared JSONL files. Defaults to "
             "<assets-dir>/sms_spam_prepared.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Delete the cached archive/extracted file and download the dataset again.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=5,
        help="Drop messages shorter than this many characters after normalization.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Drop messages shorter than this many whitespace tokens after normalization.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio assigned deterministically from normalized text.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio assigned deterministically from normalized text.",
    )
    parser.add_argument(
        "--inspect-html",
        type=Path,
        default=None,
        help="Optional path for an inspection HTML report over a small raw sample.",
    )
    parser.add_argument(
        "--inspect-samples",
        type=int,
        default=24,
        help="How many raw rows to load when --inspect-html is used. "
             "The same subset is used for the inspection report and the prepared output.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Use the lazy per-item region before collection-level operators materialize the stream.",
    )
    parser.add_argument(
        "--inspect-orientation",
        choices=["horizontal", "vertical"],
        default="vertical",
        help="Layout used when rendering the optional inspection report.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    assets_dir = args.assets_dir / "sms_spam_collection"
    output_dir = args.output_dir or (args.assets_dir / "sms_spam_prepared")
    sample_count = args.inspect_samples if args.inspect_html is not None and args.inspect_samples > 0 else 0
    inspect_requested_samples = args.inspect_samples if args.inspect_html is not None else None
    pipeline = sms_spam_prepare_pipeline(
        min_chars=args.min_chars,
        min_tokens=args.min_tokens,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        lazy=args.lazy,
    )
    input_fn = sms_spam_collection_input(
        assets_dir=assets_dir,
        force_download=int(args.force_download),
        sample_count=sample_count,
    )
    _, raw_rows, _, input_metadata = input_fn()
    dataset_path = Path(input_metadata["dataset_path"])
    input_selection = {
        "strategy": "head" if sample_count > 0 else "full",
        "requested_rows": inspect_requested_samples,
        "selected_rows": len(raw_rows),
    }

    if args.inspect_html is not None and sample_count > 0:
        inspection = pipeline.inspect(raw_rows)
        saved = PipelineInspector().save_to_html(
            inspection,
            args.inspect_html,
            orientation=args.inspect_orientation,
        )
        print(f"Inspection report saved to: {saved}")

    pipeline_description = repr(pipeline)
    trace_collector = CaptureCollector()
    pipeline.set_tracing(trace_collector)
    deduped = _materialize_records(pipeline(raw_rows))

    if trace_collector.last_trace is None:
        raise RuntimeError("SMS spam prepare pipeline did not emit a trace.")

    raw_count, before_dedupe_count, _ = _read_prepare_counts(trace_collector.last_trace)

    lineage_report = _analyze_sms_spam_lineage(
        raw_rows,
        min_chars=args.min_chars,
        min_tokens=args.min_tokens,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    _validate_lineage_against_records(lineage_report, deduped)

    summary = write_prepared_sms_dataset(
        deduped,
        raw_count=raw_count,
        before_dedupe_count=before_dedupe_count,
        dataset_path=dataset_path,
        output_dir=output_dir,
        min_chars=args.min_chars,
        min_tokens=args.min_tokens,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        lazy=args.lazy,
    )
    _write_sms_spam_lineage_artifacts(
        records=deduped,
        lineage_report=lineage_report,
        dataset_path=dataset_path,
        output_dir=output_dir,
        pipeline_description=pipeline_description,
        run_args={
            "assets_dir": str(assets_dir),
            "output_dir": str(output_dir),
            "force_download": args.force_download,
            "min_chars": args.min_chars,
            "min_tokens": args.min_tokens,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "lazy": args.lazy,
            "inspect_html": str(args.inspect_html) if args.inspect_html is not None else None,
            "inspect_orientation": args.inspect_orientation if args.inspect_html is not None else None,
            "inspect_samples_requested": inspect_requested_samples,
            "selected_input_rows": len(raw_rows),
        },
        input_selection=input_selection,
    )

    print(f"Dataset cached at: {dataset_path}")
    print(f"Prepared output written to: {output_dir}")
    print(f"Execution mode: {summary['execution_mode']}")
    print(f"Raw rows: {summary['raw_records']}")
    print(f"Rows after filtering: {summary['prepared_records_before_dedupe']}")
    print(f"Rows after dedupe: {summary['prepared_records']}")
    print(f"Duplicates removed: {summary['duplicates_removed']}")
    print(
        "Rows filtered out before dedupe: "
        f"{summary['raw_records'] - summary['prepared_records_before_dedupe']}"
    )
    print(f"Split counts: {summary['split_counts']}")
    print(f"Label counts: {summary['label_counts']['all']}")
    print(f"Prepared rows materialized: {summary['prepared_records']}")
    print(f"Rows surviving before dedupe: {summary['prepared_records_before_dedupe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
