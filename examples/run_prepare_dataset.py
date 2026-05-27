"""
Streaming dataset-preparation example built with ml-pipes.

Usage:
    python -m ml_pipes run examples.run_prepare_dataset \
        --data-arg input_path=dataset/collected \
        --arg output_path=dataset/generated/prepared.json \
        --arg where='content.gw == "rakuten"' \
        --arg label_limits=ham=2000,spam=200 \
        --arg dedupe_key=normalized \
        --arg message_format=normalized \
        --arg shuffle=42 \
        --arg sort_labels=true \
        --arg overwrite=true

    python -m ml_pipes run examples.run_prepare_dataset:build_prepare_dataset_collection_pipeline \
        --data-arg input_path=dataset/collected \
        --arg output_path=dataset/generated/prepared_collection.json \
        --arg overwrite=true
"""
from __future__ import annotations

import ast
import json
import operator
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ml_pipes import (
    InputFn,
    InvocationTrace,
    Pipeline,
    RegionCloser,
    RegionOpener,
    SideEffectOp,
    StepSpan,
    data_factory,
    pipeline_factory,
)
from ml_pipes.tracing import _NoOpTrace, merge_traces


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_DATASET_TOP_LEVEL_KEY = "submissions"


EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
HTTP_URL_RE = re.compile(r"\bhttps?://[^\s]+", flags=re.IGNORECASE)
BARE_DOMAIN_RE = re.compile(
    r"\b(?!(?:<EMAIL>))(?:[a-zA-Z0-9-]+\.)+(?:[a-zA-Z]{2,})(?:/[^\s]*)?",
    flags=re.IGNORECASE,
)
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")
DATETIME_RE = re.compile(r"\b(?:\d{4}|\d{2})[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?\b")
DATE_RE = re.compile(r"\b(?:\d{4}|\d{2})[-/]\d{1,2}[-/]\d{1,2}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
PIN_KEYWORDS = r"(?:pin|otp|one[- ]time pin|one[- ]time password|verification code|code)"
PIN_WITH_NUM_RE = re.compile(
    rf"(?P<prefix>{PIN_KEYWORDS}[^0-9]{{0,10}})(?P<pin>[0-9]{{4,8}})",
    flags=re.IGNORECASE,
)
BOOKING_KEYWORDS = r"(?:reservation|booking|ref(?:erence)? ?(?:no\.?)?)"
BOOKING_CODE_RE = re.compile(
    rf"(?P<prefix>{BOOKING_KEYWORDS}[^A-Za-z0-9]{{0,10}})(?P<code>[A-Za-z0-9]{{5,12}})",
    flags=re.IGNORECASE,
)
LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
ALNUM_CODE_RE = re.compile(
    r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,32}\b"
)
SLASH_CODE_RE = re.compile(r"\b(?=.*[A-Za-z])[A-Za-z0-9]{2,}/[A-Za-z0-9]{2,}\b")
PARAM_STRING_RE = re.compile(
    r"^[A-Za-z0-9_-]+\?(?:[A-Za-z0-9_-]+=[^;]+)(?:;[A-Za-z0-9_-]+=[^;]+)*$"
)


class FilterSyntaxError(ValueError):
    """Raised when a filter expression uses unsupported syntax."""


Submission = dict[str, object]


@dataclass(frozen=True)
class PreparedSubmission:
    label: str
    sort_key: str
    output_submission: Submission


@dataclass(frozen=True)
class ShuffleConfig:
    enabled: bool
    seed: int | None
    seed_source: str


@dataclass(frozen=True)
class ScannedItem:
    scan_index: int
    value: object


@dataclass(frozen=True)
class ScannedSubmission:
    scan_index: int
    submission: Submission


@dataclass(frozen=True)
class LabeledSubmission:
    scan_index: int
    submission: Submission
    label: str


@dataclass(frozen=True)
class MessageSubmission:
    scan_index: int
    submission: Submission
    label: str
    raw_message: str


@dataclass(frozen=True)
class GroupedSubmission:
    scan_index: int
    submission: Submission
    label: str
    raw_message: str
    cleaned_message: str


@dataclass(frozen=True)
class NormalizedSubmission:
    scan_index: int
    submission: Submission
    label: str
    raw_message: str
    cleaned_message: str
    normalized_message: str


@dataclass(frozen=True)
class DedupeReadySubmission:
    scan_index: int
    submission: Submission
    label: str
    raw_message: str
    cleaned_message: str
    normalized_message: str
    dedupe_key: str


@dataclass(frozen=True)
class PreparedCandidate:
    label: str
    dedupe_key: str
    record: PreparedSubmission


@dataclass
class PreparationRun:
    records: list[PreparedSubmission] = field(default_factory=list)


def _replace_pin(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}<PIN>"


def _replace_booking_code(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}<CODE>"


def normalize_text(text: str, *, to_lower: bool = True, is_jp: bool = True) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    if to_lower:
        text = text.lower()

    if PARAM_STRING_RE.match(text):
        return "<PARAM_STRING>"

    text = EMAIL_RE.sub("<EMAIL>", text)
    text = HTTP_URL_RE.sub("<URL>", text)
    text = BARE_DOMAIN_RE.sub("<URL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = DATETIME_RE.sub("<DATETIME>", text)
    text = DATE_RE.sub("<DATE>", text)
    text = TIME_RE.sub("<TIME>", text)
    text = PIN_WITH_NUM_RE.sub(_replace_pin, text)
    text = BOOKING_CODE_RE.sub(_replace_booking_code, text)
    text = LONG_NUMBER_RE.sub("<NUM>", text)
    text = SLASH_CODE_RE.sub("<CODE>", text)
    text = ALNUM_CODE_RE.sub("<CODE>", text)

    if is_jp:
        text = re.sub(r"(?<=[\u4e00-\u9fff\u3040-\u30ff\u30fc]),", "、", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff\u3040-\u30ff\u30fc])\.(?=\s|$)", "。", text)

    return text


def clean_whitespace(text: str) -> str:
    text = text.strip()
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_garbled_text(text: str) -> str:
    cleaned_chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"L", "N", "P", "S", "Z"}:
            cleaned_chars.append(char)
        elif category == "Cc" and char in {"\n", "\t", "\r"}:
            cleaned_chars.append(char)
    return "".join(cleaned_chars)


def clean_escape_sequences(text: str) -> str:
    text = text.replace("\\\\", "\\")
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    text = text.replace("\t", " ")
    return text


def prepare_message_for_grouping(msg: str | None) -> str:
    if msg is None:
        return ""

    cleaned = remove_garbled_text(msg)
    cleaned = clean_escape_sequences(cleaned)
    cleaned = clean_whitespace(cleaned)
    return cleaned


def normalize_message_for_training(
    msg: str | None,
    *,
    is_jp: bool = False,
    already_prepared: bool = False,
) -> str:
    prepared = msg if already_prepared else prepare_message_for_grouping(msg)
    return normalize_text(prepared, is_jp=is_jp)


def _coerce_path_string(path_str: str | Path) -> tuple[str, Path]:
    raw = str(path_str)
    return raw, Path(raw).expanduser()


def resolve_input_path(path_str: str | Path) -> Path:
    raw, path = _coerce_path_string(path_str)
    if path.is_absolute():
        return path
    if raw.startswith(("./", "../")):
        return (Path.cwd() / path).resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (PROJECT_ROOT / path).resolve()


def resolve_output_path(path_str: str | Path) -> Path:
    raw, path = _coerce_path_string(path_str)
    if path.is_absolute():
        return path
    if raw.startswith(("./", "../")):
        return (Path.cwd() / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_label_limits(raw_limits: str | Mapping[str, int] | None) -> dict[str, int]:
    if raw_limits is None:
        return {}
    if isinstance(raw_limits, Mapping):
        return {str(label): int(count) for label, count in raw_limits.items()}

    limits: dict[str, int] = {}
    for raw_limit in str(raw_limits).replace(";", ",").split(","):
        candidate = raw_limit.strip()
        if not candidate:
            continue

        if "=" in candidate:
            label, count_str = candidate.split("=", 1)
        elif ":" in candidate:
            label, count_str = candidate.split(":", 1)
        else:
            raise ValueError(f"Invalid label limit: {candidate}. Expected label=count.")

        label = label.strip()
        if not label:
            raise ValueError(f"Label cannot be empty in label limit: {candidate}")

        count = int(count_str.strip())
        if count < 0:
            raise ValueError(f"Label limit must be >= 0: {candidate}")
        limits[label] = count

    return limits


def parse_bool(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def normalize_choice(name: str, value: str, allowed: set[str]) -> str:
    candidate = value.strip().lower()
    if candidate not in allowed:
        raise ValueError(f"Invalid {name}: {value!r}. Expected one of {sorted(allowed)}.")
    return candidate


def resolve_shuffle_config(raw_value: int | str | None) -> ShuffleConfig:
    if raw_value is None:
        return ShuffleConfig(enabled=False, seed=None, seed_source="disabled")
    if isinstance(raw_value, bool):
        if raw_value:
            raise ValueError("shuffle must be 'disabled', 'auto', or a non-negative integer seed.")
        return ShuffleConfig(enabled=False, seed=None, seed_source="disabled")
    if isinstance(raw_value, int):
        if raw_value < 0:
            raise ValueError("shuffle seed must be >= 0.")
        return ShuffleConfig(enabled=True, seed=raw_value, seed_source="provided")

    normalized = str(raw_value).strip().lower()
    if normalized in {"", "disabled", "none", "false"}:
        return ShuffleConfig(enabled=False, seed=None, seed_source="disabled")
    if normalized == "auto":
        seed = random.SystemRandom().randrange(0, 2**32)
        return ShuffleConfig(enabled=True, seed=seed, seed_source="generated")

    seed = int(normalized)
    if seed < 0:
        raise ValueError("shuffle seed must be >= 0.")
    return ShuffleConfig(enabled=True, seed=seed, seed_source="provided")


@lru_cache(maxsize=128)
def parse_filter_expression(expression: str) -> ast.AST:
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FilterSyntaxError(f"Invalid filter expression: {expression}") from exc


def extract_attribute_path(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))

    return None


def resolve_field_path(submission: Mapping[str, object], field_path: str) -> object:
    path = field_path.strip()
    if not path:
        return None

    parts = [part for part in path.split(".") if part]
    if not parts:
        return None

    if parts[0] in {"submission", "submissions"}:
        parts = parts[1:]

    current: object = submission
    for part in parts:
        if current is submission and part == "content":
            current = submission.get("content", {})
            continue

        if isinstance(current, dict) and part in current:
            current = current[part]
            continue

        if current is submission:
            content = submission.get("content", {})
            if isinstance(content, dict) and part in content:
                current = content[part]
                continue

        return None

    return current


def call_allowed_function(func_name: str, args: list[object], submission: Mapping[str, object]) -> object:
    if func_name == "field":
        if len(args) != 1 or not isinstance(args[0], str):
            raise FilterSyntaxError("field() expects one string path argument.")
        return resolve_field_path(submission, args[0])

    if func_name == "exists":
        if len(args) != 1 or not isinstance(args[0], str):
            raise FilterSyntaxError("exists() expects one string path argument.")
        return resolve_field_path(submission, args[0]) is not None

    if func_name == "contains":
        if len(args) != 2:
            raise FilterSyntaxError("contains() expects two arguments.")
        haystack, needle = args
        return needle in haystack if haystack is not None else False

    if func_name == "len":
        if len(args) != 1:
            raise FilterSyntaxError("len() expects one argument.")
        return len(args[0])

    if func_name == "lower":
        if len(args) != 1:
            raise FilterSyntaxError("lower() expects one argument.")
        return str(args[0]).lower()

    if func_name == "upper":
        if len(args) != 1:
            raise FilterSyntaxError("upper() expects one argument.")
        return str(args[0]).upper()

    raise FilterSyntaxError(f"Unsupported function in filter: {func_name}")


def evaluate_comparison(op_node: ast.cmpop, left: object, right: object) -> bool:
    operations: dict[type[ast.cmpop], Any] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    op_type = type(op_node)
    if op_type in operations:
        return operations[op_type](left, right)

    if isinstance(op_node, ast.In):
        try:
            return left in right
        except TypeError:
            return False

    if isinstance(op_node, ast.NotIn):
        try:
            return left not in right
        except TypeError:
            return True

    raise FilterSyntaxError(
        f"Unsupported comparison operator: {ast.dump(op_node, include_attributes=False)}"
    )


def evaluate_filter_node(node: ast.AST, submission: Mapping[str, object]) -> object:
    if isinstance(node, ast.Expression):
        return evaluate_filter_node(node.body, submission)

    if isinstance(node, ast.BoolOp):
        values = [evaluate_filter_node(value, submission) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise FilterSyntaxError("Unsupported boolean operator in filter.")

    if isinstance(node, ast.UnaryOp):
        operand = evaluate_filter_node(node.operand, submission)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise FilterSyntaxError("Unsupported unary operator in filter.")

    if isinstance(node, ast.Compare):
        left = evaluate_filter_node(node.left, submission)
        for operator_node, comparator in zip(node.ops, node.comparators, strict=False):
            right = evaluate_filter_node(comparator, submission)
            if not evaluate_comparison(operator_node, left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Name):
        if node.id in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[node.id]
        return resolve_field_path(submission, node.id)

    if isinstance(node, ast.Attribute):
        path = extract_attribute_path(node)
        if path is None:
            raise FilterSyntaxError("Unsupported attribute usage in filter.")
        return resolve_field_path(submission, ".".join(path))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FilterSyntaxError("Only simple helper functions are allowed in filters.")
        args = [evaluate_filter_node(arg, submission) for arg in node.args]
        return call_allowed_function(node.func.id, args, submission)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [evaluate_filter_node(element, submission) for element in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(evaluate_filter_node(element, submission) for element in node.elts)

    if isinstance(node, ast.Set):
        return {evaluate_filter_node(element, submission) for element in node.elts}

    raise FilterSyntaxError(
        f"Unsupported filter syntax: {ast.dump(node, include_attributes=False)}"
    )


def matches_filter(expression: str | None, submission: Mapping[str, object]) -> bool:
    if not expression:
        return True
    tree = parse_filter_expression(expression)
    return bool(evaluate_filter_node(tree, submission))


def label_targets_met(label_limits: dict[str, int], kept_by_label: Counter[str]) -> bool:
    if not label_limits:
        return False
    return all(kept_by_label.get(label, 0) >= limit for label, limit in label_limits.items())


def choose_output_message(
    *,
    raw_message: str,
    cleaned_message: str,
    normalized_message: str,
    message_format: str,
) -> str:
    if message_format == "cleaned":
        return cleaned_message
    if message_format == "normalized":
        return normalized_message
    return raw_message


def build_output_submission(submission: Mapping[str, object], output_message: str) -> Submission:
    output_submission: Submission = dict(submission)
    content = submission.get("content")
    if isinstance(content, dict):
        output_submission["content"] = dict(content)
        output_submission["content"]["msg"] = output_message
    else:
        output_submission["content"] = {"msg": output_message}
    return output_submission


def apply_sorted_label_shuffle(records: list[PreparedSubmission], *, seed: int) -> list[PreparedSubmission]:
    sorted_records = sorted(records, key=lambda item: (item.label, item.sort_key))
    rng = random.Random(seed)
    rng.shuffle(sorted_records)
    return sorted_records


def write_json(payload: dict[str, Any], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use overwrite=true to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def close_iterable(iterator: object) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


class ResolveInputFiles:
    """Resolve one file or directory input into a sorted list of dataset JSON files."""

    def __call__(self, raw_input: str | Path) -> list[Path]:
        path = resolve_input_path(raw_input)
        if path.is_dir():
            input_files = sorted(candidate for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.lower() == ".json")
            if not input_files:
                raise FileNotFoundError(f"No JSON files found in input directory: {path}")
            return input_files

        if path.is_file():
            return [path]

        raise FileNotFoundError(f"Input path not found: {path}")


class StreamSubmissions:
    """Stream submission records from repo-compatible dataset JSON files."""

    def __call__(self, input_files: list[Path]) -> Iterable[Submission]:
        try:
            import ijson
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "run_prepare_dataset.py requires 'ijson' for streaming reads."
            ) from exc

        def iterator() -> Iterator[Submission]:
            for input_file in input_files:
                with input_file.open("rb") as handle:
                    yield from ijson.items(handle, f"{SUPPORTED_DATASET_TOP_LEVEL_KEY}.item")

        return iterator()


class LoadSubmissionCollection:
    """Load all submissions from the input files into one in-memory collection."""

    def __call__(self, input_files: list[Path]) -> list[object]:
        submissions: list[object] = []
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Dataset file must contain a JSON object at top level: {input_file}"
                )
            collection = payload.get(SUPPORTED_DATASET_TOP_LEVEL_KEY)
            if not isinstance(collection, list):
                raise TypeError(
                    f"Dataset field {SUPPORTED_DATASET_TOP_LEVEL_KEY!r} must be a list in {input_file}"
                )
            submissions.extend(collection)
        return submissions


class EndForEachSubmission(RegionCloser):
    """Region boundary for the end of per-submission processing."""

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        output_type = list[current_output] if current_output is not None else list[Any]
        return (Any,), output_type


class ForEachSubmission(RegionOpener):
    """Run the enclosed operators once per source submission."""

    closing_type = EndForEachSubmission

    def run_region(
        self,
        current: Iterable[object],
        label: str,
        execute_region: Any,
        trace: InvocationTrace | _NoOpTrace,
        cfg: Any,
    ) -> list[Any]:
        collecting = isinstance(trace, InvocationTrace)
        source = iter(current)
        results: list[Any] = []
        child_traces: list[InvocationTrace] = []
        t_region = time.perf_counter()

        try:
            for scan_index, value in enumerate(source, start=1):
                child_trace = InvocationTrace() if collecting else _NoOpTrace()
                result, child_trace = execute_region(
                    ScannedItem(scan_index=scan_index, value=value),
                    child_trace,
                )
                results.append(result)
                if collecting:
                    child_traces.append(child_trace)
        except Exception:
            merged_trace = merge_traces(child_traces) if child_traces else None
            trace.spans.append(
                StepSpan(
                    label,
                    t_region,
                    time.perf_counter() - t_region,
                    error=True,
                    child_trace=merged_trace if collecting else None,
                    operator_type=type(self),
                )
            )
            raise
        finally:
            close_iterable(source)

        merged_trace = merge_traces(child_traces) if child_traces else None
        trace.spans.append(
            StepSpan(
                label,
                t_region,
                time.perf_counter() - t_region,
                child_trace=merged_trace if collecting else None,
                operator_type=type(self),
            )
        )
        return results

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        return (Any,), ScannedItem


class RequireSubmissionMappings:
    """Drop non-mapping values and retain submission dictionaries only."""

    def __call__(self, scanned: ScannedItem | None) -> ScannedSubmission | None:
        if scanned is None:
            return None
        if not isinstance(scanned, ScannedItem):
            raise TypeError(f"RequireSubmissionMappings expected ScannedItem, got {type(scanned)!r}")
        if not isinstance(scanned.value, dict):
            return None
        return ScannedSubmission(scan_index=scanned.scan_index, submission=scanned.value)


class ApplySubmissionFilter:
    """Apply the configured filter expression to each submission."""

    def __init__(self, where: str = ""):
        self.where = where.strip() or None

    def __call__(self, scanned: ScannedSubmission | None) -> ScannedSubmission | None:
        if scanned is None:
            return None
        if not isinstance(scanned, ScannedSubmission):
            raise TypeError(f"ApplySubmissionFilter expected ScannedSubmission, got {type(scanned)!r}")
        if not matches_filter(self.where, scanned.submission):
            return None
        return scanned


class ExtractSubmissionLabel:
    """Extract and validate labels from matched submissions."""

    def __call__(self, scanned: ScannedSubmission | None) -> LabeledSubmission | None:
        if scanned is None:
            return None
        if not isinstance(scanned, ScannedSubmission):
            raise TypeError(f"ExtractSubmissionLabel expected ScannedSubmission, got {type(scanned)!r}")
        label_raw = scanned.submission.get("label")
        if label_raw is None:
            return None
        label = str(label_raw).strip()
        if not label:
            return None
        return LabeledSubmission(
            scan_index=scanned.scan_index,
            submission=scanned.submission,
            label=label,
        )


class FilterConfiguredLabels:
    """Restrict the stream to configured labels when label limits are provided."""

    def __init__(self, label_limits: str | Mapping[str, int] | None = ""):
        self.label_limits = parse_label_limits(label_limits)

    def __call__(self, labeled: LabeledSubmission | None) -> LabeledSubmission | None:
        if labeled is None or not self.label_limits:
            return labeled
        if not isinstance(labeled, LabeledSubmission):
            raise TypeError(f"FilterConfiguredLabels expected LabeledSubmission, got {type(labeled)!r}")
        if labeled.label not in self.label_limits:
            return None
        return labeled


class ExtractSubmissionMessage:
    """Extract and validate the source message payload for each labeled submission."""

    def __call__(self, labeled: LabeledSubmission | None) -> MessageSubmission | None:
        if labeled is None:
            return None
        if not isinstance(labeled, LabeledSubmission):
            raise TypeError(f"ExtractSubmissionMessage expected LabeledSubmission, got {type(labeled)!r}")
        content = labeled.submission.get("content")
        raw_message = content.get("msg") if isinstance(content, dict) else None
        if not isinstance(raw_message, str):
            return None
        return MessageSubmission(
            scan_index=labeled.scan_index,
            submission=labeled.submission,
            label=labeled.label,
            raw_message=raw_message,
        )


class PrepareGroupingText:
    """Apply lightweight cleanup to the raw message."""

    def __call__(self, message: MessageSubmission | None) -> GroupedSubmission | None:
        if message is None:
            return None
        if not isinstance(message, MessageSubmission):
            raise TypeError(f"PrepareGroupingText expected MessageSubmission, got {type(message)!r}")
        return GroupedSubmission(
            scan_index=message.scan_index,
            submission=message.submission,
            label=message.label,
            raw_message=message.raw_message,
            cleaned_message=prepare_message_for_grouping(message.raw_message),
        )


class RequireMinimumMessageLength:
    """Drop messages whose cleaned form is shorter than the configured threshold."""

    def __init__(self, min_length: int | str = 2):
        self.min_length = int(min_length)
        if self.min_length < 0:
            raise ValueError("min_length must be >= 0.")

    def __call__(self, grouped: GroupedSubmission | None) -> GroupedSubmission | None:
        if grouped is None:
            return None
        if not isinstance(grouped, GroupedSubmission):
            raise TypeError(f"RequireMinimumMessageLength expected GroupedSubmission, got {type(grouped)!r}")
        if len(grouped.cleaned_message.strip()) < self.min_length:
            return None
        return grouped


class NormalizeTrainingText:
    """Normalize cleaned messages into the training-time representation."""

    def __init__(self, is_jp: bool | str = True):
        self.is_jp = parse_bool(is_jp)

    def __call__(self, grouped: GroupedSubmission | None) -> NormalizedSubmission | None:
        if grouped is None:
            return None
        if not isinstance(grouped, GroupedSubmission):
            raise TypeError(f"NormalizeTrainingText expected GroupedSubmission, got {type(grouped)!r}")
        return NormalizedSubmission(
            scan_index=grouped.scan_index,
            submission=grouped.submission,
            label=grouped.label,
            raw_message=grouped.raw_message,
            cleaned_message=grouped.cleaned_message,
            normalized_message=normalize_message_for_training(
                grouped.cleaned_message,
                is_jp=self.is_jp,
                already_prepared=True,
            ),
        )


class SelectDedupeKey:
    """Choose the text representation used for deduplication."""

    def __init__(self, dedupe_key: str = "normalized"):
        self.dedupe_key = normalize_choice("dedupe_key", dedupe_key, {"cleaned", "normalized"})

    def __call__(self, normalized: NormalizedSubmission | None) -> DedupeReadySubmission | None:
        if normalized is None:
            return None
        if not isinstance(normalized, NormalizedSubmission):
            raise TypeError(f"SelectDedupeKey expected NormalizedSubmission, got {type(normalized)!r}")
        dedupe_key = (
            normalized.cleaned_message
            if self.dedupe_key == "cleaned"
            else normalized.normalized_message
        )
        return DedupeReadySubmission(
            scan_index=normalized.scan_index,
            submission=normalized.submission,
            label=normalized.label,
            raw_message=normalized.raw_message,
            cleaned_message=normalized.cleaned_message,
            normalized_message=normalized.normalized_message,
            dedupe_key=dedupe_key,
        )


class RequireMinimumDedupeLength:
    """Drop submissions whose dedupe key is shorter than the configured threshold."""

    def __init__(self, min_length: int | str = 2):
        self.min_length = int(min_length)
        if self.min_length < 0:
            raise ValueError("min_length must be >= 0.")

    def __call__(self, ready: DedupeReadySubmission | None) -> DedupeReadySubmission | None:
        if ready is None:
            return None
        if not isinstance(ready, DedupeReadySubmission):
            raise TypeError(f"RequireMinimumDedupeLength expected DedupeReadySubmission, got {type(ready)!r}")
        if len(ready.dedupe_key.strip()) < self.min_length:
            return None
        return ready


class BuildPreparedCandidate:
    """Build the final output record for a dedupe-ready submission."""

    def __init__(self, message_format: str = "raw"):
        self.message_format = normalize_choice(
            "message_format",
            message_format,
            {"raw", "cleaned", "normalized"},
        )

    def __call__(self, ready: DedupeReadySubmission | None) -> PreparedCandidate | None:
        if ready is None:
            return None
        if not isinstance(ready, DedupeReadySubmission):
            raise TypeError(f"BuildPreparedCandidate expected DedupeReadySubmission, got {type(ready)!r}")
        output_message = choose_output_message(
            raw_message=ready.raw_message,
            cleaned_message=ready.cleaned_message,
            normalized_message=ready.normalized_message,
            message_format=self.message_format,
        )
        return PreparedCandidate(
            label=ready.label,
            dedupe_key=ready.dedupe_key,
            record=PreparedSubmission(
                label=ready.label,
                sort_key=f"{ready.label}\t{ready.dedupe_key}\t{ready.scan_index:012d}",
                output_submission=build_output_submission(ready.submission, output_message),
            ),
        )


class CompactDroppedCandidates:
    """Remove dropped entries emitted as None by per-item operators."""

    def __call__(self, candidates: list[PreparedCandidate | None]) -> list[PreparedCandidate]:
        kept: list[PreparedCandidate] = []
        for candidate in candidates:
            if candidate is None:
                continue
            if not isinstance(candidate, PreparedCandidate):
                raise TypeError(f"CompactDroppedCandidates expected PreparedCandidate | None, got {type(candidate)!r}")
            kept.append(candidate)
        return kept


class DeduplicateSubmissions:
    """Drop repeated records using the configured dedupe key."""

    def __call__(self, candidates: list[PreparedCandidate]) -> list[PreparedCandidate]:
        seen_keys: set[str] = set()
        deduped: list[PreparedCandidate] = []
        for candidate in candidates:
            if not isinstance(candidate, PreparedCandidate):
                raise TypeError(f"DeduplicateSubmissions expected PreparedCandidate, got {type(candidate)!r}")
            if candidate.dedupe_key in seen_keys:
                continue
            seen_keys.add(candidate.dedupe_key)
            deduped.append(candidate)
        return deduped


class ApplyLabelLimits:
    """Keep the first configured number of records per label."""

    def __init__(self, label_limits: str | Mapping[str, int] | None = ""):
        self.label_limits = parse_label_limits(label_limits)

    def __call__(self, candidates: list[PreparedCandidate]) -> list[PreparedCandidate]:
        if not self.label_limits:
            return candidates

        kept_by_label: Counter[str] = Counter()
        limited: list[PreparedCandidate] = []
        for candidate in candidates:
            if not isinstance(candidate, PreparedCandidate):
                raise TypeError(f"ApplyLabelLimits expected PreparedCandidate, got {type(candidate)!r}")
            if candidate.label not in self.label_limits:
                continue
            if kept_by_label[candidate.label] >= self.label_limits[candidate.label]:
                continue
            limited.append(candidate)
            kept_by_label[candidate.label] += 1
            if label_targets_met(self.label_limits, kept_by_label):
                break

        if not label_targets_met(self.label_limits, kept_by_label):
            missing = {
                label: limit - kept_by_label.get(label, 0)
                for label, limit in self.label_limits.items()
                if kept_by_label.get(label, 0) < limit
            }
            raise ValueError(
                "Unable to satisfy all label limits from the provided inputs. "
                f"Missing counts: {missing}"
            )
        return limited


class CollectPreparedRecords:
    """Materialize prepared records into the final pipeline result."""

    def __call__(self, candidates: list[PreparedCandidate]) -> PreparationRun:
        records: list[PreparedSubmission] = []
        for candidate in candidates:
            if not isinstance(candidate, PreparedCandidate):
                raise TypeError(f"CollectPreparedRecords expected PreparedCandidate, got {type(candidate)!r}")
            records.append(candidate.record)
        return PreparationRun(records=records)


class FinalizeOrdering:
    """Apply the final deterministic or seeded ordering policy to prepared records."""

    def __init__(
        self,
        *,
        shuffle: int | str | None = "disabled",
        sort_labels: bool | str = False,
    ):
        self.shuffle = resolve_shuffle_config(shuffle)
        self.sort_labels = parse_bool(sort_labels)
        if self.sort_labels and not self.shuffle.enabled:
            raise ValueError("sort_labels requires shuffle to be enabled.")

    def __call__(self, run: PreparationRun) -> PreparationRun:
        if not self.shuffle.enabled or not run.records:
            return run

        if self.sort_labels:
            run.records = apply_sorted_label_shuffle(run.records, seed=self.shuffle.seed or 0)
        else:
            records = list(run.records)
            rng = random.Random(self.shuffle.seed)
            rng.shuffle(records)
            run.records = records
        return run


class WritePreparedDataset(SideEffectOp):
    """Write prepared submissions to disk and pass the result through unchanged."""

    def __init__(self, output_path: str | Path, overwrite: bool | str = False):
        self.output_path = resolve_output_path(output_path)
        self.overwrite = parse_bool(overwrite)

    def effect(self, run: PreparationRun) -> None:
        payload = {"submissions": [record.output_submission for record in run.records]}
        write_json(payload, self.output_path, self.overwrite)
        print(f"Final summary: kept={len(run.records):,}", flush=True)
        print(f"Output written to: {self.output_path}", flush=True)

@data_factory
def prepare_dataset_input(input_path: str) -> InputFn:
    label = Path(str(input_path)).name or str(input_path)

    def fn() -> tuple[str, Any, str | None, dict | None]:
        return (label, str(input_path), None, None)

    return fn


def _build_prepare_dataset_processing_pipeline(
    *,
    output_path: str,
    where: str = "",
    label_limits: str = "",
    dedupe_key: str = "normalized",
    message_format: str = "raw",
    min_length: int | str = 2,
    is_jp: bool | str = True,
    shuffle: int | str | None = "disabled",
    sort_labels: bool | str = False,
    overwrite: bool | str = False,
) -> Pipeline:
    pipeline = Pipeline(
        [
            ForEachSubmission(),
            RequireSubmissionMappings(),
            ApplySubmissionFilter(where=where),
            ExtractSubmissionLabel(),
            FilterConfiguredLabels(label_limits=label_limits),
            ExtractSubmissionMessage(),
            PrepareGroupingText(),
            RequireMinimumMessageLength(min_length=min_length),
            NormalizeTrainingText(is_jp=is_jp),
            SelectDedupeKey(dedupe_key=dedupe_key),
            RequireMinimumDedupeLength(min_length=min_length),
            BuildPreparedCandidate(message_format=message_format),
            EndForEachSubmission(),
            CompactDroppedCandidates(),
            DeduplicateSubmissions(),
            ApplyLabelLimits(label_limits=label_limits),
            CollectPreparedRecords(),
            FinalizeOrdering(shuffle=shuffle, sort_labels=sort_labels),
            WritePreparedDataset(output_path=output_path, overwrite=overwrite),
        ]
    )
    return pipeline


@pipeline_factory
def build_prepare_dataset_pipeline(
    output_path: str,
    where: str = "",
    label_limits: str = "",
    dedupe_key: str = "normalized",
    message_format: str = "raw",
    min_length: int | str = 2,
    is_jp: bool | str = True,
    shuffle: int | str | None = "disabled",
    sort_labels: bool | str = False,
    overwrite: bool | str = False,
) -> Pipeline:
    pipeline = Pipeline(
        [
            ResolveInputFiles(),
            StreamSubmissions(),
        ]
    )
    processing_pipeline = _build_prepare_dataset_processing_pipeline(
        output_path=output_path,
        where=where,
        label_limits=label_limits,
        dedupe_key=dedupe_key,
        message_format=message_format,
        min_length=min_length,
        is_jp=is_jp,
        shuffle=shuffle,
        sort_labels=sort_labels,
        overwrite=overwrite,
    )
    pipeline.extend(processing_pipeline.operators)
    pipeline.validate()
    return pipeline


def build_prepare_dataset_collection_pipeline(
    output_path: str,
    where: str = "",
    label_limits: str = "",
    dedupe_key: str = "normalized",
    message_format: str = "raw",
    min_length: int | str = 2,
    is_jp: bool | str = True,
    shuffle: int | str | None = "disabled",
    sort_labels: bool | str = False,
    overwrite: bool | str = False,
) -> Pipeline:
    pipeline = Pipeline(
        [
            ResolveInputFiles(),
            LoadSubmissionCollection(),
        ]
    )
    processing_pipeline = _build_prepare_dataset_processing_pipeline(
        output_path=output_path,
        where=where,
        label_limits=label_limits,
        dedupe_key=dedupe_key,
        message_format=message_format,
        min_length=min_length,
        is_jp=is_jp,
        shuffle=shuffle,
        sort_labels=sort_labels,
        overwrite=overwrite,
    )
    pipeline.extend(processing_pipeline.operators)
    pipeline.validate()
    return pipeline
