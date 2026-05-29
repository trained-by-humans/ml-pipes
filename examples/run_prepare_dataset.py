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
import unicodedata
from collections.abc import Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ml_pipes import (
    Distinct,
    EndForEachItem,
    Filter as FilterPrimitive,
    FilterNotNull as FilterNotNullPrimitive,
    ForEachItem,
    InputFn,
    MapValue as MapValuePrimitive,
    Pipeline,
    WrapMappingInObject,
    SideEffectOp,
    SHORT_CIRCUIT,
    data_factory,
    pipeline_factory,
)


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
SelectorPart = str | int
Selector = SelectorPart | tuple[SelectorPart, ...]


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


@dataclass
class MessageState:
    submission: Submission | None = None
    cleaned_text: str | None = None
    normalized_text: str | None = None


def _replace_pin(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}<PIN>"


def _replace_booking_code(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}<CODE>"


def normalize_text_content(text: str, *, to_lower: bool = True, is_jp: bool = True) -> str:
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


def prepare_text_content(msg: str | None) -> str:
    if msg is None:
        return ""

    cleaned = remove_garbled_text(msg)
    cleaned = clean_escape_sequences(cleaned)
    cleaned = clean_whitespace(cleaned)
    return cleaned


def normalize_text_for_training(
    msg: str | None,
    *,
    is_jp: bool = False,
    already_prepared: bool = False,
) -> str:
    prepared = msg if already_prepared else prepare_text_content(msg)
    return normalize_text_content(prepared, is_jp=is_jp)


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


def limits_satisfied(limits: Mapping[str, int], kept_by_value: Counter[str]) -> bool:
    if not limits:
        return False
    return all(kept_by_value.get(value, 0) >= limit for value, limit in limits.items())


_MISSING = object()


def _normalize_selector(selector: Selector) -> tuple[SelectorPart, ...]:
    if isinstance(selector, tuple):
        parts = selector
    elif isinstance(selector, int):
        parts = (selector,)
    else:
        raw_parts = selector.split(".")
        if any(part == "" for part in raw_parts):
            raise ValueError(f"Selector cannot contain empty path segments: {selector!r}")
        parts = tuple(int(part) if part.isdigit() else part for part in raw_parts)
    if not parts:
        raise ValueError("Selector cannot be empty.")
    return parts


def _select_part(current: object, part: SelectorPart) -> object:
    if isinstance(part, int):
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                return current[part]
            except IndexError:
                return _MISSING
        return _MISSING
    if isinstance(current, Mapping):
        return current.get(part, _MISSING)
    return getattr(current, part, _MISSING)


def _read_selector_or_missing(value: object, selector: Selector) -> object:
    current = value
    for part in _normalize_selector(selector):
        current = _select_part(current, part)
        if current is _MISSING:
            return _MISSING
    return current


def _read_selector(value: object, selector: Selector, operator_name: str) -> object:
    selected = _read_selector_or_missing(value, selector)
    if selected is _MISSING:
        raise TypeError(
            f"{operator_name} requires selector {_normalize_selector(selector)!r} on {type(value)!r}."
        )
    return selected


def _require_mapping_selector(value: object, selector: Selector, operator_name: str) -> Mapping[str, object]:
    selected = _read_selector(value, selector, operator_name)
    if not isinstance(selected, Mapping):
        raise TypeError(
            f"{operator_name} requires mapping at selector {_normalize_selector(selector)!r}, "
            f"got {type(selected)!r}."
        )
    return selected


def _require_text_selector(
    value: object,
    selector: Selector,
    operator_name: str,
    semantic_name: str,
    *,
    strip: bool = False,
) -> str:
    selected = _read_selector(value, selector, operator_name)
    if selected is None:
        raise ValueError(f"{operator_name} requires {semantic_name} first.")
    if not isinstance(selected, str):
        raise TypeError(
            f"{operator_name} requires {semantic_name} to be str, got {type(selected)!r}."
        )
    return selected.strip() if strip else selected


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

    def __call__(self, input_files: list[Path]) -> list[Submission]:
        submissions: list[Submission] = []
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


class FilterByExpression:
    """Apply the configured filter expression to the selected mapping."""

    def __init__(self, src: Selector, expression: str = ""):
        normalized_expression = expression.strip() or None

        def predicate(value: object) -> bool:
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"{type(self).__name__} requires mapping at selector {_normalize_selector(src)!r}, "
                    f"got {type(value)!r}."
                )
            return matches_filter(normalized_expression, value)

        self._inner = FilterPrimitive(
            src=src,
            predicate=predicate,
        )

    def __call__(self, state: object | None) -> object | None:
        return self._inner(state)


class RequireTextValue:
    """Drop values whose selected text is missing, non-string, or blank when configured."""

    def __init__(
        self,
        src: Selector,
        *,
        strip: bool = False,
        allow_blank: bool = True,
    ):
        self._filter_not_null = FilterNotNullPrimitive(src=src)

        def predicate(value: object) -> bool:
            if not isinstance(value, str):
                return False
            candidate = value.strip() if strip else value
            return allow_blank or candidate != ""

        self._inner = FilterPrimitive(src=src, predicate=predicate)

    def __call__(self, state: object | None) -> object:
        state = self._filter_not_null(state)
        if state is SHORT_CIRCUIT:
            return SHORT_CIRCUIT
        return self._inner(state)


class FilterByAllowedValues:
    """Drop records whose selected value is not present in the allowed set."""

    def __init__(
        self,
        src: Selector,
        allowed_values: Mapping[object, Any] | set[object] | list[object] | tuple[object, ...] = (),
    ):
        if isinstance(allowed_values, Mapping):
            normalized_values = set(allowed_values.keys())
        else:
            normalized_values = set(allowed_values)
        self._inner = (
            None
            if not normalized_values
            else FilterPrimitive(src=src, predicate=lambda value: value in normalized_values)
        )

    def __call__(self, state: object | None) -> object | None:
        if self._inner is None:
            return state
        return self._inner(state)


class MapValue:
    """Apply a transformation function to a selected value and store the result at as_."""

    def __init__(self, src: Selector, fn: Callable[[Any], Any], as_: Selector):
        self._inner = MapValuePrimitive(fn=fn, src=src, as_=as_)

    def __call__(self, state: object | None) -> object | None:
        return self._inner(state)


class RequireMinimumLength:
    """Drop values whose selected text is shorter than the configured threshold."""

    def __init__(self, src: Selector, min_length: int | str = 2, *, strip: bool = True):
        parsed_min_length = int(min_length)
        if parsed_min_length < 0:
            raise ValueError("min_length must be >= 0.")

        def predicate(value: object) -> bool:
            if not isinstance(value, str):
                raise TypeError(
                    f"{type(self).__name__} requires selector {_normalize_selector(src)!r} to be str, "
                    f"got {type(value)!r}."
                )
            return len(value.strip() if strip else value) >= parsed_min_length

        self._inner = FilterPrimitive(
            src=src,
            predicate=predicate,
        )

    def __call__(self, state: object | None) -> object | None:
        return self._inner(state)


class LimitByValue:
    """Keep the first configured number of records per selected value."""

    def __init__(
        self,
        src: Selector,
        limits: str | Mapping[str, int] | None = "",
    ):
        self.src = src
        self.limits = parse_label_limits(limits)

    def __call__(self, states: list[object]) -> list[object]:
        if not self.limits:
            return states

        kept_by_value: Counter[str] = Counter()
        limited: list[object] = []
        for state in states:
            value = _require_text_selector(
                state,
                self.src,
                type(self).__name__,
                "value",
                strip=True,
            )
            if value not in self.limits:
                continue
            if kept_by_value[value] >= self.limits[value]:
                continue
            limited.append(state)
            kept_by_value[value] += 1
            if limits_satisfied(self.limits, kept_by_value):
                break

        if not limits_satisfied(self.limits, kept_by_value):
            missing = {
                label: limit - kept_by_value.get(label, 0)
                for label, limit in self.limits.items()
                if kept_by_value.get(label, 0) < limit
            }
            raise ValueError(
                "Unable to satisfy all configured value limits from the provided inputs. "
                f"Missing counts: {missing}"
            )
        return limited


class BuildPreparedRecords:
    """Build the final output records from selected message state fields."""

    def __init__(
        self,
        submission: Selector = "submission",
        label: Selector = "submission.label",
        text: Selector = "submission.content.msg",
        sort_text: Selector = "normalized_text",
    ):
        self.submission = submission
        self.label = label
        self.text = text
        self.sort_text = sort_text

    def __call__(self, states: list[object]) -> list[PreparedSubmission]:
        records: list[PreparedSubmission] = []
        for state in states:
            submission = _require_mapping_selector(state, self.submission, type(self).__name__)
            label = _require_text_selector(
                state,
                self.label,
                type(self).__name__,
                "label",
                strip=True,
            )
            output_message = _require_text_selector(
                state,
                self.text,
                type(self).__name__,
                "text",
            )
            sort_text = _require_text_selector(
                state,
                self.sort_text,
                type(self).__name__,
                "sort_text",
            )
            records.append(
                PreparedSubmission(
                    label=label,
                    sort_key=f"{label}\t{sort_text}",
                    output_submission=build_output_submission(submission, output_message),
                )
            )
        return records


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

    def __call__(self, records: list[PreparedSubmission]) -> list[PreparedSubmission]:
        if not self.shuffle.enabled or not records:
            return records

        if self.sort_labels:
            return apply_sorted_label_shuffle(records, seed=self.shuffle.seed or 0)

        shuffled = list(records)
        rng = random.Random(self.shuffle.seed)
        rng.shuffle(shuffled)
        return shuffled


class WritePreparedDataset(SideEffectOp):
    """Write prepared submissions to disk and pass the result through unchanged."""

    def __init__(self, output_path: str | Path, overwrite: bool | str = False):
        self.output_path = resolve_output_path(output_path)
        self.overwrite = parse_bool(overwrite)

    def effect(self, records: list[PreparedSubmission]) -> None:
        payload = {"submissions": [record.output_submission for record in records]}
        write_json(payload, self.output_path, self.overwrite)
        print(f"Final summary: kept={len(records):,}", flush=True)
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
    parsed_limits = parse_label_limits(label_limits)
    normalized_dedupe_key = normalize_choice("dedupe_key", dedupe_key, {"cleaned", "normalized"})
    normalized_message_format = normalize_choice(
        "message_format",
        message_format,
        {"raw", "cleaned", "normalized"},
    )
    parsed_is_jp = parse_bool(is_jp)
    dedupe_selector = "cleaned_text" if normalized_dedupe_key == "cleaned" else "normalized_text"
    output_message_selector = {
        "raw": "submission.content.msg",
        "cleaned": "cleaned_text",
        "normalized": "normalized_text",
    }[normalized_message_format]
    normalize_training_text = partial(
        normalize_text_for_training,
        is_jp=parsed_is_jp,
        already_prepared=True,
    )

    pipeline = Pipeline(
        [
            ForEachItem(),
            WrapMappingInObject(as_="submission", state_factory=MessageState),
            FilterByExpression(src="submission", expression=where),
            RequireTextValue(src="submission.label", strip=True, allow_blank=False),
            FilterByAllowedValues(src="submission.label", allowed_values=parsed_limits),
            RequireTextValue(src="submission.content.msg"),
            MapValue(src="submission.content.msg", fn=prepare_text_content, as_="cleaned_text"),
            RequireMinimumLength(src="cleaned_text", min_length=min_length),
            MapValue(src="cleaned_text", fn=normalize_training_text, as_="normalized_text"),
            RequireMinimumLength(src=dedupe_selector, min_length=min_length),
            EndForEachItem(),
            Distinct(src=dedupe_selector),
            LimitByValue(src="submission.label", limits=parsed_limits),
            BuildPreparedRecords(
                submission="submission",
                label="submission.label",
                text=output_message_selector,
                sort_text=dedupe_selector,
            ),
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
