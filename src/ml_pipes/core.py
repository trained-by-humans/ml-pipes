from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, Iterable, get_args, get_origin, get_type_hints

_log = logging.getLogger(__name__)

from .context import Context, ContextOp
from .tracing import InvocationTrace, StepSpan, TraceCollector, TracingConfig, _NoOpTrace, _extract_shape, merge_traces


class PipelineValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TypeContract:
    input_type: Any
    output_type: Any


@dataclass(frozen=True)
class Region:
    opening_op: type
    closing_op: type
    name: str


class Pipeline:
    def __init__(
        self,
        operators: Iterable[Callable[..., Any] | ContextOp],
        auto_validate: bool = False,
        strict: bool = False,
        tracing: TracingConfig | None = None,
        input_type: Any = None,
    ):
        self.operators = self._flatten(list(operators))
        self._tracing_config = tracing
        self._auto_validate = auto_validate
        self._strict = strict
        self._input_type = input_type
        if auto_validate:
            self.validate()

    def set_tracing(
        self,
        collector: TraceCollector | None,
        operator_labels: list[str] | None = None,
        capture_shapes: bool = False,
    ) -> None:
        """Attach or replace tracing. Pass collector=None to disable."""
        self._tracing_config = TracingConfig(collector, operator_labels, capture_shapes) if collector is not None else None

    def extend(self, operators: Iterable[Callable[..., Any] | ContextOp]) -> Pipeline:
        """Append *operators* to this pipeline in place and return self."""
        self.operators.extend(self._flatten(list(operators)))
        if self._auto_validate:
            self.validate()
        return self

    def __rshift__(self, other: Pipeline) -> Pipeline:
        """Join two pipelines as isolated blocks: a >> b.

        If self is already a flat join chain (all Embed operators), extend it
        rather than wrapping it, so a >> b >> c stays flat: [Embed(a), Embed(b), Embed(c)].
        """
        if self.operators and all(isinstance(op, Embed) for op in self.operators):
            return Pipeline([*self.operators, Embed(other)])
        return Pipeline([Embed(self), Embed(other)])

    def __add__(self, other: Pipeline) -> Pipeline:
        """Combine two pipelines into one flat pipeline with shared context: a + b."""
        return Pipeline([*self.operators, *other.operators])

    def __call__(self, value: Any) -> Any:
        cfg = self._tracing_config
        trace = InvocationTrace() if cfg is not None else _NoOpTrace()
        try:
            result, trace = self._execute(value, trace=trace)
        finally:
            if cfg is not None:
                try:
                    cfg.collector.on_trace(trace)
                except Exception:
                    _log.exception("TraceCollector.on_trace raised; trace dropped")
        return result

    def validate(self) -> TypeContract | None:
        if not self.operators:
            return None
        self._validate_regions()
        self._validate_context_interactions()
        return self._resolve_type_contract(strict=self._strict)

    def _validate_regions(self) -> None:
        from .region import RegionCloser, RegionOpener

        # Regions cannot interleave — each closer must match the top opener.
        # Same-type nesting (Batch inside Batch, Scatter inside Scatter) is forbidden.
        stack: list[tuple[RegionOpener, int]] = []  # (opener, open_position)
        for i, op in enumerate(self.operators):
            match op:
                case RegionOpener() if stack and type(stack[-1][0]) is type(op):
                    raise PipelineValidationError(
                        f"Directly nested {type(op).__name__} regions are not supported — "
                        f"a {type(op).__name__} region may not open inside another {type(op).__name__} region"
                    )
                case RegionOpener():
                    stack.append((op, i))
                case RegionCloser() if not stack:
                    raise PipelineValidationError(
                        f"{type(op).__name__} at position {i} has no matching opener"
                    )
                case RegionCloser() if not isinstance(op, stack[-1][0].closing_type):
                    top_opener, top_pos = stack[-1]
                    raise PipelineValidationError(
                        f"{type(op).__name__} at position {i} closes {type(top_opener).__name__} "
                        f"opened at position {top_pos} — regions cannot interleave"
                    )
                case RegionCloser():
                    stack.pop()

        for opener, pos in stack:
            raise PipelineValidationError(
                f"{type(opener).__name__} at position {pos} has no matching {opener.closing_type.__name__}"
            )

    def _validate_context_interactions(self) -> None:
        from .context import Store, Recall
        from .region import RegionCloser, RegionOpener

        stored_keys: set[str] = set()
        stack: list[set[str]] = []

        for i, operator in enumerate(self.operators):
            if isinstance(operator, RegionOpener):
                stack.append(stored_keys)
                stored_keys = set()
            elif isinstance(operator, RegionCloser):
                stored_keys = stack.pop()
            elif isinstance(operator, Store):
                stored_keys.add(operator.name)
            elif isinstance(operator, Recall):
                if operator.name not in stored_keys:
                    available = sorted(stored_keys)
                    raise PipelineValidationError(
                        f"Recall({operator.name!r}) at {self._label_for(i)} "
                        f"references a key that was not stored. "
                        f"Keys available at this point: {available if available else '(none)'}"
                    )
            elif isinstance(operator, Embed):
                try:
                    operator.pipeline._validate_context_interactions()
                except PipelineValidationError as exc:
                    raise PipelineValidationError(
                        f"Validation error inside {self._label_for(i)}: {exc}"
                    ) from exc

    def _resolve_type_contract(self, strict: bool = False) -> TypeContract:
        from .region import RegionOpener, RegionCloser

        if not self.operators:
            raise PipelineValidationError("Cannot resolve type contract of an empty pipeline")

        first_input_type: Any | None = None
        previous_output_type: Any | None = self._input_type
        previous_name: str | None = None
        stored_annotations: dict[str, Any] = {}
        stack: list[dict[str, Any]] = []

        for i, operator in enumerate(self.operators):
            if isinstance(operator, RegionOpener):
                stack.append(stored_annotations)
                stored_annotations = {}
            elif isinstance(operator, RegionCloser):
                stored_annotations = stack.pop()

            if hasattr(operator, "resolve_contract"):
                input_types, output_type = operator.resolve_contract(
                    previous_output_type,
                    stored_annotations,
                    self._expand_output_annotation,
                    PipelineValidationError,
                )
                if strict and not self._is_concrete(output_type):
                    raise PipelineValidationError(
                        f"Strict mode violation at {self._label_for(i)}: output type is unresolved (Any).\n"
                        f"  Fix: annotate the return type with a concrete type, or implement resolve_contract "
                        f"to return the upstream type (e.g. passthrough: return (Any,), current_output)."
                    )
            else:
                input_types, output_type = self._resolve_operator_contract(operator)
                name = operator.__class__.__name__

                if previous_output_type is not None and not self._is_annotation_compatible(
                    previous_output_type, input_types
                ):
                    raise PipelineValidationError(
                        f"Pipeline contract mismatch at {self._label_for(i)}: "
                        f"{previous_name} returns {self._format_annotation(previous_output_type)} "
                        f"but {name} expects {self._format_parameter_annotations(input_types)}"
                    )

                if strict:
                    if any(not self._is_concrete(t) for t in input_types):
                        raise PipelineValidationError(
                            f"Strict mode violation at {self._label_for(i)}: input type is unresolved (Any).\n"
                            f"  Fix: annotate the parameter with a concrete type, or implement resolve_contract "
                            f"to accept and thread the upstream type dynamically."
                        )
                    if not self._is_concrete(output_type):
                        raise PipelineValidationError(
                            f"Strict mode violation at {self._label_for(i)}: output type is unresolved (Any).\n"
                            f"  Fix: annotate the return type with a concrete type, or implement resolve_contract "
                            f"to return the upstream type (e.g. passthrough: return (Any,), current_output)."
                        )

            if first_input_type is None:
                first_input_type = input_types[0] if len(input_types) == 1 else input_types
            previous_output_type = output_type
            previous_name = operator.__class__.__name__

        return TypeContract(input_type=first_input_type, output_type=previous_output_type)

    @staticmethod
    def _resolve_operator_contract(operator: Callable[..., Any]) -> tuple[tuple[Any, ...], Any]:
        target = Pipeline._get_signature_target(operator)
        hints = get_type_hints(target)
        signature = inspect.signature(target)
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if not parameters:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} must define at least one positional input parameter in __call__"
            )

        input_types: list[Any] = []
        for parameter in parameters:
            if parameter.name not in hints:
                raise PipelineValidationError(
                    f"{operator.__class__.__name__} is missing a type annotation for __call__ input"
                )
            input_types.append(hints[parameter.name])
        if "return" not in hints:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} is missing a return type annotation for __call__"
            )

        return tuple(input_types), hints["return"]

    @classmethod
    def _is_annotation_compatible(cls, produced: Any, expected_inputs: tuple[Any, ...]) -> bool:
        # When the next operator takes a single parameter, _build_call_args passes
        # the whole current value without unpacking — so compare the unexpanded
        # produced type against that single expected type.
        if len(expected_inputs) == 1:
            return cls._is_single_annotation_compatible(produced, expected_inputs[0])

        produced_types = cls._expand_output_annotation(produced)
        if len(produced_types) != len(expected_inputs):
            return False

        return all(
            cls._is_single_annotation_compatible(produced_type, expected_type)
            for produced_type, expected_type in zip(produced_types, expected_inputs, strict=True)
        )

    @classmethod
    def _is_single_annotation_compatible(cls, produced: Any, expected: Any) -> bool:
        if expected is Any or produced is Any:
            return True
        if produced == expected:
            return True

        if cls._is_concrete_assignable(produced, expected):
            return True

        produced_origin = get_origin(produced)
        expected_origin = get_origin(expected)

        if cls._is_union_annotation(expected):
            return any(cls._is_single_annotation_compatible(produced, option) for option in get_args(expected))
        if cls._is_union_annotation(produced):
            return all(cls._is_single_annotation_compatible(option, expected) for option in get_args(produced))

        if produced_origin is None:
            return False
        if expected_origin is None:
            # e.g. tuple[int, str] is assignable to bare tuple
            return cls._is_concrete_assignable(produced_origin, expected)
        if produced_origin != expected_origin:
            return False

        produced_args = get_args(produced)
        expected_args = get_args(expected)
        if len(produced_args) != len(expected_args):
            return False

        return all(
            cls._is_single_annotation_compatible(produced_arg, expected_arg)
            for produced_arg, expected_arg in zip(produced_args, expected_args, strict=True)
        )

    @staticmethod
    def _expand_output_annotation(annotation: Any) -> tuple[Any, ...]:
        origin = get_origin(annotation)
        if origin is tuple:
            return get_args(annotation)
        if isinstance(annotation, tuple):
            return annotation
        return (annotation,)

    @staticmethod
    def _is_concrete_assignable(produced: Any, expected: Any) -> bool:
        if not isinstance(produced, type) or not isinstance(expected, type):
            return False
        try:
            return issubclass(produced, expected)
        except TypeError:
            return False

    @staticmethod
    def _is_concrete(annotation: Any) -> bool:
        if annotation is Any:
            return False
        return all(Pipeline._is_concrete(arg) for arg in get_args(annotation))

    @staticmethod
    def _is_union_annotation(annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in (UnionType, getattr(__import__("typing"), "Union"))

    @staticmethod
    def _format_annotation(annotation: Any) -> str:
        return str(annotation).replace("typing.", "")

    @classmethod
    def _format_parameter_annotations(cls, annotations: tuple[Any, ...]) -> str:
        if len(annotations) == 1:
            return cls._format_annotation(annotations[0])
        return "(" + ", ".join(cls._format_annotation(annotation) for annotation in annotations) + ")"

    def _execute(self, value: Any, trace: Any, region: tuple[int, int] | None = None) -> tuple[Any, Any]:
        from .region import RegionOpener
        cfg = self._tracing_config  # snapshot once — set_tracing() may race on another thread
        start, end = region if region is not None else (0, len(self.operators))
        context = Context()
        current = value
        t_start = time.perf_counter()
        try:
            i = start
            while i < end:
                operator = self.operators[i]
                if isinstance(operator, RegionOpener):
                    current, context, i = self._step_into_region(operator, current, context, i, trace, cfg)
                else:
                    current, context = self._step(i, current, context, trace, cfg)
                    i += 1
        finally:
            trace.total_duration_s = time.perf_counter() - t_start
        return current, trace

    def _step_into_region(self, operator: Any, current: Any, context: Any, i: int, trace: Any, cfg: Any) -> tuple[Any, Any, int]:
        return operator.execute_region(self, current, context, i, trace, cfg)

    def _label_for(self, i: int, custom_labels: list[str] | None = None) -> str:
        if custom_labels and i < len(custom_labels):
            return custom_labels[i]
        op = self.operators[i]
        name = op.__name__ if inspect.isfunction(op) or inspect.ismethod(op) else type(op).__name__
        return f"{i}:{name}"

    def _step(self, i: int, current: Any, context: Context, trace: Any, cfg: TracingConfig | None) -> tuple[Any, Context]:
        operator = self.operators[i]
        label = self._label_for(i, cfg.operator_labels if cfg else None)
        capture = cfg.capture_shapes if cfg else False
        t = time.perf_counter()
        try:
            if isinstance(operator, ContextOp):
                result, ctx_out = operator.apply(current, context)
            else:
                args = self._build_call_args(operator, current)
                result = operator(*args)
                ctx_out = context
        except Exception:
            trace.spans.append(StepSpan(label, t, time.perf_counter() - t, error=True))
            raise
        trace.spans.append(StepSpan(
            label, t, time.perf_counter() - t,
            input_shape=_extract_shape(current) if capture else None,
            output_shape=_extract_shape(result) if capture else None,
        ))
        return result, ctx_out

    def _find_region_end(self, start: int, opening_op: type, closing_op: type) -> int:
        """Return the index of the closing op that matches the opener at *start*."""
        depth = 1
        j = start
        while j < len(self.operators) and depth > 0:
            op = self.operators[j]
            if isinstance(op, opening_op): depth += 1
            elif isinstance(op, closing_op): depth -= 1
            j += 1
        return j - 1

    @staticmethod
    def _flatten(operators: list) -> list:
        flat = []
        for op in operators:
            if isinstance(op, Inline):
                flat += Pipeline._flatten(op.pipeline.operators)
            else:
                flat.append(op)
        return flat

    @staticmethod
    def _build_call_args(operator: Callable[..., Any], current: Any) -> tuple[Any, ...]:
        signature = inspect.signature(Pipeline._get_signature_target(operator))
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if len(parameters) == 1:
            return (current,)
        if isinstance(current, tuple):
            if len(current) != len(parameters):
                raise TypeError(
                    f"{operator.__class__.__name__} expects {len(parameters)} positional arguments, "
                    f"got tuple of length {len(current)}"
                )
            return current
        raise TypeError(
            f"{operator.__class__.__name__} expects {len(parameters)} positional arguments, got 1"
        )

    @staticmethod
    def _get_signature_target(operator: Callable[..., Any]) -> Any:
        if inspect.isfunction(operator) or inspect.ismethod(operator):
            return operator
        return getattr(operator, "__call__")


class Inline:
    """
    Marker that expands a pipeline's operators into the parent at construction time.

    Example::

        preprocess = Pipeline([Resize(), Normalize()])

        full = Pipeline([
            Decode(),
            Inline(preprocess),
            Infer(...),
        ])
        # full.operators == [Decode(), Resize(), Normalize(), Infer(...)]
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline


def inline(pipeline: Pipeline) -> Inline:
    """Inline *pipeline* as a flattening marker inside a pipeline definition (a + b equivalent)."""
    return Inline(pipeline)


class Embed:
    """
    Embed a pipeline as a single isolated step inside an outer pipeline.

    The inner pipeline always starts with a fresh context (that is how
    Pipeline.__call__ works), so inner Store/Recall values are naturally
    invisible outside and outer context values are invisible inside.

    Example::

        preprocess = Pipeline([Decode(), Resize(), Normalize()])
        infer      = Pipeline([Infer("model.onnx"), Extract("output0")])

        full = Pipeline([
            embed(preprocess),
            embed(infer),
            NMS(...),
        ])
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def __call__(self, value: Any) -> Any:
        return self.pipeline(value)

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        try:
            type_contract = self.pipeline.validate()
        except PipelineValidationError as exc:
            raise validation_error_type(
                f"Validation error inside Embed: {exc}"
            ) from exc

        if type_contract is None:
            return (Any,), current_output

        if current_output is not None and not Pipeline._is_annotation_compatible(
            current_output, (type_contract.input_type,)
        ):
            raise validation_error_type(
                f"Pipeline contract mismatch: incoming type "
                f"{Pipeline._format_annotation(current_output)} is incompatible with "
                f"embed() input {Pipeline._format_annotation(type_contract.input_type)}"
            )

        return (type_contract.input_type,), type_contract.output_type


def embed(pipeline: Pipeline) -> Embed:
    """Embed *pipeline* as an isolated step inside another pipeline."""
    return Embed(pipeline)

