from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

_log = logging.getLogger(__name__)

from .context import Context, ContextOp
from .region import RegionCloser, RegionOpener
from .tracing import InvocationTrace, StepSpan, TraceCollector, TracingConfig, _NoOpTrace, _extract_shape
from .validation import PipelineValidationError, PipelineValidator, TypeContract, format_annotation, is_annotation_compatible


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
        tracing: TracingConfig | None = None,
    ):
        self.operators = self._flatten(list(operators))
        self._tracing_config = tracing
        self._auto_validate = auto_validate
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

    def validate(self, strict: bool = False) -> TypeContract | None:
        if not self.operators:
            return None
        return PipelineValidator(self.operators).validate(strict=strict)

    def _execute(self, value: Any, trace: Any, region: tuple[int, int] | None = None) -> tuple[Any, Any]:
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
                    current, context = self._step_into_region(i, current, context, trace, cfg)
                    i = self._find_region_end(i + 1, type(operator), operator.closing_type) + 1
                else:
                    current, context = self._step(i, current, context, trace, cfg)
                    i += 1
        finally:
            trace.total_duration_s = time.perf_counter() - t_start
        return current, trace

    def _step_into_region(self, i: int, current: Any, context: Context, trace: Any, cfg: TracingConfig | None) -> tuple[Any, Context]:
        operator = self.operators[i]
        label = self._label_for(i, cfg.operator_labels if cfg else None)
        region_start = i + 1
        region_end = self._find_region_end(region_start, type(operator), operator.closing_type)
        # Bounded executor: the operator can only run operators within its own region.
        def execute_region(value: Any, child_trace: Any) -> Any:
            return self._execute(value, trace=child_trace, region=(region_start, region_end))
        result = operator.run_region(current, label, execute_region, trace, cfg)
        return result, context

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

        if current_output is not None and not is_annotation_compatible(
            current_output, (type_contract.input_type,)
        ):
            raise validation_error_type(
                f"Pipeline contract mismatch: incoming type "
                f"{format_annotation(current_output)} is incompatible with "
                f"embed() input {format_annotation(type_contract.input_type)}"
            )

        return (type_contract.input_type,), type_contract.output_type


def embed(pipeline: Pipeline) -> Embed:
    """Embed *pipeline* as an isolated step inside another pipeline."""
    return Embed(pipeline)
