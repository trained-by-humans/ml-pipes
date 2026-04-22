from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, Iterable, get_args, get_origin, get_type_hints

from .context import Context, ContextOp
from .tracing import InvocationTrace, StepSpan, TraceCollector, TracingConfig, _NoOpTrace, _extract_shape


class PipelineValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TypeContract:
    input_type: Any
    output_type: Any


class Pipeline:
    def __init__(
        self,
        operators: Iterable[Callable[..., Any] | ContextOp],
        validate_on_init: bool = False,
        tracing: TracingConfig | None = None,
    ):
        self.operators = self._flatten(list(operators))
        self._tracing_config = tracing
        if validate_on_init:
            self.validate()

    def set_tracing(
        self,
        collector: TraceCollector | None,
        operator_labels: list[str] | None = None,
        capture_shapes: bool = False,
    ) -> None:
        """Attach or replace tracing. Pass collector=None to disable."""
        self._tracing_config = TracingConfig(collector, operator_labels, capture_shapes) if collector is not None else None

    @staticmethod
    def _flatten(operators: list) -> list:
        flat = []
        for op in operators:
            if isinstance(op, Inline):
                flat += Pipeline._flatten(op.pipeline.operators)
            else:
                flat.append(op)
        return flat

    def extend(self, operators: Iterable[Callable[..., Any] | ContextOp]) -> Pipeline:
        """Append *operators* to this pipeline in place and return self."""
        self.operators.extend(self._flatten(list(operators)))
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
        return self._execute(value, trace=None)

    def _execute(self, value: Any, trace: InvocationTrace | None) -> Any:
        from .ops import Batch  # local import avoids circular dependency
        current = value
        context = Context()
        i = 0
        collecting = trace is None and self._tracing_config is not None
        if trace is None:
            trace = InvocationTrace() if collecting else _NoOpTrace()
        t_start = time.perf_counter()
        try:
            while i < len(self.operators):
                operator = self.operators[i]
                if isinstance(operator, Batch):
                    current, context, i = self._step_into_batch(current, context, i, trace)
                else:
                    current, context = self._step(i, current, context, trace)
                    i += 1
        finally:
            if collecting:
                trace.total_duration_s = time.perf_counter() - t_start
                self._tracing_config.collector.on_trace(trace)
        return current

    def _label_for(self, i: int) -> str:
        if self._tracing_config and self._tracing_config.operator_labels and i < len(self._tracing_config.operator_labels):
            return self._tracing_config.operator_labels[i]
        op = self.operators[i]
        name = op.__name__ if inspect.isfunction(op) or inspect.ismethod(op) else type(op).__name__
        return f"{i}:{name}"

    def _step(self, i: int, current: Any, context: Context, trace: Any) -> tuple[Any, Context]:
        operator = self.operators[i]
        label = self._label_for(i)
        capture = self._tracing_config.capture_shapes if self._tracing_config else False
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

    def _step_into_batch(self, current: Any, context: Context, i: int, trace: Any) -> tuple[Any, Context, int]:
        from .batch import LeaderBatch
        from .ops import UnBatch

        gate = self.operators[i].gate
        batch_label = self._label_for(i)

        # Span 1: gate wait — each thread records its own blocking time
        t_wait = time.perf_counter()
        outcome = gate.enter(current)
        trace.spans.append(StepSpan(f"{batch_label}[wait]", t_wait, time.perf_counter() - t_wait))
        i += 1  # move past the Batch operator itself

        # Follower: skip region, receive leader's batch span via gate
        if not isinstance(outcome, LeaderBatch):
            while not isinstance(self.operators[i], UnBatch):
                i += 1
            if outcome.batch_span is not None:
                trace.spans.append(outcome.batch_span)
            return outcome.result, context, i + 1

        # Leader: run region operators into a child trace
        current = outcome.inputs
        batch_size = len(current) if hasattr(current, "__len__") else None
        collecting = isinstance(trace, InvocationTrace)
        child_trace = InvocationTrace(batch_size=batch_size) if collecting else _NoOpTrace(batch_size=batch_size)
        batch_context = Context()

        t_region = time.perf_counter()
        try:
            while not isinstance(self.operators[i], UnBatch):
                current, batch_context = self._step(i, current, batch_context, child_trace)
                i += 1
        except Exception as exc:
            child_trace.total_duration_s = time.perf_counter() - t_region
            batch_span = StepSpan(
                batch_label, t_region, child_trace.total_duration_s,
                error=True, child_trace=child_trace if collecting else None,
            )
            trace.spans.append(batch_span)
            gate.distribute_exception(exc)
            raise

        child_trace.total_duration_s = time.perf_counter() - t_region

        # Span 2: batch region — leader appends, then passes to followers via gate
        batch_span = StepSpan(
            batch_label, t_region, child_trace.total_duration_s,
            child_trace=child_trace if collecting else None,
        )
        trace.spans.append(batch_span)

        current = gate.distribute(current, batch_span=batch_span if collecting else None)
        return current, context, i + 1  # resume after UnBatch with the original outer context

    def _validate_batch_pairs(self) -> None:
        from .ops import Batch, UnBatch

        stack: list[int] = []
        for i, op in enumerate(self.operators):
            if isinstance(op, Batch):
                if stack:
                    raise PipelineValidationError(
                        "Nested Batch regions are not supported"
                    )
                stack.append(i)
            elif isinstance(op, UnBatch):
                if not stack:
                    raise PipelineValidationError(
                        f"UnBatch at position {i} has no matching Batch"
                    )
                stack.pop()
        if stack:
            raise PipelineValidationError(
                f"Batch at position {stack[0]} has no matching UnBatch"
            )

    def resolve_type_contract(self) -> TypeContract:
        from .ops import Batch, UnBatch

        if not self.operators:
            raise PipelineValidationError("Cannot resolve type contract of an empty pipeline")

        self._validate_batch_pairs()

        first_input_type: Any | None = None
        previous_output_type: Any | None = None
        previous_name: str | None = None
        stored_annotations: dict[str, Any] = {}
        outer_stored_annotations: dict[str, Any] | None = None

        for operator in self.operators:
            if isinstance(operator, Batch):
                # Enter batch region: swap to an isolated annotation scope.
                outer_stored_annotations = stored_annotations
                stored_annotations = {}
                continue
            elif isinstance(operator, UnBatch):
                # Exit batch region: discard the isolated scope, restore the outer one.
                stored_annotations = outer_stored_annotations  # type: ignore[assignment]
                outer_stored_annotations = None
                continue

            if isinstance(operator, ContextOp) or hasattr(operator, "resolve_contract"):
                input_types, output_type = operator.resolve_contract(
                    previous_output_type,
                    stored_annotations,
                    self._expand_output_annotation,
                    PipelineValidationError,
                )
            else:
                input_types, output_type = self._resolve_operator_contract(operator)
                name = operator.__class__.__name__

                if previous_output_type is not None and not self._is_annotation_compatible(
                    previous_output_type, input_types
                ):
                    raise PipelineValidationError(
                        f"Pipeline contract mismatch: {previous_name} returns "
                        f"{self._format_annotation(previous_output_type)} but {name} expects "
                        f"{self._format_parameter_annotations(input_types)}"
                    )

            if first_input_type is None:
                first_input_type = input_types[0] if len(input_types) == 1 else input_types
            previous_output_type = output_type
            previous_name = operator.__class__.__name__

        return TypeContract(input_type=first_input_type, output_type=previous_output_type)

    def validate(self) -> None:
        if not self.operators:
            return
        self.resolve_type_contract()

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
        type_contract = self.pipeline.resolve_type_contract()

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

