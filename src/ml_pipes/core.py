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
    ):
        self.operators = self._flatten(list(operators))
        self._tracing_config = tracing
        self._auto_validate = auto_validate
        self._strict = strict
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
        return self._execute(value)

    def validate(self) -> TypeContract | None:
        if not self.operators:
            return None
        self._validate_regions()
        self._validate_context_interactions()
        return self._resolve_type_contract(strict=self._strict)

    def _validate_regions(self) -> None:
        from .ops import Batch, Gather, Scatter, UnBatch

        # Regions cannot interleave: Scatter→Batch→UnBatch→Gather is valid but
        # Scatter→Batch→Gather (crossing) is not — each closer must match the top opener.
        BATCH_REGION = Region(opening_op=Batch, closing_op=UnBatch, name="Batch")
        SCATTER_REGION = Region(opening_op=Scatter, closing_op=Gather, name="Scatter")
        _opener_to_region = {Batch: BATCH_REGION, Scatter: SCATTER_REGION}
        _closer_to_region = {UnBatch: BATCH_REGION, Gather: SCATTER_REGION}

        stack: list[tuple[Region, int]] = []  # (region, open_position)
        for i, op in enumerate(self.operators):
            op_type = type(op)
            if op_type in _opener_to_region:
                region = _opener_to_region[op_type]
                if stack and stack[-1][0] is region:
                    raise PipelineValidationError(
                        f"Nested {region.name} regions are not supported"
                    )
                stack.append((region, i))
            elif op_type in _closer_to_region:
                region = _closer_to_region[op_type]
                if not stack:
                    raise PipelineValidationError(
                        f"{op_type.__name__} at position {i} has no matching {region.opening_op.__name__}"
                    )
                top_region, top_pos = stack[-1]
                if top_region is not region:
                    raise PipelineValidationError(
                        f"{op_type.__name__} at position {i} closes {top_region.opening_op.__name__} "
                        f"opened at position {top_pos} — regions cannot interleave"
                    )
                stack.pop()

        for region, pos in stack:
            raise PipelineValidationError(
                f"{region.opening_op.__name__} at position {pos} has no matching {region.closing_op.__name__}"
            )

    def _validate_context_interactions(self) -> None:
        from .context import Store, Recall
        from .ops import Batch, Gather, Scatter, UnBatch

        stored_keys: set[str] = set()
        stack: list[set[str]] = []

        for i, operator in enumerate(self.operators):
            if isinstance(operator, (Batch, Scatter)):
                stack.append(stored_keys)
                stored_keys = set()
            elif isinstance(operator, (UnBatch, Gather)):
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
        from .ops import Batch, Gather, Scatter, UnBatch

        if not self.operators:
            raise PipelineValidationError("Cannot resolve type contract of an empty pipeline")

        first_input_type: Any | None = None
        previous_output_type: Any | None = None
        previous_name: str | None = None
        stored_annotations: dict[str, Any] = {}
        stack: list[dict[str, Any]] = []

        for i, operator in enumerate(self.operators):
            if isinstance(operator, Batch):
                stack.append(stored_annotations)
                stored_annotations = {}
                continue  # transparent to type contract — region ops are what matter
            elif isinstance(operator, UnBatch):
                stored_annotations = stack.pop()
                continue
            elif isinstance(operator, Scatter):
                stack.append(stored_annotations)
                stored_annotations = {}
                # fall through: Scatter.resolve_contract transforms list[T] → T
            elif isinstance(operator, Gather):
                stored_annotations = stack.pop()
                # fall through: Gather.resolve_contract transforms T → list[T]

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

    def _execute(self, value: Any, region: tuple[int, int] | None = None, trace: Any = None) -> Any:
        from .ops import Batch, Scatter  # local import avoids circular dependency
        cfg = self._tracing_config  # snapshot once — set_tracing() may race on another thread
        collecting = trace is None and cfg is not None
        start, end = region if region is not None else (0, len(self.operators))
        if trace is None:
            trace = InvocationTrace() if collecting else _NoOpTrace()
        context = Context()
        current = value
        t_start = time.perf_counter()
        try:
            i = start
            while i < end:
                operator = self.operators[i]
                if isinstance(operator, Batch):
                    current, context, i = self._step_into_batch(current, context, i, trace, cfg)
                elif isinstance(operator, Scatter):
                    current, context, i = self._step_into_scatter(current, context, i, trace, cfg)
                else:
                    current, context = self._step(i, current, context, trace, cfg)
                    i += 1
        finally:
            trace.total_duration_s = time.perf_counter() - t_start
            if collecting:
                try:
                    cfg.collector.on_trace(trace)
                except Exception:
                    _log.exception("TraceCollector.on_trace raised; trace dropped")
        return current

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

    def _step_into_batch(self, current: Any, context: Context, i: int, trace: Any, cfg: TracingConfig | None) -> tuple[Any, Context, int]:
        from .batch import LeaderBatch
        from .ops import UnBatch

        gate = self.operators[i].gate
        batch_label = self._label_for(i, cfg.operator_labels if cfg else None)

        # Span 1: gate wait — each thread records its own lobby accumulation time
        t_gate_enter = time.perf_counter()
        outcome = gate.enter(current)
        gate_blocked_duration = time.perf_counter() - t_gate_enter
        i += 1  # move past the Batch operator itself

        # Follower: skip region, receive leader's batch span (or error) via gate
        if not isinstance(outcome, LeaderBatch):
            while not isinstance(self.operators[i], UnBatch):
                i += 1
            # gate_blocked_duration includes lobby wait + batch execution for followers.
            # Subtract the batch region duration to isolate the lobby accumulation time,
            # making it comparable to the leader's wait.
            batch_region_duration = outcome.batch_span.duration_s if outcome.batch_span is not None else 0.0
            lobby_wait_duration = gate_blocked_duration - batch_region_duration
            trace.spans.append(StepSpan(f"{batch_label}[wait]", t_gate_enter, lobby_wait_duration))
            if outcome.batch_span is not None:
                trace.spans.append(outcome.batch_span)
            if outcome.exception is not None:
                raise outcome.exception
            return outcome.result, context, i + 1

        trace.spans.append(StepSpan(f"{batch_label}[wait]", t_gate_enter, gate_blocked_duration))

        # Leader: run region operators into a child trace
        current = outcome.inputs
        batch_size = len(current) if hasattr(current, "__len__") else None
        collecting = isinstance(trace, InvocationTrace)
        child_trace = InvocationTrace(batch_size=batch_size) if collecting else _NoOpTrace(batch_size=batch_size)
        batch_context = Context()

        t_region = time.perf_counter()
        try:
            while not isinstance(self.operators[i], UnBatch):
                current, batch_context = self._step(i, current, batch_context, child_trace, cfg)
                i += 1
        except Exception as exc:
            child_trace.total_duration_s = time.perf_counter() - t_region
            batch_span = StepSpan(
                batch_label, t_region, child_trace.total_duration_s,
                error=True, child_trace=child_trace if collecting else None,
            )
            trace.spans.append(batch_span)
            gate.distribute_exception(exc, batch_span=batch_span if collecting else None)
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

    def _step_into_scatter(self, current: Any, context: Context, i: int, trace: Any, cfg: TracingConfig | None) -> tuple[Any, Context, int]:
        from .ops import Gather, Scatter

        gate = self.operators[i].gate
        scatter_label = self._label_for(i, cfg.operator_labels if cfg else None)
        items: list[Any] = current
        region_start = i + 1  # first op after Scatter

        # Find the matching Gather by tracking Scatter depth.
        depth = 1
        j = region_start
        while j < len(self.operators) and depth > 0:
            op = self.operators[j]
            if isinstance(op, Scatter):
                depth += 1
            elif isinstance(op, Gather):
                depth -= 1
            j += 1
        gather_pos = j - 1

        collecting = isinstance(trace, InvocationTrace)
        t_scatter = time.perf_counter()

        n_items = len(items)

        def run_region(entry: Any) -> None:
            child_trace = InvocationTrace(batch_size=n_items, scatter_workers=gate.max_concurrency) if collecting else None
            try:
                result = self._execute(entry.value, trace=child_trace, region=(region_start, gather_pos))
                entry.deposit(result, child_trace)
            except BaseException as exc:
                entry.deposit_exception(exc, child_trace)

        gate.scatter(items, run_region)

        try:
            entries = gate.gather()
        except BaseException as exc:
            trace.spans.append(StepSpan(scatter_label, t_scatter, time.perf_counter() - t_scatter, error=True))
            raise

        child_traces = [e.child_trace for e in entries if e.child_trace is not None]
        child_trace = merge_traces(child_traces) if child_traces else None
        scatter_span = StepSpan(
            scatter_label, t_scatter, time.perf_counter() - t_scatter,
            child_trace=child_trace if collecting else None,
        )
        trace.spans.append(scatter_span)
        return [e.result for e in entries], context, gather_pos + 1

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

