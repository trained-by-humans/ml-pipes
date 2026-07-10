from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar, get_args, get_origin

from ml_pipes.control import SHORT_CIRCUIT
from ml_pipes.operator import Operator
from ml_pipes.region import RegionCloser, RegionExecutor, RegionOpener, RegionTraceLike
from ml_pipes.tracing import InvocationTrace, StepSpan, TracingConfig, _NoOpTrace, merge_traces


class _ScatterEntry:
    __slots__ = ("value", "index", "event", "result", "child_trace", "exception")

    def __init__(self, value: Any, index: int) -> None:
        self.value = value
        self.index = index
        self.event: threading.Event = threading.Event()
        self.result: Any = None
        self.child_trace: Any = None
        self.exception: BaseException | None = None

    def deposit(self, result: Any, child_trace: Any = None) -> None:
        self.result = result
        self.child_trace = child_trace
        self.event.set()

    def deposit_exception(self, exc: BaseException, child_trace: Any = None) -> None:
        self.exception = exc
        self.child_trace = child_trace
        self.event.set()


class ScatterGate:
    """
    Coordination primitive for the Scatter/Gather operator pair.

    One caller fans out a list of items to N worker threads; each worker runs
    the scatter region independently with a fresh Context, then deposits its
    result. The original thread blocks at gather() until all workers have
    deposited, then collects results in submission order.
    """

    def __init__(self, max_concurrency: int) -> None:
        self.max_concurrency = max_concurrency
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self._local = threading.local()

    def scatter(self, items: list[Any], run_region: Any) -> None:
        entries = [_ScatterEntry(item, index) for index, item in enumerate(items)]
        self._local.entries = entries
        for entry in entries:
            self._executor.submit(run_region, entry)

    def gather(self) -> tuple[list[_ScatterEntry], BaseException | None]:
        entries: list[_ScatterEntry] = self._local.entries
        del self._local.entries

        first_exc: BaseException | None = None
        for entry in entries:
            entry.event.wait()
            if entry.exception is not None and first_exc is None:
                first_exc = entry.exception

        return entries, first_exc


ScatterItemT = TypeVar("ScatterItemT")


@Operator
class Gather(RegionCloser[ScatterItemT, list[ScatterItemT]]):
    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        out = list[current_output] if current_output is not None else list[Any]
        return (Any,), out


@Operator
class Scatter(RegionOpener[list[ScatterItemT], ScatterItemT]):
    closing_type = Gather

    def __init__(self, max_concurrency: int = 1) -> None:
        self.gate = ScatterGate(max_concurrency)

    def run_region(
        self,
        current: list[ScatterItemT],
        label: str,
        execute_region: RegionExecutor[ScatterItemT, Any],
        trace: RegionTraceLike,
        cfg: TracingConfig | None,
    ) -> Any:
        del cfg
        gate = self.gate
        collecting = isinstance(trace, InvocationTrace)
        items = current
        n_items = len(items)

        def run_region(entry: Any) -> None:
            child_trace = (
                InvocationTrace(batch_size=n_items, workers=gate.max_concurrency)
                if collecting
                else _NoOpTrace()
            )
            try:
                result, child_trace = execute_region(entry.value, child_trace)
                entry.deposit(result, child_trace if collecting else None)
            except BaseException as exc:
                entry.deposit_exception(exc, child_trace if collecting else None)

        gate.scatter(items, run_region)
        t_gather = time.perf_counter()
        entries, first_exc = gate.gather()
        child_traces = [entry.child_trace for entry in entries if entry.child_trace is not None]
        child_trace = merge_traces(child_traces) if child_traces else None

        if first_exc is not None:
            trace.spans.append(
                StepSpan(
                    label,
                    t_gather,
                    time.perf_counter() - t_gather,
                    error=True,
                    child_trace=child_trace if collecting else None,
                    operator_type=type(self),
                )
            )
            raise first_exc

        trace.spans.append(
            StepSpan(
                label,
                t_gather,
                time.perf_counter() - t_gather,
                child_trace=child_trace if collecting else None,
                operator_type=type(self),
            )
        )
        return [entry.result for entry in entries if entry.result is not SHORT_CIRCUIT]

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        if current_output is not None and get_origin(current_output) is list:
            args = get_args(current_output)
            return (list[Any],), args[0] if args else Any
        return (list[Any],), Any
