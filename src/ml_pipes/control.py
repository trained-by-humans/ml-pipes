from __future__ import annotations


class _ShortCircuitSignal:
    """Internal sentinel meaning the current pipeline invocation stopped early.

    At the top-level pipeline boundary the caller receives this sentinel and can
    decide how to handle the incomplete invocation. Inside collection-oriented
    regions such as ``ForEachItem``, a short-circuited item is treated as
    dropped rather than emitted as a collection element.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "SHORT_CIRCUIT"

    def __str__(self) -> str:
        return "SHORT_CIRCUIT"

    def __copy__(self) -> _ShortCircuitSignal:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _ShortCircuitSignal:
        del memo
        return self

    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_restore_short_circuit, ())


def _restore_short_circuit() -> _ShortCircuitSignal:
    return SHORT_CIRCUIT

# Public singleton used by operators and the runtime to signal early pipeline
# termination without raising an exception.
SHORT_CIRCUIT = _ShortCircuitSignal()
