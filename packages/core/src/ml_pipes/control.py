from __future__ import annotations


class _ShortCircuitSignal:
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


SHORT_CIRCUIT = _ShortCircuitSignal()

