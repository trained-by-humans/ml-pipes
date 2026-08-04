from __future__ import annotations

from collections.abc import Sequence


def resolve_multi_output_names(
    operator_name: str,
    srcs: Sequence[str],
    as_: str | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if not srcs:
        raise ValueError(f"{operator_name} requires at least one source tensor")
    if len(srcs) == 1:
        src = srcs[0]
        if as_ is not None and not isinstance(as_, str):
            raise ValueError(f"{operator_name} as_ must be a string when operating on one tensor")
        return (as_ or src,)
    if as_ is None:
        return tuple(srcs)
    if isinstance(as_, str):
        raise ValueError(f"{operator_name} as_ must be a tuple when operating on more than one tensor")
    if len(as_) != len(srcs):
        raise ValueError(f"{operator_name} as_ tuple must match the number of source tensors")
    return tuple(as_)
