"""Private runtime rules for generic annotation compatibility.

Most generic classes retain their TypeVars at runtime, so their declared
variance is sufficient.  A small number of Python and third-party generics
erase that metadata; those live in this registry instead of the matcher.
"""
from __future__ import annotations

from collections.abc import (
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from typing import Any

import numpy as np


COVARIANT = "covariant"
INVARIANT = "invariant"
CONTRAVARIANT = "contravariant"


_BARE_ARGUMENTS: dict[object, tuple[object, ...]] = {
    AbstractSet: (Any,),
    Collection: (Any,),
    Iterable: (Any,),
    MutableSequence: (Any,),
    MutableSet: (Any,),
    Sequence: (Any,),
    frozenset: (Any,),
    list: (Any,),
    set: (Any,),
    type: (Any,),
    Mapping: (Any, Any),
    MutableMapping: (Any, Any),
    dict: (Any, Any),
    tuple: (Any, Ellipsis),
    np.dtype: (Any,),
    np.ndarray: (tuple[Any, ...], np.dtype[Any]),
}

_VARIANCES: dict[object, tuple[str, ...]] = {
    AbstractSet: (COVARIANT,),
    Collection: (COVARIANT,),
    Iterable: (COVARIANT,),
    Sequence: (COVARIANT,),
    frozenset: (COVARIANT,),
    tuple: (COVARIANT,),
    type: (COVARIANT,),
    Mapping: (INVARIANT, COVARIANT),
    np.dtype: (COVARIANT,),
    np.ndarray: (COVARIANT, COVARIANT),
}

# Array shape is intentionally not a strict pipeline contract.  Dtype is.
_STRICT_ARGUMENT_INDICES: dict[object, tuple[int, ...]] = {
    np.ndarray: (1,),
}


def bare_arguments(origin: object) -> tuple[object, ...] | None:
    return _BARE_ARGUMENTS.get(origin)


def is_partial_fixed_arity(
    origin: object,
    supplied_argument_count: int,
    runtime_parameter_count: int,
) -> bool:
    """Whether a supported fixed-arity generic has missing arguments.

    Registry entries describe erased-runtime generics.  For ordinary runtime
    generics, the retained parameter count supplies the same arity metadata.
    A bare form containing ``Ellipsis`` uses the supported variable-arity
    grammar and is exempt from fixed-arity validation.
    """
    bare_args = bare_arguments(origin)
    if bare_args is not None:
        if Ellipsis in bare_args:
            return False
        expected_argument_count = len(bare_args)
    else:
        expected_argument_count = runtime_parameter_count

    return (
        expected_argument_count > 0
        and supplied_argument_count != expected_argument_count
    )


def variances(origin: object, parameter_count: int) -> tuple[str, ...]:
    registered = _VARIANCES.get(origin)
    if registered is not None:
        if len(registered) == parameter_count:
            return registered
        if len(registered) == 1:
            return registered * parameter_count

    parameters = getattr(origin, "__type_params__", ()) or getattr(origin, "__parameters__", ())
    if not isinstance(parameters, tuple):
        parameters = ()
    if len(parameters) == parameter_count:
        return tuple(
            COVARIANT if getattr(parameter, "__covariant__", False)
            else CONTRAVARIANT if getattr(parameter, "__contravariant__", False)
            else INVARIANT
            for parameter in parameters
        )
    return (INVARIANT,) * parameter_count


def strict_argument_indices(origin: object, argument_count: int) -> tuple[int, ...]:
    return _STRICT_ARGUMENT_INDICES.get(origin, tuple(range(argument_count)))
