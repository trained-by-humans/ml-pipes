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
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np


Variance: TypeAlias = Literal["covariant", "invariant", "contravariant"]

COVARIANT: Variance = "covariant"
INVARIANT: Variance = "invariant"
CONTRAVARIANT: Variance = "contravariant"


@dataclass(frozen=True)
class GenericSemantics:
    """Compatibility rules retained privately for an erased runtime generic.

    Attributes:
        bare_arguments: Canonical arguments used when the origin is bare.
        variances: Per-parameter compatibility direction:

            * ``covariant`` accepts ``G[Child]`` where ``G[Base]`` is expected.
            * ``contravariant`` reverses that direction.
            * ``invariant`` requires arguments to be mutually compatible.

            A single entry applies to every parameter when the generic has
            variable arity.
        strict_argument_indices: Argument positions that must be concrete in
            strict validation. ``None`` requires every argument to be concrete.
    """

    bare_arguments: tuple[object, ...]
    variances: tuple[Variance, ...]
    strict_argument_indices: tuple[int, ...] | None = None

    @property
    def is_variable_arity(self) -> bool:
        """Whether the bare form marks a variable-arity generic grammar."""
        return Ellipsis in self.bare_arguments

_GENERIC_SEMANTICS: dict[object, GenericSemantics] = {
    AbstractSet: GenericSemantics((Any,), (COVARIANT,)),
    Collection: GenericSemantics((Any,), (COVARIANT,)),
    Iterable: GenericSemantics((Any,), (COVARIANT,)),
    MutableSequence: GenericSemantics((Any,), (INVARIANT,)),
    MutableSet: GenericSemantics((Any,), (INVARIANT,)),
    Sequence: GenericSemantics((Any,), (COVARIANT,)),
    frozenset: GenericSemantics((Any,), (COVARIANT,)),
    list: GenericSemantics((Any,), (INVARIANT,)),
    set: GenericSemantics((Any,), (INVARIANT,)),
    type: GenericSemantics((Any,), (COVARIANT,)),
    Mapping: GenericSemantics((Any, Any), (INVARIANT, COVARIANT)),
    MutableMapping: GenericSemantics((Any, Any), (INVARIANT, INVARIANT)),
    dict: GenericSemantics((Any, Any), (INVARIANT, INVARIANT)),
    tuple: GenericSemantics(
        (Any, Ellipsis),
        (COVARIANT,),
    ),
    np.dtype: GenericSemantics((Any,), (COVARIANT,)),
    # Array shape is intentionally not a strict pipeline contract. Dtype is.
    np.ndarray: GenericSemantics(
        bare_arguments=(tuple[Any, ...], np.dtype[Any]),
        variances=(COVARIANT, COVARIANT),
        strict_argument_indices=(1,),
    ),
}


def bare_arguments(origin: object) -> tuple[object, ...] | None:
    semantics = _semantics_for(origin)
    return semantics.bare_arguments if semantics is not None else None


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
    semantics = _semantics_for(origin)
    if semantics is not None:
        return (
            not semantics.is_variable_arity
            and supplied_argument_count != len(semantics.bare_arguments)
        )
    return 0 < runtime_parameter_count != supplied_argument_count


def variances(origin: object, parameter_count: int) -> tuple[Variance, ...]:
    semantics = _semantics_for(origin)
    if semantics is not None:
        if len(semantics.variances) == parameter_count:
            return semantics.variances
        if len(semantics.variances) == 1:
            return semantics.variances * parameter_count

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
    semantics = _semantics_for(origin)
    if semantics is not None and semantics.strict_argument_indices is not None:
        return semantics.strict_argument_indices
    return tuple(range(argument_count))


def _semantics_for(origin: object) -> GenericSemantics | None:
    return _GENERIC_SEMANTICS.get(origin)
