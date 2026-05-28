from collections.abc import Iterable

import pytest
from typing import Any, TypeVar

from ml_pipes import Pipeline, PipelineValidationError, Scatter, Gather, Batch, UnBatch
from ml_pipes.validation import _resolve_typevar_output, is_single_annotation_compatible


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class FloatToBool:
    def __call__(self, value: float) -> bool:
        return value > 0


class BoolToBytes:
    def __call__(self, value: bool) -> bytes:
        return b"1" if value else b"0"


class ListIntToStr:
    def __call__(self, value: list[int]) -> str:
        return "list of ints"


class ObjectConsumer:
    def __call__(self, value: object) -> object:
        return value


class ObjectProducer:
    def __call__(self, value: int) -> object:
        return value


class IntToPair:
    def __call__(self, value: int) -> tuple[int, str]:
        return value, str(value)


class PairToBool:
    def __call__(self, number: int, text: str) -> bool:
        return text == str(number)


class PartialOp1:
    def __call__(self, value: tuple[int, Any]) -> tuple[int, Any]:
        return value


class PartialOp2:
    def __call__(self, number: Any, text: str) -> bool:
        return text == str(number)


class TripleConsumer:
    def __call__(self, x: int, y: str, z: float) -> str:
        return f"{x}-{y}-{z}"


class VagueOp:
    def __call__(self, value: Any) -> Any:
        return value

class VagueListOp:
    def __call__(self, value: list[Any]) -> list[Any]:
        return value


class VagueToListOp:
    def __call__(self, value: Any) -> list[Any]:
        return value


class VagueListToSingleOp:
    def __call__(self, value: list[Any]) -> Any:
        return value


class StringConsumer:
    def __call__(self, value: str) -> str:
        return value


class DictConsumer:
    def __call__(self, value: dict[str, int]) -> bool:
        return True


class ContractPassthrough:
    def __call__(self, value: Any) -> Any:
        return value

    def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, validation_error_type):
        return (Any,), current_output


class DynamicFixedDictOutput:
    def __call__(self, value: Any) -> dict[str, int]:
        return {"x": 1}

    def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, validation_error_type):
        return (Any,), dict[str, int]


VALIDATION_MODE_CASES = [
    pytest.param(
        {"strict": False},
        Any,
        str,
        id="non-strict-forward-only",
    ),
    pytest.param(
        {"pipeline_input_type": bool, "strict": False},
        bool,
        str,
        id="non-strict-forward-with-pipeline-input",
    ),
    pytest.param(
        {"inference": True, "strict": False},
        int,
        str,
        id="non-strict-backward-inference",
    ),
    pytest.param(
        {"strict": True},
        Any,
        str,
        id="strict-forward-only",
    ),
    pytest.param(
        {"pipeline_input_type": bool, "strict": True},
        bool,
        str,
        id="strict-forward-with-pipeline-input",
    ),
    pytest.param(
        {"inference": True, "strict": True},
        int,
        str,
        id="strict-backward-inference",
    ),
]


STRICT_FAILURE_MODE_CASES = [
    pytest.param(
        {"strict": True},
        id="strict-forward-only",
    ),
    pytest.param(
        {"pipeline_input_type": bool, "strict": True},
        id="strict-forward-with-pipeline-input",
    ),
    pytest.param(
        {"inference": True, "strict": True},
        id="strict-backward-inference",
    ),
]


def test_validate_accepts_empty_pipeline():
    Pipeline([]).validate()  # must not raise


def test_pipeline_validate_accepts_compatible_operator_chain():
    pipeline = Pipeline([IntToString(), StringToFloat(), FloatToBool()])

    pipeline.validate()


def test_pipeline_validate_rejects_incompatible_operator_chain():
    pipeline = Pipeline([IntToString(), BoolToBytes()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_auto_validate_raises_during_initialization():
    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([IntToString(), BoolToBytes()], auto_validate=True)


def test_pipeline_validate_requires_operator_annotations():
    class UntypedOp:
        def __call__(self, value):
            return value

    pipeline = Pipeline([UntypedOp()])

    with pytest.raises(PipelineValidationError, match="missing a type annotation"):
        pipeline.validate()


def test_pipeline_validate_does_not_swallow_non_annotation_static_signature_errors():
    class BrokenStaticButDynamic:
        def __call__(self, value: "MissingType") -> int:
            return 1

        def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, validation_error_type):
            return (Any,), int

    pipeline = Pipeline([BrokenStaticButDynamic()])

    with pytest.raises(NameError, match="MissingType"):
        pipeline.validate()


def test_pipeline_validate_allows_broader_downstream_input_type():
    pipeline = Pipeline([IntToString(), ObjectConsumer()])

    pipeline.validate()


def test_pipeline_validate_rejects_tighter_downstream_input_type():
    pipeline = Pipeline([ObjectProducer(), StringConsumer()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_validate_accepts_tuple_output_for_multi_arg_operator():
    pipeline = Pipeline([IntToPair(), PairToBool()])

    pipeline.validate()


def test_pipeline_validate_rejects_tuple_output_with_wrong_arity():
    pipeline = Pipeline([IntToPair(), TripleConsumer()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_validate_vague_op_between_typed_ops_is_accepted():
    pipeline = Pipeline([IntToString(), VagueOp(), BoolToBytes()])

    pipeline.validate()  # must not raise


def test_resolved_input_type_uses_operator_boundary():
    contract = Pipeline([IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int
    assert contract.output_type is str


def test_resolved_input_type_defaults_to_any_when_no_constraint_is_known():
    contract = Pipeline([ContractPassthrough()]).validate()

    assert contract is not None
    assert contract.input_type is Any


def test_resolved_input_type_skips_contract_passthrough():
    inner = Pipeline([ContractPassthrough(), IntToString()])
    contract = inner.validate(inference=True)

    assert contract is not None
    assert contract.input_type is int


def test_resolved_input_type_preserves_strongest_known_boundary_through_vague_ops():
    contract = Pipeline([ContractPassthrough(), VagueListOp(), ListIntToStr()]).validate(inference=True)

    assert contract is not None
    assert contract.input_type == list[Any]


def test_resolved_input_type_backpropagates_through_contract_passthrough():
    inner = Pipeline([
        ContractPassthrough(),
        Scatter(),
        IntToString(),
        Gather()
    ])
    contract = inner.validate(inference=True)

    assert contract.input_type == list[int]


def test_resolved_input_type_stops_backpropagating_once_the_boundary_is_concrete():
    inner = Pipeline([
        ContractPassthrough(),
        Scatter(),
        IntToString(),
        StringToFloat(),
        Gather()
    ])
    contract = inner.validate(inference=True)

    assert contract.input_type == list[int]


def test_resolved_input_type_stops_backpropagating_at_vague_operator():
    inner = Pipeline([
        ContractPassthrough(),
        Scatter(),
        VagueOp(),
        Batch(size=2),
        ListIntToStr(),
        UnBatch(),
        Gather(),
    ])
    contract = inner.validate(inference=True)

    assert contract.input_type == list[Any]


def test_validate_does_not_run_backward_inference_by_default():
    contract = Pipeline([ContractPassthrough(), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is Any


def test_validate_prefers_more_concrete_explicit_pipeline_input_type():
    contract = Pipeline([IntToString()]).validate(pipeline_input_type=bool)

    assert contract is not None
    assert contract.input_type is bool


def test_validate_merges_complementary_partial_constraints():
    contract = Pipeline([PartialOp1(), PartialOp2()]).validate(
        pipeline_input_type=tuple[Any, str]
    )

    assert contract is not None
    assert contract.input_type == tuple[int, str]
    assert contract.output_type is bool


def test_inference_does_not_narrow_input_through_dynamic_fixed_output():
    contract = Pipeline([
        ContractPassthrough(),
        DynamicFixedDictOutput(),
        DictConsumer(),
    ]).validate(inference=True)

    assert contract is not None
    assert contract.input_type is Any
    assert contract.output_type is bool


@pytest.mark.parametrize(
    ("validate_kwargs", "expected_input_type", "expected_output_type"),
    VALIDATION_MODE_CASES,
)
def test_validate_matrix_for_transitive_pipeline(validate_kwargs, expected_input_type, expected_output_type):
    contract = Pipeline([ContractPassthrough(), IntToString()]).validate(**validate_kwargs)

    assert contract is not None
    assert contract.input_type == expected_input_type
    assert contract.output_type is expected_output_type


@pytest.mark.parametrize("validate_kwargs", STRICT_FAILURE_MODE_CASES)
def test_validate_matrix_strict_rejects_vague_pipeline_in_all_modes(validate_kwargs):
    with pytest.raises(PipelineValidationError, match="Strict mode violation"):
        Pipeline([VagueOp()]).validate(**validate_kwargs)


def test_declared_pipeline_input_can_surface_incompatible_entry_type():
    pipeline = Pipeline([ContractPassthrough(), IntToString()])

    contract = pipeline.validate()
    assert contract is not None
    assert contract.input_type is Any

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate(pipeline_input_type=str)


# ---------------------------------------------------------------------------
# TypeVar compatibility
# ---------------------------------------------------------------------------

class _Base:
    pass

class _Child(_Base):
    pass

class _Unrelated:
    pass

_T = TypeVar("_T", bound=_Base)
_U = TypeVar("_U")  # unbound


def test_typevar_in_expected_accepts_bound_subclass():
    # produced=_Child, expected=~_T (bound=_Base) → _Child is subclass of _Base
    assert is_single_annotation_compatible(_Child, _T)


def test_typevar_in_expected_accepts_exact_bound():
    assert is_single_annotation_compatible(_Base, _T)


def test_typevar_in_expected_rejects_unrelated():
    assert not is_single_annotation_compatible(_Unrelated, _T)


def test_typevar_in_produced_accepts_when_bound_subclass_of_expected():
    # produced=~_T (bound=_Base), expected=_Base → bound is assignable to expected
    assert is_single_annotation_compatible(_T, _Base)


def test_typevar_in_produced_rejects_when_expected_is_subtype_of_bound():
    # produced=~_T (bound=_Base), expected=_Child → _Child < _Base but _Base is not assignable to _Child
    assert not is_single_annotation_compatible(_T, _Child)


def test_typevar_in_produced_rejects_fully_unrelated():
    assert not is_single_annotation_compatible(_T, _Unrelated)


def test_unbound_typevar_in_expected_accepts_anything():
    assert is_single_annotation_compatible(int, _U)
    assert is_single_annotation_compatible(_Base, _U)


def test_unbound_typevar_in_produced_accepts_anything():
    assert is_single_annotation_compatible(_U, int)
    assert is_single_annotation_compatible(_U, _Base)


def test_generic_subtyping_accepts_list_as_iterable():
    assert is_single_annotation_compatible(list[int], Iterable[int])


def test_generic_covariance_accepts_child_list_as_base_iterable():
    assert is_single_annotation_compatible(list[_Child], Iterable[_Base])


def test_generic_invariance_rejects_child_list_as_base_list():
    assert not is_single_annotation_compatible(list[_Child], list[_Base])


def test_typevar_output_resolved_when_same_typevar_flows_through_input():
    # ~_T in both input and output preserves the concrete subtype.
    class IdentityTypeVar:
        def __call__(self, x: _T) -> _T: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    pipeline = Pipeline([IdentityTypeVar(), ConsumesChild()])
    pipeline.validate(pipeline_input_type=_Child)


def test_typevar_output_not_resolved_from_bound_only():
    # A bound alone (_Base -> ~_T) must not collapse to the concrete input subtype.
    class ProducesTypeVar:
        def __call__(self, x: _Base) -> _T: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([ProducesTypeVar(), ConsumesChild()], auto_validate=True).validate(
            pipeline_input_type=_Child
        )


def test_typevar_input_to_base_output_does_not_preserve_subtype():
    class TypeVarToBase:
        def __call__(self, x: _T) -> _Base: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([TypeVarToBase(), ConsumesChild()], auto_validate=True).validate(
            pipeline_input_type=_Child
        )


def test_declared_base_input_through_identity_typevar_stays_base():
    class IdentityTypeVar:
        def __call__(self, x: _T) -> _T: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([IdentityTypeVar(), ConsumesChild()]).validate(
            pipeline_input_type=_Base
        )


def test_unbound_identity_typevar_preserves_declared_input_type():
    class IdentityUnboundTypeVar:
        def __call__(self, x: _U) -> _U: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    pipeline = Pipeline([IdentityUnboundTypeVar(), ConsumesChild()])
    pipeline.validate(pipeline_input_type=_Child)


def test_resolve_typevar_output_recursively_specializes_nested_output():
    assert _resolve_typevar_output(list[_T], _Child, (_T,)) == list[_Child]


def test_resolve_typevar_output_through_generic_subtyping():
    assert _resolve_typevar_output(
        list[_U],
        list[int | None],
        (Iterable[_U | None],),
    ) == list[int]


def test_typevar_output_resolved_from_multi_parameter_signature():
    class MultiInputTypeVar:
        def __call__(self, x: _T, y: int) -> _T: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    Pipeline([MultiInputTypeVar(), ConsumesChild()]).validate(
        pipeline_input_type=tuple[_Child, int]
    )


def test_nested_typevar_output_is_recursively_specialized():
    class WrapTypeVarInList:
        def __call__(self, x: _T) -> list[_T]: ...  # type: ignore[empty-body]

    class ConsumesChildList:
        def __call__(self, x: list[_Child]) -> str: ...  # type: ignore[empty-body]

    Pipeline([WrapTypeVarInList(), ConsumesChildList()]).validate(
        pipeline_input_type=_Child
    )


def test_typevar_pipeline_rejects_narrower_consumer():
    # ~_T resolved to _Base; _Child is a strict subtype of _Base → _Base not assignable to _Child
    class ProducesTypeVar:
        def __call__(self, x: _Base) -> _T: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([ProducesTypeVar(), ConsumesChild()], auto_validate=True)


def test_typevar_pipeline_rejects_incompatible_consumer():
    class ProducesTypeVar:
        def __call__(self, x: _Base) -> _T: ...  # type: ignore[empty-body]

    class ConsumesUnrelated:
        def __call__(self, x: _Unrelated) -> str: ...  # type: ignore[empty-body]

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([ProducesTypeVar(), ConsumesUnrelated()], auto_validate=True)
