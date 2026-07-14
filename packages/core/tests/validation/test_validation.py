from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence, Sequence
import ml_pipes.validation as validation_module
import warnings

import pytest
from typing import Any, Generic, MutableMapping as TypingMutableMapping, MutableSequence as TypingMutableSequence, TypeVar

from ml_pipes.core import Pipeline
from ml_pipes._typing.annotation import expand_annotation_parts
from ml_pipes.standard import (
    Batch,
    Gather,
    Recall,
    Scatter,
    Store,
    UnBatch,
)
from ml_pipes.validation import PipelineValidationError
from ml_pipes.validation import (
    PipelineValidator,
    PipelineValidationWarning,
    _BoundarySignature,
    _OperatorBoundary,
)

_VariadicT = TypeVar("_VariadicT")
_BoxT = TypeVar("_BoxT")


class _Box(Generic[_BoxT]):
    pass


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


class IntToListInt:
    def __call__(self, value: int) -> list[int]:
        return [value]


class BareListConsumer:
    def __call__(self, value: list) -> str:
        return "list"


class BareListProducer:
    def __call__(self, value: int) -> list:
        return [value]


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


class PairConsumer:
    def __call__(self, left: int, right: int) -> str:
        return f"{left}-{right}"


class VariadicTupleConsumer:
    def __call__(self, value: tuple[int, ...]) -> str:
        return ",".join(str(item) for item in value)


class GenericVariadicTupleConsumer:
    def __call__(self, value: tuple[_VariadicT, ...]) -> str:
        return ",".join(str(item) for item in value)


class IntIterableConsumer:
    def __call__(self, value: Iterable[int]) -> str:
        return ",".join(str(item) for item in value)


class IntToIterableInt:
    def __call__(self, value: int) -> Iterable[int]:
        return [value]


class BareIterableConsumer:
    def __call__(self, value: Iterable) -> str:
        return ",".join(str(item) for item in value)


class BareIterableProducer:
    def __call__(self, value: int) -> Iterable:
        return [value]


class BareBoxConsumer:
    def __call__(self, value: _Box) -> str:
        return "box"


class BareBoxProducer:
    def __call__(self, value: int) -> _Box:
        return _Box()


class GenericIterableConsumer:
    def __call__(self, value: Iterable[_VariadicT]) -> str:
        return ",".join(str(item) for item in value)


class VariadicCollector:
    def __call__(self, *values: object) -> tuple[object, ...]:
        return values


class MixedVariadicConsumer:
    def __call__(self, value: int, *rest: int) -> tuple[int, ...]:
        return (value, *rest)


class KeywordOnlyConsumer:
    def __call__(self, value: int, *, scale: int) -> int:
        return value * scale


class VarKeywordConsumer:
    def __call__(self, value: int, **metadata: object) -> int:
        del metadata
        return value


class MultiArgDefaultConsumer:
    def __call__(self, value: tuple[int, int], scale: int = 0) -> int:
        return sum(value) + scale


class SingleArgDefaultConsumer:
    def __call__(self, value: int = 0) -> int:
        return value


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

    def resolve_contract(self, upstream_annotation, validation_error_type):
        return (Any,), upstream_annotation


class DynamicFixedDictOutput:
    def __call__(self, value: Any) -> dict[str, int]:
        return {"x": 1}

    def resolve_contract(self, upstream_annotation, validation_error_type):
        return (Any,), dict[str, int]


class PlainTupleProjection:
    def __call__(self, value: Any) -> tuple[Any, Any]:
        return value, value

    def resolve_contract(self, upstream_annotation, validation_error_type):
        return (Any,), (upstream_annotation, upstream_annotation)


class PartiallyResolvedTupleOutput:
    def __call__(self, value: Any) -> tuple[Any, Any]:
        return value, value

    def resolve_contract(self, upstream_annotation, validation_error_type):
        return (Any,), (upstream_annotation, Any)


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

    with pytest.raises(
        PipelineValidationError,
        match=r"Pipeline step 0:UntypedOp is missing a type annotation",
    ):
        pipeline.validate()


def test_pipeline_validate_rejects_non_callable_operator_with_explicit_message():
    pipeline = Pipeline([object()])

    with pytest.raises(
        PipelineValidationError,
        match=r"Pipeline step 0:object must define __call__",
    ):
        pipeline.validate()


def test_pipeline_validate_does_not_swallow_non_annotation_static_signature_errors():
    class BrokenStaticButDynamic:
        def __call__(self, value: "MissingType") -> int:
            return 1

        def resolve_contract(self, upstream_annotation, validation_error_type):
            return (Any,), int

    pipeline = Pipeline([BrokenStaticButDynamic()])

    with pytest.raises(NameError, match="MissingType"):
        pipeline.validate()


def test_pipeline_validate_allows_broader_downstream_input_type():
    pipeline = Pipeline([IntToString(), ObjectConsumer()])

    pipeline.validate()


def test_pipeline_validate_allows_parameterized_output_for_object_consumer():
    pipeline = Pipeline([IntToListInt(), ObjectConsumer()])

    pipeline.validate(pipeline_input_type=int)


def test_pipeline_validate_rejects_tighter_downstream_input_type():
    pipeline = Pipeline([ObjectProducer(), StringConsumer()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


@pytest.mark.parametrize(
    ("producer", "consumer"),
    [
        pytest.param(IntToListInt(), BareListConsumer(), id="list[int]-to-list"),
        pytest.param(IntToIterableInt(), BareIterableConsumer(), id="Iterable[int]-to-Iterable"),
    ],
)
def test_pipeline_validate_accepts_parameterized_generic_output_for_bare_consumer(
    producer,
    consumer,
):
    Pipeline([producer, consumer]).validate(pipeline_input_type=int)


@pytest.mark.parametrize(
    ("producer", "consumer"),
    [
        pytest.param(BareListProducer(), ListIntToStr(), id="list-to-list[int]"),
        pytest.param(BareIterableProducer(), IntIterableConsumer(), id="Iterable-to-Iterable[int]"),
    ],
)
def test_pipeline_validate_accepts_bare_generic_output_for_parameterized_consumer(
    producer,
    consumer,
):
    Pipeline([producer, consumer]).validate(pipeline_input_type=int)


@pytest.mark.parametrize(
    ("consumer", "pipeline_input_type"),
    [
        pytest.param(BareListConsumer(), list[int], id="list[int]-to-list"),
        pytest.param(BareIterableConsumer(), Iterable[int], id="Iterable[int]-to-Iterable"),
    ],
)
def test_pipeline_validate_accepts_parameterized_pipeline_input_for_bare_consumer(
    consumer,
    pipeline_input_type,
):
    Pipeline([consumer]).validate(pipeline_input_type=pipeline_input_type)


@pytest.mark.parametrize(
    ("consumer", "pipeline_input_type"),
    [
        pytest.param(ListIntToStr(), list, id="list-to-list[int]"),
        pytest.param(IntIterableConsumer(), Iterable, id="Iterable-to-Iterable[int]"),
    ],
)
def test_pipeline_validate_accepts_bare_pipeline_input_for_parameterized_consumer(
    consumer,
    pipeline_input_type,
):
    Pipeline([consumer]).validate(pipeline_input_type=pipeline_input_type)


@pytest.mark.parametrize(
    ("operator", "pipeline_input_type", "expected_input_type"),
    [
        pytest.param(BareListConsumer(), list, list[Any], id="list"),
        pytest.param(BareIterableConsumer(), Iterable, Iterable[Any], id="Iterable"),
    ],
)
def test_pipeline_validate_publishes_bare_generic_input_as_any_parameterized_annotation(
    operator,
    pipeline_input_type,
    expected_input_type,
):
    contract = Pipeline([operator]).validate(pipeline_input_type=pipeline_input_type)

    assert contract.input_type == expected_input_type
    assert contract.output_type is str


@pytest.mark.parametrize(
    ("operator", "expected_output_type"),
    [
        pytest.param(BareListProducer(), list[Any], id="list"),
        pytest.param(BareIterableProducer(), Iterable[Any], id="Iterable"),
    ],
)
def test_pipeline_validate_publishes_bare_generic_output_as_any_parameterized_annotation(
    operator,
    expected_output_type,
):
    contract = Pipeline([operator]).validate(pipeline_input_type=int)

    assert contract.input_type is int
    assert contract.output_type == expected_output_type


def test_pipeline_validate_accepts_bare_mutable_generic_aliases() -> None:
    class SequenceConsumer:
        def __call__(self, value: Sequence) -> str:
            return "ok"

    class MappingConsumer:
        def __call__(self, value: Mapping) -> str:
            return "ok"

    typing_sequence_contract = Pipeline([SequenceConsumer()]).validate(
        pipeline_input_type=TypingMutableSequence
    )
    collections_sequence_contract = Pipeline([SequenceConsumer()]).validate(
        pipeline_input_type=MutableSequence
    )
    typing_mapping_contract = Pipeline([MappingConsumer()]).validate(
        pipeline_input_type=TypingMutableMapping
    )
    collections_mapping_contract = Pipeline([MappingConsumer()]).validate(
        pipeline_input_type=MutableMapping
    )

    assert typing_sequence_contract.input_type == MutableSequence[Any]
    assert collections_sequence_contract.input_type == MutableSequence[Any]
    assert typing_mapping_contract.input_type == MutableMapping[Any, Any]
    assert collections_mapping_contract.input_type == MutableMapping[Any, Any]


@pytest.mark.parametrize(
    ("pipeline", "pipeline_input_type"),
    [
        pytest.param(Pipeline([BareBoxProducer()]), int, id="output"),
        pytest.param(Pipeline([BareBoxConsumer()]), _Box, id="input"),
    ],
)
def test_pipeline_validate_rejects_unsupported_bare_generic_annotations(
    pipeline,
    pipeline_input_type,
):
    with pytest.raises(ValueError, match="Unsupported bare generic annotation"):
        pipeline.validate(pipeline_input_type=pipeline_input_type)


def test_pipeline_validate_accepts_tuple_output_for_multi_arg_operator():
    pipeline = Pipeline([IntToPair(), PairToBool()])

    pipeline.validate()


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(VariadicTupleConsumer(), id="tuple[int,...]-to-tuple[int,...]"),
        pytest.param(GenericVariadicTupleConsumer(), id="tuple[int,...]-to-tuple[T,...]"),
        pytest.param(IntIterableConsumer(), id="tuple[int,...]-to-Iterable[int]"),
        pytest.param(GenericIterableConsumer(), id="tuple[int,...]-to-Iterable[T]"),
    ],
)
def test_pipeline_validate_keeps_variadic_tuple_as_single_input_boundary(operator):
    contract = Pipeline([operator]).validate(pipeline_input_type=tuple[int, ...])

    assert contract.input_type == tuple[int, ...]
    assert contract.output_type is str


@pytest.mark.parametrize(
    ("operator", "parameter_name"),
    [
        pytest.param(VariadicCollector(), "values", id="variadic-only"),
        pytest.param(MixedVariadicConsumer(), "rest", id="mixed-fixed-and-variadic"),
    ],
)
def test_pipeline_validate_rejects_variadic_positional_operator_parameters(operator, parameter_name):
    with pytest.raises(
        PipelineValidationError,
        match=rf"variadic positional parameters.*{parameter_name}",
    ):
        Pipeline([operator]).validate()


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(KeywordOnlyConsumer(), id="keyword-only"),
        pytest.param(VarKeywordConsumer(), id="var-keyword"),
    ],
)
def test_pipeline_validate_rejects_other_non_positional_operator_parameters(operator):
    with pytest.raises(PipelineValidationError, match="chains operators by argument position"):
        Pipeline([operator]).validate()


def test_pipeline_validate_warns_when_multi_arg_operator_uses_positional_defaults():
    with pytest.warns(
        PipelineValidationWarning,
        match=r"positional defaults \(scale\).*requiring 2 positional pipeline inputs",
    ):
        Pipeline([MultiArgDefaultConsumer()]).validate(
            pipeline_input_type=tuple[tuple[int, int], int]
        )


def test_pipeline_validate_does_not_warn_for_single_arg_positional_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineValidationWarning)
        Pipeline([SingleArgDefaultConsumer()]).validate()


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


def test_resolved_input_type_backpropagates_through_store():
    contract = Pipeline([Store("x"), IntToString()]).validate(inference=True)

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


def test_resolved_input_type_backpropagates_through_plain_tuple_contract_projection():
    contract = Pipeline([
        ContractPassthrough(),
        PlainTupleProjection(),
        PairConsumer(),
    ]).validate(inference=True)

    assert contract.input_type is int


def test_resolved_input_type_does_not_backpropagate_through_partially_unresolved_tuple_output():
    contract = Pipeline([
        ContractPassthrough(),
        PartiallyResolvedTupleOutput(),
        PairConsumer(),
    ]).validate(inference=True)

    assert contract.input_type is Any


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


def test_validate_accepts_concrete_iterable_subtype_for_bare_iterable_consumer():
    contract = Pipeline([IntToString(), BareIterableConsumer()]).validate(
        pipeline_input_type=int
    )

    assert contract is not None
    assert contract.input_type is int
    assert contract.output_type is str


def test_validate_preserves_fixed_tuple_input_shape_against_sequence_supertype():
    class SequenceConsumer:
        def __call__(self, value: Sequence[int | str]) -> str:
            return ",".join(str(item) for item in value)

    contract = Pipeline([SequenceConsumer()]).validate(
        pipeline_input_type=tuple[int, str]
    )

    assert contract is not None
    assert contract.input_type == tuple[int, str]


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


def test_entry_contract_mismatch_reports_pipeline_input():
    with pytest.raises(PipelineValidationError, match="Pipeline input provides"):
        Pipeline([IntToString()]).validate(pipeline_input_type=str)


class _Base:
    pass


class _Child(_Base):
    pass


class _Unrelated:
    pass


_T = TypeVar("_T", bound=_Base)
_U = TypeVar("_U")  # unbound


def test_typevar_output_resolved_when_same_typevar_flows_through_input():
    class IdentityTypeVar:
        def __call__(self, x: _T) -> _T: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    pipeline = Pipeline([IdentityTypeVar(), ConsumesChild()])
    pipeline.validate(pipeline_input_type=_Child)


def test_typevar_output_not_resolved_from_bound_only():
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


def test_validate_publishes_bound_for_unresolved_typevar_boundary():
    class IdentityTypeVar:
        def __call__(self, x: _T) -> _T: ...  # type: ignore[empty-body]

    contract = Pipeline([IdentityTypeVar()]).validate()

    assert contract.input_type is _Base
    assert contract.output_type is _Base


def test_validate_recursively_publishes_bound_inside_generic_output():
    class WrapTypeVarInList:
        def __call__(self, x: _T) -> list[_T]: ...  # type: ignore[empty-body]

    contract = Pipeline([WrapTypeVarInList()]).validate()

    assert contract.input_type is _Base
    assert contract.output_type == list[_Base]


def test_validate_recursively_publishes_bound_inside_dynamic_tuple_output():
    class WrapTypeVarInList:
        def __call__(self, x: _T) -> list[_T]: ...  # type: ignore[empty-body]

    typed_recall: Recall[list[_T], None] = Recall("saved")
    contract = Pipeline([WrapTypeVarInList(), Store("saved"), typed_recall]).validate()

    assert contract.input_type is _Base
    assert contract.output_type == (list[_Base], list[_Base])


def test_typevar_output_resolved_from_multi_parameter_signature():
    class MultiInputTypeVar:
        def __call__(self, x: _T, y: int) -> _T: ...  # type: ignore[empty-body]

    class ConsumesChild:
        def __call__(self, x: _Child) -> str: ...  # type: ignore[empty-body]

    Pipeline([MultiInputTypeVar(), ConsumesChild()]).validate(
        pipeline_input_type=tuple[_Child, int]
    )


def test_single_tuple_parameter_typevar_pipeline_rejects_mixed_tuple_input():
    class IterableToList:
        def __call__(self, x: Iterable[_U]) -> list[_U]: ...  # type: ignore[empty-body]

    class ConsumesChildList:
        def __call__(self, x: list[_Child]) -> str: ...  # type: ignore[empty-body]

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([IterableToList(), ConsumesChildList()]).validate(
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


def _make_projection_boundary(
    operator: Any,
    *,
    input_types: tuple[Any, ...],
    output_type: Any,
) -> _OperatorBoundary:
    return _OperatorBoundary(
        operator=operator,
        previous_output_type=Any,
        context_inputs=None,
        dynamic_boundary=_BoundarySignature(
            input_types=input_types,
            output_type=output_type,
        ),
        static_boundary=None,
    )


def test_project_input_annotation_from_output_template_matches_plain_tuple_template():
    class PlainTupleProjectionContract:
        def resolve_contract(
            self,
            upstream_annotation,
            validation_error_type,
        ):
            del validation_error_type
            return (upstream_annotation,), (upstream_annotation, str)

    boundary = _make_projection_boundary(
        PlainTupleProjectionContract(),
        input_types=(Any,),
        output_type=(Any, str),
    )

    assert PipelineValidator._project_input_annotation_from_output_template(
        boundary,
        tuple[int, str],
    ) is int


def test_project_input_annotation_from_output_template_supports_ordered_any_bindings():
    class OrderedListProjectionContract:
        def resolve_contract(
            self,
            upstream_annotation,
            validation_error_type,
        ):
            del validation_error_type
            left_annotation, right_annotation = expand_annotation_parts(upstream_annotation)
            return expand_annotation_parts(upstream_annotation), tuple[list[left_annotation], list[right_annotation]]

    boundary = _make_projection_boundary(
        OrderedListProjectionContract(),
        input_types=(Any, Any),
        output_type=tuple[list[Any], list[Any]],
    )

    assert PipelineValidator._project_input_annotation_from_output_template(
        boundary,
        tuple[list[int], list[str]],
    ) == tuple[int, str]


def test_project_input_annotation_from_output_template_rejects_placeholder_count_mismatch():
    class OrderedListProjectionContract:
        def resolve_contract(
            self,
            upstream_annotation,
            validation_error_type,
        ):
            del validation_error_type
            left_annotation, right_annotation = expand_annotation_parts(upstream_annotation)
            return expand_annotation_parts(upstream_annotation), tuple[list[left_annotation], list[right_annotation]]

    boundary = _make_projection_boundary(
        OrderedListProjectionContract(),
        input_types=(Any,),
        output_type=tuple[list[Any], list[Any]],
    )

    assert PipelineValidator._project_input_annotation_from_output_template(
        boundary,
        tuple[list[int], list[str]],
    ) is None


def test_project_input_annotation_from_output_template_returns_none_when_projection_fails_confirmation():
    class RejectingProjectionContract:
        def resolve_contract(
            self,
            upstream_annotation,
            validation_error_type,
        ):
            del validation_error_type
            return (upstream_annotation,), (str, upstream_annotation)

    boundary = _make_projection_boundary(
        RejectingProjectionContract(),
        input_types=(Any,),
        output_type=(Any, str),
    )

    assert PipelineValidator._project_input_annotation_from_output_template(
        boundary,
        tuple[int, str],
    ) is None


def test_project_input_annotation_from_output_template_returns_none_when_binding_collection_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    class PlainTupleProjectionContract:
        def resolve_contract(
            self,
            upstream_annotation,
            validation_error_type,
        ):
            del validation_error_type
            return (upstream_annotation,), (upstream_annotation, str)

    boundary = _make_projection_boundary(
        PlainTupleProjectionContract(),
        input_types=(Any,),
        output_type=(Any, str),
    )
    monkeypatch.setattr(
        validation_module,
        "_collect_any_placeholder_bindings",
        lambda *args: None,
    )

    assert PipelineValidator._project_input_annotation_from_output_template(
        boundary,
        tuple[int, str],
    ) is None


def test_project_input_annotation_from_output_template_returns_none_when_placeholder_replacement_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    class PlainTupleProjectionContract:
        def resolve_contract(
            self,
            upstream_annotation,
            validation_error_type,
        ):
            del validation_error_type
            return (upstream_annotation,), (upstream_annotation, str)

    boundary = _make_projection_boundary(
        PlainTupleProjectionContract(),
        input_types=(Any,),
        output_type=(Any, str),
    )
    monkeypatch.setattr(
        validation_module,
        "_replace_any_placeholders_in_order",
        lambda *args: None,
    )

    assert PipelineValidator._project_input_annotation_from_output_template(
        boundary,
        tuple[int, str],
    ) is None


def test_validation_rejects_variadic_tuple_projection_with_pipeline_error():
    class IterableConsumer:
        def __call__(self, value: Iterable[int]) -> str:
            return "ok"

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([Store("saved"), Recall("saved"), IterableConsumer()]).validate(
            pipeline_input_type=tuple[int, ...]
        )
