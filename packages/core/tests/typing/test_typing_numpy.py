from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from ml_pipes._typing.annotation import is_assignable
from ml_pipes.core import Pipeline
from ml_pipes.validation import PipelineValidationError


_NATIVE_UINT8 = np.ndarray[tuple[Any, ...], np.dtype[np.uint8]]
_NATIVE_GENERIC = np.ndarray[tuple[Any, ...], np.dtype[np.generic]]


# The expected results are NumPy's public typing contract.  The companion
# mypy test verifies that contract against the installed NumPy stubs.
_NUMPY_COMPATIBILITY_CASES = [
    (
        npt.NDArray[np.uint8],
        npt.NDArray[np.generic],
        "npt.NDArray[np.uint8]",
        "npt.NDArray[np.generic]",
        True,
        True,
    ),
    (
        npt.NDArray[np.generic],
        npt.NDArray[np.uint8],
        "npt.NDArray[np.generic]",
        "npt.NDArray[np.uint8]",
        False,
        False,
    ),
    (
        npt.NDArray[np.float32],
        npt.NDArray[np.generic],
        "npt.NDArray[np.float32]",
        "npt.NDArray[np.generic]",
        True,
        True,
    ),
    (
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        "npt.NDArray[np.uint8]",
        "npt.NDArray[np.float32]",
        False,
        False,
    ),
    (
        npt.NDArray[np.uint8],
        _NATIVE_GENERIC,
        "npt.NDArray[np.uint8]",
        "np.ndarray[tuple[Any, ...], np.dtype[np.generic]]",
        True,
        True,
    ),
    (
        _NATIVE_UINT8,
        npt.NDArray[np.generic],
        "np.ndarray[tuple[Any, ...], np.dtype[np.uint8]]",
        "npt.NDArray[np.generic]",
        True,
        True,
    ),
    (
        _NATIVE_UINT8,
        _NATIVE_GENERIC,
        "np.ndarray[tuple[Any, ...], np.dtype[np.uint8]]",
        "np.ndarray[tuple[Any, ...], np.dtype[np.generic]]",
        True,
        True,
    ),
    (
        _NATIVE_GENERIC,
        _NATIVE_UINT8,
        "np.ndarray[tuple[Any, ...], np.dtype[np.generic]]",
        "np.ndarray[tuple[Any, ...], np.dtype[np.uint8]]",
        False,
        False,
    ),
    (
        npt.NDArray[np.uint8],
        np.ndarray,
        "npt.NDArray[np.uint8]",
        "np.ndarray",
        True,
        True,
    ),
    (
        np.ndarray,
        npt.NDArray[np.uint8],
        "np.ndarray",
        "npt.NDArray[np.uint8]",
        True,
        True,
    ),
    (
        np.dtype[np.uint8],
        np.dtype[np.generic],
        "np.dtype[np.uint8]",
        "np.dtype[np.generic]",
        True,
        True,
    ),
    (
        np.dtype[np.generic],
        np.dtype[np.uint8],
        "np.dtype[np.generic]",
        "np.dtype[np.uint8]",
        False,
        False,
    ),
]

_MATCHER_COMPATIBILITY_CASES = _NUMPY_COMPATIBILITY_CASES


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation", "source_expression", "target_expression", "expected", "matcher_gap"),
    _MATCHER_COMPATIBILITY_CASES,
)
def test_numpy_annotation_assignability_matches_numpy_typing(
    source_annotation: object,
    target_annotation: object,
    source_expression: str,
    target_expression: str,
    expected: bool,
    matcher_gap: bool,
) -> None:
    del source_expression, target_expression
    del matcher_gap
    assert is_assignable(source_annotation, target_annotation) is expected


def _producer(output_annotation: object) -> object:
    class Producer:
        def __call__(self, value: object) -> object:
            return value

    Producer.__call__.__annotations__["return"] = output_annotation
    return Producer()


def _consumer(input_annotation: object) -> object:
    class Consumer:
        def __call__(self, value: object) -> object:
            return value

    Consumer.__call__.__annotations__["value"] = input_annotation
    return Consumer()


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation", "source_expression", "target_expression", "expected", "matcher_gap"),
    _MATCHER_COMPATIBILITY_CASES,
)
def test_numpy_annotation_compatibility_at_pipeline_boundary(
    source_annotation: object,
    target_annotation: object,
    source_expression: str,
    target_expression: str,
    expected: bool,
    matcher_gap: bool,
) -> None:
    del source_expression, target_expression
    pipeline = Pipeline([_producer(source_annotation), _consumer(target_annotation)])

    del matcher_gap
    if expected:
        pipeline.validate()
    else:
        with pytest.raises(PipelineValidationError, match="Pipeline contract mismatch"):
            pipeline.validate()


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation", "source_expression", "target_expression", "expected", "matcher_gap"),
    _NUMPY_COMPATIBILITY_CASES,
)
def test_numpy_stub_assignability_oracle(
    source_annotation: object,
    target_annotation: object,
    source_expression: str,
    target_expression: str,
    expected: bool,
    matcher_gap: bool,
) -> None:
    """Keep the matrix aligned with the installed NumPy type-checker stubs."""
    del source_annotation, target_annotation, matcher_gap
    from mypy import api as mypy_api
    program = (
        "from typing import Any\n"
        "import numpy as np\n"
        "import numpy.typing as npt\n\n"
        f"source: {source_expression}\n"
        f"def target(value: {target_expression}) -> None: ...\n"
        "target(source)\n"
    )

    _, _, exit_status = mypy_api.run(
        ["-c", program, "--no-error-summary", "--show-error-codes"],
    )

    assert (exit_status == 0) is expected


def test_strict_treats_ndarray_shape_as_non_contractual_but_requires_dtype() -> None:
    class TypedArray:
        def __call__(self, value: npt.NDArray[np.uint8]) -> _NATIVE_UINT8:
            return value

    class BareArray:
        def __call__(self, value: np.ndarray) -> np.ndarray:
            return value

    class TypedDType:
        def __call__(self, value: np.dtype[np.uint8]) -> np.dtype[np.uint8]:
            return value

    class BareDType:
        def __call__(self, value: np.dtype) -> np.dtype:
            return value

    Pipeline([TypedArray()]).validate(strict=True)
    Pipeline([TypedDType()]).validate(strict=True)
    with pytest.raises(PipelineValidationError, match="unresolved"):
        Pipeline([BareArray()]).validate(strict=True)
    with pytest.raises(PipelineValidationError, match="unresolved"):
        Pipeline([BareDType()]).validate(strict=True)
