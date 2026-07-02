from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt

from ml_pipes._typing.annotation import is_assignable
from ml_pipes.operator import Operator
from .types import TensorPayload, TensorRegistry

__all__ = [
    "ApplyTensorMask",
    "ArgMax",
    "AsType",
    "BinarizeTensor",
    "BinarizeTensorByThreshold",
    "Collate",
    "CreateTensorMask",
    "CreateTensorMaskByThreshold",
    "FilterTensors",
    "GatherRows",
    "GatherScores",
    "MapTensor",
    "MultiplyTensors",
    "Scale",
    "SelectTensors",
    "Sigmoid",
    "Slice",
    "Softmax",
    "SortTensorsBy",
    "Squeeze",
    "TensorPayload",
    "TensorRegistry",
    "TopK",
    "TopKIndices2D",
    "Transpose",
]

TensorLike: TypeAlias = (
    TensorPayload
    | np.ndarray
    | tuple[TensorPayload, ...]
    | tuple[np.ndarray, ...]
    | list[TensorPayload]
    | list[np.ndarray]
)
TensorMask: TypeAlias = npt.NDArray[np.bool_]
TensorInput: TypeAlias = TensorLike | TensorRegistry
TensorValueT = TypeVar("TensorValueT", bound=TensorLike)
AsTypeModeT = TypeVar("AsTypeModeT", bound=bool)


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis if axis >= 0 else axis + ndim
    if normalized < 0 or normalized >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return normalized


def _shape_without_axis(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    return shape[:axis] + shape[axis + 1 :]


def _flatten_leading_dim(array: np.ndarray) -> np.ndarray:
    leading = int(array.shape[0])
    trailing = int(np.prod(array.shape[1:], dtype=np.int64))
    return array.reshape(leading, trailing)


def _resolve_multi_output_names(
    operator_name: str,
    srcs: tuple[str, ...],
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


@Operator
class AsType(Generic[AsTypeModeT]):
    @overload
    def __init__(
        self: "AsType[Literal[False]]",
        dtype: str,
        src: None = None,
        as_: None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self: "AsType[Literal[True]]",
        dtype: str,
        src: str,
        as_: str | None = None,
    ) -> None:
        ...

    def __init__(self, dtype: str, src: str | None = None, as_: str | None = None):
        if src is None and as_ is not None:
            raise ValueError("AsType as_ requires src.")
        self.dtype = np.dtype(dtype)
        self.src = src
        self.as_ = as_ or src

    def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, error_type):
        if self.src is not None:
            return (TensorRegistry,), TensorRegistry
        if current_output is not Any and is_assignable(current_output, TensorLike):
            return (current_output,), current_output
        return (TensorLike,), TensorLike

    @overload
    def __call__(self: "AsType[Literal[False]]", value: TensorValueT) -> TensorValueT:
        ...

    @overload
    def __call__(self: "AsType[Literal[True]]", value: TensorRegistry) -> TensorRegistry:
        ...

    def __call__(self, value: Any) -> Any:
        if self.src is not None:
            if not isinstance(value, TensorRegistry):
                raise TypeError(f"AsType src={self.src!r} requires TensorRegistry, got {type(value)!r}")
            value[self.as_] = self._cast_array(value[self.src])
            return value
        return self._cast_value(value)

    def _cast_value(self, value: TensorValueT) -> TensorValueT:
        if isinstance(value, TensorPayload):
            return cast(
                TensorValueT,
                TensorPayload(array=value.array.astype(self.dtype, copy=False), layout=value.layout, dtype=str(self.dtype)),
            )
        if isinstance(value, np.ndarray):
            return cast(TensorValueT, self._cast_array(value))
        if isinstance(value, tuple):
            return cast(TensorValueT, tuple(self._cast_sequence_item(item) for item in value))
        if isinstance(value, list):
            return cast(TensorValueT, [self._cast_sequence_item(item) for item in value])
        raise TypeError(f"AsType does not support value type {type(value)!r}")

    def _cast_sequence_item(self, value: object) -> TensorPayload | np.ndarray:
        if isinstance(value, TensorPayload):
            return TensorPayload(array=value.array.astype(self.dtype, copy=False), layout=value.layout, dtype=str(self.dtype))
        if isinstance(value, np.ndarray):
            return self._cast_array(value)
        raise TypeError(f"AsType does not support sequence item type {type(value)!r}")

    def _cast_array(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value).astype(self.dtype, copy=False)


@Operator
class Squeeze:
    def __init__(self, src: str, axis: int | tuple[int, ...] | None = None, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        tensor = registry[self.src]
        registry[self.as_] = np.squeeze(tensor, axis=self.axis) if self.axis is not None else np.squeeze(tensor)
        return registry


@Operator
class Transpose:
    def __init__(self, src: str, axes: tuple[int, ...] | None = None, as_: str | None = None):
        self.src = src
        self.axes = axes
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = np.transpose(registry[self.src], self.axes)
        return registry


@Operator
class Slice:
    def __init__(self, src: str, at: slice, as_: str | None = None):
        self.src = src
        self.at = at
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = registry[self.src][:, self.at]
        return registry


@Operator
class GatherRows:
    def __init__(self, src: str, indices: str, as_: str | None = None):
        self.src = src
        self.indices = indices
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        src = registry[self.src]
        idx = registry[self.indices]
        registry[self.as_] = src[np.arange(src.shape[0]), idx]
        return registry


GatherScores = GatherRows


@Operator
class TopK:
    def __init__(self, src: str, k: int, values_as: str = "top_values", indices_as: str = "top_indices"):
        self.src = src
        self.k = k
        self.values_as = values_as
        self.indices_as = indices_as

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        values = registry[self.src]
        if values.ndim != 1:
            raise ValueError(f"TopK expects a 1D tensor, got shape {values.shape}")
        size = int(values.shape[0])
        top_k = min(self.k, size)
        if top_k == 0:
            registry[self.values_as] = values[:0]
            registry[self.indices_as] = np.zeros((0,), dtype=np.int64)
            return registry
        top_indices = np.argpartition(values, -top_k)[-top_k:]
        order = np.argsort(values[top_indices])[::-1]
        top_indices = top_indices[order].astype(np.int64, copy=False)
        registry[self.values_as] = values[top_indices]
        registry[self.indices_as] = top_indices
        return registry


@Operator
class TopKIndices2D:
    def __init__(
        self,
        src: str,
        k: int,
        values_as: str = "top_values",
        row_indices_as: str = "row_indices",
        col_indices_as: str = "col_indices",
    ):
        self.src = src
        self.k = k
        self.values_as = values_as
        self.row_indices_as = row_indices_as
        self.col_indices_as = col_indices_as

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        values = registry[self.src]
        if values.ndim != 2:
            raise ValueError(f"TopKIndices2D expects a 2D tensor, got shape {values.shape}")
        rows, cols = values.shape
        flat = values.reshape(-1)
        top_k = min(self.k, int(flat.shape[0]))
        if top_k == 0:
            registry[self.values_as] = flat[:0]
            registry[self.row_indices_as] = np.zeros((0,), dtype=np.int64)
            registry[self.col_indices_as] = np.zeros((0,), dtype=np.int64)
            return registry
        top_indices = np.argpartition(flat, -top_k)[-top_k:]
        order = np.argsort(flat[top_indices])[::-1]
        top_indices = top_indices[order].astype(np.int64, copy=False)
        registry[self.values_as] = flat[top_indices]
        registry[self.row_indices_as] = (top_indices // cols).astype(np.int64, copy=False)
        registry[self.col_indices_as] = (top_indices % cols).astype(np.int64, copy=False)
        return registry


@Operator
class ArgMax:
    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        tensor = registry[self.src]
        axis = _normalize_axis(self.axis, tensor.ndim)
        if tensor.shape[axis] == 0:
            registry[self.as_] = np.zeros(_shape_without_axis(tensor.shape, axis), dtype=np.int32)
            return registry
        registry[self.as_] = np.argmax(tensor, axis=axis).astype(np.int32)
        return registry


@Operator
class Softmax:
    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        x = registry[self.src]
        axis = _normalize_axis(self.axis, x.ndim)
        if x.shape[axis] == 0:
            registry[self.as_] = x.copy()
            return registry
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        registry[self.as_] = exp / np.sum(exp, axis=axis, keepdims=True)
        return registry


@Operator
class Sigmoid:
    def __init__(self, src: str, as_: str | None = None):
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        x = registry[self.src]
        positive = x >= 0
        result = np.empty_like(x)
        result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_values = np.exp(x[~positive])
        result[~positive] = exp_values / (1.0 + exp_values)
        registry[self.as_] = result
        return registry


@Operator
class MultiplyTensors:
    def __init__(self, left: str, right: str, as_: str | None = None):
        self.left = left
        self.right = right
        self.as_ = as_ or left

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = registry[self.left] * registry[self.right]
        return registry


@Operator
class CreateTensorMask:
    def __init__(self, src: str, predicate: Callable[[np.ndarray], TensorMask], as_: str):
        self.src = src
        self.as_ = as_
        self.predicate = predicate

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = np.asarray(self.predicate(registry[self.src]), dtype=bool)
        return registry


BinarizeTensor = CreateTensorMask


@Operator
class CreateTensorMaskByThreshold:
    def __init__(self, src: str, threshold: float, as_: str | None = None):
        self._inner = CreateTensorMask(
            src=src,
            as_=as_ or src,
            predicate=lambda tensor: tensor >= threshold,
        )

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        return self._inner(registry)


BinarizeTensorByThreshold = CreateTensorMaskByThreshold


@Operator
class ApplyTensorMask:
    def __init__(self, *srcs: str, mask: str, as_: str | tuple[str, ...] | None = None):
        self.srcs = srcs
        self.mask = mask
        self.dst_names = _resolve_multi_output_names("ApplyTensorMask", srcs, as_)

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        mask = registry[self.mask]
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][mask]
        return registry


@Operator
class SelectTensors:
    def __init__(self, *srcs: str, indices: str, as_: str | tuple[str, ...] | None = None):
        self.srcs = srcs
        self.indices = indices
        self.dst_names = _resolve_multi_output_names("SelectTensors", srcs, as_)

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        indices = registry[self.indices]
        if np.issubdtype(indices.dtype, np.bool_):
            raise TypeError("SelectTensors indices must be integers; use ApplyTensorMask for boolean masks.")
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][indices]
        return registry


@Operator
class FilterTensors:
    def __init__(
        self,
        *srcs: str,
        by: str,
        predicate: Callable[[np.ndarray], TensorMask],
        as_: str | tuple[str, ...] | None = None,
    ):
        self.srcs = srcs
        self.by = by
        self.predicate = predicate
        self.dst_names = _resolve_multi_output_names("FilterTensors", srcs, as_)

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        mask = np.asarray(self.predicate(registry[self.by]))
        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError("FilterTensors predicate must return a boolean mask.")
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][mask]
        return registry


@Operator
class MapTensor:
    def __init__(self, src: str, fn: Callable[[np.ndarray], np.ndarray], as_: str | None = None):
        self.src = src
        self.fn = fn
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = self.fn(registry[self.src])
        return registry


@Operator
class SortTensorsBy:
    def __init__(
        self,
        *srcs: str,
        by: str,
        descending: bool = True,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (by,) + tuple(src for src in srcs if src != by)
        self.srcs = all_srcs
        self.by = by
        self.descending = descending
        self.dst_names = _resolve_multi_output_names("SortTensorsBy", all_srcs, as_)

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        order = np.argsort(registry[self.by])
        if self.descending:
            order = order[::-1]
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][order]
        return registry


@Operator
class Collate:
    def __call__(self, tensors: list[TensorPayload]) -> TensorPayload:
        if not tensors:
            raise ValueError("Collate received an empty list")
        arrays = [t.array for t in tensors]
        if arrays[0].ndim == 4 and arrays[0].shape[0] == 1:
            batched = np.concatenate(arrays, axis=0)
        else:
            batched = np.stack(arrays, axis=0)
        return TensorPayload(array=batched, layout=tensors[0].layout, dtype=tensors[0].dtype)


@Operator
class Scale:
    def __init__(self, src: str, by: float | tuple | list, as_: str | None = None):
        self.src = src
        self.by = np.asarray(by)
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        tensor = registry[self.src]
        registry[self.as_] = tensor * self.by.astype(tensor.dtype)
        return registry
