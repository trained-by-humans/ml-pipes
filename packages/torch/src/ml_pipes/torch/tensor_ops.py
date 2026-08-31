from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import torch

from ml_pipes._operator_utils import resolve_multi_output_names
from ml_pipes._typing.annotation import is_assignable
from ml_pipes.operator import Operator
from .types import (
    TorchTensorPayload,
    TorchTensorRegistry,
    canonical_torch_device,
    canonical_torch_dtype,
    resolve_torch_dtype,
)

__all__ = [
    "AsType",
    "ArgMax",
    "Squeeze",
    "Transpose",
    "GatherRows",
    "TopK",
    "TopKIndices2D",
    "GatherScores",
    "Slice",
    "Softmax",
    "Sigmoid",
    "MultiplyTensors",
    "CreateTensorMask",
    "BinarizeTensor",
    "CreateTensorMaskByThreshold",
    "BinarizeTensorByThreshold",
    "ApplyTensorMask",
    "SelectTensors",
    "FilterTensors",
    "MapTensor",
    "SortTensorsBy",
    "Scale",
]

TorchTensorLike: TypeAlias = (
    TorchTensorPayload
    | torch.Tensor
    | tuple[TorchTensorPayload, ...]
    | tuple[torch.Tensor, ...]
    | list[TorchTensorPayload]
    | list[torch.Tensor]
)
TorchTensorMask: TypeAlias = torch.Tensor | npt.NDArray[np.bool_]
TorchTensorValueT = TypeVar("TorchTensorValueT", bound=TorchTensorLike)
TorchAsTypeModeT = TypeVar("TorchAsTypeModeT", bound=bool)


def _normalize_torch_axis(axis: int, ndim: int) -> int:
    normalized = axis if axis >= 0 else axis + ndim
    if normalized < 0 or normalized >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], got {axis})"
        )
    return normalized


def _normalize_torch_axes(axis: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, int) else axis
    normalized = tuple(_normalize_torch_axis(item, ndim) for item in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate axis in squeeze: {axis!r}")
    return normalized


@Operator
class AsType(Generic[TorchAsTypeModeT]):
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
        self.dtype = dtype
        self._torch_dtype = resolve_torch_dtype(dtype)
        self.src = src
        self.as_ = as_ or src

    def resolve_contract(self, upstream_annotation, error_type):
        if self.src is not None:
            return (TorchTensorRegistry,), TorchTensorRegistry
        if upstream_annotation is not Any and is_assignable(
            upstream_annotation,
            TorchTensorLike,
        ):
            return (upstream_annotation,), upstream_annotation
        return (TorchTensorLike,), TorchTensorLike

    @overload
    def __call__(self: "AsType[Literal[False]]", value: TorchTensorValueT) -> TorchTensorValueT:
        ...

    @overload
    def __call__(self: "AsType[Literal[True]]", value: TorchTensorRegistry) -> TorchTensorRegistry:
        ...

    def __call__(self, value: Any) -> Any:
        if self.src is not None:
            if not isinstance(value, TorchTensorRegistry):
                raise TypeError(
                    f"AsType src={self.src!r} requires TorchTensorRegistry, got {type(value)!r}"
                )
            value[self.as_] = self._cast_tensor(value[self.src])
            return value
        return self._cast_value(value)

    def _cast_value(self, value: TorchTensorValueT) -> TorchTensorValueT:
        if isinstance(value, TorchTensorPayload):
            tensor = self._cast_tensor(value.array)
            return cast(
                TorchTensorValueT,
                TorchTensorPayload(
                    array=tensor,
                    layout=value.layout,
                    dtype=canonical_torch_dtype(tensor.dtype),
                    device=canonical_torch_device(tensor.device),
                ),
            )
        if isinstance(value, torch.Tensor):
            return cast(TorchTensorValueT, self._cast_tensor(value))
        if isinstance(value, tuple):
            return cast(TorchTensorValueT, tuple(self._cast_sequence_item(item) for item in value))
        if isinstance(value, list):
            return cast(TorchTensorValueT, [self._cast_sequence_item(item) for item in value])
        raise TypeError(f"AsType does not support value type {type(value)!r}")

    def _cast_sequence_item(self, value: object) -> TorchTensorPayload | torch.Tensor:
        if isinstance(value, TorchTensorPayload):
            tensor = self._cast_tensor(value.array)
            return TorchTensorPayload(
                array=tensor,
                layout=value.layout,
                dtype=canonical_torch_dtype(tensor.dtype),
                device=canonical_torch_device(tensor.device),
            )
        if isinstance(value, torch.Tensor):
            return self._cast_tensor(value)
        raise TypeError(f"AsType does not support sequence item type {type(value)!r}")

    def _cast_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype=self._torch_dtype)


@Operator
class ArgMax:
    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        tensor = registry[self.src]
        axis = self.axis if self.axis >= 0 else self.axis + tensor.ndim
        if axis < 0 or axis >= tensor.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-tensor.ndim}, {tensor.ndim - 1}], "
                f"but got {self.axis})"
            )
        if tensor.shape[axis] == 0:
            shape = tensor.shape[:axis] + tensor.shape[axis + 1 :]
            registry[self.as_] = torch.zeros(shape, dtype=torch.int64, device=tensor.device)
            return registry
        registry[self.as_] = torch.argmax(tensor, dim=axis)
        return registry


@Operator
class Squeeze:
    def __init__(self, src: str, axis: int | tuple[int, ...] | None = None, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        tensor = registry[self.src]
        if self.axis is None:
            registry[self.as_] = torch.squeeze(tensor)
            return registry

        axes = sorted(_normalize_torch_axes(self.axis, tensor.ndim), reverse=True)
        squeezed = tensor
        for axis in axes:
            if squeezed.shape[axis] != 1:
                raise ValueError(
                    f"cannot squeeze axis {axis} with size {squeezed.shape[axis]} for tensor {self.src!r}"
                )
            squeezed = torch.squeeze(squeezed, dim=axis)
        registry[self.as_] = squeezed
        return registry


@Operator
class Transpose:
    def __init__(self, src: str, axes: tuple[int, ...] | None = None, as_: str | None = None):
        self.src = src
        self.axes = axes
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        tensor = registry[self.src]
        dims = self.axes if self.axes is not None else tuple(range(tensor.ndim - 1, -1, -1))
        registry[self.as_] = torch.permute(tensor, dims)
        return registry


@Operator
class GatherRows:
    def __init__(self, src: str, indices: str, as_: str | None = None):
        self.src = src
        self.indices = indices
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        src = registry[self.src]
        idx = registry[self.indices]
        registry[self.as_] = src[torch.arange(src.shape[0], device=src.device), idx]
        return registry


@Operator
class TopK:
    def __init__(self, src: str, k: int, values_as: str = "top_values", indices_as: str = "top_indices"):
        self.src = src
        self.k = k
        self.values_as = values_as
        self.indices_as = indices_as

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        values = registry[self.src]
        if values.ndim != 1:
            raise ValueError(f"TopK expects a 1D tensor, got shape {tuple(values.shape)}")
        top_k = min(self.k, int(values.numel()))
        if top_k == 0:
            registry[self.values_as] = values[:0]
            registry[self.indices_as] = torch.zeros((0,), dtype=torch.int64, device=values.device)
            return registry
        top_values, top_indices = torch.topk(values, k=top_k)
        registry[self.values_as] = top_values
        registry[self.indices_as] = top_indices.to(torch.int64)
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

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        values = registry[self.src]
        if values.ndim != 2:
            raise ValueError(f"TopKIndices2D expects a 2D tensor, got shape {tuple(values.shape)}")
        _, cols = values.shape
        flat = values.reshape(-1)
        top_k = min(self.k, int(flat.numel()))
        if top_k == 0:
            registry[self.values_as] = flat[:0]
            empty = torch.zeros((0,), dtype=torch.int64, device=values.device)
            registry[self.row_indices_as] = empty
            registry[self.col_indices_as] = empty
            return registry
        top_values, top_indices = torch.topk(flat, k=top_k)
        top_indices = top_indices.to(torch.int64)
        registry[self.values_as] = top_values
        registry[self.row_indices_as] = torch.div(top_indices, cols, rounding_mode="floor")
        registry[self.col_indices_as] = top_indices % cols
        return registry


GatherScores = GatherRows


@Operator
class Slice:
    def __init__(self, src: str, at: slice, as_: str | None = None):
        self.src = src
        self.at = at
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = registry[self.src][:, self.at]
        return registry


@Operator
class Softmax:
    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = torch.softmax(registry[self.src], dim=self.axis)
        return registry


@Operator
class Sigmoid:
    def __init__(self, src: str, as_: str | None = None):
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = torch.sigmoid(registry[self.src])
        return registry


@Operator
class MultiplyTensors:
    def __init__(self, left: str, right: str, as_: str | None = None):
        self.left = left
        self.right = right
        self.as_ = as_ or left

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = registry[self.left] * registry[self.right]
        return registry


@Operator
class CreateTensorMask:
    def __init__(self, src: str, predicate: Callable[[torch.Tensor], TorchTensorMask], as_: str):
        self.src = src
        self.as_ = as_
        self.predicate = predicate

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        src = registry[self.src]
        mask = self.predicate(src)
        registry[self.as_] = (
            mask.to(device=src.device, dtype=torch.bool)
            if isinstance(mask, torch.Tensor)
            else torch.as_tensor(mask, dtype=torch.bool, device=src.device)
        )
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

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        return self._inner(registry)


BinarizeTensorByThreshold = CreateTensorMaskByThreshold


@Operator
class ApplyTensorMask:
    def __init__(self, *srcs: str, mask: str, as_: str | tuple[str, ...] | None = None):
        self.srcs = srcs
        self.mask = mask
        self.dst_names = resolve_multi_output_names("ApplyTensorMask", srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        mask = registry[self.mask]
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][mask]
        return registry


@Operator
class SelectTensors:
    def __init__(self, *srcs: str, indices: str, as_: str | tuple[str, ...] | None = None):
        self.srcs = srcs
        self.indices = indices
        self.dst_names = resolve_multi_output_names("SelectTensors", srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        indices = registry[self.indices]
        if indices.dtype == torch.bool:
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
        predicate: Callable[[torch.Tensor], TorchTensorMask],
        as_: str | tuple[str, ...] | None = None,
    ):
        self.srcs = srcs
        self.by = by
        self.predicate = predicate
        self.dst_names = resolve_multi_output_names("FilterTensors", srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        source = registry[self.by]
        raw_mask = self.predicate(source)
        if isinstance(raw_mask, torch.Tensor):
            if raw_mask.dtype != torch.bool:
                raise TypeError("FilterTensors predicate must return a boolean mask.")
            mask = raw_mask.to(device=source.device)
        else:
            mask_array = np.asarray(raw_mask)
            if not np.issubdtype(mask_array.dtype, np.bool_):
                raise TypeError("FilterTensors predicate must return a boolean mask.")
            mask = torch.as_tensor(mask_array, dtype=torch.bool, device=source.device)
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][mask]
        return registry


@Operator
class MapTensor:
    def __init__(self, src: str, fn: Callable[[torch.Tensor], torch.Tensor], as_: str | None = None):
        self.src = src
        self.fn = fn
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
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
        self.dst_names = resolve_multi_output_names("SortTensorsBy", all_srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        order = torch.argsort(registry[self.by], descending=self.descending)
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][order]
        return registry


@Operator
class Scale:
    def __init__(self, src: str, by: float | tuple | list, as_: str | None = None):
        self.src = src
        self.by = torch.as_tensor(by)
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        tensor = registry[self.src]
        registry[self.as_] = tensor * self.by.to(device=tensor.device, dtype=tensor.dtype)
        return registry
