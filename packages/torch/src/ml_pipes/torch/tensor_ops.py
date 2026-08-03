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
    "TorchAsType",
    "TorchArgMax",
    "TorchSqueeze",
    "TorchGatherRows",
    "TorchTopK",
    "TorchTopKIndices2D",
    "TorchGatherScores",
    "TorchSlice",
    "TorchSoftmax",
    "TorchSigmoid",
    "TorchMultiplyTensors",
    "TorchCreateTensorMask",
    "TorchBinarizeTensor",
    "TorchCreateTensorMaskByThreshold",
    "TorchBinarizeTensorByThreshold",
    "TorchApplyTensorMask",
    "TorchSelectTensors",
    "TorchSortTensorsBy",
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
class TorchAsType(Generic[TorchAsTypeModeT]):
    @overload
    def __init__(
        self: "TorchAsType[Literal[False]]",
        dtype: str,
        src: None = None,
        as_: None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self: "TorchAsType[Literal[True]]",
        dtype: str,
        src: str,
        as_: str | None = None,
    ) -> None:
        ...

    def __init__(self, dtype: str, src: str | None = None, as_: str | None = None):
        if src is None and as_ is not None:
            raise ValueError("TorchAsType as_ requires src.")
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
    def __call__(self: "TorchAsType[Literal[False]]", value: TorchTensorValueT) -> TorchTensorValueT:
        ...

    @overload
    def __call__(self: "TorchAsType[Literal[True]]", value: TorchTensorRegistry) -> TorchTensorRegistry:
        ...

    def __call__(self, value: Any) -> Any:
        if self.src is not None:
            if not isinstance(value, TorchTensorRegistry):
                raise TypeError(
                    f"TorchAsType src={self.src!r} requires TorchTensorRegistry, got {type(value)!r}"
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
        raise TypeError(f"TorchAsType does not support value type {type(value)!r}")

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
        raise TypeError(f"TorchAsType does not support sequence item type {type(value)!r}")

    def _cast_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype=self._torch_dtype)


@Operator
class TorchArgMax:
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
class TorchSqueeze:
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
class TorchGatherRows:
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
class TorchTopK:
    def __init__(self, src: str, k: int, values_as: str = "top_values", indices_as: str = "top_indices"):
        self.src = src
        self.k = k
        self.values_as = values_as
        self.indices_as = indices_as

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        values = registry[self.src]
        if values.ndim != 1:
            raise ValueError(f"TorchTopK expects a 1D tensor, got shape {tuple(values.shape)}")
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
class TorchTopKIndices2D:
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
            raise ValueError(f"TorchTopKIndices2D expects a 2D tensor, got shape {tuple(values.shape)}")
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


TorchGatherScores = TorchGatherRows


@Operator
class TorchSlice:
    def __init__(self, src: str, at: slice, as_: str | None = None):
        self.src = src
        self.at = at
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = registry[self.src][:, self.at]
        return registry


@Operator
class TorchSoftmax:
    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = torch.softmax(registry[self.src], dim=self.axis)
        return registry


@Operator
class TorchSigmoid:
    def __init__(self, src: str, as_: str | None = None):
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = torch.sigmoid(registry[self.src])
        return registry


@Operator
class TorchMultiplyTensors:
    def __init__(self, left: str, right: str, as_: str | None = None):
        self.left = left
        self.right = right
        self.as_ = as_ or left

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = registry[self.left] * registry[self.right]
        return registry


@Operator
class TorchCreateTensorMask:
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


TorchBinarizeTensor = TorchCreateTensorMask


@Operator
class TorchCreateTensorMaskByThreshold:
    def __init__(self, src: str, threshold: float, as_: str | None = None):
        self._inner = TorchCreateTensorMask(
            src=src,
            as_=as_ or src,
            predicate=lambda tensor: tensor >= threshold,
        )

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        return self._inner(registry)


TorchBinarizeTensorByThreshold = TorchCreateTensorMaskByThreshold


@Operator
class TorchApplyTensorMask:
    def __init__(self, *srcs: str, mask: str, as_: str | tuple[str, ...] | None = None):
        self.srcs = srcs
        self.mask = mask
        self.dst_names = resolve_multi_output_names("TorchApplyTensorMask", srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        mask = registry[self.mask]
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][mask]
        return registry


@Operator
class TorchSelectTensors:
    def __init__(self, *srcs: str, indices: str, as_: str | tuple[str, ...] | None = None):
        self.srcs = srcs
        self.indices = indices
        self.dst_names = resolve_multi_output_names("TorchSelectTensors", srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        indices = registry[self.indices]
        if indices.dtype == torch.bool:
            raise TypeError("TorchSelectTensors indices must be integers; use TorchApplyTensorMask for boolean masks.")
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][indices]
        return registry


@Operator
class TorchSortTensorsBy:
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
        self.dst_names = resolve_multi_output_names("TorchSortTensorsBy", all_srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        order = torch.argsort(registry[self.by], descending=self.descending)
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][order]
        return registry
