from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Collection, Sequence
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import torch

from ml_pipes._typing.annotation import is_assignable
from ml_pipes.operator import Operator
from ml_pipes.tensor import (
    TensorPayload,
    TensorRegistry,
)
from .types import (
    TorchRuntimeOutputs,
    TorchTensorPayload,
    TorchTensorRegistry,
    canonical_torch_device,
    canonical_torch_dtype,
    resolve_torch_dtype,
)

TorchTensorLike: TypeAlias = (
    TorchTensorPayload
    | torch.Tensor
    | tuple[TorchTensorPayload, ...]
    | tuple[torch.Tensor, ...]
    | list[TorchTensorPayload]
    | list[torch.Tensor]
)
TorchTensorInput: TypeAlias = TorchTensorLike | TorchTensorRegistry
TorchTransferInput: TypeAlias = TorchTensorInput | TorchRuntimeOutputs
TorchTensorMask: TypeAlias = torch.Tensor | npt.NDArray[np.bool_]
TorchInferOutput: TypeAlias = torch.Tensor | Sequence[torch.Tensor]
TorchTensorValueT = TypeVar("TorchTensorValueT", bound=TorchTensorLike)
TorchAsTypeModeT = TypeVar("TorchAsTypeModeT", bound=bool)
TorchTransferInputT = TypeVar("TorchTransferInputT", bound=TorchTransferInput)


def _synchronize_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _collect_torch_devices(value: object) -> set[torch.device]:
    if isinstance(value, TorchTensorPayload):
        return {value.array.device}
    if isinstance(value, TorchTensorRegistry):
        return {tensor.device for tensor in value._tensors.values()}
    if isinstance(value, TorchRuntimeOutputs):
        return {tensor.array.device for tensor in value.tensors}
    if isinstance(value, torch.Tensor):
        return {value.device}
    if isinstance(value, tuple | list):
        devices: set[torch.device] = set()
        for item in value:
            devices.update(_collect_torch_devices(item))
        return devices
    return set()


def _torch_conversion_can_alias_numpy_source(
    device: str,
    source_dtype: torch.dtype,
    target_dtype: torch.dtype | None,
) -> bool:
    return device == "cpu" and (target_dtype is None or source_dtype == target_dtype)


def _numpy_conversion_can_alias_torch_source(
    source_device_type: str,
    source_dtype: np.dtype,
    target_dtype: np.dtype | None,
) -> bool:
    return source_device_type == "cpu" and (target_dtype is None or source_dtype == target_dtype)


def _convert_numpy_array_to_torch(
    source: np.ndarray,
    *,
    device: str,
    target_dtype: torch.dtype | None,
    copy: bool,
) -> torch.Tensor:
    array = np.asarray(source)
    source_dtype = torch.as_tensor(array).dtype
    conversion_can_alias_source = _torch_conversion_can_alias_numpy_source(
        device=device,
        source_dtype=source_dtype,
        target_dtype=target_dtype,
    )
    tensor = torch.as_tensor(array, dtype=target_dtype, device=device)
    detached_from_source = not conversion_can_alias_source
    if copy and not detached_from_source:
        tensor = tensor.clone()
    return tensor


def _convert_torch_tensor_to_numpy(
    source: torch.Tensor,
    *,
    target_dtype: np.dtype | None,
    copy: bool,
) -> np.ndarray:
    base_array = source.detach().cpu().numpy()
    conversion_can_alias_source = _numpy_conversion_can_alias_torch_source(
        source_device_type=source.device.type,
        source_dtype=base_array.dtype,
        target_dtype=target_dtype,
    )
    array = base_array
    if target_dtype is not None:
        array = array.astype(target_dtype, copy=False)
    detached_from_source = not conversion_can_alias_source or array is not base_array
    if copy and not detached_from_source:
        array = array.copy()
    return array


@Operator
class ToTorch:
    def __init__(self, device: str = "cpu", dtype: str | None = None, copy: bool = False):
        self.device = canonical_torch_device(device)
        self.dtype = dtype
        self.copy = copy

    def __call__(self, tensor_payload: TensorPayload) -> TorchTensorPayload:
        target_dtype = resolve_torch_dtype(self.dtype) if self.dtype is not None else None
        tensor = _convert_numpy_array_to_torch(
            tensor_payload.array,
            device=self.device,
            target_dtype=target_dtype,
            copy=self.copy,
        )
        return TorchTensorPayload(
            array=tensor,
            layout=tensor_payload.layout,
            dtype=canonical_torch_dtype(tensor.dtype),
            device=canonical_torch_device(tensor.device),
        )


@Operator
class ToNumpy:
    def __init__(self, dtype: str | None = None, copy: bool = False):
        self.dtype = dtype
        self.copy = copy

    def __call__(self, tensor_payload: TorchTensorPayload) -> TensorPayload:
        target_dtype = np.dtype(self.dtype) if self.dtype is not None else None
        array = _convert_torch_tensor_to_numpy(
            tensor_payload.array,
            target_dtype=target_dtype,
            copy=self.copy,
        )
        return TensorPayload(array=array, layout=tensor_payload.layout, dtype=str(array.dtype))


@Operator
class ToTorchRegistry:
    def __init__(self, device: str = "cpu", dtype: str | None = None, copy: bool = False):
        self.device = canonical_torch_device(device)
        self.dtype = dtype
        self.copy = copy

    def __call__(self, registry: TensorRegistry) -> TorchTensorRegistry:
        target_dtype = resolve_torch_dtype(self.dtype) if self.dtype is not None else None
        tensors = {}
        for name, value in registry._tensors.items():
            tensors[name] = _convert_numpy_array_to_torch(
                value,
                device=self.device,
                target_dtype=target_dtype,
                copy=self.copy,
            )
        return TorchTensorRegistry(tensors)


@Operator
class ToNumpyRegistry:
    def __init__(self, dtype: str | None = None, copy: bool = False):
        self.dtype = dtype
        self.copy = copy

    def __call__(self, registry: TorchTensorRegistry) -> TensorRegistry:
        arrays = {}
        for name, tensor in registry._tensors.items():
            target_dtype = np.dtype(self.dtype) if self.dtype is not None else None
            arrays[name] = _convert_torch_tensor_to_numpy(
                tensor,
                target_dtype=target_dtype,
                copy=self.copy,
            )
        return TensorRegistry(arrays)


@Operator
class ToDevice:
    def __init__(self, device: str):
        self.device = canonical_torch_device(device)

    def resolve_contract(self, upstream_annotation, error_type):
        if upstream_annotation is not Any and is_assignable(
            upstream_annotation,
            TorchTransferInput,
        ):
            return (upstream_annotation,), upstream_annotation
        return (TorchTransferInput,), TorchTransferInput

    def __call__(self, value: TorchTransferInputT) -> TorchTransferInputT:
        return cast(TorchTransferInputT, self._move_value(value))

    def _move_value(self, value: TorchTransferInput) -> TorchTransferInput:
        if isinstance(value, TorchTensorPayload):
            return self._move_payload(value)
        if isinstance(value, TorchTensorRegistry):
            for name, tensor in value._tensors.items():
                value[name] = tensor.to(device=self.device)
            return value
        if isinstance(value, TorchRuntimeOutputs):
            return TorchRuntimeOutputs(
                tensors=tuple(self._move_payload(tensor) for tensor in value.tensors),
                names=value.names,
            )
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device)
        if isinstance(value, tuple):
            return cast(TorchTensorLike, tuple(self._move_sequence_item(item) for item in value))
        if isinstance(value, list):
            return cast(TorchTensorLike, [self._move_sequence_item(item) for item in value])
        raise TypeError(f"ToDevice does not support value type {type(value)!r}")

    def _move_sequence_item(self, value: object) -> TorchTensorPayload | torch.Tensor:
        if isinstance(value, TorchTensorPayload):
            return self._move_payload(value)
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device)
        raise TypeError(f"ToDevice does not support sequence item type {type(value)!r}")

    def _move_payload(self, value: TorchTensorPayload) -> TorchTensorPayload:
        tensor = value.array.to(device=self.device)
        return TorchTensorPayload(
            array=tensor,
            layout=value.layout,
            dtype=canonical_torch_dtype(tensor.dtype),
            device=canonical_torch_device(tensor.device),
        )


@Operator
class TorchSynchronizeTensors:
    def resolve_contract(self, upstream_annotation, error_type):
        if upstream_annotation is not Any and is_assignable(
            upstream_annotation,
            TorchTransferInput,
        ):
            return (upstream_annotation,), upstream_annotation
        return (TorchTransferInput,), TorchTransferInput

    def __call__(self, value: TorchTransferInputT) -> TorchTransferInputT:
        devices = _collect_torch_devices(value)
        if not devices:
            raise TypeError(f"TorchSynchronizeTensors does not support value type {type(value)!r}")
        for device in sorted(devices, key=str):
            _synchronize_torch_device(device)
        return value


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
            shape = tensor.shape[:axis] + tensor.shape[axis + 1:]
            registry[self.as_] = torch.zeros(shape, dtype=torch.int64, device=tensor.device)
            return registry
        registry[self.as_] = torch.argmax(tensor, dim=axis)
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


def _resolve_multi_output_names(
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


@Operator
class TorchApplyTensorMask:
    def __init__(self, *srcs: str, mask: str, as_: str | tuple[str, ...] | None = None):
        self.srcs = srcs
        self.mask = mask
        self.dst_names = _resolve_multi_output_names("TorchApplyTensorMask", srcs, as_)

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
        self.dst_names = _resolve_multi_output_names("TorchSelectTensors", srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        indices = registry[self.indices]
        if indices.dtype == torch.bool:
            raise TypeError("TorchSelectTensors indices must be integers; use TorchApplyTensorMask for boolean masks.")
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][indices]
        return registry


@Operator
class TorchFilterTensorsByScore:
    def __init__(
        self,
        *srcs: str,
        score: str,
        min_score: float,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (score,) + tuple(src for src in srcs if src != score)
        self.srcs = all_srcs
        self.score = score
        self.min_score = min_score
        self.dst_names = _resolve_multi_output_names("TorchFilterTensorsByScore", all_srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        keep = registry[self.score] >= self.min_score
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][keep]
        return registry


@Operator
class TorchFilterTensorsByClasses:
    """Filters tensors by class id membership."""

    def __init__(
        self,
        *srcs: str,
        classes: str = "classes",
        keep_classes: Collection[int],
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (classes,) + tuple(src for src in srcs if src != classes)
        self.srcs = all_srcs
        self.classes = classes
        self.keep_classes = tuple(keep_classes)
        self.dst_names = _resolve_multi_output_names("TorchFilterTensorsByClasses", all_srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        allowed = torch.as_tensor(
            self.keep_classes,
            device=registry[self.classes].device,
            dtype=registry[self.classes].dtype,
        )
        keep = torch.isin(registry[self.classes], allowed)
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][keep]
        return registry


@Operator
class TorchFilterTensorsByMasksArea:
    def __init__(
        self,
        *srcs: str,
        masks: str = "masks",
        min_area: int = 1,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (masks,) + tuple(src for src in srcs if src != masks)
        self.srcs = all_srcs
        self.masks = masks
        self.min_area = min_area
        self.dst_names = _resolve_multi_output_names("TorchFilterTensorsByMasksArea", all_srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        masks = registry[self.masks]
        areas = masks.to(dtype=torch.bool).flatten(1).sum(dim=1)
        keep = areas >= self.min_area
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][keep]
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
        self.dst_names = _resolve_multi_output_names("TorchSortTensorsBy", all_srcs, as_)

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        order = torch.argsort(registry[self.by], descending=self.descending)
        for src, dst in zip(self.srcs, self.dst_names, strict=True):
            registry[dst] = registry[src][order]
        return registry


@Operator
class TorchWeightMasksByScores:
    def __init__(self, masks: str = "masks", scores: str = "scores", *, as_: str):
        self.masks = masks
        self.scores = scores
        self.as_ = as_

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        scores = registry[self.scores]
        masks = registry[self.masks]
        expanded_scores = scores.reshape((scores.shape[0],) + (1,) * (masks.ndim - 1))
        registry[self.as_] = expanded_scores * masks
        return registry


@Operator
class TorchResizeMasks:
    """Resizes a stack of masks to a target shape."""

    def __init__(self, masks: str = "masks", as_: str | None = None):
        self.masks = masks
        self.as_ = as_ or masks

    def __call__(self, registry: TorchTensorRegistry, image_shape: tuple[int, int]) -> TorchTensorRegistry:
        masks = registry[self.masks]
        resized = torch.nn.functional.interpolate(
            masks[:, None, :, :],
            size=image_shape,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        registry[self.as_] = resized
        return registry


@Operator
class TorchMeanMaskScores:
    """Computes one mean score per mask from dense mask values.

    If `binary_masks` is provided, the mean is computed only over pixels where
    the binary mask is True. If `binary_masks` is None, the mean is computed
    over all pixels in each dense mask.
    """

    def __init__(
        self,
        masks: str = "masks",
        binary_masks: str | None = "binary_masks",
        *,
        as_: str,
    ):
        self.masks = masks
        self.binary_masks = binary_masks
        self.as_ = as_

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        masks = registry[self.masks]
        if self.binary_masks is None:
            registry[self.as_] = masks.flatten(1).mean(dim=1)
            return registry

        binary_masks = registry[self.binary_masks]
        areas = binary_masks.flatten(1).sum(dim=1)
        registry[self.as_] = torch.where(
            areas > 0,
            (masks * binary_masks).flatten(1).sum(dim=1) / areas.clamp_min(1).to(masks.dtype),
            torch.zeros((masks.shape[0],), dtype=masks.dtype, device=masks.device),
        )
        return registry


@Operator
class TorchMasksToBoxes:
    def __init__(self, masks: str = "masks", *, as_: str):
        self.masks = masks
        self.as_ = as_

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        masks = registry[self.masks]
        count = masks.shape[0]
        if count == 0:
            registry[self.as_] = torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
            return registry

        _, height, width = masks.shape
        xs = torch.arange(width, dtype=torch.float32, device=masks.device).view(1, 1, width)
        ys = torch.arange(height, dtype=torch.float32, device=masks.device).view(1, height, 1)
        x1 = torch.where(masks, xs, float(width)).amin(dim=(-2, -1))
        y1 = torch.where(masks, ys, float(height)).amin(dim=(-2, -1))
        x2 = torch.where(masks, xs, -1.0).amax(dim=(-2, -1)) + 1.0
        y2 = torch.where(masks, ys, -1.0).amax(dim=(-2, -1)) + 1.0
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        empty = ~masks.any(dim=(-2, -1))
        boxes[empty] = 0.0
        registry[self.as_] = boxes
        return registry


@Operator
class TorchInfer:
    def __init__(
        self,
        module_or_callable: Callable[[torch.Tensor], TorchInferOutput],
        input_layout: str = "NCHW",
        dtype: str | None = None,
        output_names: Sequence[str] | None = None,
        output_layouts: Sequence[str] | None = None,
        serialize: bool = False,
    ):
        if not callable(module_or_callable):
            raise TypeError("TorchInfer requires a callable module or function")
        self.module_or_callable = module_or_callable
        self.input_layout = input_layout
        self.model_dtype = dtype
        self.output_names = tuple(output_names) if output_names is not None else None
        self.output_layouts = tuple(output_layouts) if output_layouts is not None else None
        self._lock = threading.Lock() if serialize else contextlib.nullcontext()

    def __call__(self, tensor_payload: TorchTensorPayload) -> TorchRuntimeOutputs:
        if tensor_payload.layout != self.input_layout:
            raise ValueError(
                f"TorchInfer expects {self.input_layout} tensor layout, got {tensor_payload.layout}"
            )
        if self.model_dtype is not None and tensor_payload.dtype != self.model_dtype:
            raise ValueError(
                f"TorchInfer expects model dtype {self.model_dtype}, got {tensor_payload.dtype}"
            )

        with torch.inference_mode():
            with self._lock:
                outputs = self.module_or_callable(tensor_payload.array)

        if isinstance(outputs, torch.Tensor):
            output_tensors = (outputs,)
        elif isinstance(outputs, Sequence):
            output_tensors = tuple(outputs)
            if any(not isinstance(output, torch.Tensor) for output in output_tensors):
                raise TypeError(
                    "TorchInfer supports only torch.Tensor outputs or sequences of torch.Tensor outputs"
                )
        else:
            raise TypeError(
                "TorchInfer supports only torch.Tensor outputs or sequences of torch.Tensor outputs"
            )

        if self.output_layouts is None:
            output_layouts = tuple("UNKNOWN" for _ in output_tensors)
        else:
            if len(self.output_layouts) != len(output_tensors):
                raise ValueError(
                    f"TorchInfer expected {len(self.output_layouts)} output layouts, got {len(output_tensors)} outputs"
                )
            output_layouts = self.output_layouts

        if self.output_names is None:
            output_names = tuple(f"output_{index}" for index in range(len(output_tensors)))
        else:
            if len(self.output_names) != len(output_tensors):
                raise ValueError(
                    f"TorchInfer expected {len(self.output_names)} output names, got {len(output_tensors)} outputs"
                )
            output_names = self.output_names

        tensors = tuple(
            TorchTensorPayload(
                array=output,
                layout=layout,
                dtype=canonical_torch_dtype(output.dtype),
                device=canonical_torch_device(output.device),
            )
            for output, layout in zip(output_tensors, output_layouts, strict=True)
        )
        return TorchRuntimeOutputs(tensors=tensors, names=output_names)


@Operator
class TorchExtract:
    def __init__(self, *names: str, as_: str | tuple[str, ...] | None = None):
        if not names:
            raise ValueError("TorchExtract requires at least one output name")
        if as_ is not None:
            aliases: tuple[str, ...] = (as_,) if isinstance(as_, str) else tuple(as_)
            if len(aliases) != len(names):
                raise ValueError(
                    f"TorchExtract: as_ length ({len(aliases)}) must match names length ({len(names)})"
                )
        else:
            aliases = names
        self._mapping: dict[str, str] = dict(zip(names, aliases))

    def __call__(self, outputs: TorchRuntimeOutputs) -> TorchTensorRegistry:
        registry = TorchTensorRegistry()
        for src, dst in self._mapping.items():
            if src not in outputs.names:
                raise KeyError(
                    f"TorchExtract: output {src!r} not found. Available: {list(outputs.names)}"
                )
            idx = list(outputs.names).index(src)
            registry[dst] = outputs.tensors[idx].array
        return registry


@Operator
class TorchCollate:
    def __call__(self, tensors: list[TorchTensorPayload]) -> TorchTensorPayload:
        if not tensors:
            raise ValueError("TorchCollate received an empty list")
        arrays = [tensor.array for tensor in tensors]
        if arrays[0].ndim == 4 and arrays[0].shape[0] == 1:
            batched = torch.cat(arrays, dim=0)
        else:
            batched = torch.stack(arrays, dim=0)
        return TorchTensorPayload(
            array=batched,
            layout=tensors[0].layout,
            dtype=canonical_torch_dtype(batched.dtype),
            device=canonical_torch_device(batched.device),
        )


@Operator
class TorchDistribute:
    def __call__(self, outputs: TorchRuntimeOutputs) -> list[TorchRuntimeOutputs]:
        n = outputs.tensors[0].array.shape[0]
        result = []
        for i in range(n):
            sample_tensors = tuple(
                TorchTensorPayload(
                    array=t.array[i : i + 1].clone(),
                    layout=t.layout,
                    dtype=t.dtype,
                    device=t.device,
                )
                for t in outputs.tensors
            )
            result.append(TorchRuntimeOutputs(tensors=sample_tensors, names=outputs.names))
        return result


@Operator
class TorchNMS:
    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        kept_as: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.kept_as = kept_as
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        boxes = registry[self.boxes]
        scores = registry[self.scores]
        classes = registry[self.classes]

        conf_mask = scores >= self.conf_threshold
        filtered_boxes = boxes[conf_mask]
        filtered_scores = scores[conf_mask]
        filtered_classes = classes[conf_mask]
        original_indices = torch.nonzero(conf_mask, as_tuple=False).squeeze(1)

        if filtered_boxes.numel() == 0:
            kept_original = torch.zeros((0,), dtype=torch.int64, device=boxes.device)
        else:
            kept_filtered = self._nms_indices(filtered_boxes, filtered_scores, filtered_classes)
            kept_original = original_indices[kept_filtered]

        registry[self.boxes] = boxes[kept_original]
        registry[self.scores] = scores[kept_original]
        registry[self.classes] = classes[kept_original]
        if self.kept_as is not None:
            registry[self.kept_as] = kept_original.to(dtype=torch.int64)
        return registry

    def _nms_indices(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
    ) -> torch.Tensor:
        from torchvision.ops import nms

        kept_groups: list[torch.Tensor] = []
        for class_id in torch.unique(classes):
            class_indices = torch.nonzero(classes == class_id, as_tuple=False).squeeze(1)
            if class_indices.numel() == 0:
                continue
            kept_groups.append(class_indices[nms(boxes[class_indices], scores[class_indices], self.iou_threshold)])

        if not kept_groups:
            return torch.zeros((0,), dtype=torch.int64, device=boxes.device)

        kept = torch.cat(kept_groups)
        ordered = kept[torch.argsort(scores[kept], descending=True)]
        return ordered[: self.max_detections].to(dtype=torch.int64)
