from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from ml_pipes.types import TensorPayload, TensorRegistry
from ml_pipes.validation import is_annotation_compatible

from .types import (
    TorchRuntimeOutputs,
    TorchTensorPayload,
    TorchTensorRegistry,
    canonical_torch_device,
    canonical_torch_dtype,
    resolve_torch_dtype,
)


class ToTorch:
    def __init__(self, device: str = "cpu", dtype: str | None = None):
        self.device = canonical_torch_device(device)
        self.dtype = dtype

    def __call__(self, tensor_payload: TensorPayload) -> TorchTensorPayload:
        target_dtype = resolve_torch_dtype(self.dtype) if self.dtype is not None else None
        tensor = torch.tensor(
            np.asarray(tensor_payload.array),
            dtype=target_dtype,
            device=self.device,
        )
        return TorchTensorPayload(
            array=tensor,
            layout=tensor_payload.layout,
            dtype=canonical_torch_dtype(tensor.dtype),
            device=canonical_torch_device(tensor.device),
        )


class ToNumpy:
    def __init__(self, dtype: str | None = None):
        self.dtype = dtype

    def __call__(self, tensor_payload: TorchTensorPayload) -> TensorPayload:
        array = tensor_payload.array.detach().cpu().numpy().copy()
        if self.dtype is not None:
            array = array.astype(np.dtype(self.dtype), copy=False)
        return TensorPayload(array=array, layout=tensor_payload.layout, dtype=str(array.dtype))


class ToTorchRegistry:
    def __init__(self, device: str = "cpu", dtype: str | None = None):
        self.device = canonical_torch_device(device)
        self.dtype = dtype

    def __call__(self, registry: TensorRegistry) -> TorchTensorRegistry:
        target_dtype = resolve_torch_dtype(self.dtype) if self.dtype is not None else None
        tensors = {
            name: torch.tensor(np.asarray(value), dtype=target_dtype, device=self.device)
            for name, value in registry._tensors.items()
        }
        return TorchTensorRegistry(tensors)


class ToNumpyRegistry:
    def __init__(self, dtype: str | None = None):
        self.dtype = dtype

    def __call__(self, registry: TorchTensorRegistry) -> TensorRegistry:
        arrays = {}
        for name, tensor in registry._tensors.items():
            array = tensor.detach().cpu().numpy().copy()
            if self.dtype is not None:
                array = array.astype(np.dtype(self.dtype), copy=False)
            arrays[name] = array
        return TensorRegistry(arrays)


class ToDevice:
    def __init__(self, device: str):
        self.device = canonical_torch_device(device)

    def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, error_type):
        torch_like = TorchTensorPayload | TorchTensorRegistry
        if current_output is not Any and is_annotation_compatible(current_output, (torch_like,)):
            return (current_output,), current_output
        return (torch_like,), torch_like

    def __call__(self, value: object) -> object:
        if isinstance(value, TorchTensorPayload):
            tensor = value.array.to(device=self.device)
            return TorchTensorPayload(
                array=tensor,
                layout=value.layout,
                dtype=canonical_torch_dtype(tensor.dtype),
                device=canonical_torch_device(tensor.device),
            )
        if isinstance(value, TorchTensorRegistry):
            for name, tensor in value._tensors.items():
                value[name] = tensor.to(device=self.device)
            return value
        raise TypeError(f"ToDevice does not support value type {type(value)!r}")


class TorchAsType:
    def __init__(self, dtype: str, src: str | None = None, as_: str | None = None):
        self.dtype = dtype
        self._torch_dtype = resolve_torch_dtype(dtype)
        self.src = src
        self.as_ = as_ or src

    def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, error_type):
        if self.src is not None:
            return (TorchTensorRegistry,), TorchTensorRegistry
        tensor_like = (
            TorchTensorPayload
            | torch.Tensor
            | tuple[TorchTensorPayload, ...]
            | tuple[torch.Tensor, ...]
            | list[TorchTensorPayload]
            | list[torch.Tensor]
        )
        if current_output is not Any and is_annotation_compatible(current_output, (tensor_like,)):
            return (current_output,), current_output
        return (tensor_like,), tensor_like

    def __call__(self, value: object) -> object:
        if self.src is not None:
            if not isinstance(value, TorchTensorRegistry):
                raise TypeError(
                    f"TorchAsType src={self.src!r} requires TorchTensorRegistry, got {type(value)!r}"
                )
            value[self.as_] = self._cast_tensor(value[self.src])
            return value
        return self._cast_value(value)

    def _cast_value(self, value: object) -> object:
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
        if isinstance(value, tuple):
            return tuple(self._cast_sequence_item(item) for item in value)
        if isinstance(value, list):
            return [self._cast_sequence_item(item) for item in value]
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


class TorchArgMax:
    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = torch.argmax(registry[self.src], dim=self.axis)
        return registry


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


TorchGatherScores = TorchGatherRows


class TorchSlice:
    def __init__(self, src: str, at: slice, as_: str | None = None):
        self.src = src
        self.at = at
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = registry[self.src][:, self.at]
        return registry


class TorchSoftmax:
    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = torch.softmax(registry[self.src], dim=self.axis)
        return registry


class TorchSigmoid:
    def __init__(self, src: str, as_: str | None = None):
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = torch.sigmoid(registry[self.src])
        return registry


class TorchMultiplyTensors:
    def __init__(self, left: str, right: str, as_: str | None = None):
        self.left = left
        self.right = right
        self.as_ = as_ or left

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry[self.as_] = registry[self.left] * registry[self.right]
        return registry


class TorchCreateTensorMask:
    def __init__(self, as_: str, predicate: Callable[[TorchTensorRegistry], Any]):
        self.as_ = as_
        self.predicate = predicate

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        mask = self.predicate(registry)
        registry[self.as_] = (
            mask.to(dtype=torch.bool)
            if isinstance(mask, torch.Tensor)
            else torch.as_tensor(mask, dtype=torch.bool)
        )
        return registry


class TorchBinarizeTensor:
    def __init__(self, src: str, threshold: float, as_: str | None = None):
        self._inner = TorchCreateTensorMask(
            as_=as_ or src,
            predicate=lambda registry: registry[src] >= threshold,
        )

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        return self._inner(registry)


class TorchApplyTensorMask:
    def __init__(self, *srcs: str, mask: str):
        self.srcs = srcs
        self.mask = mask

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        mask = registry[self.mask]
        for src in self.srcs:
            registry[src] = registry[src][mask]
        return registry


class TorchSelectTensors:
    def __init__(self, *srcs: str, indices: str):
        self.srcs = srcs
        self.indices = indices

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        indices = registry[self.indices]
        for src in self.srcs:
            registry[src] = registry[src][indices]
        return registry


class TorchFilterTensorsByScore:
    def __init__(self, *srcs: str, score: str, min_score: float):
        all_srcs = (score,) + tuple(src for src in srcs if src != score)
        self.srcs = all_srcs
        self.score = score
        self.min_score = min_score

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        keep = registry[self.score] >= self.min_score
        for src in self.srcs:
            registry[src] = registry[src][keep]
        return registry


class TorchFilterTensorsByMasksArea:
    def __init__(self, *srcs: str, masks: str = "masks", min_area: int = 1):
        all_srcs = (masks,) + tuple(src for src in srcs if src != masks)
        self.srcs = all_srcs
        self.masks = masks
        self.min_area = min_area

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        masks = registry[self.masks]
        areas = masks.to(dtype=torch.bool).flatten(1).sum(dim=1)
        keep = areas >= self.min_area
        for src in self.srcs:
            registry[src] = registry[src][keep]
        return registry


class TorchSortTensorsBy:
    def __init__(self, *srcs: str, by: str, descending: bool = True):
        all_srcs = (by,) + tuple(src for src in srcs if src != by)
        self.srcs = all_srcs
        self.by = by
        self.descending = descending

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        order = torch.argsort(registry[self.by], descending=self.descending)
        for src in self.srcs:
            registry[src] = registry[src][order]
        return registry


class TorchWeightMasksByScores:
    def __init__(self, masks: str = "masks", scores: str = "scores", as_: str = "weighted_masks"):
        self.masks = masks
        self.scores = scores
        self.as_ = as_

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        scores = registry[self.scores]
        masks = registry[self.masks]
        expanded_scores = scores.reshape((scores.shape[0],) + (1,) * (masks.ndim - 1))
        registry[self.as_] = expanded_scores * masks
        return registry


class TorchInfer:
    def __init__(
        self,
        module_or_callable: Callable[[torch.Tensor], Any],
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
        elif isinstance(outputs, (tuple, list)):
            output_tensors = tuple(outputs)
            if any(not isinstance(output, torch.Tensor) for output in output_tensors):
                raise TypeError(
                    "TorchInfer supports only torch.Tensor outputs or tuple/list of torch.Tensor outputs"
                )
        else:
            raise TypeError(
                "TorchInfer supports only torch.Tensor outputs or tuple/list of torch.Tensor outputs"
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
