from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping, Sequence

import torch

from ml_pipes.operator import Operator

from .types import (
    TorchRuntimeOutputs,
    TorchTensorPayload,
    TorchTensorRegistry,
    canonical_torch_device,
    canonical_torch_dtype,
)

__all__ = [
    "TorchInfer",
    "TorchExtract",
    "Collate",
    "TorchDistribute",
]


@Operator
class TorchInfer:
    def __init__(
        self,
        model: torch.nn.Module,
        input_name: str | None = None,
        input_layout: str = "NCHW",
        dtype: str | None = None,
        output_layouts: Sequence[str] | None = None,
        serialize: bool = False,
    ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"TorchInfer requires a torch.nn.Module, got {type(model)!r}")
        self.model = model
        self.input_name = input_name
        self.input_layout = input_layout
        self.model_dtype = dtype
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
                if self.input_name is None:
                    outputs = self.model(tensor_payload.array)
                else:
                    outputs = self.model(**{self.input_name: tensor_payload.array})

        output_names, output_tensors = self._normalize_outputs(outputs)

        if self.output_layouts is None:
            output_layouts = tuple("UNKNOWN" for _ in output_tensors)
        else:
            if len(self.output_layouts) != len(output_tensors):
                raise ValueError(
                    f"TorchInfer expected {len(self.output_layouts)} output layouts, got {len(output_tensors)} outputs"
                )
            output_layouts = self.output_layouts

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

    def _normalize_outputs(self, outputs: object) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
        if isinstance(outputs, torch.Tensor):
            return ("output_0",), (outputs,)
        if isinstance(outputs, Sequence):
            output_tensors = tuple(outputs)
            if any(not isinstance(output, torch.Tensor) for output in output_tensors):
                raise TypeError(
                    "TorchInfer supports only torch.Tensor outputs, sequences of torch.Tensor outputs, "
                    "or mappings of named torch.Tensor outputs"
                )
            return tuple(f"output_{index}" for index in range(len(output_tensors))), output_tensors
        if isinstance(outputs, Mapping):
            output_names: list[str] = []
            output_tensors: list[torch.Tensor] = []
            for name, output in outputs.items():
                if not isinstance(name, str):
                    raise TypeError(f"TorchInfer mapping output names must be strings, got {type(name)!r}")
                if not isinstance(output, torch.Tensor):
                    continue
                output_names.append(name)
                output_tensors.append(output)
            if output_tensors:
                return tuple(output_names), tuple(output_tensors)
            raise TypeError("TorchInfer mapping outputs must contain at least one torch.Tensor value")
        raise TypeError(
            "TorchInfer supports only torch.Tensor outputs, sequences of torch.Tensor outputs, "
            "or mappings of named torch.Tensor outputs"
        )


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
class Collate:
    def __call__(self, tensors: list[TorchTensorPayload]) -> TorchTensorPayload:
        if not tensors:
            raise ValueError("Collate received an empty list")
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
