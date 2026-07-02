from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import numpy as np

from .onnx_types import RuntimeOutputs
from .operator import Operator
from .tensor_types import TensorPayload, TensorRegistry

__all__ = [
    "Distribute",
    "Extract",
    "Infer",
    "RuntimeOutputs",
]


@Operator
class Infer:
    def __init__(
        self,
        model_path: str | Path,
        providers: tuple[str, ...] = ("CoreMLExecutionProvider", "CPUExecutionProvider"),
        input_name: str | None = None,
        input_layout: str = "NCHW",
        dtype: str | None = None,
        output_layouts: tuple[str, ...] | None = None,
        serialize: bool = False,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")

        import onnxruntime as ort

        self.model_path = path
        self.session = ort.InferenceSession(str(path), providers=list(providers))
        self._lock = threading.Lock() if serialize else contextlib.nullcontext()
        self.input_name = input_name or self.session.get_inputs()[0].name
        self.input_layout = input_layout
        self.model_dtype = np.dtype(dtype) if dtype is not None else None
        self.output_layouts = output_layouts
        self.output_names = tuple(output.name for output in self.session.get_outputs())

    def __call__(self, tensor_payload: TensorPayload) -> RuntimeOutputs:
        if tensor_payload.layout != self.input_layout:
            raise ValueError(
                f"Infer expects {self.input_layout} tensor layout, got {tensor_payload.layout}"
            )

        actual_dtype = np.dtype(tensor_payload.dtype)
        if self.model_dtype is not None and actual_dtype != self.model_dtype:
            raise ValueError(f"Infer expects model dtype {self.model_dtype}, got {actual_dtype}")

        with self._lock:
            outputs = self.session.run(None, {self.input_name: tensor_payload.array})

        if self.output_layouts is None:
            output_layouts = tuple("UNKNOWN" for _ in outputs)
        else:
            if len(self.output_layouts) != len(outputs):
                raise ValueError(
                    f"Infer expected {len(self.output_layouts)} output layouts, got {len(outputs)} outputs"
                )
            output_layouts = self.output_layouts

        runtime_output_names = self.output_names or tuple(f"output_{index}" for index in range(len(outputs)))
        tensors = tuple(
            TensorPayload(array=np.asarray(output), layout=layout, dtype=str(np.asarray(output).dtype))
            for output, layout in zip(outputs, output_layouts, strict=True)
        )
        return RuntimeOutputs(tensors=tensors, names=runtime_output_names)


@Operator
class Extract:
    """Extracts named tensors from RuntimeOutputs into a TensorRegistry."""

    def __init__(self, *names: str, as_: str | tuple[str, ...] | None = None):
        if not names:
            raise ValueError("Extract requires at least one output name")
        if as_ is not None:
            aliases: tuple[str, ...] = (as_,) if isinstance(as_, str) else tuple(as_)
            if len(aliases) != len(names):
                raise ValueError(
                    f"Extract: as_ length ({len(aliases)}) must match names length ({len(names)})"
                )
        else:
            aliases = names
        self._mapping: dict[str, str] = dict(zip(names, aliases))

    def __call__(self, outputs: RuntimeOutputs) -> TensorRegistry:
        registry = TensorRegistry()
        for src, dst in self._mapping.items():
            if src not in outputs.names:
                raise KeyError(
                    f"Extract: output {src!r} not found. Available: {list(outputs.names)}"
                )
            idx = list(outputs.names).index(src)
            registry[dst] = outputs.tensors[idx].array
        return registry


@Operator
class Distribute:
    """Split a batched RuntimeOutputs back into a list of per-sample outputs."""

    def __call__(self, outputs: RuntimeOutputs) -> list[RuntimeOutputs]:
        n = outputs.tensors[0].array.shape[0]
        result = []
        for i in range(n):
            sample_tensors = tuple(
                TensorPayload(
                    array=t.array[i : i + 1].copy(),
                    layout=t.layout,
                    dtype=t.dtype,
                )
                for t in outputs.tensors
            )
            result.append(RuntimeOutputs(tensors=sample_tensors, names=outputs.names))
        return result
