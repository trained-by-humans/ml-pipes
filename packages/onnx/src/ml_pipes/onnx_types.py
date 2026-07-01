from __future__ import annotations

from dataclasses import dataclass

from .tensor_types import TensorPayload

__all__ = [
    "RuntimeOutputs",
]


@dataclass(frozen=True)
class RuntimeOutputs:
    # Runtime-facing output tensors exactly as exposed by the exported graph.
    tensors: tuple[TensorPayload, ...]
    names: tuple[str, ...]
