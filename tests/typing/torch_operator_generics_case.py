from __future__ import annotations

from typing import cast

import torch

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from ml_pipes.torch import ToDevice, TorchAsType, TorchSynchronizeTensors
from ml_pipes.torch.types import TorchRuntimeOutputs, TorchTensorPayload, TorchTensorRegistry


sample_payload = cast(TorchTensorPayload, None)
sample_registry = cast(TorchTensorRegistry, None)
sample_outputs = cast(TorchRuntimeOutputs, None)
sample_tensor = cast(torch.Tensor, None)

assert_type(ToDevice("cpu")(sample_payload), TorchTensorPayload)
assert_type(ToDevice("cpu")(sample_registry), TorchTensorRegistry)
assert_type(ToDevice("cpu")(sample_outputs), TorchRuntimeOutputs)
assert_type(ToDevice("cpu")(sample_tensor), torch.Tensor)
assert_type(ToDevice("cpu")([sample_tensor]), list[torch.Tensor])

assert_type(TorchSynchronizeTensors()(sample_payload), TorchTensorPayload)
assert_type(TorchSynchronizeTensors()(sample_outputs), TorchRuntimeOutputs)

assert_type(TorchAsType("float16")(sample_payload), TorchTensorPayload)
assert_type(TorchAsType("float16")(sample_tensor), torch.Tensor)
assert_type(TorchAsType("float16")([sample_tensor]), list[torch.Tensor])
assert_type(TorchAsType("float16", src="scores")(sample_registry), TorchTensorRegistry)
