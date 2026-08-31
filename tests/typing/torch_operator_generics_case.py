from __future__ import annotations

from typing import cast

import torch

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from ml_pipes.tensor import TensorPayload as NumpyTensorPayload
from ml_pipes.tensor import TensorRegistry as NumpyTensorRegistry
from ml_pipes.torch import (
    AsType,
    Distribute,
    Extract,
    Infer,
    RuntimeOutputs,
    SynchronizeTensors,
    TensorPayload,
    TensorRegistry,
    ToDevice,
    ToNumpy,
    ToNumpyRegistry,
    ToTorch,
    ToTorchRegistry,
)


sample_payload = cast(TensorPayload, None)
sample_registry = cast(TensorRegistry, None)
sample_outputs = cast(RuntimeOutputs, None)
sample_tensor = cast(torch.Tensor, None)
sample_numpy_payload = cast(NumpyTensorPayload, None)
sample_numpy_registry = cast(NumpyTensorRegistry, None)
sample_model = cast(torch.nn.Module, None)

assert_type(ToDevice("cpu")(sample_payload), TensorPayload)
assert_type(ToDevice("cpu")(sample_registry), TensorRegistry)
assert_type(ToDevice("cpu")(sample_outputs), RuntimeOutputs)
assert_type(ToDevice("cpu")(sample_tensor), torch.Tensor)
assert_type(ToDevice("cpu")([sample_tensor]), list[torch.Tensor])

assert_type(SynchronizeTensors()(sample_payload), TensorPayload)
assert_type(SynchronizeTensors()(sample_outputs), RuntimeOutputs)

assert_type(ToTorch()(sample_numpy_payload), TensorPayload)
assert_type(ToNumpy()(sample_payload), NumpyTensorPayload)
assert_type(ToTorchRegistry()(sample_numpy_registry), TensorRegistry)
assert_type(ToNumpyRegistry()(sample_registry), NumpyTensorRegistry)
assert_type(Infer(sample_model)(sample_payload), RuntimeOutputs)
assert_type(Extract("output_0")(sample_outputs), TensorRegistry)
assert_type(Distribute()(sample_outputs), list[RuntimeOutputs])

assert_type(AsType("float16")(sample_payload), TensorPayload)
assert_type(AsType("float16")(sample_tensor), torch.Tensor)
assert_type(AsType("float16")([sample_tensor]), list[torch.Tensor])
assert_type(AsType("float16", src="scores")(sample_registry), TensorRegistry)
