from __future__ import annotations

import numpy as np

from ml_pipes.inspection import PipelineInspector, TextBlock
from ml_pipes.onnx import RuntimeOutputs
from ml_pipes.tensor import TensorPayload


def test_pipeline_inspector_formats_runtime_outputs_from_onnx_package() -> None:
    outputs = RuntimeOutputs(
        tensors=(
            TensorPayload(
                array=np.zeros((1, 3, 2, 2), dtype=np.float32),
                layout="NCHW",
                dtype="float32",
            ),
        ),
        names=("scores",),
    )

    blocks = PipelineInspector()._value_to_blocks(outputs)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "RuntimeOutputs"
    assert blocks[0].rows == [("scores", "(1, 3, 2, 2)")]
