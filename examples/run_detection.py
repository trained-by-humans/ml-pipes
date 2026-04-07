from __future__ import annotations

import json
import sys

from ml_pipes import (
    DecodeOp,
    DecodePredictionsOp,
    InferOp,
    NMSOp,
    NormalizeOp,
    Pipeline,
    ProjectToInputOp,
    ResizeOp,
)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python examples/run_detection.py path/to/model.onnx path/to/image.jpg")
        return 1

    model_path, image_path = sys.argv[1], sys.argv[2]
    pipeline = Pipeline(
        [
            DecodeOp(),
            ResizeOp((640, 640)),
            NormalizeOp(),
            InferOp(model_path),
            DecodePredictionsOp(),
            NMSOp(),
            ProjectToInputOp(),
        ]
    )
    result = pipeline(image_path)
    print(
        json.dumps(
            {
                "boxes": result.data.boxes,
                "scores": result.data.scores,
                "classes": result.data.classes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
