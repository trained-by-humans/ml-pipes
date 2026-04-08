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
    Recall,
    ResizeOp,
    Select,
    Store,
)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python examples/run_detection.py path/to/model.onnx path/to/image.jpg")
        return 1

    model_path, image_path = sys.argv[1], sys.argv[2]
    infer = InferOp(model_path)
    normalize = NormalizeOp()
    decode_predictions = DecodePredictionsOp()
    nms = NMSOp()
    pipeline = Pipeline(
        [
            DecodeOp(),
            ResizeOp((640, 640)),
            Store("resize_transform", index=1),
            Select(0),
            normalize,
            infer,
            decode_predictions,
            nms,
            Recall("resize_transform"),
            ProjectToInputOp(),
        ]
    )

    result = pipeline(image_path)
    print(
        json.dumps(
            {
                "boxes": result.boxes,
                "scores": result.scores,
                "classes": result.classes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
