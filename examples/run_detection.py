from __future__ import annotations

import json
import sys

from ml_pipes import (
    ArgMax,
    ConvertBoxFormat,
    DecodeOp,
    GatherScores,
    InferOp,
    NMS,
    NormalizeOp,
    Pick,
    Pipeline,
    Project,
    Recall,
    ResizeOp,
    Select,
    Slice,
    Squeeze,
    Store,
    ToDetections,
    Transpose,
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
            Store("resize_transform", index=1),
            Pick(0),
            NormalizeOp(),
            InferOp(model_path),
            Select("output0", as_="preds"),
            Squeeze("preds"),
            Transpose("preds"),
            Slice("preds", slice(None, 4), as_="boxes"),
            Slice("preds", slice(4, None), as_="scores"),
            ArgMax("scores", as_="classes"),
            GatherScores("scores", "classes"),
            ConvertBoxFormat("boxes", from_="cxcywh", to="xyxy"),
            NMS(),
            Recall("resize_transform"),
            Project(),
            ToDetections(),
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
