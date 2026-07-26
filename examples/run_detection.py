from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipes.core import Pipeline
from ml_pipes.onnx import (
    Infer,
    Extract,
)
from ml_pipes.standard import (
    Pick,
    Recall,
    Store,
)
from ml_pipes.tensor import (
    ArgMax,
    GatherScores,
    Slice,
    Squeeze,
    Transpose,
)
from ml_pipes.vision import (
    ConvertBoxFormat,
    Decode,
    Detections,
    LoadFile,
    NMS,
    Normalize,
    ProjectBoxes,
    Resize,
    ToDetections,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal YOLOv8-compatible ONNX detection pipeline on a local image.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a local YOLOv8-compatible ONNX model.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image path.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: model file not found: {args.model_path}")
        return 1
    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        return 1

    pipeline: Pipeline[str | Path, Detections] = Pipeline(
        [
            LoadFile(),
            Decode(),
            Resize((640, 640)),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(),
            Infer(args.model_path),
            Extract("output0", as_="preds"),
            Squeeze("preds"),
            Transpose("preds"),
            Slice("preds", slice(None, 4), as_="boxes"),
            Slice("preds", slice(4, None), as_="scores"),
            ArgMax("scores", as_="classes"),
            GatherScores("scores", "classes"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(),
            Recall("resize_transform"),
            ProjectBoxes(),
            ToDetections(),
        ],
        auto_validate=True,
    )

    result = pipeline(args.input)
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
