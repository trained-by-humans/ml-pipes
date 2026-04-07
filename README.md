# ml-pipes

Minimal YOLOv8-style ONNX Runtime inference SDK with:

- decode
- letterbox resize
- normalize
- ONNX inference
- decode predictions
- NMS
- projection to original input coordinates
- drawing bounding boxes
- saving annotated images

## Install

```bash
pip install -e .
```

## Example

```bash
python examples/run_detection.py path/to/model.onnx path/to/image.jpg
```

## Ready-To-Run Public Demo

This repo also includes a public demo that downloads:

- a YOLOv8n ONNX model from Hugging Face
- a COCO validation image from the public COCO dataset

Run it with:

```bash
python examples/run_public_demo.py
```

The first run downloads the model and image into `.example_assets/`.
It also writes an annotated image to `.example_assets/coco_000000039769_annotated.jpg`.
