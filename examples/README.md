# Examples

## Environment Setup

`ml-pipes` requires Python 3.10+.

Before creating a virtual environment, confirm Python is installed:

```bash
python3 --version
```

If `python3` is not found, install Python 3.10+ first and then come back to
these steps. On Windows, make sure Python is added to `PATH` during
installation.

If you are new to Python, create and activate a virtual environment from the
repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of
`source .venv/bin/activate`.

## Dependency Setup

Choose one dependency setup path after the environment is active. Start with
this base install for the bundled starter example and most ONNX + vision
examples. Extra setup for inspection, Torch, streaming, and endpoint examples
is listed under the relevant sections below.

### Local Workspace Setup

Use this when you cloned this repository and want the examples to run against
the checked-out code:

```bash
python -m pip install uv
uv sync --group shared-framework                          # most vision + ONNX examples
```

This syncs the workspace packages from this repository into the project
environment.

### Published Package Setup

Use this when you want to run the examples against released packages instead
of the repository checkout:

```bash
python -m pip install 'ml-pipes[onnx,vision]'             # most vision + ONNX examples
```

> [!NOTE]
> The commands above cover the bundled starter example and most ONNX + vision
> examples. Some examples below need extra setup, additional dependencies, or
> extra assets. Those requirements are listed under the relevant sections.
>
> For the package matrix and public import model behind these installs, see
> [docs/PACKAGES.md](../docs/PACKAGES.md).

> [!IMPORTANT]
> All commands and file paths below are shown relative to the repository root
> unless noted otherwise. If you run commands from `examples/`, adjust paths
> accordingly.

## Run The Bundled Starter Example

`run_yolo8_onnx.py` is the recommended first run. This repository already
includes the starter model and sample image in `examples/.example_assets/`, so
this example should run without downloading extra assets.

Once the environment is active and dependencies are installed, run it from the
repository root:

```bash
# Bundled starter image + bundled starter model
python examples/run_yolo8_onnx.py

# Your own image + bundled starter model
python examples/run_yolo8_onnx.py --input path/to/photo.jpg

# Your own image + your own local ONNX model
python examples/run_yolo8_onnx.py --input path/to/photo.jpg --model-path path/to/model.onnx
```

A successful default run writes the annotated image to
`examples/.example_assets/coco_000000039769_yolov8n.jpg`.

If you pass your own image or model, the default output still goes under
`examples/.example_assets/` and uses the input and model names to build the
filename.

## Inference On Files

| Example | Model | Task | Notable pipeline features |
|---|---|---|---|
| `run_detection.py` | any YOLOv8-compatible | detection | generic, bring your own model |
| `run_yolo8_onnx.py` | YOLOv8 | detection | baseline YOLO pipeline |
| `run_yolo11n_fp16.py` | YOLO11n FP16 | detection | `Cast` for FP16, letterbox resize |
| `run_rfdetr_nano.py` | DETR-style detector | detection | `Scale` for normalized boxes, softmax logits |
| `run_yolo11n_seg.py` | YOLO11n-seg | instance segmentation | prototype masks, `ReconstructMasks` + `FilterBy` |
| `run_maskrcnn.py` | Mask R-CNN int8 | instance segmentation | CNN family, NMS baked in, 28x28 RoI masks, BGR mean subtraction |
| `run_yolo8_batch.py` | YOLOv8 | batch detection | simple batch region usage |
| `run_yolo8_tile.py` | YOLOv8 | tiled detection | tile and merge style pipeline |

`run_yolo8_batch.py` also needs Ultralytics to export the dynamic-batch YOLOv8
model on first run:

```bash
python -m pip install ultralytics
```

```bash
python examples/run_yolo8_onnx.py
python examples/run_yolo11n_seg.py
python examples/run_rfdetr_nano.py
python examples/run_maskrcnn.py
python examples/run_yolo8_tile.py
```

## Inspection, Tracing, And Benchmarking

| Example | Focus | Notes |
|---|---|---|
| `run_inspect.py` | step-by-step inspection | renders a successful pipeline run |
| `run_inspect_errors.py` | failed-run inspection | shows how inspection captures errors |
| `run_yolo8_tracing.py` | tracing | prints or captures per-step trace data |
| `run_yolo8_batch_benchmark.py` | single benchmark | benchmarks batch throughput on one pipeline |
| `benchmarks/run_yolo8_benchmark.py` | benchmark workflow | direct `Benchmark` usage for one pipeline |
| `benchmarks/run_yolo8_benchmark_sweep.py` | benchmark sweep | compares plain and tiled pipelines side by side |
| `benchmarks/run_yolo8_benchmark_sweep_axis.py` | axis sweep | sweeps `slice_wh x overlap_wh` combinations |
| `benchmarks/run_yolo8_benchmark_variants.py` | variant sweep | compares multiple YOLOv8 model sizes |
| `benchmarks/run_yolo8_benchmark_cli.py` | CLI benchmark target | target for `python -m ml_pipes benchmark` |

Extra setup for the inspection entries (`run_inspect.py` and
`run_inspect_errors.py`):

```bash
# Local workspace setup
uv sync --group shared-framework --group inspection-otel

# Published package setup
python -m pip install 'ml-pipes[inspection,onnx,vision]'
```

`run_yolo8_tracing.py` and the benchmark entries use the base install above.
If you run `run_inspect.py --pipeline batched`, also install Ultralytics for
the dynamic-batch YOLOv8 export:

```bash
python -m pip install ultralytics
```

`run_inspect.py` also accepts `--output` to override the annotated image path
for the simple and tiled pipelines.

## Data Preparation

| Example | Domain | Notes |
|---|---|---|
| `run_sms_spam_prepare.py` | tabular / text preparation | non-vision example built from data operators |

## Streaming And Live Inference

| Example | Model | Task | Notes |
|---|---|---|---|
| `streaming/run_yolo8_webcam.py` | YOLOv8 | live detection | reads from the default camera; press Q to quit |
| `run_yolo8_video.py` | YOLOv8 | video detection | sequential baseline; auto-downloads OpenCV's `vtest.avi` sample |
| `streaming/run_shibuya_counter.py` | CSRNet + detector | crowd counting pipeline |
| `streaming/run_shibuya_csrnet.py` | CSRNet | density-map based crowd estimation |
| `streaming/run_shibuya_rf.py` | DETR-style detector | streaming detector variant |

Extra setup for specific streaming entries:

`streaming/run_shibuya_csrnet.py` and `streaming/run_shibuya_counter.py`
need the same Torch setup described in [Torch And Domain Handoff](#torch-and-domain-handoff).

`streaming/run_shibuya_rf.py` also needs:

```bash
python -m pip install supervision
```

```bash
# Live webcam — press Q to quit
python examples/streaming/run_yolo8_webcam.py

# Video file — uses bundled sample, or pass --input clip.mp4
python examples/run_yolo8_video.py
python examples/run_yolo8_video.py --input clip.mp4 --output annotated.mp4
```

## Torch And Domain Handoff

| Example | Focus | Notes |
|---|---|---|
| `torch/run_mask2former_torch_postprocess.py` | Torch-heavy postprocess | keeps mask postprocessing in Torch |
| `torch/run_mask2former_numpy_postprocess.py` | NumPy handoff | converts back earlier and finishes in NumPy |

Extra setup for these Torch examples:

```bash
# Local workspace setup
uv sync --group shared-framework --group torch

# Published package setup
python -m pip install 'ml-pipes[torch,vision]'

# Model-specific dependencies for the Mask2Former weights
python -m pip install transformers safetensors
```

## Inference Endpoint

| Example | Model | Task | Notes |
|---|---|---|---|
| `run_yolo8_endpoint.py` | YOLOv8 | HTTP detection endpoint | requires `pip install flask` |

Extra setup for the HTTP endpoint example:

```bash
python -m pip install flask
```

```bash
# Terminal 1 — start the server
python examples/run_yolo8_endpoint.py

# Terminal 2 — send a test request (uses bundled sample, or pass --input photo.jpg)
python examples/run_yolo8_endpoint.py --call
python examples/run_yolo8_endpoint.py --call --input photo.jpg

# Or with curl
curl -s -X POST http://localhost:5000/detect \
     -H "Content-Type: application/octet-stream" \
     --data-binary @examples/.example_assets/coco_000000039769.jpg | python -m json.tool
```
