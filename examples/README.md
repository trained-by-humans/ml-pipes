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

Choose one dependency setup path after the environment is active.

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
> For the package matrix and public import model behind these installs, see
> [docs/PACKAGES.md](../docs/PACKAGES.md).

## Bundled Starter Example

`run_yolo8_onnx.py` is the recommended first run. This repository already
includes the starter model and sample image in `examples/.example_assets/`, so
this example should run without downloading extra assets.

Once the environment is active and dependencies are installed, run it from the
repository root:

```bash
# Published package setup
python -m pip install 'ml-pipes[onnx,vision]'

# Local workspace setup
python -m pip install uv
uv sync --group shared-framework

# Run the bundled starter example
python examples/run_yolo8_onnx.py
```

A successful default run writes the annotated image to
`examples/.example_assets/coco_000000039769_yolov8n.jpg`.

## Running Examples

Once setup is done, you can run each example as a Python script.
Some examples need extra setup, and their sections call that out.

All commands and file paths below are shown relative to the repository root
unless noted otherwise. If you run commands from `examples/`, adjust paths
accordingly.

> [!NOTE]
> The built-in example assets always resolve under
> `examples/.example_assets`, regardless of your current working directory.
> Relative paths that you pass to flags such as `--input`, `--output`, or
> `--model-path` are still resolved from the directory where you run the
> command.

Start by running the example with its default arguments:

```bash
# Start with the default run
python examples/run_yolo8_tile.py
```

If you want to customize the run afterward, use CLI flags to override the
default model, input, output, or runtime behavior:

```bash
# Then override only the parts you want to change
python examples/run_yolo8_tile.py --input path/to/photo.jpg --slice-wh 320 320 --overlap-wh 80 80
```

### Common CLI Args

Many examples reuse the same small set of file and model selection flags:

| Arg | Common in | Meaning |
|---|---|---|
| `--model-path path/to/model.onnx` | file-based inference, selected streaming examples | Use your own local model instead of the bundled or downloaded default model. |
| `--input path/to/input-file` | file-based inference, endpoint calls | Use your own image or video instead of the bundled sample input. |
| `--output path/to/output-file` | file-based inference, inspection, video | Write the result to an explicit location instead of the example's default output path. |

Examples:

```bash
# Your own image + bundled starter model
python examples/run_yolo8_onnx.py --input path/to/photo.jpg

# Your own image + your own local ONNX model
python examples/run_yolo8_onnx.py --input path/to/photo.jpg --model-path path/to/model.onnx
```

> [!TIP]
> The exact flags vary by script, so check `python path/to/example.py --help`
> when in doubt. The tables below highlight the most useful args for each
> example, not the full CLI surface.

## File-Based Inference

| Example | Model | Task | Key args | Notable pipeline features |
|---|---|---|---|---|
| `run_yolo8_onnx.py` | YOLOv8 | detection | common args only | baseline YOLO pipeline |
| `run_yolo11n_fp16.py` | YOLO11n FP16 | detection | common args only | `Cast` for FP16, letterbox resize |
| `run_rfdetr_nano.py` | DETR-style detector | detection | common args only | `Scale` for normalized boxes, softmax logits |
| `run_yolo11n_seg.py` | YOLO11n-seg | instance segmentation | common args only | prototype masks, `ReconstructMasks` + `FilterBy` |
| `run_maskrcnn.py` | Mask R-CNN int8 | instance segmentation | common args only | CNN family, NMS baked in, 28x28 RoI masks, BGR mean subtraction |
| `run_yolo8_video.py` | YOLOv8 | video detection | common args only | sequential video-file baseline |
| `run_yolo8_batch.py` | YOLOv8 | batch detection | `--images`, `--batch-size`, `--workers` | simple batch region usage |
| `run_yolo8_tile.py` | YOLOv8 | tiled detection | `--conf-threshold`, `--slice-wh`, `--overlap-wh`, `--max-concurrency` | tile and merge style pipeline |

`run_yolo8_batch.py` also needs Ultralytics to export the dynamic-batch YOLOv8
model on first run:

```bash
python -m pip install ultralytics
```

Example command:

```bash
python examples/run_yolo8_tile.py --input path/to/photo.jpg --slice-wh 320 320 --overlap-wh 80 80
```

## Inspection And Tracing

| Example | Focus | Key args | Notes |
|---|---|---|---|
| `run_inspect.py` | step-by-step inspection | `--pipeline`, `--save-html`, `--plot`, `--dump`, `--load`, `--print-only` | renders successful runs and a synthetic failure case |
| `run_yolo8_tracing.py` | tracing | `--runs` | prints or captures per-step trace data |

Extra setup for the inspection entry (`run_inspect.py`):

```bash
# Local workspace setup
uv sync --group shared-framework --group inspection-otel

# Published package setup
python -m pip install 'ml-pipes[inspection,onnx,vision]'
```

Example command:

```bash
python examples/run_inspect.py --pipeline tiled --save-html report.html
```

## Benchmarking

| Example | Focus | Key args | Notes |
|---|---|---|---|
| `benchmarks/run_yolo8_benchmark.py` | benchmark workflow | common args only | direct `Benchmark` usage for one pipeline |
| `benchmarks/run_yolo8_benchmark_sweep.py` | benchmark sweep | common args only | compares one plain baseline against a small tiled `slice_wh` sweep |
| `benchmarks/run_yolo8_benchmark_variants.py` | variant sweep | `--variants` | compares multiple YOLOv8 model sizes |
| `benchmarks/run_yolo8_benchmark_cli.py` | CLI benchmark target | `--arg slice_wh=...`, `--arg model_path=...` | target for `python -m ml_pipes benchmark` |

Common benchmark args for the script-based examples:

| Arg | Common in | Meaning |
|---|---|---|
| `--runs N` | script-based benchmarks | Control how many repeated runs a benchmark executes. |
| `--warmup N` | script-based benchmarks | Discard warmup iterations before measurement. |
| `--save path/to/results-dir` | script-based benchmarks | Save benchmark result files under the given directory. |

The benchmark entries use the base install above.

Example command:

```bash
python examples/benchmarks/run_yolo8_benchmark_sweep.py --runs 20 --save results/
```

## Data Preparation

| Example | Domain | Key args | Notes |
|---|---|---|---|
| `run_sms_spam_prepare.py` | tabular / text preparation | `--output-dir`, `--inspect-html`, `--lazy` | non-vision example built from data operators |

Example command:

```bash
python examples/run_sms_spam_prepare.py --inspect-html report.html
```

## Streaming And Live Inference

| Example | Model | Task | Key args | Notes |
|---|---|---|---|---|
| `streaming/run_yolo8_webcam.py` | YOLOv8 | live detection | `--model-path` | reads from the default camera; press Q to quit |
| `streaming/run_shibuya_counter.py` | YOLOv8 | detection-based crowd counting | `--tile`, `--conf-threshold` | default YouTube source needs `yt-dlp`; direct stream URLs do not |
| `streaming/run_shibuya_csrnet.py` | CSRNet | density-map based crowd estimation | `--weights`, `--device` | default YouTube source needs `yt-dlp`; also requires `torch` |

Common args for the two Shibuya stream examples:

| Arg | Common in | Meaning |
|---|---|---|
| `--url <stream-url>` | Shibuya stream examples | Pass a YouTube page URL or a direct playable stream URL. |
| `--workers N` | Shibuya stream examples | Control how many inference workers run in parallel. |
| `--stride N` | Shibuya stream examples | Process every `N`th frame instead of every frame. |
| `--target-fps N` | Shibuya stream examples | Set the fallback FPS used when the stream source does not report one. |

Extra setup for specific streaming entries:

The Shibuya stream examples use a default YouTube page URL. Install `yt-dlp`
to resolve that default, or pass a direct stream URL with `--url`:

```bash
python -m pip install yt-dlp
```

`streaming/run_shibuya_csrnet.py` also requires Torch:

```bash
python -m pip install torch
```

Example command:

```bash
python examples/streaming/run_yolo8_webcam.py
```

## Torch Domain Handoff

| Example | Focus | Key args | Notes |
|---|---|---|---|
| `torch/run_mask2former_torch_postprocess.py` | Torch-heavy postprocess | common args only | keeps mask postprocessing in Torch |
| `torch/run_mask2former_numpy_postprocess.py` | NumPy handoff | common args only | converts back earlier and finishes in NumPy |

Common args for the two Mask2Former examples:

| Arg | Common in | Meaning |
|---|---|---|
| `--task {instance,panoptic}` | Mask2Former examples | Choose which Mask2Former task variant to run. |
| `--device <torch-device>` | Mask2Former examples | Select the Torch device for model execution. |
| `--input path/to/input-file` | Mask2Former examples | Use your own image instead of the sample COCO image. |
| `--output path/to/output-file` | Mask2Former examples | Set the output path prefix for the annotated result image. |

Extra setup for these Torch examples:

```bash
# Local workspace setup
uv sync --group shared-framework --group torch

# Published package setup
python -m pip install 'ml-pipes[torch,vision]'

# Model-specific dependencies for the Mask2Former weights
python -m pip install transformers safetensors scipy
```

Example command:

```bash
python examples/torch/run_mask2former_torch_postprocess.py
```

## RESTful Inference Endpoint

| Example | Model | Task | Key args | Notes |
|---|---|---|---|---|
| `run_yolo8_endpoint.py` | YOLOv8 | HTTP detection endpoint | `--call`, `--input`, `--model-path` | requires `pip install flask` |

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
