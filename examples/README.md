# Examples

If you only want to explore the runnable examples, install the smallest stack
that matches the example family you want to run:

```bash
pip install 'ml-pipes[onnx,vision]'              # most vision + ONNX examples
pip install 'ml-pipes[inspection,onnx,vision]'   # inspection examples
pip install 'ml-pipes[torch,vision]'             # torch examples
pip install flask                                # add for the HTTP endpoint example
```

Then run the example directly from this directory:

```bash
# From the repository root
python examples/run_yolo8_onnx.py

# Or from the examples directory
cd examples
python run_yolo8_onnx.py
```

The commands below are shown from the repository root. If you prefer to work
from `examples/`, drop the leading `examples/` prefix from script paths and
use `.example_assets/` instead of `examples/.example_assets/`.

Most self-contained examples download the models and sample assets they need
into the shared `.example_assets/` cache next to this directory on demand.
Generic entry points such as `run_detection.py` do not download defaults;
provide the model path and input path explicitly.

For the package matrix and the component import model behind these install
profiles, see [docs/PACKAGES.md](../docs/PACKAGES.md).

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

## Inference Endpoint

| Example | Model | Task | Notes |
|---|---|---|---|
| `run_yolo8_endpoint.py` | YOLOv8 | HTTP detection endpoint | requires `pip install flask` |

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
