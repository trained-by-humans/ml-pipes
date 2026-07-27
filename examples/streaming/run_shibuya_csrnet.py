from __future__ import annotations

import argparse
import collections
import sys
import urllib.error
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.streaming"

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from ..common import ASSETS_DIR, download_if_missing
from .stream_common import FrameReader, add_streaming_args, resolve_stream_source
from ml_pipes.collectors import ThroughputCollector
from ml_pipes.core import Pipeline
from ml_pipes.onnx import (
    Extract,
    RuntimeOutputs,
)
from ml_pipes.standard import (
    Pick,
    Recall,
    Store,
)
from ml_pipes.tensor import (
    AsType,
    Squeeze,
    TensorPayload,
)
from ml_pipes.vision import (
    BlendImages,
    ClampDensity,
    DensityToHeatmap,
    ImagePayload,
    Normalize,
    SumDensity,
    ToDensityPrediction,
)

CSRNET_MODEL_NAME = "csrnet_shanghaitech_b_rootstrap.pth"
CSRNET_MODEL_URL = "https://huggingface.co/rootstrap-org/crowd-counting/resolve/main/weights.pth"

_IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
_IMAGENET_STD_RGB = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# CSRNet architecture
# ---------------------------------------------------------------------------

def build_csrnet_model() -> Any:
    def make_layers(cfg: list[int | str], in_channels: int = 3, dilation: bool = False) -> Any:
        rate = 2 if dilation else 1
        layers: list[Any] = []
        for value in cfg:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            layers.extend((
                nn.Conv2d(in_channels, int(value), kernel_size=3, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
            ))
            in_channels = int(value)
        return nn.Sequential(*layers)

    class CSRNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frontend = make_layers([64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512])
            self.backend = make_layers([512, 512, 512, 256, 128, 64], in_channels=512, dilation=True)
            self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, x: Any) -> Any:
            x = self.frontend(x)
            x = self.backend(x)
            return self.output_layer(x)

    return CSRNet()


# ---------------------------------------------------------------------------
# Torch/runtime integration
# ---------------------------------------------------------------------------

def unwrap_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    for key in ("state_dict", "model_state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and nested:
            return {
                (name[len("module."):] if name.startswith("module.") else name): value
                for name, value in nested.items()
            }
    if checkpoint:
        return {
            (name[len("module."):] if name.startswith("module.") else name): value
            for name, value in checkpoint.items()
        }
    raise ValueError("Checkpoint did not contain any weights")


def resolve_weights_path(weights_path: Path | None) -> Path:
    if weights_path is not None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"CSRNet weights not found: {weights_path}")
        return weights_path
    resolved = ASSETS_DIR / CSRNET_MODEL_NAME
    try:
        download_if_missing(CSRNET_MODEL_URL, resolved)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            "Failed to download CSRNet weights automatically. Pass --weights with a local checkpoint path."
        ) from exc
    return resolved


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_model(weights_path: Path, device: str) -> Any:
    model = build_csrnet_model()
    checkpoint = torch.load(str(weights_path), map_location=device)
    state_dict = unwrap_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"CSRNet weights mismatch. Missing={sorted(missing)} Unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model


class CSRNetInfer:
    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device

    def __call__(self, tensor_payload: TensorPayload) -> RuntimeOutputs:
        if tensor_payload.layout != "NCHW":
            raise ValueError(f"CSRNetInfer expects NCHW tensor layout, got {tensor_payload.layout}")
        tensor = torch.from_numpy(np.ascontiguousarray(tensor_payload.array)).to(self.device)
        with torch.inference_mode():
            density = self.model(tensor).cpu().numpy()
        return RuntimeOutputs(
            tensors=(TensorPayload(array=density, layout="NCHW", dtype=str(density.dtype)),),
            names=("output0",),
        )


def build_pipeline(
    model: Any,
    device: str,
) -> Pipeline[ImagePayload, tuple[ImagePayload, float]]:
    return Pipeline([
        Store("source_frame"),
        Normalize(
            scale=1.0 / 255.0,
            mean=_IMAGENET_MEAN_RGB,
            std=_IMAGENET_STD_RGB,
            output_layout="NCHW",
            output_color_space="RGB",
        ),
        CSRNetInfer(model, device),
        Extract("output0", as_="density"),
        Squeeze("density", axis=(0, 1)),
        AsType(src="density", dtype="float32"),
        ToDensityPrediction("density"),
        ClampDensity(),
        Store("density_prediction"),
        SumDensity(),
        Store("count"),
        Recall("density_prediction", prepend=True),
        Pick(0),
        Recall("source_frame", prepend=True),
        DensityToHeatmap(),
        BlendImages(),
        Recall("count"),
    ], auto_validate=True)


def run_stream(
    url: str,
    pipeline: Pipeline[ImagePayload, tuple[ImagePayload, float]],
    target_fps: float,
    workers: int,
    stride: int,
    device: str,
) -> int:
    print(f"Resolving stream source from {url} ...", file=sys.stderr)

    try:
        stream_source = resolve_stream_source(url)
        reader = FrameReader(stream_source, fallback_fps=target_fps, stride=stride)
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    throughput = ThroughputCollector(target_fps=target_fps, report_interval_s=1.0)
    pipeline.set_tracing(throughput)
    pending: collections.deque[Future[tuple[ImagePayload, float]]] = collections.deque()
    stopped = False
    throughput.target_fps = reader.stream_fps
    status = f"device={device}"
    window_title = "Shibuya Crossing - CSRNet"

    print(
        f"Streaming with {workers} worker(s), stride={stride}, {status} "
        "— press Q in the window to quit.",
        file=sys.stderr,
    )

    def infer(frame: np.ndarray) -> tuple[ImagePayload, float]:
        return pipeline(ImagePayload(array=frame))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stopped = True

            while not stopped and len(pending) < workers:
                ok, frame = reader.latest()
                if not ok or frame is None:
                    stopped = True
                    break
                pending.append(pool.submit(infer, frame))

            if not pending:
                break

            future = pending[0]
            try:
                annotated, _ = future.result(timeout=0.05)
                pending.popleft()
                cv2.imshow(window_title, annotated.array)
            except TimeoutError:
                pass

    reader.stop()
    cv2.destroyAllWindows()
    throughput.flush()
    throughput.print_summary()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Shibuya crossing with CSRNet crowd counting.")
    add_streaming_args(parser)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Fallback FPS when the stream does not expose one.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional local CSRNet checkpoint. Defaults to a cached download in the assets directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch device used for inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch is None or nn is None:
        print("Torch is required: python -m pip install torch", file=sys.stderr)
        return 1

    try:
        resolved_weights = resolve_weights_path(args.weights)
        resolved_device = choose_device(args.device)
        model = load_model(resolved_weights, resolved_device)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    pipeline = build_pipeline(model, resolved_device)
    pipeline.validate()
    pipeline.describe()

    return run_stream(
        url=args.url,
        pipeline=pipeline,
        target_fps=args.target_fps,
        workers=args.workers,
        stride=args.stride,
        device=resolved_device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
