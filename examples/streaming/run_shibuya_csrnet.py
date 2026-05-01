from __future__ import annotations

import argparse
import collections
import sys
import urllib.error
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.streaming"

import cv2
import numpy as np

from ..common import add_assets_dir_arg, download_if_missing
from .stream_common import FrameReader, add_streaming_args, get_stream_url
from ml_pipes import ImagePayload, Normalize, Pick, Pipeline, Recall, Store, TensorPayload, ThroughputCollector

CSRNET_MODEL_NAME = "csrnet_shanghaitech_b_rootstrap.pth"
CSRNET_MODEL_URL = "https://huggingface.co/rootstrap-org/crowd-counting/resolve/main/weights.pth"

_IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
_IMAGENET_STD_RGB = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Generic density/CNN result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityPrediction:
    density_map: np.ndarray


# ---------------------------------------------------------------------------
# CSRNet architecture
# ---------------------------------------------------------------------------

def build_csrnet_model() -> Any:
    _, nn = _import_torch()

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

def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "CSRNet example requires torch. Install it in the project environment before running this script."
        ) from exc
    return torch, nn


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


def resolve_weights_path(assets_dir: Path, weights_path: Path | None) -> Path:
    if weights_path is not None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"CSRNet weights not found: {weights_path}")
        return weights_path
    resolved = assets_dir / CSRNET_MODEL_NAME
    try:
        download_if_missing(CSRNET_MODEL_URL, resolved)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            "Failed to download CSRNet weights automatically. Pass --weights with a local checkpoint path."
        ) from exc
    return resolved


def choose_device(requested: str) -> str:
    torch, _ = _import_torch()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_model(weights_path: Path, device: str) -> tuple[Any, Any]:
    torch, _ = _import_torch()
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
    return model, torch


class CSRNetInfer:
    def __init__(self, model: Any, torch: Any, device: str) -> None:
        self.model = model
        self.torch = torch
        self.device = device

    def __call__(self, tensor_payload: TensorPayload) -> DensityPrediction:
        tensor = self.torch.from_numpy(np.ascontiguousarray(tensor_payload.array)).to(self.device)
        with self.torch.inference_mode():
            density = self.model(tensor).squeeze(0).squeeze(0).detach().cpu().numpy()
        return DensityPrediction(
            density_map=density.astype(np.float32, copy=False),
        )


# ---------------------------------------------------------------------------
# Generic density-map postprocess operators
# ---------------------------------------------------------------------------

class ClampDensity:
    def __call__(self, prediction: DensityPrediction) -> DensityPrediction:
        return DensityPrediction(
            density_map=np.maximum(prediction.density_map.astype(np.float32, copy=False), 0.0),
        )


class SumDensity:
    def __call__(self, prediction: DensityPrediction) -> float:
        return float(prediction.density_map.sum())


# ---------------------------------------------------------------------------
# Generic heatmap/overlay operators
# ---------------------------------------------------------------------------

class DensityToHeatmap:
    def __init__(
        self,
        colormap: int = cv2.COLORMAP_TURBO,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> None:
        self.colormap = colormap
        self.interpolation = interpolation

    def __call__(self, source_image: ImagePayload, prediction: DensityPrediction) -> tuple[ImagePayload, ImagePayload]:
        height, width = source_image.array.shape[:2]
        density = np.maximum(prediction.density_map.astype(np.float32, copy=False), 0.0)
        if density.size == 0 or float(density.max()) <= 0.0:
            heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            normalized = cv2.normalize(density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(normalized.astype(np.uint8), self.colormap)
            heatmap = cv2.resize(heatmap, (width, height), interpolation=self.interpolation)
        return source_image, ImagePayload(array=heatmap, color_space="BGR", layout="HWC")


class BlendImages:
    def __init__(self, base_weight: float = 0.60, overlay_weight: float = 0.40) -> None:
        self.base_weight = base_weight
        self.overlay_weight = overlay_weight

    def __call__(self, source_image: ImagePayload, overlay_image: ImagePayload) -> ImagePayload:
        if source_image.layout != "HWC" or overlay_image.layout != "HWC":
            raise ValueError("BlendImages expects HWC images")
        if source_image.array.shape != overlay_image.array.shape:
            raise ValueError(
                f"BlendImages requires matching shapes, got {source_image.array.shape} and {overlay_image.array.shape}"
            )
        blended = cv2.addWeighted(
            source_image.array,
            self.base_weight,
            overlay_image.array,
            self.overlay_weight,
            0.0,
        )
        return ImagePayload(
            array=blended,
            color_space=source_image.color_space,
            layout=source_image.layout,
        )


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------

def build_preprocess_pipeline() -> Pipeline:
    return Pipeline([
        Store("source_frame"),
        Normalize(
            scale=1.0 / 255.0,
            mean=_IMAGENET_MEAN_RGB,
            std=_IMAGENET_STD_RGB,
            output_layout="NCHW",
            output_color_space="RGB",
        ),
    ])


def build_postprocess_pipeline() -> Pipeline:
    return Pipeline([
        ClampDensity(),
        Store("density_prediction"),
        SumDensity(),
        Store("count"),
        Recall("density_prediction", index=0),
        Pick(0),
        Recall("source_frame", index=0),
        DensityToHeatmap(),
        BlendImages(),
        Recall("count"),
    ])


def build_frame_pipeline(infer_op: Any) -> Pipeline:
    pipeline = build_preprocess_pipeline() + Pipeline([infer_op]) + build_postprocess_pipeline()
    pipeline.validate()
    return pipeline


def run(
    url: str,
    assets_dir: Path,
    target_fps: float,
    workers: int,
    stride: int,
    weights: Path | None,
    device: str,
) -> int:
    try:
        resolved_weights = resolve_weights_path(assets_dir, weights)
        resolved_device = choose_device(device)
        model, torch = load_model(resolved_weights, resolved_device)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Resolving stream URL from {url} ...", file=sys.stderr)
    stream_url = get_stream_url(url)

    try:
        reader = FrameReader(stream_url, fallback_fps=target_fps, stride=stride)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    throughput = ThroughputCollector(target_fps=target_fps, report_interval_s=1.0)
    pending: collections.deque[Future] = collections.deque()
    stopped = False
    frame_pipeline = build_frame_pipeline(CSRNetInfer(model, torch, resolved_device))
    frame_pipeline.set_tracing(throughput)

    throughput.target_fps = reader.stream_fps

    print(
        f"Streaming with {workers} worker(s), stride={stride}, device={resolved_device} "
        "— press Q in the window to quit.",
        file=sys.stderr,
    )

    def infer(frame: Any) -> tuple[ImagePayload, float]:
        return frame_pipeline(ImagePayload(array=frame, color_space="BGR", layout="HWC"))

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
                annotated, count = future.result(timeout=0.05)
                pending.popleft()
                cv2.imshow("Shibuya Crossing - CSRNet", annotated.array)
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
    add_assets_dir_arg(parser)
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
    return run(
        url=args.url,
        assets_dir=args.assets_dir,
        target_fps=args.target_fps,
        workers=args.workers,
        stride=args.stride,
        weights=args.weights,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
