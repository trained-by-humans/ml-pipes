from __future__ import annotations

import argparse
import collections
import sys
import time
import urllib.error
from concurrent.futures import Future, ThreadPoolExecutor
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

CSRNET_MODEL_NAME = "csrnet_shanghaitech_b_rootstrap.pth"
CSRNET_MODEL_URL = "https://huggingface.co/rootstrap-org/crowd-counting/resolve/main/weights.pth"


class RollingAverage:
    def __init__(self, window: int = 12) -> None:
        self._values: collections.deque[float] = collections.deque(maxlen=max(1, window))

    def update(self, value: float) -> float:
        self._values.append(float(value))
        return float(sum(self._values) / len(self._values))


class RollingFPS:
    def __init__(self, window: int = 30) -> None:
        self._timestamps: collections.deque[float] = collections.deque(maxlen=max(2, window))

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return float((len(self._timestamps) - 1) / elapsed)


def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "CSRNet example requires torch. Install it in the project environment before running this script."
        ) from exc
    return torch, nn


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


def preprocess_frame(frame: np.ndarray, torch: Any, device: str) -> Any:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))[None, ...]
    tensor = torch.from_numpy(np.ascontiguousarray(chw))
    return tensor.to(device)


def infer_density(frame: np.ndarray, model: Any, torch: Any, device: str) -> tuple[float, np.ndarray, float]:
    started = time.perf_counter()
    tensor = preprocess_frame(frame, torch, device)
    with torch.inference_mode():
        density = model(tensor).squeeze(0).squeeze(0).detach().cpu().numpy()
    density = np.maximum(density.astype(np.float32, copy=False), 0.0)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return float(density.sum()), density, latency_ms


def density_to_heatmap(density_map: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape
    density = np.maximum(density_map.astype(np.float32, copy=False), 0.0)
    if density.size == 0 or float(density.max()) <= 0.0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    normalized = cv2.normalize(density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)


def render_overlay(frame: np.ndarray, density_map: np.ndarray, count: float, smoothed: float, latency_ms: float, fps: float) -> np.ndarray:
    heatmap = density_to_heatmap(density_map, frame.shape[:2])
    canvas = cv2.addWeighted(frame, 0.60, heatmap, 0.40, 0.0)
    cv2.rectangle(canvas, (0, 0), (430, 110), (15, 18, 28), thickness=-1)
    lines = (
        "Backend: CSRNet",
        f"Count: {count:.1f}",
        f"Smoothed: {smoothed:.1f}",
        f"Latency: {latency_ms:.1f} ms   FPS: {fps:.1f}",
    )
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (14, 28 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (236, 240, 241),
            2,
            cv2.LINE_AA,
        )
    return canvas


def run(
    url: str,
    assets_dir: Path,
    target_fps: float,
    workers: int,
    stride: int,
    weights: Path | None,
    device: str,
    smoothing_window: int,
    log_interval_s: float,
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

    count_smoother = RollingAverage(window=smoothing_window)
    fps_monitor = RollingFPS()
    last_log = 0.0
    pending: collections.deque[Future] = collections.deque()
    stopped = False

    print(
        f"Streaming with {workers} worker(s), stride={stride}, device={resolved_device} "
        "— press Q in the window to quit.",
        file=sys.stderr,
    )

    def infer(frame: Any) -> tuple[np.ndarray, float, np.ndarray, float]:
        count, density_map, latency_ms = infer_density(frame, model, torch, resolved_device)
        return frame, count, density_map, latency_ms

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
                frame, count, density_map, latency_ms = future.result(timeout=0.05)
                pending.popleft()
                smoothed = count_smoother.update(count)
                fps = fps_monitor.tick()
                annotated = render_overlay(frame, density_map, count, smoothed, latency_ms, fps)
                cv2.imshow("Shibuya Crossing - CSRNet", annotated)

                now = time.perf_counter()
                if now - last_log >= log_interval_s:
                    print(
                        f"[csrnet] count={count:.1f} smoothed={smoothed:.1f} "
                        f"latency={latency_ms:.1f}ms fps={fps:.1f}",
                        file=sys.stderr,
                    )
                    last_log = now
            except TimeoutError:
                pass

    reader.stop()
    cv2.destroyAllWindows()
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
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=12,
        help="Rolling window used to smooth the displayed count.",
    )
    parser.add_argument(
        "--log-interval-s",
        type=float,
        default=1.0,
        help="How often to print count and latency metrics.",
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
        smoothing_window=args.smoothing_window,
        log_interval_s=args.log_interval_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
