from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

import cv2
from ml_pipes.operator import Operator
from ml_pipes.vision import ImagePayload


def _is_youtube_page_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return host == "youtu.be" or host.endswith(".youtu.be") or host == "youtube.com" or host.endswith(".youtube.com")


def resolve_stream_source(url: str) -> str:
    """Resolve a YouTube page URL to a playable stream URL, or pass direct sources through unchanged."""
    if not _is_youtube_page_url(url):
        return url

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required to resolve YouTube page URLs. Install it with "
            "'python -m pip install yt-dlp', or pass a direct stream URL with --url."
        ) from exc

    try:
        with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to resolve the YouTube stream URL. Retry later or pass a direct stream URL with --url."
        ) from exc

    stream_url = info.get("url")
    if not isinstance(stream_url, str) or not stream_url:
        raise RuntimeError("yt-dlp did not return a playable stream URL.")
    return stream_url


@Operator
class DrawCount:
    def __init__(
        self,
        counter: str = "people",
        *,
        decimals: int = 1,
        origin: tuple[int, int] = (12, 28),
        font_scale: float = 0.8,
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> None:
        self.counter = counter
        self.decimals = decimals
        self.origin = origin
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness

    def __call__(self, image: ImagePayload, count: int | float) -> ImagePayload:
        if isinstance(count, int) and not isinstance(count, bool):
            rendered_count = str(count)
        else:
            rendered_count = f"{float(count):.{self.decimals}f}"

        frame = image.array.copy()
        cv2.putText(
            frame,
            f"{self.counter}: {rendered_count}",
            self.origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.color,
            self.thickness,
            cv2.LINE_AA,
        )
        return ImagePayload(array=frame, color_space=image.color_space, layout=image.layout)


class FrameReader:
    """Reads frames from a VideoCapture on a background thread into a deque.
    Each frame is tagged with its presentation time (PTS) derived from the
    stream FPS and frame index — independent of HLS burst jitter.

    Accepts a URL string (network stream, with reconnect) or an integer
    device index (webcam, no reconnect).

    Two consumption modes:
    - latest(): drops all expired frames, returns the most recent due one (eager).
    - next(): returns the oldest buffered frame in strict arrival order (lazy).

    Both block until a frame is available or the stream ends."""

    def __init__(
        self,
        source: str | int,
        fallback_fps: float = 25.0,
        reconnect_delay_s: float = 1.0,
        stride: int = 1,
    ) -> None:
        self._source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise OSError(f"Could not open source: {source}")
        stream_fps = self._cap.get(cv2.CAP_PROP_FPS) or fallback_fps
        self.stream_fps = stream_fps
        self._stride = max(1, stride)
        self._frame_interval = 1.0 / stream_fps
        self._reconnect_delay_s = reconnect_delay_s
        self._buf: collections.deque[tuple[float, Any]] = collections.deque()
        self._lock = threading.Lock()
        self._stopped = False
        self._t_start: float | None = None
        self._frame_index = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped:
            # Grab and discard stride-1 frames, then decode the strided one
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    break
            ok, frame = self._cap.read()
            if not ok:
                if self._stopped:
                    break
                # Only attempt reconnect for URL sources, not device indices
                if isinstance(self._source, int):
                    self._stopped = True
                    break
                print("\nStream lost — reconnecting...", file=sys.stderr)
                while not self._stopped:
                    time.sleep(self._reconnect_delay_s)
                    self._cap.open(self._source)
                    if self._cap.isOpened():
                        print("Reconnected.", file=sys.stderr)
                        break
                continue
            now = time.perf_counter()
            if self._t_start is None:
                self._t_start = now
            # PTS advances by stride intervals so timing stays true to stream clock
            pts = self._t_start + self._frame_index * self._frame_interval * self._stride
            self._frame_index += 1
            with self._lock:
                self._buf.append((pts, frame))

    def latest(self) -> tuple[bool, Any]:
        """Eager: drop expired frames, return the most recent due frame."""
        while True:
            with self._lock:
                if not self._buf:
                    if self._stopped:
                        return False, None
                else:
                    now = time.perf_counter()
                    current: Any = None
                    while self._buf and self._buf[0][0] <= now:
                        _, current = self._buf.popleft()
                    if current is not None:
                        return True, current
            time.sleep(0.001)

    def next(self) -> tuple[bool, Any]:
        """Lazy: return the oldest buffered frame in strict arrival order."""
        while True:
            with self._lock:
                if self._buf:
                    _, frame = self._buf.popleft()
                    return True, frame
                if self._stopped:
                    return False, None
            time.sleep(0.001)

    def stop(self) -> None:
        self._stopped = True
        self._thread.join()
        self._cap.release()


def add_streaming_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/watch?v=dfVK7ld38Ys",
        help="YouTube page URL or direct stream URL.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel inference workers.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame; intermediate frames are grabbed but not decoded.",
    )
