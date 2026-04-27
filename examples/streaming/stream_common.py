from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from typing import Any

import cv2


def get_stream_url(youtube_url: str) -> str:
    import yt_dlp
    with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True}) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


class FrameReader:
    """Reads frames from a VideoCapture on a background thread into a deque.
    Each frame is tagged with its presentation time (PTS) derived from the
    stream FPS and frame index — independent of HLS burst jitter.

    Two consumption modes:
    - latest(): drops all expired frames, returns the most recent due one (eager).
    - next(): returns the oldest buffered frame in strict arrival order (lazy).

    Both block until a frame is available or the stream ends."""

    def __init__(
        self,
        stream_url: str,
        fallback_fps: float = 25.0,
        reconnect_delay_s: float = 1.0,
        stride: int = 1,
    ) -> None:
        self._stream_url = stream_url
        self._cap = cv2.VideoCapture(stream_url)
        if not self._cap.isOpened():
            raise OSError(f"Could not open stream: {stream_url}")
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
                print("\nStream lost — reconnecting...", file=sys.stderr)
                while not self._stopped:
                    time.sleep(self._reconnect_delay_s)
                    self._cap.open(self._stream_url)
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
        help="YouTube live URL.",
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
