"""The neural sensor: a background relighting thread plus the confidence trigger with hysteresis."""

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from relight.pipeline import ImageLike, Relighter

logger = logging.getLogger(__name__)


class ConfidenceTrigger:
    """Turn relighting on when the raw detector's max confidence c_max drops below tau_low,
    and off again once it rises above tau_high (paper: 0.3 / 0.5). Frames with no detection
    (c_max is None) keep the current state."""

    def __init__(self, tau_low: float = 0.3, tau_high: float = 0.5):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.active = False

    def update(self, c_max: Optional[float]) -> bool:
        if c_max is not None:
            if not self.active and c_max < self.tau_low:
                self.active = True
            elif self.active and c_max > self.tau_high:
                self.active = False
        return self.active


class AsyncRelighter:
    """Relight frames in a daemon thread so the control loop never blocks.

    submit() is non-blocking; when the queue is full the oldest frame is dropped.
    latest() returns the most recent relit frame (PIL, RGB ndarray). One inference step,
    as in the closed-loop runs.
    """

    def __init__(self, relighter: Relighter, num_inference_steps: int = 1, max_queue_size: int = 2):
        self.relighter = relighter
        self.num_inference_steps = num_inference_steps
        self.requests: "queue.Queue" = queue.Queue(maxsize=max_queue_size)
        self.lock = threading.Lock()
        self.latest_pil = None
        self.latest_np: Optional[np.ndarray] = None
        self.processed = 0
        self.dropped = 0
        self.times = []
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> "AsyncRelighter":
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self.requests.put(None, timeout=1.0)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def submit(self, image: ImageLike) -> bool:
        if not self._running:
            return False
        try:
            self.requests.put_nowait(image)
        except queue.Full:
            try:
                self.requests.get_nowait()
                self.dropped += 1
                self.requests.put_nowait(image)
            except (queue.Empty, queue.Full):
                return False
        return True

    def latest(self):
        with self.lock:
            return self.latest_pil, self.latest_np

    def stats(self) -> dict:
        with self.lock:
            recent = self.times[-100:]
            return {"processed": self.processed, "dropped": self.dropped,
                    "avg_time": sum(recent) / len(recent) if recent else 0.0,
                    "last_time": recent[-1] if recent else 0.0}

    def _loop(self) -> None:
        while self._running:
            try:
                image = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if image is None:
                break
            try:
                t0 = time.time()
                relit = self.relighter(image, num_inference_steps=self.num_inference_steps)
                dt = time.time() - t0
                with self.lock:
                    self.latest_pil, self.latest_np = relit, np.asarray(relit)
                    self.processed += 1
                    self.times.append(dt)
            except Exception:
                logger.exception("relighting failed")
            finally:
                self.requests.task_done()
