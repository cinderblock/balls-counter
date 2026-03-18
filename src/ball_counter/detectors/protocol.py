"""Detector protocol — interface all detectors must satisfy."""

from typing import Protocol

import numpy as np

from ball_counter.counter import MotionEvent


class Detector(Protocol):
    """A single-goal ball detector. Receives pre-cropped frames, emits events."""

    def process_frame(self, crop: np.ndarray) -> MotionEvent | None:
        """Process one frame crop. Returns MotionEvent if a scoring event detected."""
        ...

    def reset(self) -> None:
        """Reset internal state (counts, background model, etc.)."""
        ...

    @property
    def count(self) -> int:
        """Total ball count accumulated so far."""
        ...

    @property
    def signal(self) -> int:
        """Current signal value (for visualization / live tuning)."""
        ...

    def draw(self, frame: np.ndarray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        """Draw detection zone overlay onto the frame in-place."""
        ...
