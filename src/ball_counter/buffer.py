"""Rolling 60-second frame buffer per goal."""

import threading
from collections import deque
from dataclasses import dataclass

from ball_counter.counter import MotionEvent


@dataclass
class BufferFrame:
    """One frame's worth of data stored in the rolling buffer."""
    timestamp: str
    jpeg: bytes           # raw crop JPEG (no overlay), full resolution before downsample
    frame_idx: int
    signal: int           # motion signal value (moving yellow pixels in zone)
    rising: bool          # whether signal was rising after this frame
    event: MotionEvent | None = None  # set if a scoring event fired on this frame


class RollingBuffer:
    """Thread-safe rolling buffer of the last N frames for a single goal.

    Default maxlen=1800 ≈ 60 seconds at 30 fps.
    Auto-evicts oldest frames when full.
    """

    def __init__(self, maxlen: int = 1800):
        self._buf: deque[BufferFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, frame: BufferFrame) -> None:
        with self._lock:
            self._buf.append(frame)

    def snapshot(self) -> list[BufferFrame]:
        """Return a copy of the current buffer contents (oldest first)."""
        with self._lock:
            return list(self._buf)

    def slice_by_index(self, start: int, end: int) -> list[BufferFrame]:
        """Return frames whose frame_idx falls in [start, end] inclusive."""
        with self._lock:
            return [f for f in self._buf if start <= f.frame_idx <= end]

    def latest(self) -> BufferFrame | None:
        """Return the most recently appended frame, or None if empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
