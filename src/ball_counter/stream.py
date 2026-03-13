"""Single stream processor: motion-based counting for one camera."""

import os

import cv2
import numpy as np

from ball_counter.config import StreamConfig
from ball_counter.counter import MotionCounter, MotionEvent


class StreamProcessor:
    """Processes a single video stream using motion-based counting."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.cap: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.counter: MotionCounter | None = None
        self.last_event: MotionEvent | None = None
        self.score_flash = 0

    def open(self) -> bool:
        source = self.config.source
        # Use TCP for RTSP streams
        if source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            return False

        # Read first frame to get dimensions and initialize counter
        ret, self.frame = self.cap.read()
        if not ret:
            return False

        h, w = self.frame.shape[:2]
        line = tuple(self.config.line) if self.config.line else None
        roi = self.config.roi_points if self.config.roi_points else None

        self.counter = MotionCounter(
            frame_shape=(h, w),
            line=line,
            roi=roi,
            ball_area=self.config.ball_area,
            band_width=self.config.band_width,
            min_peak=self.config.min_peak,
            fall_ratio=self.config.fall_ratio,
            cooldown=self.config.cooldown,
            hsv_low=self.config.hsv_low,
            hsv_high=self.config.hsv_high,
        )

        # Process the first frame we already read
        self.counter.process_frame(self.frame)
        return True

    def read_frame(self) -> bool:
        if self.cap is None:
            return False
        ret, self.frame = self.cap.read()
        return ret

    def process_frame(self) -> MotionEvent | None:
        """Run motion counting on the current frame.

        Returns a MotionEvent if scoring happened this frame.
        """
        if self.frame is None or self.counter is None:
            return None

        event = self.counter.process_frame(self.frame)

        if event is not None:
            self.last_event = event
            self.score_flash = 20

        return event

    @property
    def count(self) -> int:
        return self.counter.count if self.counter else 0

    def draw_overlay(self) -> np.ndarray | None:
        """Draw detection overlay on the current frame and return it."""
        if self.frame is None or self.counter is None:
            return None

        display = self.frame.copy()

        # Draw the detection zone
        self.counter.draw(display)

        # Header bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 50), (0, 0, 0), -1)
        label = f"{self.config.name} [{self.config.mode}] Count: {self.count}"
        cv2.putText(display, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        signal_text = f"Signal: {self.counter.signal:5d}px"
        if self.counter.rising:
            signal_text += " RISING"
        cv2.putText(display, signal_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        # Flash on score
        if self.score_flash > 0:
            alpha = self.score_flash / 20
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                          (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha * 0.1, display, 1.0 - alpha * 0.1, 0, display)
            if self.last_event:
                cv2.putText(display, f"+{self.last_event.n_balls}",
                            (display.shape[1] - 80, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.score_flash -= 1

        return display

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
