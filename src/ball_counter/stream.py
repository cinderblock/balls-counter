"""Video source and goal processors for motion-based ball counting."""

import os
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

from ball_counter.buffer import BufferFrame, RollingBuffer
from ball_counter.config import GoalConfig, SourceConfig
from ball_counter.counter import MotionCounter, MotionEvent


class GoalProcessor:
    """Counts balls for one goal zone on a shared video frame."""

    def __init__(self, config: GoalConfig, ml_model_path: str | None = None):
        self.config = config
        self.counter: MotionCounter | None = None
        self.ml_detector = None
        self.last_event: MotionEvent | None = None
        self.score_flash = 0
        self._crop_bounds: tuple[int, int, int, int] | None = None
        self._last_frame: np.ndarray | None = None
        self.buffer = RollingBuffer()

        if ml_model_path is not None:
            from ball_counter.ml_detector import MLPeakDetector
            self.ml_detector = MLPeakDetector(
                ml_model_path,
                ball_area=config.ball_area,
            )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def count(self) -> int:
        if self.ml_detector is not None:
            return self.ml_detector.count
        return self.counter.count if self.counter else 0

    @property
    def processed_frames(self) -> int:
        return self.counter.frame_idx if self.counter else 0

    def init(self, frame: np.ndarray) -> None:
        """Initialize the MotionCounter from the first frame."""
        h, w = frame.shape[:2]
        self._crop_bounds = self._compute_crop(h, w)
        x1, y1, x2, y2 = self._crop_bounds
        ds = self.config.downsample

        # Offset geometry into crop-local coordinates, then apply downsample
        def offset_scale(pts: list[list[int]]) -> list[list[int]]:
            return [[int((p[0] - x1) * ds), int((p[1] - y1) * ds)] for p in pts]

        line = tuple(offset_scale(self.config.line)) if self.config.line else None
        roi = offset_scale(self.config.roi_points) if self.config.roi_points else None

        crop_h, crop_w = y2 - y1, x2 - x1
        scaled_h = int(crop_h * ds)
        scaled_w = int(crop_w * ds)
        self.counter = MotionCounter(
            frame_shape=(scaled_h, scaled_w),
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
        crop = frame[y1:y2, x1:x2]
        if ds != 1.0:
            crop = cv2.resize(crop, (scaled_w, scaled_h))
        self.counter.process_frame(crop)

    def reset_count(self) -> None:
        if self.counter is not None:
            self.counter.count = 0
        if self.ml_detector is not None:
            self.ml_detector.count = 0

    def process(self, frame: np.ndarray, timestamp: str = "") -> MotionEvent | None:
        """Run motion counting on the crop region of the given frame."""
        if self.counter is None:
            return None
        self._last_frame = frame
        x1, y1, x2, y2 = self._crop_bounds
        crop = frame[y1:y2, x1:x2]
        ds = self.config.downsample
        if ds != 1.0:
            crop = cv2.resize(crop, (int((x2 - x1) * ds), int((y2 - y1) * ds)))

        # Always run signal extraction (also does threshold detection)
        threshold_event = self.counter.process_frame(crop)

        # Use ML detector if available, otherwise fall back to threshold
        if self.ml_detector is not None:
            event = self.ml_detector.process_signal(
                self.counter.signal,
                self.counter.frame_idx,
                peak_area=self.counter.signal,
            )
            # Sync count from ML detector back to counter for display
            if event is not None:
                self.counter.count = self.ml_detector.count
        else:
            event = threshold_event

        if event is not None:
            self.last_event = event
            self.score_flash = 20

        # Feed rolling buffer with the raw (no overlay) full-resolution crop
        raw = frame[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ok:
            self.buffer.append(BufferFrame(
                timestamp=timestamp,
                jpeg=bytes(buf),
                frame_idx=self.counter.frame_idx,
                signal=self.counter.signal,
                rising=self.counter.rising,
                event=event,
            ))

        return event

    def crop_jpeg(self, quality: int = 75) -> bytes | None:
        """Return the goal window crop with overlay as JPEG bytes."""
        if self._last_frame is None or self.counter is None:
            return None

        x1, y1, x2, y2 = self._crop_bounds
        display = self._last_frame[y1:y2, x1:x2].copy()
        self.counter.draw(display, color=self.config.draw_color)

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

        ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return bytes(buf) if ok else None

    def _compute_crop(self, h: int, w: int, padding: int = 150) -> tuple[int, int, int, int]:
        if self.config.crop_override is not None:
            x1, y1, x2, y2 = self.config.crop_override
            return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        points: list[tuple[int, int]] = []
        if self.config.line:
            points = list(self.config.line)
        elif self.config.roi_points:
            points = list(self.config.roi_points)
        if not points:
            return (0, 0, w, h)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (
            max(0, min(xs) - padding),
            max(0, min(ys) - padding),
            min(w, max(xs) + padding),
            min(h, max(ys) + padding),
        )


class SourceProcessor:
    """Opens one video source and runs all its goal processors on each frame."""

    def __init__(self, config: SourceConfig, ml_model_path: str | None = None):
        self.config = config
        self.goals: list[GoalProcessor] = [
            GoalProcessor(g, ml_model_path=ml_model_path) for g in config.goals
        ]
        self.cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._total_frames = 0
        self._is_video_file = False

    @property
    def source(self) -> str:
        return self.config.source

    @property
    def is_video_file(self) -> bool:
        return self._is_video_file

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def timestamp_str(self) -> str:
        if self._is_video_file and self.cap is not None:
            ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            td = timedelta(milliseconds=ms)
            total_s = int(td.total_seconds())
            h, rem = divmod(total_s, 3600)
            m, s = divmod(rem, 60)
            return f"{h}:{m:02d}:{s:02d}.{int(td.microseconds / 1000):03d}"
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def open(self) -> bool:
        source = self.config.source
        if source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            self._is_video_file = False
        else:
            self.cap = cv2.VideoCapture(source)
            self._is_video_file = True

        if not self.cap.isOpened():
            return False

        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, self._frame = self.cap.read()
        if not ret:
            return False

        for goal in self.goals:
            goal.init(self._frame)
        return True

    def _reopen(self) -> bool:
        source = self.config.source
        if self.cap is not None:
            self.cap.release()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        return self.cap.isOpened()

    def read_frame(self) -> bool:
        if self.cap is None:
            return False
        ret, frame = self.cap.read()
        if ret:
            self._frame = frame
            return True
        if self._is_video_file:
            return False
        print(f"[{self.source}] stream dropped, reconnecting...")
        for attempt in range(1, 6):
            time.sleep(2 * attempt)
            if self._reopen():
                ret, frame = self.cap.read()
                if ret:
                    self._frame = frame
                    print(f"[{self.source}] reconnected")
                    return True
            print(f"[{self.source}] reconnect attempt {attempt} failed")
        print(f"[{self.source}] giving up after 5 attempts")
        return False

    def process_frame(self) -> list[tuple[GoalProcessor, MotionEvent | None]]:
        """Process the current frame through all goal counters."""
        if self._frame is None:
            return []
        ts = self.timestamp_str
        return [(goal, goal.process(self._frame, ts)) for goal in self.goals]

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
