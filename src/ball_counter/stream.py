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

    def __init__(self, config: GoalConfig, ml_model_path: str | None = None,
                 yolo_model_path: str | None = None):
        self.config = config
        self.counter: MotionCounter | None = None
        self.ml_detector = None
        self.yolo_detector = None
        self.last_event: MotionEvent | None = None
        self.score_flash = 0
        self._crop_bounds: tuple[int, int, int, int] | None = None
        self._last_frame: np.ndarray | None = None
        self.buffer = RollingBuffer()

        if yolo_model_path is not None and config.roi_points:
            # YOLO takes priority over ML when both are specified
            pass  # initialized in init() once we know frame geometry
            self._yolo_model_path = yolo_model_path
        else:
            self._yolo_model_path = None

        if ml_model_path is not None and self._yolo_model_path is None:
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
        if self.yolo_detector is not None:
            return self.yolo_detector.total_count
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

        # MotionCounter takes either line or roi, not both.
        # When both are present, use line for detection (roi is stored for future use).
        line = tuple(offset_scale(self.config.line)) if self.config.line else None
        roi = offset_scale(self.config.roi_points) if self.config.roi_points and not line else None

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

        # Initialize YOLO detector with crop-local ROI
        if self._yolo_model_path is not None:
            from ball_counter.yolo_detector import YOLOBallDetector
            roi = offset_scale(self.config.roi_points)
            print(f"yolo     - {self.config.name}: crop={scaled_w}x{scaled_h} "
                  f"roi={roi} max_track_dist=60")
            self.yolo_detector = YOLOBallDetector(
                model_path=self._yolo_model_path,
                roi_points=roi,
                conf_threshold=0.7,
                max_track_distance=60.0,
                min_track_age=1,
            )
            self.yolo_detector.process_frame(crop)

    def reset_count(self) -> None:
        if self.counter is not None:
            self.counter.count = 0
        if self.ml_detector is not None:
            self.ml_detector.count = 0
        if self.yolo_detector is not None:
            self.yolo_detector.reset()

    def process(self, frame: np.ndarray, timestamp: str = "",
                alignment_offset: tuple[float, float] | None = None) -> MotionEvent | None:
        """Run motion counting on the crop region of the given frame.

        alignment_offset: (dx, dy) pixel drift from AprilTag tracking.
        When provided, the crop region is shifted to follow camera drift.
        """
        if self.counter is None:
            return None
        self._last_frame = frame
        x1, y1, x2, y2 = self._crop_bounds

        # Store for overlay drawing (don't shift crop — shift geometry instead)
        self._last_alignment_offset = alignment_offset
        crop = frame[y1:y2, x1:x2]
        ds = self.config.downsample
        if ds != 1.0:
            crop = cv2.resize(crop, (int((x2 - x1) * ds), int((y2 - y1) * ds)))

        # Always run signal extraction (also does threshold detection)
        threshold_event = self.counter.process_frame(crop)

        # Detection priority: YOLO > ML > threshold
        if self.yolo_detector is not None:
            event = self.yolo_detector.process_frame(crop)
            if event is not None:
                self.counter.count = self.yolo_detector.total_count
        elif self.ml_detector is not None:
            if self.ml_detector.in_channels > 1:
                sig = self.counter.signal_features
            else:
                sig = self.counter.signal
            event = self.ml_detector.process_signal(
                sig,
                self.counter.frame_idx,
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

    # Overlay modes: cycle with set_overlay_mode()
    OVERLAY_NONE = 0
    OVERLAY_LINE = 1
    OVERLAY_ROI = 2
    OVERLAY_CORRECTED = 3  # AprilTag-corrected ROI
    OVERLAY_ALL = 4
    _OVERLAY_COUNT = 5

    def set_overlay_mode(self, mode: int | None = None) -> int:
        """Set or cycle the overlay mode. Returns the new mode."""
        if not hasattr(self, '_overlay_mode'):
            self._overlay_mode = self.OVERLAY_LINE
        if mode is not None:
            self._overlay_mode = mode % self._OVERLAY_COUNT
        else:
            self._overlay_mode = (self._overlay_mode + 1) % self._OVERLAY_COUNT
        return self._overlay_mode

    @property
    def overlay_mode(self) -> int:
        return getattr(self, '_overlay_mode', self.OVERLAY_LINE)

    def crop_jpeg(self, quality: int = 75) -> bytes | None:
        """Return the goal window crop with overlay as JPEG bytes."""
        if self._last_frame is None or self.counter is None:
            return None

        x1, y1, x2, y2 = self._crop_bounds
        raw_crop = self._last_frame[y1:y2, x1:x2]
        # Force consistent output size
        target_w, target_h = x2 - x1, y2 - y1
        if raw_crop.shape[1] != target_w or raw_crop.shape[0] != target_h:
            display = cv2.resize(raw_crop, (target_w, target_h)).copy()
        else:
            display = raw_crop.copy()
        ds = self.config.downsample
        mode = self.overlay_mode

        def offset_scale(pts):
            return [[int((p[0] - x1) * ds), int((p[1] - y1) * ds)] for p in pts]

        # Draw based on overlay mode
        if mode == self.OVERLAY_LINE or mode == self.OVERLAY_ALL:
            self.counter.draw(display, color=self.config.draw_color)
        if mode == self.OVERLAY_ROI or mode == self.OVERLAY_ALL:
            if self.config.roi_points:
                roi_local = offset_scale(self.config.roi_points)
                roi_arr = np.array(roi_local, dtype=np.int32)
                cv2.polylines(display, [roi_arr], True, (255, 0, 255), 2)
                cv2.putText(display, "ROI", (roi_arr[0][0], roi_arr[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        if mode == self.OVERLAY_CORRECTED or mode == self.OVERLAY_ALL:
            offset = getattr(self, '_last_alignment_offset', None)
            if self.config.roi_points and offset is not None:
                dx, dy = offset
                corrected_pts = [[p[0] + dx, p[1] + dy] for p in self.config.roi_points]
                roi_local = offset_scale(corrected_pts)
                roi_arr = np.array(roi_local, dtype=np.int32)
                cv2.polylines(display, [roi_arr], True, (0, 255, 255), 2)
                cv2.putText(display, "corrected", (roi_arr[0][0], roi_arr[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if mode == self.OVERLAY_NONE:
            pass  # no overlay

        # Mode label
        labels = ["off", "line", "roi", "corrected", "all"]
        cv2.putText(display, labels[mode], (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

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

    def __init__(self, config: SourceConfig, ml_model_path: str | None = None,
                 yolo_model_path: str | None = None):
        self.config = config
        self.goals: list[GoalProcessor] = [
            GoalProcessor(g, ml_model_path=ml_model_path,
                          yolo_model_path=yolo_model_path) for g in config.goals
        ]
        self.cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._total_frames = 0
        self._is_video_file = False
        self._alignment: "AlignmentTracker | None" = None
        self._alignment_interval = 30  # frames between alignment updates
        self._frame_count = 0

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

        # Initialize AprilTag alignment tracker on the full frame
        from ball_counter.apriltag import AlignmentTracker, GOAL_MARKER_IDS
        # Build per-goal marker ID mapping from config or defaults
        goal_marker_map = {}
        for goal in self.goals:
            if goal.config.marker_ids:
                goal_marker_map[goal.name] = goal.config.marker_ids
            elif goal.name in GOAL_MARKER_IDS:
                goal_marker_map[goal.name] = GOAL_MARKER_IDS[goal.name]
        self._alignment = AlignmentTracker(goal_marker_ids=goal_marker_map)
        # Gather crop regions to search for markers near goals
        self._search_regions = []
        for goal in self.goals:
            if goal.config.crop_override:
                self._search_regions.append(tuple(goal.config.crop_override))
        self._alignment.update(self._frame, search_regions=self._search_regions)
        if self._alignment.initialized:
            print(f"apriltag - {self._alignment.status_str()}")

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
        attempt = 0
        delay = 2
        while True:
            attempt += 1
            time.sleep(delay)
            if self._reopen():
                ret, frame = self.cap.read()
                if ret:
                    self._frame = frame
                    print(f"[{self.source}] reconnected after {attempt} attempt(s)")
                    return True
            print(f"[{self.source}] reconnect attempt {attempt} failed, next retry in {delay}s")
            delay = min(delay * 2, 300)

    def process_frame(self) -> list[tuple[GoalProcessor, MotionEvent | None]]:
        """Process the current frame through all goal counters."""
        if self._frame is None:
            return []

        # Periodic alignment update on the full frame
        self._frame_count += 1
        if (self._alignment is not None
                and self._frame_count % self._alignment_interval == 0):
            self._alignment.update(self._frame, search_regions=self._search_regions)
            drift = self._alignment.drift_px
            if drift > 2.0:
                dx, dy = self._alignment.offset
                print(f"apriltag - drift: ({dx:+.1f}, {dy:+.1f})px = {drift:.1f}px")

        ts = self.timestamp_str
        results = []
        for goal in self.goals:
            offset = (self._alignment.goal_offset(goal.name)
                      if self._alignment and self._alignment.initialized else None)
            results.append((goal, goal.process(self._frame, ts, alignment_offset=offset)))
        return results

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
