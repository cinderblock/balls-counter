"""Ball counting: motion-based, ROI-based, and line-crossing approaches."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from ball_counter.detector import create_mask


@dataclass
class MotionEvent:
    """A single scoring event detected by the motion counter."""
    frame: int
    n_balls: int
    peak_area: int


class MotionCounter:
    """Count balls using background subtraction + color masking + peak detection.

    Supports two geometry modes:
    - "line": A narrow band around a counting line (for side-view outlets where
      balls drop through a line).
    - "roi": A thin ring around a polygon perimeter (for top-down views where
      balls enter a hole).
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],  # (height, width)
        *,
        # Geometry — provide either line or roi, not both
        line: tuple[list[int], list[int]] | None = None,
        roi: list[list[int]] | None = None,
        # Tuning
        ball_area: int = 900,
        band_width: int = 20,
        min_peak: int = 0,
        fall_ratio: float = 0.5,
        min_rise: int = 100,
        cooldown: int = 0,
        # HSV
        hsv_low: tuple[int, int, int] = (20, 100, 100),
        hsv_high: tuple[int, int, int] = (35, 255, 255),
        # Background subtractor
        bg_history: int = 60,
        bg_var_threshold: int = 50,
    ):
        h, w = frame_shape
        self.ball_area = ball_area
        self.min_peak = min_peak
        self.fall_ratio = fall_ratio
        self.min_rise = min_rise
        self.cooldown = cooldown
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high

        # Build the detection mask based on geometry mode
        if line is not None and roi is not None:
            raise ValueError("Provide either line or roi, not both")
        if line is None and roi is None:
            raise ValueError("Must provide either line or roi")

        self.mask = np.zeros((h, w), dtype=np.uint8)
        if line is not None:
            self.mode = "line"
            p1, p2 = line
            self.line = (p1, p2)
            band_pts = np.array([
                [p1[0], p1[1] - band_width], [p2[0], p2[1] - band_width],
                [p2[0], p2[1] + band_width], [p1[0], p1[1] + band_width]
            ], dtype=np.int32)
            cv2.fillPoly(self.mask, [band_pts], 255)
            self.draw_pts = band_pts
        else:
            self.mode = "roi"
            roi_pts = np.array(roi, dtype=np.int32)
            self.roi_pts = roi_pts
            # Ring mask: dilated minus eroded
            filled = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(filled, [roi_pts], 255)
            outer = cv2.dilate(filled, np.ones((band_width, band_width), np.uint8))
            inner = cv2.erode(filled, np.ones((band_width, band_width), np.uint8))
            self.mask = cv2.subtract(outer, inner)

        # Background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=bg_history,
            varThreshold=bg_var_threshold,
            detectShadows=False,
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Peak detection state
        self.prev_area = 0
        self.rising = False
        self.peak_val = 0
        self.cooldown_remaining = 0

        # Results
        self.count = 0
        self.frame_idx = 0
        self.events: list[MotionEvent] = []

    def process_frame(self, frame: np.ndarray) -> MotionEvent | None:
        """Process one frame. Returns a MotionEvent if scoring happened."""
        fg_mask = self.bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)

        yellow = create_mask(frame, self.hsv_low, self.hsv_high)
        moving_yellow = cv2.bitwise_and(yellow, fg_mask)
        moving_in_zone = cv2.bitwise_and(moving_yellow, self.mask)
        area = cv2.countNonZero(moving_in_zone)

        event = None

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        else:
            if area > self.prev_area and area > self.min_rise:
                self.rising = True
                self.peak_val = max(self.peak_val, area)
            elif self.rising and area < self.peak_val * self.fall_ratio:
                if self.peak_val >= self.min_peak:
                    n_balls = max(1, round(self.peak_val / self.ball_area))
                    self.count += n_balls
                    event = MotionEvent(self.frame_idx, n_balls, self.peak_val)
                    self.events.append(event)
                    self.cooldown_remaining = self.cooldown
                self.rising = False
                self.peak_val = 0

        self.prev_area = area
        self.frame_idx += 1
        return event

    @property
    def signal(self) -> int:
        """Current signal value (moving yellow pixels in zone)."""
        return self.prev_area

    def draw(self, frame: np.ndarray, alpha: float = 0.2,
             color: tuple[int, int, int] = (0, 0, 255)) -> None:
        """Draw the detection zone on a frame."""
        dim = tuple(max(0, c - 75) for c in color)  # darker fill
        overlay = frame.copy()
        if self.mode == "line":
            p1, p2 = self.line
            cv2.line(frame, tuple(p1), tuple(p2), color, 2)
            cv2.fillPoly(overlay, [self.draw_pts], dim)
        else:
            cv2.polylines(frame, [self.roi_pts], True, color, 2)
            ring_colored = np.zeros_like(frame)
            ring_colored[self.mask > 0] = dim
            overlay = cv2.add(overlay, ring_colored)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


class LineCrossingCounter:
    """
    Counts objects that cross a line defined by two points.

    Objects are counted when their centroid crosses from one side to the other.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self._previous_side: dict[int, int] = {}
        self.count_in = 0
        self.count_out = 0

    @property
    def count(self) -> int:
        return self.count_in

    def _side_of_line(self, px: int, py: int) -> int:
        cross = (self.x2 - self.x1) * (py - self.y1) - (self.y2 - self.y1) * (px - self.x1)
        return 1 if cross > 0 else -1

    def update_object(self, object_id: int, cx: int, cy: int) -> str | None:
        """Check if an object crossed the line. Returns 'in', 'out', or None."""
        current_side = self._side_of_line(cx, cy)

        if object_id not in self._previous_side:
            self._previous_side[object_id] = current_side
            return None

        previous_side = self._previous_side[object_id]
        self._previous_side[object_id] = current_side

        if previous_side != current_side:
            if current_side == 1:
                self.count_in += 1
                return "in"
            else:
                self.count_out += 1
                return "out"

        return None

    def update(self, tracked_objects: dict[int, np.ndarray]) -> list[int]:
        """Update with all tracked objects. Returns IDs that crossed 'in' this frame."""
        scored = []
        for object_id, centroid in tracked_objects.items():
            event = self.update_object(object_id, int(centroid[0]), int(centroid[1]))
            if event == "in":
                scored.append(object_id)
        return scored

    def cleanup(self, active_ids: set[int]) -> None:
        stale = set(self._previous_side.keys()) - active_ids
        for object_id in stale:
            del self._previous_side[object_id]

    def draw(self, frame: np.ndarray) -> None:
        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)


class ROICounter:
    """
    Counts balls entering/leaving a polygonal region of interest.

    Two modes:
    - "inlet": Counts balls that enter the ROI and then disappear (went through
      the hole). Balls that enter and leave back out are not counted.
    - "outlet": Counts each new unique ball that appears in the ROI (arrived
      from the scoring hole into the dump zone).
    """

    def __init__(self, roi_points: list[tuple[int, int]], mode: str = "inlet"):
        if mode not in ("inlet", "outlet"):
            raise ValueError(f"mode must be 'inlet' or 'outlet', got {mode!r}")
        self.roi_polygon = np.array(roi_points, dtype=np.int32)
        self.mode = mode
        self.count = 0

        # Track state per object ID
        # For inlet: objects currently inside the ROI
        # For outlet: objects we've already counted
        self._inside: set[int] = set()
        self._counted: set[int] = set()

    def is_inside_roi(self, x: int, y: int) -> bool:
        """Check if a point is inside the ROI polygon."""
        return cv2.pointPolygonTest(self.roi_polygon, (float(x), float(y)), False) >= 0

    def update(self, tracked_objects: dict[int, np.ndarray]) -> list[int]:
        """
        Update counts based on current tracked object positions.

        Returns list of object IDs that were just counted this frame.
        """
        newly_counted = []

        current_inside: set[int] = set()
        for object_id, centroid in tracked_objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            if self.is_inside_roi(cx, cy):
                current_inside.add(object_id)

        if self.mode == "outlet":
            # Count new balls appearing in the ROI
            for object_id in current_inside:
                if object_id not in self._counted:
                    self._counted.add(object_id)
                    self.count += 1
                    newly_counted.append(object_id)

        elif self.mode == "inlet":
            # Track balls entering the ROI
            for object_id in current_inside - self._inside:
                # Ball just entered the ROI
                pass  # We'll count it if it disappears while inside

            # Check for balls that were inside and have now disappeared entirely
            # (not just left the ROI — they must have gone through the hole)
            for object_id in self._inside:
                if object_id not in tracked_objects:
                    # Object disappeared while inside ROI = scored
                    if object_id not in self._counted:
                        self._counted.add(object_id)
                        self.count += 1
                        newly_counted.append(object_id)
                elif object_id not in current_inside:
                    # Object left the ROI (still tracked, just moved out) = not scored
                    pass

        self._inside = current_inside
        return newly_counted

    def cleanup(self, active_ids: set[int]) -> None:
        """Remove tracking data for deregistered objects."""
        self._inside &= active_ids
        # Keep _counted forever to avoid double-counting if IDs wrap

    def draw(self, frame: np.ndarray) -> None:
        """Draw the ROI polygon on a frame."""
        cv2.polylines(frame, [self.roi_polygon], True, (0, 0, 255), 2)
