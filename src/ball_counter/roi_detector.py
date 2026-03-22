"""ROI-based ball detector with simple blob tracking.

Detects balls by finding yellow blobs via background subtraction + color
masking, tracks them with nearest-centroid matching, and counts new
objects that appear inside an ROI polygon (outlet mode).

Balls that enter the ROI from outside (crossing in) are tracked but not
counted — only balls that first appear inside the ROI are scored.
"""

import cv2
import numpy as np

from ball_counter.counter import MotionEvent
from ball_counter.detector import create_mask


class BlobTracker:
    """Simple nearest-centroid tracker with ID assignment.

    Each frame: find contours → extract centroids → match to existing
    tracks by proximity → assign new IDs to unmatched detections →
    expire tracks not seen for `max_age` frames.
    """

    def __init__(self, max_distance: float = 50, max_age: int = 5):
        self.max_distance = max_distance
        self.max_age = max_age
        self._next_id = 0
        # {id: {"centroid": (cx, cy), "age": frames_since_seen, "first_frame": bool}}
        self._tracks: dict[int, dict] = {}

    def update(self, centroids: list[tuple[int, int]]) -> dict[int, np.ndarray]:
        """Match new centroids to existing tracks. Returns {id: centroid}."""
        if not centroids:
            # Age all tracks
            to_remove = []
            for tid, track in self._tracks.items():
                track["age"] += 1
                if track["age"] > self.max_age:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._tracks[tid]
            return {}

        # Build cost matrix: distance from each track to each detection
        track_ids = list(self._tracks.keys())
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        result: dict[int, np.ndarray] = {}

        if track_ids:
            track_pts = np.array([self._tracks[tid]["centroid"] for tid in track_ids])
            det_pts = np.array(centroids)

            # Greedy nearest-neighbor matching
            dists = np.linalg.norm(track_pts[:, None] - det_pts[None, :], axis=2)
            pairs = []
            for ti in range(len(track_ids)):
                for di in range(len(centroids)):
                    if dists[ti, di] < self.max_distance:
                        pairs.append((dists[ti, di], ti, di))
            pairs.sort()

            for _, ti, di in pairs:
                if ti in matched_tracks or di in matched_dets:
                    continue
                tid = track_ids[ti]
                self._tracks[tid]["centroid"] = centroids[di]
                self._tracks[tid]["age"] = 0
                self._tracks[tid]["first_frame"] = False
                matched_tracks.add(ti)
                matched_dets.add(di)
                result[tid] = np.array(centroids[di])

        # Create new tracks for unmatched detections
        for di, c in enumerate(centroids):
            if di not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"centroid": c, "age": 0, "first_frame": True}
                result[tid] = np.array(c)

        # Age unmatched tracks
        to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

        return result

    def is_new(self, track_id: int) -> bool:
        """Check if a track was just created this frame."""
        track = self._tracks.get(track_id)
        return track is not None and track["first_frame"]


class ROIBlobDetector:
    """Detects balls using blob tracking + ROI polygon.

    Pipeline per frame:
    1. Background subtraction → foreground mask
    2. HSV color filter → yellow mask
    3. AND → moving yellow pixels
    4. Find contours → blob centroids
    5. Track blobs with BlobTracker
    6. ROICounter: count new blobs appearing inside polygon

    Only blobs that FIRST APPEAR inside the ROI are counted (outlet mode).
    Blobs entering from outside are tracked but ignored.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        roi_points: list[list[int]],
        *,
        hsv_low: tuple[int, int, int] = (20, 100, 100),
        hsv_high: tuple[int, int, int] = (35, 255, 255),
        min_area: int = 50,
        max_track_distance: float = 50,
        bg_history: int = 60,
        bg_var_threshold: int = 50,
    ):
        self.roi_polygon = np.array(roi_points, dtype=np.int32)
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.min_area = min_area

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=bg_history,
            varThreshold=bg_var_threshold,
            detectShadows=False,
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.tracker = BlobTracker(max_distance=max_track_distance)

        # ROI mask for filtering
        h, w = frame_shape
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_polygon], 255)

        self.count = 0
        self.frame_idx = 0
        self.events: list[MotionEvent] = []
        self._counted_ids: set[int] = set()

    def process_frame(self, frame: np.ndarray) -> MotionEvent | None:
        """Process one frame. Returns MotionEvent if new ball(s) detected."""
        fg_mask = self.bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)

        yellow = create_mask(frame, self.hsv_low, self.hsv_high)
        moving_yellow = cv2.bitwise_and(yellow, fg_mask)

        # Find contours of moving yellow blobs
        contours, _ = cv2.findContours(moving_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract centroids of sufficiently large blobs
        centroids = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

        # Track blobs
        tracked = self.tracker.update(centroids)

        # Count new blobs that appeared inside the ROI
        event = None
        new_balls = 0
        for tid, centroid in tracked.items():
            if tid in self._counted_ids:
                continue
            if not self.tracker.is_new(tid):
                continue
            # New track — check if it appeared inside the ROI
            cx, cy = int(centroid[0]), int(centroid[1])
            if cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), False) >= 0:
                self._counted_ids.add(tid)
                new_balls += 1

        if new_balls > 0:
            self.count += new_balls
            event = MotionEvent(self.frame_idx, new_balls, 0)
            self.events.append(event)

        self.frame_idx += 1
        return event

    @property
    def signal(self) -> int:
        """Compat: return number of tracked blobs inside ROI."""
        return self.count
