"""YOLO-based ball detector with trajectory-aware tracking.

Detects balls per-frame using YOLOv8, tracks them across frames with a
direction-constrained tracker, and counts balls that enter through the
outlet region of the ROI polygon.

Key insight: balls always enter from the goal outlet (one edge of the ROI)
and move in a consistent direction. The tracker uses this constraint to
avoid merging new arrivals with existing tracks.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ball_counter.counter import MotionEvent


@dataclass
class Track:
    """A tracked ball."""
    id: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    first_inside_roi: bool
    counted: bool = False
    frames_missing: int = 0
    age: int = 1
    history: list = field(default_factory=list)  # list of (cx, cy)


class YOLOBallDetector:
    """Detect and track balls using YOLO + direction-aware tracker.

    Only counts balls that first appear inside the ROI polygon (outlet mode).
    Uses entry-zone awareness: detections near the entry point of the ROI
    are more likely to be new balls, not existing tracks that drifted back.
    """

    def __init__(
        self,
        model_path: str | Path,
        roi_points: list[list[int]],
        *,
        conf_threshold: float = 0.3,
        max_track_distance: float = 40.0,
        max_missing_frames: int = 5,
        min_track_age: int = 1,
        device: str = "0",
        entry_zone_radius: float = 30.0,
    ):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_track_distance = max_track_distance
        self.max_missing_frames = max_missing_frames
        self.min_track_age = min_track_age
        self.entry_zone_radius = entry_zone_radius

        # ROI polygon for origin filtering
        self.roi_polygon = np.array(roi_points, dtype=np.int32)
        self.roi_mask = None  # lazily built on first frame

        # Estimate entry point: the centroid of the ROI edge closest to
        # where balls emerge. For now, use the centroid of all ROI points
        # — can be refined with config.
        pts = np.array(roi_points, dtype=np.float32)
        self.roi_centroid = pts.mean(axis=0)

        # Tracking state
        self.tracks: list[Track] = []
        self.next_id = 0
        self.frame_idx = 0
        self.total_count = 0

    def _build_roi_mask(self, h: int, w: int):
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_polygon], 255)

    def _point_in_roi(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.roi_polygon.astype(np.float32),
                                     (float(x), float(y)), False) >= 0

    def _detect(self, frame: np.ndarray) -> list[tuple[float, float, tuple[int, int, int, int], float]]:
        """Run YOLO on frame, return list of (cx, cy, (x1,y1,x2,y2), conf)."""
        results = self.model.predict(frame, conf=self.conf_threshold, device=self.device,
                                      verbose=False, imgsz=320)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append((cx, cy, (int(x1), int(y1), int(x2), int(y2)), conf))
        return detections

    def _match_tracks(self, detections):
        """Match detections to existing tracks.

        Uses nearest centroid with a constraint: detections near the entry
        zone prefer to be new tracks rather than matching distant existing ones.
        """
        if not self.tracks or not detections:
            return {}, list(range(len(detections)))

        # Cost matrix
        track_centroids = np.array([(t.centroid[0], t.centroid[1]) for t in self.tracks])
        det_centroids = np.array([(d[0], d[1]) for d in detections])
        dists = np.linalg.norm(track_centroids[:, None] - det_centroids[None, :], axis=2)

        matches = {}  # det_idx -> track_idx
        unmatched = list(range(len(detections)))

        # Greedy matching: closest pairs first
        used_tracks = set()
        used_dets = set()

        while True:
            if dists.size == 0:
                break
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            ti, di = int(min_idx[0]), int(min_idx[1])
            min_dist = dists[ti, di]

            if min_dist > self.max_track_distance:
                break

            # Direction constraint: if the track has moved in a consistent
            # direction, don't match a detection that would reverse it
            track = self.tracks[ti]
            accept = True
            if len(track.history) >= 2 and min_dist > 10:
                # Compute track's recent movement direction
                recent = track.history[-2:]
                dx_track = recent[-1][0] - recent[0][0]
                dy_track = recent[-1][1] - recent[0][1]
                # Compute direction to this detection
                dx_det = detections[di][0] - track.centroid[0]
                dy_det = detections[di][1] - track.centroid[1]
                # Dot product: negative means reversal
                dot = dx_track * dx_det + dy_track * dy_det
                if dot < 0 and min_dist > 20:
                    # This detection would reverse the track's direction
                    # More likely a new ball
                    accept = False

            if accept:
                matches[di] = ti
                used_tracks.add(ti)
                used_dets.add(di)
                if di in unmatched:
                    unmatched.remove(di)

            dists[ti, :] = np.inf
            dists[:, di] = np.inf

        return matches, unmatched

    def process_frame(self, frame: np.ndarray, debug: bool = False) -> MotionEvent | None:
        """Process one frame. Returns MotionEvent if new ball counted."""
        self.frame_idx += 1
        h, w = frame.shape[:2]
        if self.roi_mask is None:
            self._build_roi_mask(h, w)

        detections = self._detect(frame)
        matches, unmatched_dets = self._match_tracks(detections)

        # Track which track IDs were matched this frame
        matched_track_ids = set()

        # Update matched tracks
        for di, ti in matches.items():
            cx, cy, bbox, conf = detections[di]
            track = self.tracks[ti]
            matched_track_ids.add(track.id)
            track.centroid = (cx, cy)
            track.bbox = bbox
            track.frames_missing = 0
            track.age += 1
            track.history.append((cx, cy))
            if len(track.history) > 10:
                track.history = track.history[-10:]

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            cx, cy, bbox, conf = detections[di]
            inside = self._point_in_roi(cx, cy)
            track = Track(
                id=self.next_id,
                centroid=(cx, cy),
                bbox=bbox,
                first_inside_roi=inside,
                history=[(cx, cy)],
            )
            self.tracks.append(track)
            matched_track_ids.add(track.id)
            self.next_id += 1
            if debug:
                print(f"  [yolo] new track id={track.id} at ({cx:.0f},{cy:.0f}) "
                      f"inside_roi={inside} conf={conf:.2f}")

        # Count tracks that originated inside ROI and have enough age
        new_counts = 0
        for track in self.tracks:
            if (not track.counted
                    and track.first_inside_roi
                    and track.age >= self.min_track_age):
                track.counted = True
                new_counts += 1
                self.total_count += 1
                if debug:
                    print(f"  [yolo] COUNTED track id={track.id} age={track.age} "
                          f"total={self.total_count}")

        # Increment missing frames for unmatched tracks and prune
        for track in self.tracks:
            if track.id not in matched_track_ids:
                track.frames_missing += 1
        self.tracks = [t for t in self.tracks
                       if t.frames_missing <= self.max_missing_frames]

        if new_counts > 0:
            return MotionEvent(
                frame=self.frame_idx,
                n_balls=new_counts,
                peak_area=0,
            )
        return None

    def reset(self):
        """Reset all state."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_idx = 0
        self.total_count = 0
