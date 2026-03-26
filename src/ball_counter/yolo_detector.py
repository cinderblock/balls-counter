"""YOLO-based ball detector with simple tracking and outlet origin filtering.

Detects balls per-frame using YOLOv8, tracks them across frames with a simple
nearest-centroid tracker, and only counts balls whose first detection was inside
the outlet polygon (ROI).
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


class YOLOBallDetector:
    """Detect and track balls using YOLO + simple nearest-centroid tracker.

    Only counts balls that first appear inside the ROI polygon (outlet mode).
    """

    def __init__(
        self,
        model_path: str | Path,
        roi_points: list[list[int]],
        *,
        conf_threshold: float = 0.3,
        max_track_distance: float = 40.0,
        max_missing_frames: int = 5,
        min_track_age: int = 2,
        device: str = "0",
    ):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_track_distance = max_track_distance
        self.max_missing_frames = max_missing_frames
        self.min_track_age = min_track_age

        # ROI polygon for origin filtering
        self.roi_polygon = np.array(roi_points, dtype=np.int32)
        self.roi_mask = None  # lazily built on first frame

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
                                      verbose=False, imgsz=160)
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
        """Match detections to existing tracks using nearest centroid."""
        if not self.tracks or not detections:
            return {}, list(range(len(detections)))

        # Cost matrix
        track_centroids = np.array([(t.centroid[0], t.centroid[1]) for t in self.tracks])
        det_centroids = np.array([(d[0], d[1]) for d in detections])
        dists = np.linalg.norm(track_centroids[:, None] - det_centroids[None, :], axis=2)

        matches = {}  # det_idx -> track_idx
        unmatched = list(range(len(detections)))

        # Greedy matching
        while True:
            if dists.size == 0:
                break
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            ti, di = int(min_idx[0]), int(min_idx[1])
            if dists[ti, di] > self.max_track_distance:
                break
            matches[di] = ti
            if di in unmatched:
                unmatched.remove(di)
            dists[ti, :] = np.inf
            dists[:, di] = np.inf

        return matches, unmatched

    def process_frame(self, frame: np.ndarray) -> MotionEvent | None:
        """Process one frame. Returns MotionEvent if new ball counted."""
        self.frame_idx += 1
        h, w = frame.shape[:2]
        if self.roi_mask is None:
            self._build_roi_mask(h, w)

        detections = self._detect(frame)
        matches, unmatched_dets = self._match_tracks(detections)

        new_counts = 0

        # Update matched tracks
        for di, ti in matches.items():
            cx, cy, bbox, conf = detections[di]
            track = self.tracks[ti]
            track.centroid = (cx, cy)
            track.bbox = bbox
            track.frames_missing = 0
            track.age += 1

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            cx, cy, bbox, conf = detections[di]
            inside = self._point_in_roi(cx, cy)
            track = Track(
                id=self.next_id,
                centroid=(cx, cy),
                bbox=bbox,
                first_inside_roi=inside,
            )
            self.tracks.append(track)
            self.next_id += 1

        # Count tracks that originated inside ROI and have enough age
        for track in self.tracks:
            if (not track.counted
                    and track.first_inside_roi
                    and track.age >= self.min_track_age):
                track.counted = True
                new_counts += 1
                self.total_count += 1

        # Increment missing frames and prune dead tracks
        for track in self.tracks:
            if track.id not in {self.tracks[ti].id for ti in matches.values()}:
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
