"""AprilTag detection and camera alignment correction.

Detects AprilTag 36h11 markers on the full RTSP frame and computes
an affine correction to compensate for camera drift (thermal, vibration).

Reference marker positions are recorded on first successful detection.
Subsequent frames compute the transform needed to align current positions
back to reference, which is applied to ROI polygons and crop coordinates.
"""

import cv2
import cv2.aruco
import numpy as np


# AprilTag 36h11 dictionary
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _make_detector() -> cv2.aruco.ArucoDetector:
    """Create an AprilTag detector tuned for small/distant markers."""
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 40
    params.adaptiveThreshWinSizeStep = 3
    params.minMarkerPerimeterRate = 0.005
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 1
    return cv2.aruco.ArucoDetector(_DICT, params)


def detect_apriltags(frame: np.ndarray) -> dict[int, np.ndarray]:
    """Detect AprilTag 36h11 markers in a frame.

    Returns dict of {marker_id: center_point} where center_point is (x, y).
    """
    detector = _make_detector()
    corners, ids, _ = detector.detectMarkers(frame)

    result = {}
    if ids is not None:
        for i, mid in enumerate(ids):
            center = corners[i][0].mean(axis=0)
            result[int(mid[0])] = center
    return result


def detect_apriltags_full(frame: np.ndarray) -> dict[int, dict]:
    """Detect AprilTags with full corner info.

    Returns dict of {marker_id: {"center": (x,y), "corners": 4x2 array}}.
    """
    detector = _make_detector()
    corners, ids, _ = detector.detectMarkers(frame)

    result = {}
    if ids is not None:
        for i, mid in enumerate(ids):
            c = corners[i][0]
            result[int(mid[0])] = {
                "center": c.mean(axis=0),
                "corners": c,
            }
    return result


# Known marker-to-goal mapping.
# Side 1 (visible from the primary camera angle):
#   Red goal: IDs 3 (left), 4 (right) — red labels on reference drawing
#   Blue goal: IDs 19 (left), 20 (right) — blue labels on reference drawing
# Side 2 (visible from the opposite camera angle):
#   Red goal: IDs 2, 11
#   Blue goal: IDs 21, 24
# The markers are side-by-side on the panel above each goal outlet.
# Physical dimensions from reference: 5.78in from left edge to left marker,
# 19.78in total panel width, 41in from outlet to marker panel.
GOAL_MARKER_IDS = {
    "red-goal": [2, 3, 4, 11],   # all IDs that may appear on red goal
    "blue-goal": [19, 20, 21, 24],  # all IDs that may appear on blue goal
}


class AlignmentTracker:
    """Track camera alignment using AprilTag markers.

    On startup, records reference marker positions. On each update,
    computes the affine transform to correct for drift.

    Can track per-goal alignment when goal_marker_ids is provided.
    """

    def __init__(self, expected_ids: list[int] | None = None,
                 goal_marker_ids: dict[str, list[int]] | None = None):
        """
        expected_ids: if set, only use these marker IDs for global alignment.
        goal_marker_ids: {goal_name: [marker_ids]} for per-goal alignment.
        """
        self.expected_ids = set(expected_ids) if expected_ids else None
        self.goal_marker_ids = goal_marker_ids or {}
        # Per-goal offsets
        self.goal_offsets: dict[str, tuple[float, float]] = {}
        self.goal_transforms: dict[str, np.ndarray] = {}
        self.reference: dict[int, np.ndarray] = {}  # id -> center (x, y)
        self.current: dict[int, np.ndarray] = {}
        self.transform: np.ndarray | None = None  # 2x3 affine matrix
        self.offset: tuple[float, float] = (0.0, 0.0)  # simple dx, dy
        self._initialized = False

    def update(self, frame: np.ndarray,
              search_regions: list[tuple[int, int, int, int]] | None = None) -> bool:
        """Detect markers and update alignment.

        search_regions: optional list of (x1, y1, x2, y2) rects to search in
        addition to the full frame. Helps find small markers near goals.

        Returns True if alignment was updated successfully.
        """
        # Detect on full frame
        tags = detect_apriltags(frame)

        # Also search in expanded regions around goals (markers are small/angled)
        if search_regions:
            h, w = frame.shape[:2]
            for rx1, ry1, rx2, ry2 in search_regions:
                # Expand by 50% to catch markers near the crop edge
                margin = max(rx2 - rx1, ry2 - ry1) // 2
                ex1 = max(0, rx1 - margin)
                ey1 = max(0, ry1 - margin)
                ex2 = min(w, rx2 + margin)
                ey2 = min(h, ry2 + margin)
                region = frame[ey1:ey2, ex1:ex2]
                region_tags = detect_apriltags(region)
                for tid, center in region_tags.items():
                    if tid not in tags:
                        # Convert back to full-frame coords
                        tags[tid] = np.array([center[0] + ex1, center[1] + ey1])

        if not tags:
            return False

        # Filter to expected IDs if specified
        if self.expected_ids:
            tags = {k: v for k, v in tags.items() if k in self.expected_ids}

        if len(tags) < 2:
            return False

        self.current = tags

        if not self._initialized:
            self.reference = dict(tags)
            self._initialized = True
            print(f"apriltag - reference set: {list(tags.keys())} "
                  f"at {', '.join(f'{k}=({v[0]:.0f},{v[1]:.0f})' for k, v in tags.items())}")
            return True

        # Find common markers between reference and current
        common = set(self.reference.keys()) & set(tags.keys())
        if len(common) < 2:
            return False

        # Compute transform from reference to current
        src = np.array([self.reference[k] for k in sorted(common)], dtype=np.float32)
        dst = np.array([tags[k] for k in sorted(common)], dtype=np.float32)

        if len(common) >= 3:
            # Full affine (translation + rotation + scale)
            self.transform, _ = cv2.estimateAffine2D(src, dst)
        else:
            # 2 points: estimate translation + rotation
            self.transform, _ = cv2.estimateAffinePartial2D(src, dst)

        # Simple offset (average displacement)
        displacements = dst - src
        self.offset = (float(displacements[:, 0].mean()),
                       float(displacements[:, 1].mean()))

        # Per-goal alignment using only that goal's markers
        for goal_name, marker_ids in self.goal_marker_ids.items():
            goal_common = set(marker_ids) & set(self.reference.keys()) & set(tags.keys())
            if len(goal_common) >= 1:
                g_src = np.array([self.reference[k] for k in sorted(goal_common)], dtype=np.float32)
                g_dst = np.array([tags[k] for k in sorted(goal_common)], dtype=np.float32)
                disp = g_dst - g_src
                self.goal_offsets[goal_name] = (
                    float(disp[:, 0].mean()),
                    float(disp[:, 1].mean()),
                )
                if len(goal_common) >= 2:
                    self.goal_transforms[goal_name], _ = cv2.estimateAffinePartial2D(g_src, g_dst)

        return True

    def correct_points(self, points: list[list[int]]) -> list[list[int]]:
        """Apply alignment correction to a set of points.

        Maps points from reference frame coordinates to current frame coordinates.
        """
        if self.transform is None:
            return points

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        corrected = cv2.transform(pts, self.transform)
        return corrected.reshape(-1, 2).astype(int).tolist()

    def correct_point(self, x: float, y: float) -> tuple[float, float]:
        """Apply alignment correction to a single point."""
        if self.transform is None:
            return (x, y)
        pt = np.array([[[x, y]]], dtype=np.float32)
        corrected = cv2.transform(pt, self.transform)
        return (float(corrected[0, 0, 0]), float(corrected[0, 0, 1]))

    @property
    def drift_px(self) -> float:
        """Current drift magnitude in pixels."""
        return (self.offset[0] ** 2 + self.offset[1] ** 2) ** 0.5

    @property
    def initialized(self) -> bool:
        return self._initialized

    def goal_offset(self, goal_name: str) -> tuple[float, float] | None:
        """Get per-goal alignment offset, falling back to global offset."""
        if goal_name in self.goal_offsets:
            return self.goal_offsets[goal_name]
        if self._initialized:
            return self.offset
        return None

    def status_str(self) -> str:
        if not self._initialized:
            return "no markers detected"
        n = len(self.current)
        dx, dy = self.offset
        parts = [f"{n} markers, drift=({dx:+.1f}, {dy:+.1f})px"]
        for gn, (gdx, gdy) in self.goal_offsets.items():
            parts.append(f"{gn}=({gdx:+.1f},{gdy:+.1f})")
        return ", ".join(parts)
