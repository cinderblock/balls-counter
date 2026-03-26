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


class AlignmentTracker:
    """Track camera alignment using AprilTag markers.

    On startup, records reference marker positions. On each update,
    computes the affine transform to correct for drift.
    """

    def __init__(self, expected_ids: list[int] | None = None):
        self.expected_ids = set(expected_ids) if expected_ids else None
        self.reference: dict[int, np.ndarray] = {}  # id -> center (x, y)
        self.current: dict[int, np.ndarray] = {}
        self.transform: np.ndarray | None = None  # 2x3 affine matrix
        self.offset: tuple[float, float] = (0.0, 0.0)  # simple dx, dy
        self._initialized = False

    def update(self, frame: np.ndarray) -> bool:
        """Detect markers and update alignment.

        Returns True if alignment was updated successfully.
        """
        tags = detect_apriltags(frame)
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

    def status_str(self) -> str:
        if not self._initialized:
            return "no markers detected"
        n = len(self.current)
        dx, dy = self.offset
        return f"{n} markers, drift=({dx:+.1f}, {dy:+.1f})px"
