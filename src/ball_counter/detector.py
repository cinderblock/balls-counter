"""Ball detection using HSV color thresholding + watershed separation."""

import cv2
import numpy as np

# Default HSV range for bright yellow balls
DEFAULT_HSV_LOW = (20, 100, 100)
DEFAULT_HSV_HIGH = (35, 255, 255)

# Default detection parameters
DEFAULT_MIN_AREA = 200
DEFAULT_MIN_CIRCULARITY = 0.3


def create_mask(
    frame: np.ndarray,
    hsv_low: tuple[int, int, int] = DEFAULT_HSV_LOW,
    hsv_high: tuple[int, int, int] = DEFAULT_HSV_HIGH,
) -> np.ndarray:
    """Create a binary mask isolating colored regions."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))

    # Light cleanup — small kernel to remove specks without merging balls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_balls(
    frame: np.ndarray,
    hsv_low: tuple[int, int, int] = DEFAULT_HSV_LOW,
    hsv_high: tuple[int, int, int] = DEFAULT_HSV_HIGH,
    min_area: int = DEFAULT_MIN_AREA,
    min_circularity: float = DEFAULT_MIN_CIRCULARITY,
) -> list[dict]:
    """
    Detect colored balls in a frame using watershed to separate touching balls.

    Returns a list of detections, each with:
        - center: (x, y) centroid
        - radius: approximate radius in pixels
        - area: contour area
    """
    mask = create_mask(frame, hsv_low, hsv_high)

    # Distance transform to find ball centers
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Threshold distance map to get sure foreground (ball centers)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find markers from sure foreground
    n_labels, markers = cv2.connectedComponents(sure_fg)

    # Watershed needs markers starting from 1, background = 0, unknown = 0
    # Increment all markers so background is 1
    markers = markers + 1
    # Mark unknown region (mask but not sure foreground) as 0
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown == 255] = 0

    # Watershed
    frame_bgr = frame.copy()
    markers = cv2.watershed(frame_bgr, markers)

    # Extract detections from watershed regions
    detections = []
    for label in range(2, n_labels + 1):  # skip background (1)
        region_mask = np.uint8(markers == label) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)

            detections.append(
                {
                    "center": (int(cx), int(cy)),
                    "radius": int(radius),
                    "area": area,
                }
            )

    return detections
