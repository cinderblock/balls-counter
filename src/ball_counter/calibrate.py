"""Interactive HSV threshold calibration tool.

Opens a frame from the video source with trackbars to adjust HSV thresholds
in real-time. Can update an existing config file with the tuned values.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def nothing(_):
    pass


def main():
    parser = argparse.ArgumentParser(description="Calibrate HSV thresholds for ball detection")
    parser.add_argument("source", help="Video source: file path or RTSP URL")
    parser.add_argument(
        "--config",
        help="Config file to update with calibrated values",
    )
    parser.add_argument(
        "--stream",
        help="Stream name in config to update (required with --config)",
    )
    parser.add_argument(
        "--hsv-low",
        type=int,
        nargs=3,
        default=[20, 100, 100],
        metavar=("H", "S", "V"),
        help="Starting HSV low values (default: 20 100 100)",
    )
    parser.add_argument(
        "--hsv-high",
        type=int,
        nargs=3,
        default=[35, 255, 255],
        metavar=("H", "S", "V"),
        help="Starting HSV high values (default: 35 255 255)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: cannot open video source: {args.source}", file=sys.stderr)
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read from video source", file=sys.stderr)
        sys.exit(1)

    cv2.namedWindow("Calibrate")
    cv2.createTrackbar("H Low", "Calibrate", args.hsv_low[0], 179, nothing)
    cv2.createTrackbar("S Low", "Calibrate", args.hsv_low[1], 255, nothing)
    cv2.createTrackbar("V Low", "Calibrate", args.hsv_low[2], 255, nothing)
    cv2.createTrackbar("H High", "Calibrate", args.hsv_high[0], 179, nothing)
    cv2.createTrackbar("S High", "Calibrate", args.hsv_high[1], 255, nothing)
    cv2.createTrackbar("V High", "Calibrate", args.hsv_high[2], 255, nothing)
    cv2.createTrackbar("Min Area", "Calibrate", 500, 5000, nothing)
    cv2.createTrackbar("Circularity", "Calibrate", 50, 100, nothing)

    print("Adjust trackbars to isolate the yellow balls.")
    print("  'n' = next frame, 'q' = done")

    while True:
        h_lo = cv2.getTrackbarPos("H Low", "Calibrate")
        s_lo = cv2.getTrackbarPos("S Low", "Calibrate")
        v_lo = cv2.getTrackbarPos("V Low", "Calibrate")
        h_hi = cv2.getTrackbarPos("H High", "Calibrate")
        s_hi = cv2.getTrackbarPos("S High", "Calibrate")
        v_hi = cv2.getTrackbarPos("V High", "Calibrate")
        min_area = cv2.getTrackbarPos("Min Area", "Calibrate")
        circularity_pct = cv2.getTrackbarPos("Circularity", "Calibrate")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low = np.array([h_lo, s_lo, v_lo])
        high = np.array([h_hi, s_hi, v_hi])
        mask = cv2.inRange(hsv, low, high)

        # Apply morphological cleanup like detector does
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Show detections on the original frame
        display = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_circ = circularity_pct / 100.0
        ball_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circ = 4 * np.pi * area / (perimeter * perimeter)
            if circ < min_circ:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(display, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
            cv2.circle(display, (int(cx), int(cy)), 3, (0, 255, 0), -1)
            ball_count += 1

        cv2.putText(
            display,
            f"Detected: {ball_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Side by side: original with detections | mask
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([display, mask_bgr])

        max_width = 1920
        if combined.shape[1] > max_width:
            scale = max_width / combined.shape[1]
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow("Calibrate", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nhsv_low: [{h_lo}, {s_lo}, {v_lo}]")
    print(f"hsv_high: [{h_hi}, {s_hi}, {v_hi}]")
    print(f"min_area: {min_area}")
    print(f"min_circularity: {min_circ}")

    # Update config file if specified
    if args.config and args.stream:
        config_path = Path(args.config)
        with open(config_path) as f:
            data = json.load(f)

        for stream in data["streams"]:
            if stream["name"] == args.stream:
                stream["hsv_low"] = [h_lo, s_lo, v_lo]
                stream["hsv_high"] = [h_hi, s_hi, v_hi]
                stream["min_area"] = min_area
                stream["min_circularity"] = min_circ
                print(f"\nUpdated '{args.stream}' in {config_path}")
                break
        else:
            print(f"\nWarning: stream '{args.stream}' not found in config")

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
