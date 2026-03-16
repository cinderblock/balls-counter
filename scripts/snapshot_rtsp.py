"""Grab a snapshot from an RTSP stream."""

import os
import sys
import cv2

url = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else "snapshot.png"

print(f"Connecting to {url}...")
if "rtsp" in url.lower():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"Cannot open {url}", file=sys.stderr)
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Stream: {w}x{h} @ {fps}fps")

# Skip warmup frames
for _ in range(150):
    ret, frame = cap.read()
    if not ret:
        print("Lost stream during warmup", file=sys.stderr)
        sys.exit(1)

cv2.imwrite(output, frame)
print(f"Saved to {output}")
cap.release()
