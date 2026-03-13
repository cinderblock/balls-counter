"""Grab a snapshot from RTSP stream, with TCP transport and longer warmup."""

import sys
import os
import cv2

url = "rtsp://10.255.9.97:8554/the-field"
print(f"Connecting to {url}...")

# Force TCP transport for more reliable RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: cannot open stream", file=sys.stderr)
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Stream: {w}x{h}")

# Skip 150 frames to ensure we hit a keyframe
for i in range(150):
    ret, frame = cap.read()
    if not ret:
        print(f"Lost stream at frame {i}")
        sys.exit(1)
    if i % 50 == 0:
        print(f"  frame {i}...")

print("Saving...")
cv2.imwrite("samples/field_snapshot.png", frame)
print("Saved samples/field_snapshot.png")
cap.release()
