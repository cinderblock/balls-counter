"""Grab a single frame from a video file and save it."""

import sys
import cv2

video = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else "snapshot.png"
frame_num = int(sys.argv[3]) if len(sys.argv) > 3 else 50

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
ret, frame = cap.read()
if ret:
    cv2.imwrite(output, frame)
    print(f"Saved frame {frame_num} to {output} ({frame.shape[1]}x{frame.shape[0]})")
else:
    print("Failed to read frame")
cap.release()
