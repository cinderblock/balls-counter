"""Grab wider frames from red full video to see the complete drop path."""

import cv2

cap = cv2.VideoCapture("samples/red/sample1-full.mp4")

for idx in [345, 370, 400, 430, 460, 500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        frame = frame[:, : w // 2]
        path = f"samples/red/wide_f{idx:04d}.png"
        cv2.imwrite(path, frame)
        print(f"Saved {path}")

cap.release()
