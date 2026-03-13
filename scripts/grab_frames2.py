"""Grab frames from sample2, cropped to left half."""

import cv2

cap = cv2.VideoCapture("sample2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps:.1f} fps, {total} frames, {total/fps:.1f}s")

for pct in [0, 25, 50, 75, 90]:
    frame_num = int(total * pct / 100)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        frame = frame[:, : w // 2]
        path = f"red_frame_{pct:02d}.png"
        cv2.imwrite(path, frame)
        print(f"Saved {path} (frame {frame_num}, cropped to {frame.shape[1]}x{frame.shape[0]})")

cap.release()
