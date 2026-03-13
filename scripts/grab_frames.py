import cv2
import sys

cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps:.1f} fps, {total} frames, {total/fps:.1f}s")

# Grab frames at 0%, 25%, 50%, 75%, and 90%
for pct in [0, 25, 50, 75, 90]:
    frame_num = int(total * pct / 100)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        path = f"frame_{pct:02d}.png"
        cv2.imwrite(path, frame)
        print(f"Saved {path} (frame {frame_num})")

cap.release()
