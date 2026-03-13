import cv2
import sys

cap = cv2.VideoCapture(sys.argv[1])
ret, frame = cap.read()
cap.release()
if ret:
    cv2.imwrite("sample_frame.png", frame)
    print(f"Saved sample_frame.png ({frame.shape[1]}x{frame.shape[0]})")
else:
    print("Failed to read frame")
