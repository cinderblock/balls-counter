"""Test RTSP connection to goal camera."""

import os
import sys
import cv2

urls = [
    # Port 7447 (plain RTSP)
    "rtsp://10.255.0.2:7447/IJ9qWeF1mnsM4C7X",
    # Original (SRTP)
    "rtsps://10.255.0.2:7441/IJ9qWeF1mnsM4C7X?enableSrtp",
    # Plain RTSP on 7441
    "rtsp://10.255.0.2:7441/IJ9qWeF1mnsM4C7X",
]

for url in urls:
    print(f"\nTrying: {url}")
    if url.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if ret:
            print(f"  SUCCESS: {w}x{h} @ {fps}fps")
            cv2.imwrite("samples/new-angle/rtsp_snapshot.png", frame)
            print(f"  Saved snapshot to samples/new-angle/rtsp_snapshot.png")
            cap.release()
            sys.exit(0)
        else:
            print(f"  Opened but cannot read frame")
    else:
        print(f"  Cannot open")
    cap.release()

print("\nAll URLs failed")
sys.exit(1)
