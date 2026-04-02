"""Test dual CuvidCropReader against the live RTSP stream."""
import sys
import time
sys.path.insert(0, "src")

from ball_counter.stream import CuvidCropReader

URL = "rtsp://10.255.0.2:7447/W1OqgGiim6LfiPtt"
red_crop = (501, 268, 647, 548)
blue_crop = (2086, 279, 2248, 570)

red = CuvidCropReader(URL, red_crop)
blue = CuvidCropReader(URL, blue_crop)
print(f"Red:  {red.crop_w}x{red.crop_h}")
print(f"Blue: {blue.crop_w}x{blue.crop_h}")

red.open()
blue.open()

print("Reading 120 frames from each...")
t0 = time.monotonic()
for i in range(120):
    r = red.read()
    b = blue.read()
    if r is None or b is None:
        print(f"  Frame {i}: FAILED (red={r is not None}, blue={b is not None})")
        break
    if i == 0:
        print(f"  Red: {r.shape}, Blue: {b.shape}")
dt = time.monotonic() - t0
print(f"  {i+1} frames in {dt:.2f}s ({(i+1)/dt:.1f} fps)")

red.release()
blue.release()
print("Done")
