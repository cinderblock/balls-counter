"""Visual test of the motion-based counter."""

import argparse
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ball_counter.detector import create_mask

with open("samples/red/sample1-goal.line.json") as f:
    line_data = json.load(f)
p1, p2 = line_data["line"]

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Frame number to start on")
args = parser.parse_args()

VIDEO_PATH = "samples/red/sample1-goal.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get dimensions
vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Band mask around the line
band_mask = np.zeros((vh, vw), dtype=np.uint8)
band_pts = np.array([
    [p1[0], p1[1] - 20], [p2[0], p2[1] - 20],
    [p2[0], p2[1] + 20], [p1[0], p1[1] + 20]
], dtype=np.int32)
cv2.fillPoly(band_mask, [band_pts], 255)

bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=60,
    varThreshold=50,
    detectShadows=False,
)

# If starting mid-video, warm up the background subtractor
if args.start > 0:
    print(f"Warming up background subtractor to frame {args.start}...")
    for i in range(args.start):
        ret, frame = cap.read()
        if not ret:
            break
        bg_sub.apply(frame)
        yellow = create_mask(frame)
        fg = bg_sub.apply(frame)
    print("Ready.")

cv2.namedWindow("Motion Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Motion Counter", vw * 3, vh * 3)

frame_idx = args.start
paused = False
speed = 1.0
count = 0
BALL_AREA = 900
prev_area = 0
rising = False
peak_val = 0
blips: list[list] = []  # [frame_scored, n_balls, frames_remaining]
BLIP_DURATION = 20
frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_sub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        yellow = create_mask(frame)
        moving_yellow = cv2.bitwise_and(yellow, fg_mask)
        moving_in_band = cv2.bitwise_and(moving_yellow, band_mask)
        area = cv2.countNonZero(moving_in_band)

        # Peak detection
        if area > prev_area and area > 100:
            rising = True
            peak_val = max(peak_val, area)
        elif rising and area < peak_val * 0.5:
            n_balls = max(1, round(peak_val / BALL_AREA))
            count += n_balls
            blips.append([count, n_balls, BLIP_DURATION])
            rising = False
            peak_val = 0

        prev_area = area
        frame_idx += 1

    if frame is None:
        continue

    display = frame.copy()

    # Draw the counting line
    cv2.line(display, tuple(p1), tuple(p2), (0, 0, 255), 2)

    # Draw the band (semi-transparent)
    band_overlay = display.copy()
    cv2.fillPoly(band_overlay, [band_pts], (0, 0, 180))
    cv2.addWeighted(band_overlay, 0.2, display, 0.8, 0, display)

    # Draw moving yellow in band (bright green overlay)
    if not paused:
        moving_bgr = cv2.cvtColor(moving_in_band, cv2.COLOR_GRAY2BGR)
        green_highlight = np.zeros_like(display)
        green_highlight[:, :, 1] = moving_in_band  # green channel
        cv2.addWeighted(green_highlight, 0.7, display, 1.0, 0, display)

    # Score blips
    active_blips = []
    for blip in blips:
        total_at_score, n, remaining = blip
        if remaining > 0:
            alpha = remaining / BLIP_DURATION
            # Flash the line green
            overlay = display.copy()
            cv2.line(overlay, tuple(p1), tuple(p2), (0, 255, 0), 4)
            cv2.addWeighted(overlay, alpha * 0.5, display, 1.0 - alpha * 0.5, 0, display)
            blip[2] -= 1
            active_blips.append(blip)
    blips = active_blips

    # Stats with black background
    cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(display, f"Frame {frame_idx}  Count: {count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    signal_text = f"Signal: {prev_area:5d}px  {'RISING' if rising else ''}"
    cv2.putText(display, signal_text,
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

    speed_label = f"{speed:.1f}x" if speed != 1.0 else "1x"
    state = "PAUSED" if paused else speed_label
    cv2.putText(display, state, (display.shape[1] - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 128, 255) if paused else (0, 255, 0), 2)

    cv2.imshow("Motion Counter", display)
    delay = max(1, int((1000 / fps) / speed)) if not paused else 50
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        paused = not paused
    elif key == ord("d") and paused:
        paused = False  # let one frame through
        # hack: unpause for one iteration
    elif key == ord(","):
        speed = max(0.1, speed / 2)
    elif key == ord("."):
        speed = min(8.0, speed * 2)

cap.release()
cv2.destroyAllWindows()
