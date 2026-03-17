"""Visual debug: see which balls the exit line catches/misses on sample2.

Single panel showing the goal area with:
- Red line + band overlay
- Score count near the line
- Green flash when a score is detected
- Yellow GT marker when ground truth says a score happened
- Signal strength bar

Controls: Space=pause, D/A=step, W/S=±10, Q=quit, 1-4=speed

Pass --headless to skip the GUI and just print results.
"""

import argparse
import json
import sys

sys.path.insert(0, "src")

import cv2
import numpy as np
from ball_counter.counter import MotionCounter

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Skip GUI, print results only")
args = parser.parse_args()

VIDEO = "samples/blue/sample2-full.mp4"
GEOM_FILE = "samples/blue/sample2-full.geometry.json"
SCORES_FILE = "samples/blue/sample2-full.scores.json"

with open(GEOM_FILE) as f:
    geometries = json.load(f)
with open(SCORES_FILE) as f:
    gt_scores = json.load(f)

cap = cv2.VideoCapture(VIDEO)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Find line geometry
line_pts = None
poly_pts = None
for g in geometries:
    if g["type"] == "line":
        line_pts = g["points"]
    elif g["type"] == "polygon":
        poly_pts = g["points"]

if not line_pts:
    print("No line geometry found")
    sys.exit(1)

BALL_AREA = 500
BAND_WIDTH = 15


def make_counter():
    return MotionCounter(
        frame_shape=(h, w),
        line=tuple(line_pts),
        ball_area=BALL_AREA,
        band_width=BAND_WIDTH,
        fall_ratio=0.7,
        min_rise=50,
    )


counter = make_counter()

# Crop around the goal area
all_x = [p[0] for p in line_pts]
all_y = [p[1] for p in line_pts]
if poly_pts:
    all_x += [p[0] for p in poly_pts]
    all_y += [p[1] for p in poly_pts]
pad = 250
crop_x1 = max(0, min(all_x) - pad)
crop_y1 = max(0, min(all_y) - pad)
crop_x2 = min(w, max(all_x) + pad)
crop_y2 = min(h, max(all_y) + pad)

cw = crop_x2 - crop_x1
ch = crop_y2 - crop_y1
display_scale = 2
dw = cw * display_scale
dh = ch * display_scale

WINDOW = "Exit Line Debug"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, min(dw, 1920), min(dh, 1080))

paused = False
delay = int(1000 / fps)
speeds = {ord("1"): 1.0, ord("2"): 0.5, ord("3"): 0.25, ord("4"): 0.1}
speed = 1.0
blip_frames = 0
gt_flash = 0


def process_and_draw(frame):
    """Process frame, draw overlays, return cropped display."""
    global blip_frames, gt_flash

    event = counter.process_frame(frame)
    if event:
        blip_frames = 20

    fi = counter.frame_idx - 1

    # GT flash
    if fi in gt_scores:
        gt_flash = 20

    display = frame.copy()

    # Draw the detection band (subtle red tint)
    overlay = display.copy()
    cv2.fillPoly(overlay, [counter.draw_pts], (0, 0, 120))
    cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

    # Draw the line
    p1, p2 = counter.line
    cv2.line(display, tuple(p1), tuple(p2), (0, 0, 255), 3)

    # Draw polygon outline if present
    if poly_pts:
        cv2.polylines(display, [np.array(poly_pts, dtype=np.int32)], True, (0, 180, 0), 2)

    # Score flash — green glow on the band
    if blip_frames > 0:
        glow = np.zeros_like(display)
        cv2.fillPoly(glow, [counter.draw_pts], (0, 255, 0))
        alpha = blip_frames / 20
        display = cv2.addWeighted(display, 1.0, glow, alpha * 0.4, 0)
        blip_frames -= 1

    # GT flash — yellow border flash
    if gt_flash > 0:
        alpha = gt_flash / 20
        # Yellow circle near top-left of crop
        cv2.circle(display, (crop_x1 + 40, crop_y1 + 40), 25, (0, 255, 255), -1)
        cv2.putText(display, "GT", (crop_x1 + 22, crop_y1 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        gt_flash -= 1

    # Score count near the line (midpoint, offset above)
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2 - 40
    count_text = str(counter.count)
    # Background box
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(display, (mid_x - tw // 2 - 10, mid_y - th - 10),
                  (mid_x + tw // 2 + 10, mid_y + 10), (0, 0, 0), -1)
    cv2.putText(display, count_text, (mid_x - tw // 2, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    # Signal bar on the right side
    bar_x = crop_x2 - 50
    bar_h = crop_y2 - crop_y1 - 40
    bar_y_top = crop_y1 + 20
    signal = counter.signal
    max_signal = max(5000, signal)
    fill_h = int(bar_h * min(1.0, signal / max_signal))
    cv2.rectangle(display, (bar_x, bar_y_top), (bar_x + 30, bar_y_top + bar_h), (60, 60, 60), -1)
    bar_color = (0, 255, 0) if counter.rising else (0, 140, 0)
    cv2.rectangle(display, (bar_x, bar_y_top + bar_h - fill_h),
                  (bar_x + 30, bar_y_top + bar_h), bar_color, -1)
    cv2.putText(display, f"{signal}", (bar_x - 10, bar_y_top + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Frame info at bottom
    info_y = crop_y2 - 15
    cv2.rectangle(display, (crop_x1, info_y - 18), (crop_x2, crop_y2), (0, 0, 0), -1)
    cv2.putText(display, f"Frame {fi}/{total}  Speed:{speed:.1f}x  Space=pause D/A=step W/S=+/-10 Q=quit",
                (crop_x1 + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Crop
    cropped = display[crop_y1:crop_y2, crop_x1:crop_x2]
    return cv2.resize(cropped, (dw, dh))


def reset_to(target_frame):
    """Rewind and replay to target_frame."""
    global counter, blip_frames, gt_flash
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    counter = make_counter()
    blip_frames = 0
    gt_flash = 0
    display = None
    frame = None
    for _ in range(target_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        display = process_and_draw(frame)
    return frame, display


# Determine interesting frame ranges from GT (skip blank periods in playback)
# Add padding around each GT burst
gt_ranges = []
if gt_scores:
    start = gt_scores[0] - 30
    end = gt_scores[0] + 30
    for g in gt_scores[1:]:
        if g - end > 60:
            gt_ranges.append((max(0, start), end))
            start = g - 30
        end = g + 30
    gt_ranges.append((max(0, start), min(total, end)))


def is_interesting_frame(fi):
    """Is this frame near a GT event?"""
    for s, e in gt_ranges:
        if s <= fi <= e:
            return True
    return False


if args.headless:
    # Process all frames without GUI
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        counter.process_frame(frame)
    cap.release()
else:
    frame = None
    display = None
    skip_label_frames = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                paused = True
                continue

            fi = counter.frame_idx  # frame about to be processed

            # Skip blank periods: process silently until next interesting frame
            if not is_interesting_frame(fi):
                counter.process_frame(frame)
                # Check if next interesting range is coming
                next_interesting = None
                for s, e in gt_ranges:
                    if s > fi:
                        next_interesting = s
                        break
                if next_interesting and next_interesting - fi > 5:
                    # Fast-forward: process frames without display
                    skip_to = next_interesting - 5
                    while counter.frame_idx < skip_to:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        counter.process_frame(frame)
                    skip_label_frames = 30
                continue

            display = process_and_draw(frame)

            # Show skip indicator
            if skip_label_frames > 0:
                cv2.putText(display, ">> SKIPPED BLANK >>", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                skip_label_frames -= 1

            cv2.imshow(WINDOW, display)

        actual_delay = max(1, int(delay / speed)) if not paused else 50
        key = cv2.waitKey(actual_delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d") and paused:
            ret, frame = cap.read()
            if ret:
                display = process_and_draw(frame)
                cv2.imshow(WINDOW, display)
        elif key == ord("a") and paused:
            target = max(0, counter.frame_idx - 2)
            frame, display = reset_to(target)
            if display is not None:
                cv2.imshow(WINDOW, display)
        elif key == ord("w") and paused:
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                display = process_and_draw(frame)
            if display is not None:
                cv2.imshow(WINDOW, display)
        elif key == ord("s") and paused:
            target = max(0, counter.frame_idx - 11)
            frame, display = reset_to(target)
            if display is not None:
                cv2.imshow(WINDOW, display)
        elif key in speeds:
            speed = speeds[key]

    cap.release()
    cv2.destroyAllWindows()

# Summary
print(f"\nTotal count: {counter.count}")
print(f"Events: {len(counter.events)}")
print(f"Ground truth: {len(gt_scores)}")

matched_gt = set()
matched_ev = set()
for i, ev in enumerate(counter.events):
    for j, gt in enumerate(gt_scores):
        if j not in matched_gt and abs(ev.frame - gt) <= 15:
            matched_gt.add(j)
            matched_ev.add(i)
            break

print(f"Matched: {len(matched_gt)}/{len(gt_scores)}")
print(f"FP: {len(counter.events) - len(matched_ev)}")
missed = [gt_scores[j] for j in range(len(gt_scores)) if j not in matched_gt]
if missed:
    print(f"Missed GT frames: {missed}")
