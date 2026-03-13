"""Visual test: replay video with line, detections, and tracking overlaid."""

import argparse
import json
import sys

import cv2

sys.path.insert(0, "src")
from ball_counter.counter import LineCrossingCounter
from ball_counter.detector import detect_balls
from ball_counter.tracker import CentroidTracker

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


def process_up_to(target_frame):
    """Replay from frame 0 to target_frame, returning fresh state."""
    line = LineCrossingCounter(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
    trk = CentroidTracker(max_disappeared=30)
    c = cv2.VideoCapture(VIDEO_PATH)

    frame = None
    dets = []
    for i in range(target_frame + 1):
        ret, frame = c.read()
        if not ret:
            break
        dets = detect_balls(frame)
        centroids = [d["center"] for d in dets]
        objects = trk.update(centroids)
        for object_id, centroid in objects.items():
            line.update_object(object_id, int(centroid[0]), int(centroid[1]))
        line.cleanup(set(objects.keys()))

    c.release()
    return line, trk, frame, dets


# Initial state — if starting mid-video, replay from 0 to build correct state
if args.start > 0:
    print(f"Replaying frames 0-{args.start} to build state...")
    counting_line, tracker, _, _ = process_up_to(args.start - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx_init = args.start
else:
    counting_line = LineCrossingCounter(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
    tracker = CentroidTracker(max_disappeared=30)
    frame_idx_init = 0

cv2.namedWindow("Line Test", cv2.WINDOW_NORMAL)
# Get video dimensions and size window to 6x
probe = cv2.VideoCapture(VIDEO_PATH)
vw = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
vh = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
probe.release()
cv2.resizeWindow("Line Test", vw * 3, vh * 3)
frame_idx = frame_idx_init
paused = False
speed = 1.0
blips: list[list[int]] = []
BLIP_DURATION = 20
frame = None
detections = []

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_balls(frame)
        centroids = [d["center"] for d in detections]
        objects = tracker.update(centroids)

        for object_id, centroid in objects.items():
            event = counting_line.update_object(object_id, int(centroid[0]), int(centroid[1]))
            if event == "in":
                blips.append([int(centroid[0]), int(centroid[1]), BLIP_DURATION])
            elif event == "out":
                blips.append([int(centroid[0]), int(centroid[1]), BLIP_DURATION])
        counting_line.cleanup(set(objects.keys()))
        frame_idx += 1

    if frame is None:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = 0

    display = frame.copy()

    # Draw line
    counting_line.draw(display)

    # Draw tracked objects with IDs
    for object_id, centroid in tracker.objects.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(display, (cx, cy), 8, (0, 255, 0), 2)
        cv2.putText(display, str(object_id), (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw detection circles (unfilled, yellow)
    for d in detections:
        cx, cy = d["center"]
        cv2.circle(display, (cx, cy), d["radius"], (0, 255, 255), 1)

    # Draw score blips (expanding semi-transparent circle that fades out)
    active_blips = []
    for blip in blips:
        bx, by, remaining = blip
        t = 1.0 - (remaining / BLIP_DURATION)
        radius = int(15 + t * 30)
        alpha = remaining / BLIP_DURATION

        overlay = display.copy()
        cv2.circle(overlay, (bx, by), radius, (0, 255, 0), 2)
        cv2.circle(overlay, (bx, by), int(radius * 0.4), (0, 255, 0), -1)
        cv2.addWeighted(overlay, alpha * 0.6, display, 1.0 - alpha * 0.6, 0, display)

        blip[2] -= 1
        if blip[2] > 0:
            active_blips.append(blip)
    blips = active_blips

    # Stats with black background
    stats_text = f"Frame {frame_idx}  IN:{counting_line.count_in} OUT:{counting_line.count_out} NET:{counting_line.count_in - counting_line.count_out}"
    speed_label = f"{speed:.1f}x" if speed != 1.0 else "1x"
    state = f"{'PAUSED' if paused else speed_label}"
    cv2.rectangle(display, (0, 0), (display.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(display, stats_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, state, (display.shape[1] - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255) if paused else (0, 255, 0), 2)

    cv2.imshow("Line Test", display)
    delay = max(1, int((1000 / fps) / speed)) if not paused else 50
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        paused = not paused
    elif key == ord("d") and paused:
        # Step forward one frame
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_balls(frame)
        centroids = [d["center"] for d in detections]
        objects = tracker.update(centroids)
        for object_id, centroid in objects.items():
            event = counting_line.update_object(object_id, int(centroid[0]), int(centroid[1]))
            if event == "in":
                blips.append([int(centroid[0]), int(centroid[1]), BLIP_DURATION])
            elif event == "out":
                blips.append([int(centroid[0]), int(centroid[1]), BLIP_DURATION])
        counting_line.cleanup(set(objects.keys()))
        frame_idx += 1
    elif key == ord("a") and paused:
        # Jump back 1 second — replay from start to rebuild state
        target = max(0, frame_idx - int(fps))
        print(f"Rewinding to frame {target} (replaying from 0)...")
        cap.release()
        counting_line, tracker, frame, detections = process_up_to(target)
        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target + 1)
        frame_idx = target
        blips.clear()
    elif key == ord(","):
        speed = max(0.1, speed / 2)
        print(f"Speed: {speed:.1f}x")
    elif key == ord("."):
        speed = min(8.0, speed * 2)
        print(f"Speed: {speed:.1f}x")

cap.release()
cv2.destroyAllWindows()
