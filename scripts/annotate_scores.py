"""Annotate scores on a slow-mo video, windowed on a goal region.

Usage:
  uv run python scripts/annotate_scores.py <video> --crop x1,y1,x2,y2 [--output scores.json]

Controls:
  Space     - play/pause
  D / Right - step forward 1 frame
  A / Left  - step back 1 frame
  W / Up    - step forward 10 frames
  S / Down  - step back 10 frames
  1-5       - playback speed (1=0.1x, 2=0.25x, 3=0.5x, 4=1x, 5=2x)
  Enter     - mark score at current frame
  Backspace - unmark nearest score (within 15 frames)
  Q         - save and quit
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SPEEDS = {
    ord("1"): (0.1, "0.1x"),
    ord("2"): (0.25, "0.25x"),
    ord("3"): (0.5, "0.5x"),
    ord("4"): (1.0, "1x"),
    ord("5"): (2.0, "2x"),
}


def main():
    parser = argparse.ArgumentParser(description="Annotate scores on a windowed video")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--crop", required=True, help="Crop region: x1,y1,x2,y2")
    parser.add_argument("--output", help="Output JSON path (default: <video>.scores.json)")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    args = parser.parse_args()

    x1, y1, x2, y2 = map(int, args.crop.split(","))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cw, ch = x2 - x1, y2 - y1

    out_path = args.output or str(Path(args.video).with_suffix(".scores.json"))

    # Load existing scores
    scores = []
    if Path(out_path).exists():
        with open(out_path) as f:
            scores = json.load(f)
        print(f"Loaded {len(scores)} existing scores from {out_path}")

    window = "Score annotator - Enter=mark Backspace=unmark Q=save+quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = min(1920 / cw, 1080 / ch, 4.0)
    cv2.resizeWindow(window, int(cw * scale), int(ch * scale))

    playing = False
    speed = 0.25
    speed_label = "0.25x"
    frame_idx = args.start

    def read_frame(idx):
        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame[y1:y2, x1:x2]

    def draw(frame, idx):
        display = frame.copy()
        h, w = display.shape[:2]

        # Highlight if this frame is scored
        is_scored = idx in scores
        if is_scored:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)

        # Score markers on timeline
        bar_y = h - 20
        cv2.rectangle(display, (0, bar_y), (w, h), (0, 0, 0), -1)
        for s in scores:
            sx = int(s / total_frames * w)
            cv2.line(display, (sx, bar_y), (sx, h), (0, 255, 0), 1)
        # Current position
        cx = int(idx / total_frames * w)
        cv2.line(display, (cx, bar_y), (cx, h), (0, 0, 255), 2)

        # Header
        cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
        status = "PLAY" if playing else "PAUSE"
        cv2.putText(display, f"Frame {idx}/{total_frames}  {speed_label}  {status}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Scores: {len(scores)}  {'* SCORED *' if is_scored else ''}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if is_scored else (200, 200, 200), 1)

        cv2.imshow(window, display)

    frame = read_frame(frame_idx)
    if frame is None:
        print("Cannot read initial frame", file=sys.stderr)
        sys.exit(1)
    draw(frame, frame_idx)

    while True:
        wait_ms = max(1, int(1000 / fps / speed)) if playing else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            scores.sort()
            with open(out_path, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"Saved {len(scores)} scores to {out_path}")
            break

        elif key == ord(" "):
            playing = not playing

        elif key in (ord("d"), 83):  # D or Right
            playing = False
            frame_idx = min(frame_idx + 1, total_frames - 1)
            frame = read_frame(frame_idx)

        elif key in (ord("a"), 81):  # A or Left
            playing = False
            frame_idx = max(frame_idx - 1, 0)
            frame = read_frame(frame_idx)

        elif key in (ord("w"), 82):  # W or Up
            playing = False
            frame_idx = min(frame_idx + 10, total_frames - 1)
            frame = read_frame(frame_idx)

        elif key in (ord("s"), 84):  # S or Down
            playing = False
            frame_idx = max(frame_idx - 10, 0)
            frame = read_frame(frame_idx)

        elif key in SPEEDS:
            speed, speed_label = SPEEDS[key]

        elif key == 13:  # Enter
            if frame_idx not in scores:
                scores.append(frame_idx)
                scores.sort()
                print(f"Score #{len(scores)} at frame {frame_idx}")

        elif key == 8:  # Backspace
            nearest = None
            nearest_dist = 16
            for s in scores:
                d = abs(s - frame_idx)
                if d < nearest_dist:
                    nearest = s
                    nearest_dist = d
            if nearest is not None:
                scores.remove(nearest)
                print(f"Removed score at frame {nearest}, {len(scores)} remaining")

        elif playing:
            frame_idx = min(frame_idx + 1, total_frames - 1)
            frame = read_frame(frame_idx)
            if frame_idx >= total_frames - 1:
                playing = False

        if frame is not None:
            draw(frame, frame_idx)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
