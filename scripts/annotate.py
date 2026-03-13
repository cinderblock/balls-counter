"""Simple video player for annotating scoring events.

Controls:
    Space   - play/pause
    D       - step forward 1 frame (while paused)
    A       - step backward 1 frame (while paused)
    W       - step forward 10 frames (while paused)
    S       - step backward 10 frames (while paused)
    Enter   - mark current frame as a score event
    Backspace - remove score mark from current frame
    Q       - quit and save
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Annotate scoring events in a video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--output",
        help="Output JSON file (default: <video>-scores.json)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else video_path.with_suffix(".scores.json")

    # Load existing annotations if present
    scores: list[int] = []
    if output_path.exists():
        with open(output_path) as f:
            scores = json.load(f)
        print(f"Loaded {len(scores)} existing annotations from {output_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    playing = False
    frame = None

    window = "Annotate - Space:play/pause D/A:step Enter:mark Q:quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    def draw(f, idx):
        display = f.copy()
        h, w = display.shape[:2]

        # Frame info
        time_s = idx / fps
        is_scored = idx in scores
        color = (0, 0, 255) if is_scored else (255, 255, 255)

        cv2.putText(
            display,
            f"Frame {idx}/{total_frames - 1}  Time {time_s:.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        if is_scored:
            cv2.putText(display, "SCORED", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Score count
        cv2.putText(
            display,
            f"Total scores: {len(scores)}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Playback state
        state = "PLAYING" if playing else "PAUSED"
        cv2.putText(
            display,
            state,
            (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if playing else (0, 128, 255),
            2,
        )

        # Timeline bar
        bar_y = h - 5
        bar_h = 8
        cv2.rectangle(display, (0, bar_y - bar_h), (w, bar_y), (50, 50, 50), -1)

        # Draw score marks on timeline
        for s in scores:
            sx = int(s / total_frames * w)
            cv2.line(display, (sx, bar_y - bar_h), (sx, bar_y), (0, 0, 255), 2)

        # Current position
        px = int(idx / total_frames * w)
        cv2.line(display, (px, bar_y - bar_h - 4), (px, bar_y + 2), (0, 255, 0), 2)

        cv2.imshow(window, display)

    frame = read_frame(0)
    if frame is None:
        print("Error: cannot read first frame", file=sys.stderr)
        sys.exit(1)

    draw(frame, frame_idx)

    while True:
        wait_ms = int(1000 / fps) if playing else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == 13:  # Enter
            if frame_idx not in scores:
                scores.append(frame_idx)
                scores.sort()
                print(f"Marked frame {frame_idx} ({frame_idx / fps:.2f}s) - total: {len(scores)}")
        elif key == 8:  # Backspace
            if frame_idx in scores:
                scores.remove(frame_idx)
                print(f"Unmarked frame {frame_idx} - total: {len(scores)}")
        elif not playing:
            if key == ord("d"):
                frame_idx = min(frame_idx + 1, total_frames - 1)
                frame = read_frame(frame_idx)
            elif key == ord("a"):
                frame_idx = max(frame_idx - 1, 0)
                frame = read_frame(frame_idx)
            elif key == ord("w"):
                frame_idx = min(frame_idx + 10, total_frames - 1)
                frame = read_frame(frame_idx)
            elif key == ord("s"):
                frame_idx = max(frame_idx - 10, 0)
                frame = read_frame(frame_idx)

        if playing:
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
                playing = False
            else:
                frame = read_frame(frame_idx)

        if frame is not None:
            draw(frame, frame_idx)

    cap.release()
    cv2.destroyAllWindows()

    # Save
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nSaved {len(scores)} score annotations to {output_path}")


if __name__ == "__main__":
    main()
