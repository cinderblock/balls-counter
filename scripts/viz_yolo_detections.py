"""Visualize YOLO detections on clip frames.

Runs the YOLO model on frames from annotated clips and saves images
with detected bounding boxes + centroids overlaid.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO detections on clip frames")
    parser.add_argument("--clips-dir", type=Path, default=Path("configs/clips"))
    parser.add_argument("--model", type=Path, default=Path("models/yolo_ball_detector.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_viz"))
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Show all detections above this conf (color-coded by confidence)")
    parser.add_argument("--n-clips", type=int, default=20, help="Number of clips to sample")
    parser.add_argument("--frames-per-clip", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(args.model))

    # Find annotated clips
    clips = []
    for jp in sorted(args.clips_dir.glob("*.json")):
        if jp.stem == "reviewers":
            continue
        mp4 = jp.with_suffix(".mp4")
        if not mp4.exists():
            continue
        clip = json.loads(jp.read_text())
        if not clip.get("annotations"):
            continue
        times = []
        for ann in clip.get("annotations", {}).values():
            for m in ann.get("marks", []):
                times.append(float(m["video_time"]))
        clips.append((jp, mp4, clip, sorted(times)))

    random.shuffle(clips)
    clips = clips[:args.n_clips]
    print(f"Processing {len(clips)} clips...")

    total = 0
    for jp, mp4, clip, mark_times in clips:
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick frames: near marks + a couple random
        target_frames = set()
        for t in mark_times:
            fi = int(t * fps)
            target_frames.add(max(0, min(fi, n_frames - 1)))
        # Add some random frames
        if n_frames > 0:
            extras = random.sample(range(n_frames), min(2, n_frames))
            target_frames.update(extras)

        # Limit per clip
        target_frames = sorted(target_frames)[:args.frames_per_clip]

        fi = 0
        for target in target_frames:
            while fi < target:
                cap.grab()
                fi += 1
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            fi += 1

            # Run YOLO
            results = model.predict(frame, conf=args.conf, device="0",
                                     verbose=False, imgsz=320)

            # Draw detections
            viz = frame.copy()
            h, w = viz.shape[:2]

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Color by confidence: red < 0.5, yellow 0.5-0.7, green > 0.7
                    if conf >= 0.7:
                        color = (0, 255, 0)
                    elif conf >= 0.5:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)

                    # Bounding box
                    cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    # Centroid crosshair
                    r_cross = 8
                    cv2.line(viz, (int(cx) - r_cross, int(cy)), (int(cx) + r_cross, int(cy)), color, 2)
                    cv2.line(viz, (int(cx), int(cy) - r_cross), (int(cx), int(cy) + r_cross), color, 2)
                    # Confidence label
                    cv2.putText(viz, f"{conf:.2f}", (int(x1), int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Mark human annotations on the frame
            for t in mark_times:
                mark_fi = int(t * fps)
                if abs(mark_fi - target) <= 1:
                    # This frame has a human mark — draw a small magenta diamond at bottom
                    cv2.putText(viz, "* HUMAN MARK *", (4, h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
                    break

            # Frame info
            cv2.putText(viz, f"{jp.stem} f{target}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            out_path = args.output / f"{jp.stem}_f{target:06d}.jpg"
            cv2.imwrite(str(out_path), viz)
            total += 1

        cap.release()

    print(f"Saved {total} visualizations to {args.output}/")


if __name__ == "__main__":
    main()
