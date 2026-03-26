"""Generate YOLO training data from annotated clips.

Uses HSV color masking + watershed to auto-generate bounding boxes around balls
near annotated timestamps. Also extracts negative frames (no balls) for background.

Output: YOLO-format dataset in data/yolo_balls/ with train/val splits.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ball_counter.config import load_configs
from ball_counter.detector import create_mask


def extract_bboxes(frame, hsv_low, hsv_high, min_area=80):
    """Extract bounding boxes from color-masked frame using contours."""
    mask = create_mask(frame, tuple(hsv_low), tuple(hsv_high))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Pad slightly
        pad = max(2, int(max(w, h) * 0.1))
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        bboxes.append((x, y, w, h))
    return bboxes


def get_all_marks(clip):
    """Get deduplicated mark times from all annotators."""
    times = []
    for ann in clip.get("annotations", {}).values():
        for m in ann.get("marks", []):
            t = float(m["video_time"])
            n = m.get("n_balls", 1)
            for _ in range(n):
                times.append(t)
    times.sort()
    deduped = []
    for t in times:
        if not deduped or t - deduped[-1] > 0.15:
            deduped.append(t)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO training data from annotated clips")
    parser.add_argument("clips_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/yolo_balls"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--neg-ratio", type=float, default=0.3,
                        help="Ratio of negative (no-ball) frames to include")
    parser.add_argument("--window", type=float, default=0.2,
                        help="Seconds around each mark to extract frames")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    sources, _ = load_configs(args.config)
    goal_configs = {}
    for src in sources:
        for g in src.goals:
            goal_configs[g.name] = g

    # Collect all annotated clips
    clips = []
    for jp in sorted(args.clips_dir.glob("*.json")):
        if jp.stem == "reviewers":
            continue
        mp4 = jp.with_suffix(".mp4")
        if not mp4.exists():
            continue
        clip = json.loads(jp.read_text())
        goal_name = clip.get("goal", "")
        gc = goal_configs.get(goal_name)
        if gc is None:
            continue
        marks = get_all_marks(clip)
        if not marks and not clip.get("annotations"):
            continue  # Skip unannotated clips — can't trust auto-labels without human ground truth
        clips.append((jp, mp4, clip, gc, marks))

    print(f"Found {len(clips)} clips with configs")
    if not clips:
        print("No clips to process!")
        return

    # Setup output dirs
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process clips
    random.shuffle(clips)
    n_val = int(len(clips) * args.val_split)
    val_clips = set(c[0].stem for c in clips[:n_val])

    total_pos = 0
    total_neg = 0
    total_boxes = 0

    for jp, mp4, clip, gc, marks in clips:
        split = "val" if jp.stem in val_clips else "train"
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Peek first frame to check dimensions (counts as frame 0)
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            continue
        fh, fw = first_frame.shape[:2]

        crop = gc.crop_override
        ds = gc.downsample
        already_cropped = False
        if crop:
            # If crop coordinates exceed frame dimensions, clip is already cropped
            if crop[2] > fw or crop[3] > fh:
                already_cropped = True
                crop = None
                ds = 1.0

        # Frames near marks (positive)
        mark_frames = set()
        for t in marks:
            center = int(t * fps)
            window = int(args.window * fps)
            for f in range(max(0, center - window), min(n_frames, center + window + 1)):
                mark_frames.add(f)

        # Negative frames: random frames far from any mark
        neg_frames = set()
        if marks:
            n_neg = int(len(mark_frames) * args.neg_ratio)
            all_frames = set(range(n_frames))
            # Exclude frames within 1s of any mark
            exclude = set()
            for t in marks:
                center = int(t * fps)
                for f in range(max(0, center - int(fps)), min(n_frames, center + int(fps) + 1)):
                    exclude.add(f)
            candidates = list(all_frames - exclude)
            if candidates:
                neg_frames = set(random.sample(candidates, min(n_neg, len(candidates))))
        else:
            # Zero-score clip: sample some frames as negatives
            n_neg = min(10, n_frames)
            neg_frames = set(random.sample(range(n_frames), n_neg))

        target_frames = sorted(mark_frames | neg_frames)
        if not target_frames:
            cap.release()
            continue
        target_set = set(target_frames)

        # Read all frames sequentially; frame 0 is already in first_frame
        max_target = max(target_frames)
        for fi in range(n_frames):
            if fi > max_target:
                break
            if fi == 0:
                frame = first_frame
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            if fi not in target_set:
                continue
            target = fi

            # Crop
            if crop:
                x1, y1, x2, y2 = crop
                fh, fw = frame.shape[:2]
                # Clamp crop to frame bounds
                cx1, cy1 = max(0, min(x1, fw)), max(0, min(y1, fh))
                cx2, cy2 = max(0, min(x2, fw)), max(0, min(y2, fh))
                frame = frame[cy1:cy2, cx1:cx2]

            if frame.size == 0:
                continue

            # Downsample
            if ds and ds != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (max(1, int(w * ds)), max(1, int(h * ds))))

            img_h, img_w = frame.shape[:2]
            if img_h == 0 or img_w == 0:
                continue

            is_positive = target in mark_frames

            # Extract bboxes
            bboxes = extract_bboxes(frame, gc.hsv_low, gc.hsv_high)

            # Save image
            img_name = f"{jp.stem}_f{target:06d}"
            img_path = args.output / "images" / split / f"{img_name}.jpg"
            cv2.imwrite(str(img_path), frame)

            # Save YOLO label (class 0 = ball)
            label_path = args.output / "labels" / split / f"{img_name}.txt"
            lines = []
            for (bx, by, bw, bh) in bboxes:
                # YOLO format: class cx cy w h (all normalized)
                cx = (bx + bw / 2) / img_w
                cy = (by + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                total_boxes += 1

            label_path.write_text("\n".join(lines))

            if is_positive:
                total_pos += 1
            else:
                total_neg += 1

        cap.release()

    # Write dataset YAML
    yaml_path = args.output / "dataset.yaml"
    yaml_path.write_text(f"""path: {args.output.resolve()}
train: images/train
val: images/val
names:
  0: ball
""")

    print(f"Done: {total_pos} positive frames, {total_neg} negative frames, {total_boxes} bounding boxes")
    print(f"Train/val split by clip ({1-args.val_split:.0%}/{args.val_split:.0%})")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
