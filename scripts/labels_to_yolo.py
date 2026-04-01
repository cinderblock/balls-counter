"""Convert human centroid labels to YOLO bounding box format.

Reads data/labels/index.json (centroids from the /balls UI) and
extracts frames from clip MP4s, writing YOLO-format dataset.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Convert centroid labels to YOLO format")
    parser.add_argument("--labels", type=Path, default=Path("data/labels/index.json"))
    parser.add_argument("--clips-dir", type=Path, default=Path("configs/clips"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_balls"))
    parser.add_argument("--ball-radius", type=int, default=16,
                        help="Radius in image pixels for bounding box generation")
    parser.add_argument("--edge-margin", type=int, default=8,
                        help="Skip marks within this many pixels of the frame edge")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    index = json.loads(args.labels.read_text())
    frames = index.get("frames", {})

    # Only use labeled frames
    labeled = {fid: info for fid, info in frames.items() if info.get("labeled")}
    print(f"{len(labeled)} labeled frames, {sum(len(f.get('marks', [])) for f in labeled.values())} centroids")

    if not labeled:
        print("No labeled frames found!")
        return

    # Split by clip (not by frame) to avoid leakage
    clips = list(set(f["clip"] for f in labeled.values()))
    random.shuffle(clips)
    n_val = max(1, int(len(clips) * args.val_split))
    val_clips = set(clips[:n_val])
    print(f"Train/val split: {len(clips) - n_val} / {n_val} clips")

    # Setup output dirs
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0
    skipped_edge = 0
    r = args.ball_radius

    # Group frames by clip for efficient video reading
    by_clip = {}
    for fid, info in labeled.items():
        by_clip.setdefault(info["clip"], []).append((fid, info))

    for clip_name, clip_frames in sorted(by_clip.items()):
        mp4 = args.clips_dir / (clip_name + ".mp4")
        if not mp4.exists():
            print(f"  Skipping {clip_name}: no MP4")
            continue

        split = "val" if clip_name in val_clips else "train"

        # Sort by frame number for sequential reading
        clip_frames.sort(key=lambda x: x[1]["frame_num"])
        target_map = {info["frame_num"]: (fid, info) for fid, info in clip_frames}

        cap = cv2.VideoCapture(str(mp4))
        max_frame = max(target_map.keys())

        fi = 0
        while fi <= max_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if fi in target_map:
                fid, info = target_map[fi]
                img_h, img_w = frame.shape[:2]
                marks = info.get("marks", [])

                # Save image
                img_path = args.output / "images" / split / f"{fid}.jpg"
                cv2.imwrite(str(img_path), frame)

                # Convert centroids to YOLO bboxes
                lines = []
                m = args.edge_margin
                for cx, cy in marks:
                    # Skip marks too close to frame edge (likely partial balls)
                    if cx < m or cy < m or cx >= img_w - m or cy >= img_h - m:
                        skipped_edge += 1
                        continue
                    # Bounding box: square of radius r around centroid
                    bx = max(0, cx - r)
                    by = max(0, cy - r)
                    bw = min(img_w, cx + r) - bx
                    bh = min(img_h, cy + r) - by
                    # YOLO normalized format
                    ncx = (bx + bw / 2) / img_w
                    ncy = (by + bh / 2) / img_h
                    nw = bw / img_w
                    nh = bh / img_h
                    lines.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                    total_boxes += 1

                label_path = args.output / "labels" / split / f"{fid}.txt"
                label_path.write_text("\n".join(lines))
                total_images += 1

            fi += 1

        cap.release()

    # Write dataset YAML
    yaml_path = args.output / "dataset.yaml"
    yaml_path.write_text(f"""path: {args.output.resolve()}
train: images/train
val: images/val
names:
  0: ball
""")

    print(f"Done: {total_images} images, {total_boxes} bounding boxes"
          f" ({skipped_edge} edge marks skipped, margin={args.edge_margin}px)")
    print(f"Dataset: {yaml_path}")


if __name__ == "__main__":
    main()
