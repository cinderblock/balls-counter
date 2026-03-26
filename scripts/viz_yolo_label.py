"""Visualize YOLO labels on training images."""

import sys
from pathlib import Path

import cv2


def main():
    img_path = Path(sys.argv[1])
    label_path = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    if label_path.exists():
        for line in label_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert from normalized to pixel
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    out = img_path.parent.parent.parent / "viz" / (img_path.stem + "_viz.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
