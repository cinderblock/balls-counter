"""Train YOLOv8-nano on auto-labeled ball detection data."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-nano ball detector")
    parser.add_argument("--data", type=Path, default=Path("data/yolo_balls/dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=160,
                        help="Image size (crops are small, no need for large)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="ball_detector")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")  # nano pretrained on COCO

    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="models/yolo",
        patience=20,
        # Augmentation tuned for small ball crops
        hsv_h=0.01,  # minimal hue shift (balls are a specific color)
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=0,  # no rotation (gravity matters)
        translate=0.1,
        scale=0.3,
        flipud=0.0,  # no vertical flip
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        # Training params
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        # Save
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights to models/ (use actual save dir, not assumed path)
    import shutil
    save_dir = Path(results.save_dir) if results else None
    best = save_dir / "weights" / "best.pt" if save_dir else None
    dest = Path("models") / "yolo_ball_detector.pt"
    if best and best.exists():
        prev = dest.with_suffix(".prev.pt")
        if dest.exists():
            shutil.copy2(dest, prev)
            print(f"Previous model backed up to: {prev}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"Best model saved to: {dest}")
    else:
        print(f"WARNING: best.pt not found at {best}")

    return results


if __name__ == "__main__":
    main()
