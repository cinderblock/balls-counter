"""Save a goal's rolling buffer to an MP4 + JSON sidecar."""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ball_counter.buffer import BufferFrame


def save_clip(
    frames: list[BufferFrame],
    goal_name: str,
    clips_dir: Path,
    fps: float = 30.0,
) -> tuple[Path, Path]:
    """Encode frames to MP4 + JSON sidecar. Returns (mp4_path, json_path)."""
    if not frames:
        raise ValueError("No frames to save")

    clips_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{goal_name}"
    mp4_path = clips_dir / f"{stem}.mp4"
    json_path = clips_dir / f"{stem}.json"

    writer: cv2.VideoWriter | None = None
    sidecar: list[dict] = []

    for bf in frames:
        arr = np.frombuffer(bf.jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    str(mp4_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
            writer.write(img)

        ev = None
        if bf.event is not None:
            ev = {
                "frame": bf.event.frame,
                "n_balls": bf.event.n_balls,
                "peak_area": bf.event.peak_area,
            }
        sidecar.append({
            "frame_idx": bf.frame_idx,
            "timestamp": bf.timestamp,
            "signal": bf.signal,
            "rising": bf.rising,
            "event": ev,
        })

    if writer is None:
        raise ValueError("No decodable frames in buffer")
    writer.release()

    with open(json_path, "w") as f:
        json.dump(
            {
                "goal": goal_name,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "fps": fps,
                "n_frames": len(frames),
                "frames": sidecar,
            },
            f,
            indent=2,
        )

    return mp4_path, json_path
