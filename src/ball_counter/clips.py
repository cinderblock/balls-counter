"""Save a goal's rolling buffer to an MP4 + JSON sidecar."""

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ball_counter.buffer import BufferFrame


def _reencode_h264(src: Path, dst: Path, fps: float) -> None:
    """Re-encode src (any codec) to dst as H.264 with faststart, for browser playback."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", str(src),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def save_clip(
    frames: list[BufferFrame],
    goal_name: str,
    clips_dir: Path,
    fps: float = 30.0,
) -> tuple[Path, Path]:
    """Encode frames to MP4 (H.264) + JSON sidecar. Returns (mp4_path, json_path)."""
    if not frames:
        raise ValueError("No frames to save")

    clips_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{goal_name}"
    mp4_path = clips_dir / f"{stem}.mp4"
    json_path = clips_dir / f"{stem}.json"

    writer: cv2.VideoWriter | None = None
    sidecar: list[dict] = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        for bf in frames:
            arr = np.frombuffer(bf.jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(
                        str(tmp_path),
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
        writer = None

        _reencode_h264(tmp_path, mp4_path, fps)
    finally:
        if writer is not None:
            writer.release()
        tmp_path.unlink(missing_ok=True)

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


def trim_clip(
    clip_id: str,
    clips_dir: Path,
    segments: list[dict],
    delete_original: bool = False,
) -> list[str]:
    """Split a clip into one or more segments, creating new MP4+JSON for each.

    Each segment dict must have ``start_frame`` and ``end_frame`` keys.
    Returns the list of newly created clip IDs.
    """
    mp4_src = clips_dir / f"{clip_id}.mp4"
    json_src = clips_dir / f"{clip_id}.json"
    if not mp4_src.exists() or not json_src.exists():
        raise FileNotFoundError(f"Clip {clip_id} not found")

    import copy

    with open(json_src) as f:
        data = json.load(f)

    src_frames = data.get("frames", [])
    src_fps = data.get("fps", 30.0)
    goal = data.get("goal", "unknown")

    new_ids: list[str] = []
    for i, seg in enumerate(segments):
        # start_frame / end_frame are 0-based position indices into the frames
        # array (matching the timeline UI), NOT the stored frame_idx values.
        start_pos = seg["start_frame"]
        end_pos = seg["end_frame"]
        if end_pos <= start_pos:
            continue

        seg_frames = src_frames[start_pos:end_pos + 1]
        if not seg_frames:
            continue

        start_sec = start_pos / src_fps
        duration_sec = (end_pos - start_pos) / src_fps

        suffix = f"_p{i + 1}" if len(segments) > 1 else "_trimmed"
        new_id = f"{clip_id}{suffix}"
        new_mp4 = clips_dir / f"{new_id}.mp4"
        new_json = clips_dir / f"{new_id}.json"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp4_src),
                "-ss", str(start_sec),
                "-t", str(duration_sec),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                str(new_mp4),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Build a lookup from original frame_idx → new 0-based position
        orig_idx_set = {fr["frame_idx"] for fr in seg_frames}

        new_sidecar_frames = []
        for pos, fr in enumerate(seg_frames):
            nf = copy.deepcopy(fr)
            nf["frame_idx"] = pos
            if nf.get("event") and "frame" in nf["event"]:
                nf["event"]["frame"] = pos
            new_sidecar_frames.append(nf)

        new_captures = []
        for cap in data.get("captures", []):
            if cap.get("frame_idx") in orig_idx_set:
                nc = dict(cap)
                # Map original frame_idx to new position
                for pos, fr in enumerate(seg_frames):
                    if fr["frame_idx"] == cap["frame_idx"]:
                        nc["frame_idx"] = pos
                        break
                new_captures.append(nc)

        new_annotations: dict = {}
        for token, anno in (data.get("annotations") or {}).items():
            new_marks = []
            for m in (anno.get("marks") or []):
                # Annotations use video_time which maps to position-based time
                vt = m.get("video_time", 0)
                if start_pos / src_fps <= vt <= end_pos / src_fps:
                    nm = dict(m)
                    nm["video_time"] = vt - start_pos / src_fps
                    nm["frame_idx"] = round(nm["video_time"] * src_fps)
                    new_marks.append(nm)
            if new_marks:
                new_annotations[token] = {
                    "label": anno.get("label", token),
                    "saved_at": anno.get("saved_at", ""),
                    "marks": new_marks,
                }

        new_data = {
            "goal": goal,
            "saved_at": data.get("saved_at", ""),
            "fps": src_fps,
            "n_frames": len(new_sidecar_frames),
            "frames": new_sidecar_frames,
            "trimmed_from": clip_id,
            "trim_range": {"start_pos": start_pos, "end_pos": end_pos},
        }
        if new_captures:
            new_data["captures"] = new_captures
        if new_annotations:
            new_data["annotations"] = new_annotations

        with open(new_json, "w") as f_out:
            json.dump(new_data, f_out, indent=2)

        new_ids.append(new_id)

    if delete_original and new_ids:
        mp4_src.unlink()
        json_src.unlink()

    return new_ids
