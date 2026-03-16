"""Annotate a video with counting geometry (lines and polygons) and manual score marks.

Usage:
  uv run python scripts/annotate_geometry.py samples/red/sample2-full.mp4

Controls:
  Drawing:
    L        - Start drawing a LINE (click 2 points, Enter to confirm)
    P        - Start drawing a POLYGON/ROI (click points, Enter to confirm)
    R        - Reset current drawing
    Tab      - Cycle through saved geometries
    Delete   - Remove selected geometry

  Playback:
    Space    - Play/pause
    D / A    - Step forward/back 1 frame
    W / S    - Step forward/back 10 frames
    1-5      - Playback speed (1=0.25x, 2=0.5x, 3=1x, 4=2x, 5=4x)

  Scoring:
    Enter    - Mark a score event at current frame
    Backspace- Unmark score at current frame (±5 frame tolerance)

  Q        - Save and quit
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


class GeometryAnnotator:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"Cannot open {video_path}", file=sys.stderr)
            sys.exit(1)

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_idx = 0
        self.frame = None
        self.playing = False
        self.speed_idx = 2  # index into speeds list
        self.speeds = [0.25, 0.5, 1.0, 2.0, 4.0]

        # Geometry storage
        self.geometries = []  # list of {"type": "line"|"polygon", "points": [...], "name": "..."}
        self.selected_geom = -1

        # Current drawing state
        self.drawing_mode = None  # None, "line", "polygon"
        self.drawing_points = []

        # Score marks
        self.scores = []  # list of frame numbers

        # Mouse state
        self.mouse_pos = (0, 0)

        # Load existing annotations
        self.geom_path = self.video_path.with_suffix(".geometry.json")
        self.scores_path = self.video_path.with_suffix(".scores.json")
        self._load()

        # Window
        self.window = "Annotate - L=line P=polygon Space=play Q=quit"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        scale = min(1920 / self.w, 1080 / self.h, 2.0)
        cv2.resizeWindow(self.window, int(self.w * scale), int(self.h * scale))
        cv2.setMouseCallback(self.window, self._on_mouse)

        self._read_frame(0)

    def _load(self):
        if self.geom_path.exists():
            with open(self.geom_path) as f:
                self.geometries = json.load(f)
            print(f"Loaded {len(self.geometries)} geometries from {self.geom_path}")
        if self.scores_path.exists():
            with open(self.scores_path) as f:
                self.scores = json.load(f)
            print(f"Loaded {len(self.scores)} scores from {self.scores_path}")

    def _save(self):
        with open(self.geom_path, "w") as f:
            json.dump(self.geometries, f, indent=2)
        with open(self.scores_path, "w") as f:
            json.dump(sorted(self.scores), f, indent=2)
        print(f"Saved {len(self.geometries)} geometries to {self.geom_path}")
        print(f"Saved {len(self.scores)} scores to {self.scores_path}")

    def _read_frame(self, idx):
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.frame_idx = idx

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self.drawing_mode:
            self.drawing_points.append([x, y])

    def _draw_overlay(self):
        display = self.frame.copy()

        # Draw saved geometries
        for i, geom in enumerate(self.geometries):
            color = (0, 255, 0) if i == self.selected_geom else (0, 180, 255)
            pts = geom["points"]
            if geom["type"] == "line":
                cv2.line(display, tuple(pts[0]), tuple(pts[1]), color, 2)
                # Draw band preview
                p1, p2 = pts
                bw = 20
                band_pts = np.array([
                    [p1[0], p1[1] - bw], [p2[0], p2[1] - bw],
                    [p2[0], p2[1] + bw], [p1[0], p1[1] + bw]
                ], dtype=np.int32)
                overlay = display.copy()
                cv2.fillPoly(overlay, [band_pts], color)
                cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            elif geom["type"] == "polygon":
                arr = np.array(pts, dtype=np.int32)
                overlay = display.copy()
                cv2.fillPoly(overlay, [arr], color)
                cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
                cv2.polylines(display, [arr], True, color, 2)
            # Label
            label_pt = pts[0]
            name = geom.get("name", f"{geom['type']} {i+1}")
            cv2.putText(display, name, (label_pt[0] + 5, label_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw current drawing in progress
        if self.drawing_mode and self.drawing_points:
            color = (0, 0, 255)
            for pt in self.drawing_points:
                cv2.circle(display, tuple(pt), 5, color, -1)
            if len(self.drawing_points) >= 2:
                if self.drawing_mode == "line":
                    cv2.line(display, tuple(self.drawing_points[0]),
                             tuple(self.drawing_points[1]), color, 2)
                else:
                    pts = np.array(self.drawing_points, dtype=np.int32)
                    cv2.polylines(display, [pts], True, color, 2)

        # Score markers on timeline
        near_score = any(abs(s - self.frame_idx) <= 3 for s in self.scores)

        # Header bar
        bar_h = 80
        cv2.rectangle(display, (0, 0), (self.w, bar_h), (0, 0, 0), -1)

        # Frame info
        time_s = self.frame_idx / self.fps if self.fps > 0 else 0
        info = f"Frame {self.frame_idx}/{self.total_frames}  Time {time_s:.1f}s  Speed {self.speeds[self.speed_idx]}x"
        cv2.putText(display, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Score count and mode
        score_text = f"Scores: {len(self.scores)}"
        if near_score:
            score_text += "  ** SCORED HERE **"
            cv2.putText(display, score_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, score_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Drawing mode indicator
        if self.drawing_mode:
            mode_text = f"DRAWING {self.drawing_mode.upper()} ({len(self.drawing_points)} pts) - Enter=save R=reset"
            cv2.putText(display, mode_text, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            geom_info = f"Geometries: {len(self.geometries)}"
            if self.selected_geom >= 0 and self.selected_geom < len(self.geometries):
                g = self.geometries[self.selected_geom]
                geom_info += f"  Selected: {g.get('name', g['type'])} (Del to remove)"
            cv2.putText(display, geom_info, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Timeline bar at bottom
        timeline_y = self.h - 20
        cv2.rectangle(display, (0, timeline_y), (self.w, self.h), (40, 40, 40), -1)
        # Progress
        if self.total_frames > 0:
            progress_x = int(self.frame_idx / self.total_frames * self.w)
            cv2.rectangle(display, (0, timeline_y), (progress_x, self.h), (80, 80, 80), -1)
        # Score dots on timeline
        for s in self.scores:
            sx = int(s / self.total_frames * self.w) if self.total_frames > 0 else 0
            cv2.circle(display, (sx, timeline_y + 10), 4, (0, 255, 0), -1)
        # Current position marker
        cv2.line(display, (progress_x, timeline_y), (progress_x, self.h), (0, 0, 255), 2)

        return display

    def _finish_drawing(self):
        if self.drawing_mode == "line" and len(self.drawing_points) == 2:
            name = f"line-{len(self.geometries)+1}"
            self.geometries.append({
                "type": "line",
                "points": self.drawing_points[:2],
                "name": name,
            })
            print(f"Added line: {name}")
        elif self.drawing_mode == "polygon" and len(self.drawing_points) >= 3:
            name = f"polygon-{len(self.geometries)+1}"
            self.geometries.append({
                "type": "polygon",
                "points": self.drawing_points,
                "name": name,
            })
            print(f"Added polygon: {name} ({len(self.drawing_points)} points)")
        else:
            print(f"Not enough points for {self.drawing_mode}")
        self.drawing_mode = None
        self.drawing_points = []

    def run(self):
        while True:
            display = self._draw_overlay()
            cv2.imshow(self.window, display)

            wait_ms = 1 if not self.playing else max(1, int(1000 / self.fps / self.speeds[self.speed_idx]))
            key = cv2.waitKey(wait_ms) & 0xFF

            if self.playing and key == 255:
                self._read_frame(self.frame_idx + 1)
                if self.frame_idx >= self.total_frames - 1:
                    self.playing = False
                continue

            if key == ord("q"):
                self._save()
                break
            elif key == ord(" "):
                self.playing = not self.playing
            elif key == ord("d"):
                self.playing = False
                self._read_frame(self.frame_idx + 1)
            elif key == ord("a"):
                self.playing = False
                self._read_frame(self.frame_idx - 1)
            elif key == ord("w"):
                self.playing = False
                self._read_frame(self.frame_idx + 10)
            elif key == ord("s"):
                self.playing = False
                self._read_frame(self.frame_idx - 10)
            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                self.speed_idx = key - ord("1")
            elif key == ord("l") and not self.drawing_mode:
                self.drawing_mode = "line"
                self.drawing_points = []
                print("Drawing LINE - click 2 points, Enter to confirm")
            elif key == ord("p") and not self.drawing_mode:
                self.drawing_mode = "polygon"
                self.drawing_points = []
                print("Drawing POLYGON - click points, Enter to confirm")
            elif key == ord("r"):
                self.drawing_points = []
                if not self.drawing_mode:
                    # Reset to beginning
                    self._read_frame(0)
            elif key == 13:  # Enter
                if self.drawing_mode:
                    self._finish_drawing()
                else:
                    # Mark score
                    self.scores.append(self.frame_idx)
                    self.scores.sort()
                    print(f"Score #{len(self.scores)} at frame {self.frame_idx}")
            elif key == 8:  # Backspace
                # Remove nearest score within ±5 frames
                nearest = None
                min_dist = 6
                for s in self.scores:
                    d = abs(s - self.frame_idx)
                    if d < min_dist:
                        min_dist = d
                        nearest = s
                if nearest is not None:
                    self.scores.remove(nearest)
                    print(f"Removed score at frame {nearest}, {len(self.scores)} remaining")
            elif key == 9:  # Tab
                if self.geometries:
                    self.selected_geom = (self.selected_geom + 1) % len(self.geometries)
            elif key == 0 or key == 255:
                # Delete key or no key
                pass

            # Check for Delete key (different codes on different platforms)
            if key == 46 or key == 127:  # Delete
                if 0 <= self.selected_geom < len(self.geometries):
                    removed = self.geometries.pop(self.selected_geom)
                    print(f"Removed {removed.get('name', removed['type'])}")
                    self.selected_geom = min(self.selected_geom, len(self.geometries) - 1)

        cv2.destroyAllWindows()
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Annotate video with geometry and scores")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()
    annotator = GeometryAnnotator(args.video)
    annotator.run()


if __name__ == "__main__":
    main()
