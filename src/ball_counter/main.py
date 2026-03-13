"""Main entry point: multi-stream motion-based ball counter."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ball_counter.config import load_configs
from ball_counter.stream import StreamProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count ball scoring events from video streams")
    parser.add_argument("config", help="Path to JSON config file defining streams")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless without visualization windows")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    configs = load_configs(config_path)

    # Validate all streams have geometry defined
    ready_configs = []
    for config in configs:
        if not config.line and not config.roi_points:
            print(f"Warning: {config.name} has no line or ROI defined. Skipping.")
        else:
            ready_configs.append(config)

    if not ready_configs:
        print("No streams ready. Define line or roi_points in config.", file=sys.stderr)
        sys.exit(1)

    # Open all streams
    processors: list[StreamProcessor] = []
    for config in ready_configs:
        proc = StreamProcessor(config)
        if not proc.open():
            print(f"Error: cannot open stream {config.name} ({config.source})", file=sys.stderr)
            continue
        geom = "line" if config.line else "roi"
        processors.append(proc)
        print(f"Opened: {config.name} [{config.mode}, {geom}] from {config.source}")

    if not processors:
        print("No streams could be opened.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning {len(processors)} stream(s). Press 'q' to quit.\n")

    while True:
        frames_read = 0
        display_tiles: list[np.ndarray] = []

        for proc in processors:
            if not proc.read_frame():
                continue
            frames_read += 1

            event = proc.process_frame()
            if event:
                print(f"[{proc.config.name}] +{event.n_balls} "
                      f"(peak={event.peak_area}) Total: {proc.count}")

            if not args.no_display:
                overlay = proc.draw_overlay()
                if overlay is not None:
                    display_tiles.append(overlay)

        if frames_read == 0:
            break

        if not args.no_display and display_tiles:
            target_h = 480
            resized = []
            for tile in display_tiles:
                scale = target_h / tile.shape[0]
                resized.append(cv2.resize(tile, None, fx=scale, fy=scale))

            # Arrange in a 2x2 grid
            while len(resized) < 4:
                h = resized[0].shape[0]
                w = resized[0].shape[1]
                resized.append(np.zeros((h, w, 3), dtype=np.uint8))

            row1 = np.hstack(resized[:2])
            row2 = np.hstack(resized[2:4])
            grid = np.vstack([row1, row2])

            cv2.imshow("Ball Counter", grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    print("\n--- Final Counts ---")
    total = 0
    for proc in processors:
        print(f"  {proc.config.name} [{proc.config.mode}]: {proc.count}")
        total += proc.count
        proc.release()
    print(f"  Combined total: {total}")

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
