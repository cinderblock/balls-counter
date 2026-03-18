"""Main entry point: multi-stream motion-based ball counter (headless daemon)."""

import argparse
import sys
from pathlib import Path

from ball_counter.config import load_configs
from ball_counter.stream import SourceProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count ball scoring events from video streams")
    parser.add_argument("config", help="Path to JSON config file defining streams")
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Enable HTTP API and MJPEG server on this port (e.g. 8080)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=300,
        help="Print video-file progress every N processed frames (0 = off)",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    configs = load_configs(Path(args.config))

    sources: list[SourceProcessor] = []
    for config in configs:
        ready_goals = [g for g in config.goals if g.line or g.roi_points]
        skipped = len(config.goals) - len(ready_goals)
        if skipped:
            print(f"[{config.source}] WARNING: {skipped} goal(s) have no line or ROI, skipping them")
        if not ready_goals:
            print(f"[{config.source}] WARNING: no goals ready, skipping source")
            continue
        config.goals[:] = ready_goals

        proc = SourceProcessor(config)
        if not proc.open():
            print(f"ERROR: cannot open source: {config.source}", file=sys.stderr)
            continue
        goal_names = ", ".join(g.name for g in proc.goals)
        print(f"[{config.source}] connected — goals: {goal_names}")
        sources.append(proc)

    if not sources:
        print("ERROR: no sources could be opened", file=sys.stderr)
        sys.exit(1)

    state = None
    if args.web_port is not None:
        from ball_counter.web import AppState, start_server_thread
        state = AppState()
        for proc in sources:
            for goal in proc.goals:
                state.update_count(goal.name, 0)
        start_server_thread(state, args.web_port)
        print(f"Web API listening on http://0.0.0.0:{args.web_port}")

    progress_last: dict[str, int] = {}
    all_live = all(not s.is_video_file for s in sources)

    while True:
        frames_read = 0
        file_sources_done = 0

        for proc in sources:
            if not proc.read_frame():
                if proc.is_video_file:
                    file_sources_done += 1
                continue
            frames_read += 1

            ts = proc.timestamp_str
            results = proc.process_frame()

            # Progress logging for video files
            if proc.is_video_file and proc.total_frames > 0 and args.progress_interval > 0:
                frame_idx = results[0][0].processed_frames if results else 0
                last = progress_last.get(proc.source, 0)
                if frame_idx == 1 or frame_idx - last >= args.progress_interval or frame_idx >= proc.total_frames:
                    pct = frame_idx / proc.total_frames * 100
                    print(f"[{proc.source}] progress: {frame_idx}/{proc.total_frames} ({pct:.1f}%)")
                    progress_last[proc.source] = frame_idx

            if state is not None:
                for name in state.pop_resets():
                    for goal in proc.goals:
                        if goal.name == name:
                            goal.reset_count()
                            state.update_count(goal.name, 0)
                            print(f"[{goal.name}] count reset to 0")

            for goal, event in results:
                if state is not None:
                    state.update_count(goal.name, goal.count)
                    jpeg = goal.crop_jpeg()
                    if jpeg is not None:
                        state.update_frame(goal.name, jpeg)

                if event:
                    print(f"[{goal.name}] score at {ts}: +{event.n_balls} (total: {goal.count})")
                    if state is not None:
                        state.emit_event(goal.name, event.n_balls, goal.count, ts)

        file_sources = sum(1 for s in sources if s.is_video_file)
        if frames_read == 0 and (all_live or file_sources_done >= file_sources):
            break

    print("\n--- final counts ---")
    total = 0
    for proc in sources:
        for goal in proc.goals:
            print(f"  {goal.name}: {goal.count}")
            total += goal.count
        proc.release()
    print(f"  combined: {total}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
