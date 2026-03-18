"""Main entry point: multi-stream motion-based ball counter (headless daemon)."""

import argparse
import sys
import threading
from pathlib import Path

from ball_counter.config import load_configs
from ball_counter.stream import SourceProcessor

_DEFAULT_WIZARD_PORT = 8080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count ball scoring events from video streams")
    parser.add_argument("config", help="Path to JSON config file defining streams")
    parser.add_argument(
        "--web-port",
        default=None,
        metavar="PORT_OR_SOCKET",
        help="Enable HTTP API server on a TCP port (e.g. 8080) or Unix socket path (e.g. /run/ball-counter.sock)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Interface to bind the web server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--trusted-proxies",
        default=None,
        metavar="IPS",
        help="Comma-separated IPs (or '*') to trust for X-Forwarded-* headers (e.g. for Caddy)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=300,
        help="Print video-file progress every N processed frames (0 = off)",
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Launch setup wizard even if config already exists",
    )
    return parser.parse_args()


def _parse_web_binding(value: str) -> tuple[int | None, str | None]:
    """Return (port, None) for a TCP port or (None, socket_path) for a socket path."""
    try:
        return int(value), None
    except ValueError:
        return None, value


def _web_url(host: str, port: int | None, socket: str | None) -> str:
    if socket:
        return f"unix:{socket}"
    return f"http://{host}:{port}"


def _start_sources(config_path: Path) -> list[SourceProcessor]:
    configs = load_configs(config_path)
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
    return sources


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config_missing = not config_path.exists()
    wizard_mode = config_missing or args.wizard

    # Parse web binding; when config is missing the wizard always needs a port
    if args.web_port:
        web_port, web_socket = _parse_web_binding(args.web_port)
    elif config_missing:
        web_port, web_socket = _DEFAULT_WIZARD_PORT, None
    else:
        web_port, web_socket = None, None

    trusted = [p.strip() for p in args.trusted_proxies.split(",")] if args.trusted_proxies else None

    web_state = None
    wizard_done: threading.Event | None = None

    if wizard_mode:
        from ball_counter import web as _web
        from ball_counter.web import AppState, start_server_thread
        existing = None
        if not config_missing:
            existing = load_configs(config_path)
        wizard_done = threading.Event()
        _web.set_wizard_state(str(config_path), existing, wizard_done)
        web_state = AppState()
        start_server_thread(web_state, port=web_port, host=args.host,
                            socket_path=web_socket, trusted_proxies=trusted)
        url = _web_url(args.host, web_port, web_socket)
        if config_missing:
            print(f"Config not found. Open {url}/wizard to set up.")
            wizard_done.wait()  # block until wizard saves the config
            print("Wizard saved config — starting processors...")
        else:
            print(f"Wizard active. Open {url}/wizard (counting continues with existing config).")

    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    sources = _start_sources(config_path)

    if not sources:
        print("ERROR: no sources could be opened", file=sys.stderr)
        sys.exit(1)

    state = web_state
    if state is None and web_port is not None:
        from ball_counter.web import AppState, start_server_thread
        state = AppState()
        start_server_thread(state, port=web_port, host=args.host,
                            socket_path=web_socket, trusted_proxies=trusted)
        print(f"Web API listening on {_web_url(args.host, web_port, web_socket)}")

    if state is not None:
        state.set_clips_dir(config_path.parent / "clips")
        for proc in sources:
            for goal in proc.goals:
                state.update_count(goal.name, 0)
                state.register_buffer(goal.name, goal.buffer)

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
