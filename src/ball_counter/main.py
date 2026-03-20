"""Main entry point: multi-stream motion-based ball counter (headless daemon)."""

import argparse
import sys
import threading
from pathlib import Path

from ball_counter.config import load_configs
from ball_counter.stream import SourceProcessor


class AutoRecorder:
    """Automatically saves short clips around detected scoring events.

    When an event fires, waits ``tail_sec`` for the signal to settle, then
    slices ``pad_sec`` of quiet on each side of the event from the rolling
    buffer and saves to the clips directory.  Stops recording once
    ``max_bytes`` of MP4 data has been written.
    """

    def __init__(
        self,
        clips_dir: Path,
        fps: float = 30.0,
        pad_sec: float = 2.0,
        tail_sec: float = 2.0,
        max_bytes: int = 1_000_000_000,  # 1 GB
    ):
        self._clips_dir = clips_dir
        self._fps = fps
        self._pad_sec = pad_sec
        self._tail_sec = tail_sec
        self._max_bytes = max_bytes
        self._bytes_written = 0
        self._full = False
        self._lock = threading.Lock()
        # Pending sessions: goal_name -> {event_frames, timer, buffer}
        self._pending: dict[str, dict] = {}

        # Sum existing auto-recorded clips so restarts respect the budget
        # Sum existing auto-recorded clips so restarts respect the budget.
        # Auto clips are tagged with "auto_recorded" in their JSON sidecar.
        if clips_dir.exists():
            for jsn in clips_dir.glob("*.json"):
                if jsn.stem == "reviewers":
                    continue
                try:
                    import json as _json
                    data = _json.loads(jsn.read_text())
                    if data.get("auto_recorded"):
                        mp4 = clips_dir / (jsn.stem + ".mp4")
                        if mp4.exists():
                            self._bytes_written += mp4.stat().st_size
                except Exception:
                    pass
            if self._bytes_written >= self._max_bytes:
                self._full = True
                print(f"[auto-record] budget exhausted ({self._bytes_written / 1e6:.0f} MB used)")

    @property
    def full(self) -> bool:
        return self._full

    def on_activity(self, goal_name: str, frame_idx: int, buffer) -> None:
        """Record any signal activity. Coalesces nearby frames into one clip."""
        if self._full:
            return
        with self._lock:
            session = self._pending.get(goal_name)
            if session is not None:
                # More activity within the tail window — coalesce
                session["event_frames"].append(frame_idx)
                session["timer"].cancel()
            else:
                session = {"event_frames": [frame_idx], "buffer": buffer}
                self._pending[goal_name] = session
            t = threading.Timer(
                self._tail_sec,
                self._save,
                args=(goal_name,),
            )
            t.daemon = True
            t.start()
            session["timer"] = t

    def _save(self, goal_name: str) -> None:
        if self._full:
            return
        with self._lock:
            session = self._pending.pop(goal_name, None)
        if session is None:
            return
        event_frames = session["event_frames"]
        buffer = session["buffer"]
        pad_frames = int(self._pad_sec * self._fps)
        start = min(event_frames) - pad_frames
        end = max(event_frames) + pad_frames
        frames = buffer.slice_by_index(start, end)
        if not frames:
            return

        self._clips_dir.mkdir(parents=True, exist_ok=True)

        import json as _json
        from ball_counter.clips import save_clip
        try:
            mp4, jsn_path = save_clip(frames, goal_name, self._clips_dir, fps=self._fps)
            # Tag the sidecar so we can identify auto-recorded clips
            data = _json.loads(jsn_path.read_text())
            data["auto_recorded"] = True
            jsn_path.write_text(_json.dumps(data, indent=2))
            size = mp4.stat().st_size
            duration = len(frames) / self._fps
            with self._lock:
                self._bytes_written += size
                total_mb = self._bytes_written / 1e6
                if self._bytes_written >= self._max_bytes:
                    self._full = True
                    print(f"[auto-record] saved {mp4.name} ({duration:.1f}s, {size/1e3:.0f} kB) — "
                          f"budget full ({total_mb:.0f} MB), stopping")
                else:
                    remaining_mb = (self._max_bytes - self._bytes_written) / 1e6
                    print(f"[auto-record] saved {mp4.name} ({duration:.1f}s, {size/1e3:.0f} kB, "
                          f"{remaining_mb:.0f} MB remaining)")
        except Exception as e:
            print(f"[auto-record] save failed for {goal_name}: {e}")

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


def _start_sources(config_path: Path) -> tuple[list[SourceProcessor], object]:
    configs, pfms = load_configs(config_path)
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
    return sources, pfms


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
        existing_pfms = None
        if not config_missing:
            existing, existing_pfms = load_configs(config_path)
        wizard_done = threading.Event()
        _web.set_wizard_state(str(config_path), existing, wizard_done, pfms=existing_pfms)
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

    sources, pfms_cfg = _start_sources(config_path)

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

    forwarder = None
    if pfms_cfg is not None:
        from ball_counter.pfms import PfmsForwarder
        forwarder = PfmsForwarder(pfms_cfg.url, pfms_cfg.key, pfms_cfg.source)
        print(f"PFMS integration enabled → {pfms_cfg.url}")

    if state is not None:
        state.set_clips_dir(config_path.parent / "clips")
        for proc in sources:
            for goal in proc.goals:
                state.update_count(goal.name, 0)
                state.register_buffer(goal.name, goal.buffer)

    # Auto-record short clips around detected events (live sources only)
    clips_dir = config_path.parent / "clips"
    recorder = None
    if all(not s.is_video_file for s in sources):
        import shutil
        free_bytes = shutil.disk_usage(clips_dir if clips_dir.exists() else clips_dir.parent).free
        free_gb = free_bytes / 1e9
        if free_gb < 10:
            print(f"[auto-record] DISABLED — only {free_gb:.1f} GB free disk space (need 10 GB)")
        else:
            recorder = AutoRecorder(clips_dir)
            if not recorder.full:
                print(f"[auto-record] enabled → {clips_dir} (1 GB budget, {free_gb:.0f} GB free)")


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
                            goal.buffer.clear()
                            state.update_count(goal.name, 0)
                            state.emit_reset(goal.name)
                            print(f"[{goal.name}] count reset to 0")

                for score_name, score_n in state.pop_scores():
                    for goal in proc.goals:
                        if goal.name == score_name:
                            goal.counter.count += score_n
                            state.update_count(goal.name, goal.count)
                            state.emit_event(goal.name, score_n, goal.count, ts)
                            print(f"[{goal.name}] manual +{score_n} (total: {goal.count})")
                            if forwarder and goal.config.pfms_element:
                                alliance = ("red" if "red" in goal.name
                                            else "blue" if "blue" in goal.name
                                            else None)
                                if alliance:
                                    forwarder.send(alliance, goal.config.pfms_element, score_n)

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
                    if forwarder and goal.config.pfms_element:
                        alliance = ("red" if "red" in goal.name
                                    else "blue" if "blue" in goal.name
                                    else None)
                        if alliance:
                            forwarder.send(alliance, goal.config.pfms_element, event.n_balls)
                        else:
                            print(f"[{goal.name}] WARNING: cannot determine alliance for PFMS")

                # Auto-record on signal activity above 10% of ball_area —
                # more sensitive than scoring but filters out sensor noise
                if recorder and not recorder.full:
                    threshold = goal.config.ball_area * 0.1
                    if goal.counter.signal > threshold:
                        recorder.on_activity(goal.name, goal.counter.frame_idx, goal.buffer)

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
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
