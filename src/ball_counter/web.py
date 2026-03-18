"""Optional HTTP API and MJPEG server for real-time ball counting web view."""

import asyncio
import json
import queue
import threading
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse


class AppState:
    """Thread-safe shared state between the processing loop and the web server."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._frames: dict[str, bytes] = {}  # latest JPEG per stream
        self._event_queues: list[queue.Queue] = []
        self._pending_resets: set[str] = set()

    def request_reset(self, name: str) -> None:
        with self._lock:
            self._pending_resets.add(name)

    def pop_resets(self) -> set[str]:
        """Return and clear any pending reset requests."""
        with self._lock:
            resets = self._pending_resets
            self._pending_resets = set()
            return resets

    def update_count(self, name: str, count: int) -> None:
        with self._lock:
            self._counts[name] = count

    def update_frame(self, name: str, jpeg: bytes) -> None:
        with self._lock:
            self._frames[name] = jpeg

    def emit_event(self, name: str, n_balls: int, total: int, timestamp: str) -> None:
        event = {"stream": name, "n_balls": n_balls, "total": total, "time": timestamp}
        with self._lock:
            for q in self._event_queues:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def get_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def get_frame(self, name: str) -> bytes | None:
        with self._lock:
            return self._frames.get(name)

    def get_stream_names(self) -> list[str]:
        with self._lock:
            return list(self._counts.keys())

    def _subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self._lock:
            self._event_queues.append(q)
        return q

    def _unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._event_queues.remove(q)
            except ValueError:
                pass


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Ball Counter API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        streams = state.get_stream_names()
        stream_cards = ""
        for name in streams:
            color = "red" if "red" in name else ("blue" if "blue" in name else "#888")
            stream_cards += f"""
            <div class="card">
              <div class="goal-label" style="color:{color}">{name}</div>
              <div class="count" id="count-{name}">0</div>
              <button class="clear-btn" onclick="resetGoal('{name}')">Clear</button>
              <img src="/api/stream/{name}.mjpeg" onerror="this.style.opacity='0.3'" />
            </div>"""

        return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Ball Counter</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #111; color: #eee; font-family: sans-serif; padding: 1rem; }}
    h1 {{ text-align: center; font-size: 1.4rem; margin-bottom: 1rem; color: #aaa; }}
    #goals {{ display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }}
    .card {{ background: #1e1e1e; border-radius: 8px; padding: 1rem; text-align: center; min-width: 280px; }}
    .goal-label {{ font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.4rem; }}
    .count {{ font-size: 4rem; font-weight: bold; line-height: 1; margin-bottom: 0.6rem; }}
    .card img {{ width: 100%; border-radius: 4px; background: #000; display: block; }}
    #log {{ margin-top: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto; }}
    #log h2 {{ font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; }}
    #events {{ list-style: none; max-height: 200px; overflow-y: auto; }}
    #events li {{ padding: 0.3rem 0.5rem; border-bottom: 1px solid #2a2a2a; font-size: 0.85rem; color: #ccc; }}
    #events li span.flash {{ color: #0f0; font-weight: bold; }}
    #status {{ text-align: center; font-size: 0.75rem; color: #444; margin-top: 1rem; }}
    .clear-btn {{ margin-bottom: 0.6rem; padding: 0.3rem 1rem; background: #333; color: #aaa; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }}
    .clear-btn:hover {{ background: #500; color: #fff; border-color: #a00; }}
  </style>
</head>
<body>
  <h1>Ball Counter</h1>
  <div id="goals">{stream_cards}</div>
  <div id="log">
    <h2>Recent scores</h2>
    <ul id="events"></ul>
  </div>
  <div id="status">connecting...</div>
  <script>
    const counts = {{}};
    const evtList = document.getElementById('events');
    const statusEl = document.getElementById('status');

    // Load initial counts
    fetch('/api/status').then(r => r.json()).then(data => {{
      for (const [name, count] of Object.entries(data.streams)) {{
        counts[name] = count;
        const el = document.getElementById('count-' + name);
        if (el) el.textContent = count;
      }}
    }});

    function resetGoal(name) {{
      fetch('/api/reset/' + name, {{method: 'POST'}}).then(() => {{
        const el = document.getElementById('count-' + name);
        if (el) el.textContent = '0';
      }});
    }}

    // Listen for score events
    const es = new EventSource('/api/events');
    es.onopen = () => statusEl.textContent = 'live';
    es.onerror = () => statusEl.textContent = 'disconnected — retrying...';
    es.onmessage = e => {{
      const ev = JSON.parse(e.data);
      counts[ev.stream] = ev.total;
      const el = document.getElementById('count-' + ev.stream);
      if (el) el.textContent = ev.total;

      const li = document.createElement('li');
      li.innerHTML = `<span class="flash">+${{ev.n_balls}}</span> ${{ev.stream}} &mdash; total ${{ev.total}} &nbsp;<small style="color:#555">${{ev.time}}</small>`;
      evtList.prepend(li);
      if (evtList.children.length > 50) evtList.lastChild.remove();
    }};
  </script>
</body>
</html>""")

    @app.get("/api/status")
    def status():
        return {"streams": state.get_counts()}

    @app.post("/api/reset/{name}")
    def reset(name: str):
        if name not in state.get_stream_names():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Stream '{name}' not found")
        state.request_reset(name)
        return {"ok": True}

    @app.get("/api/events")
    async def events():
        """Server-Sent Events stream of score events."""
        q = state._subscribe()

        async def generator() -> AsyncGenerator[str, None]:
            try:
                while True:
                    try:
                        event = q.get_nowait()
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass
            finally:
                state._unsubscribe(q)

        return StreamingResponse(generator(), media_type="text/event-stream")

    @app.get("/api/stream/{name}.mjpeg")
    async def mjpeg(name: str):
        """MJPEG stream for the goal window crop of a named stream."""

        async def generator() -> AsyncGenerator[bytes, None]:
            boundary = b"--frame"
            try:
                while True:
                    frame = state.get_frame(name)
                    if frame is not None:
                        yield (
                            boundary
                            + b"\r\nContent-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                            + frame
                            + b"\r\n"
                        )
                    await asyncio.sleep(1 / 30)
            except asyncio.CancelledError:
                pass

        return StreamingResponse(
            generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def start_server(state: AppState, port: int) -> None:
    """Start the uvicorn server in the current thread (blocks until done)."""
    app = create_app(state)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def start_server_thread(state: AppState, port: int) -> threading.Thread:
    """Launch the web server in a background daemon thread."""
    t = threading.Thread(target=start_server, args=(state, port), daemon=True)
    t.start()
    return t
