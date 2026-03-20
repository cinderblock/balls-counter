"""Optional HTTP API and MJPEG server for real-time ball counting web view."""

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

# Wizard state
_wizard_frames: dict[str, tuple[bytes, float]] = {}  # token -> (jpeg, created_at)
_wizard_config_path: str | None = None
_wizard_existing_configs: list | None = None  # list[SourceConfig] for pre-population
_wizard_existing_pfms = None                  # PfmsConfig | None for pre-population
_wizard_saved: bool = False  # set True after save so redirect deactivates
_wizard_done_event: "threading.Event | None" = None  # signalled when wizard saves
_WIZARD_TOKEN_TTL = 600  # seconds


def set_wizard_state(
    config_path: str,
    existing_configs: list | None = None,
    done_event: "threading.Event | None" = None,
    pfms=None,
) -> None:
    global _wizard_config_path, _wizard_existing_configs, _wizard_existing_pfms
    global _wizard_saved, _wizard_done_event
    _wizard_config_path = config_path
    _wizard_existing_configs = existing_configs
    _wizard_existing_pfms = pfms
    _wizard_saved = False
    _wizard_done_event = done_event


def _wizard_active() -> bool:
    """True when wizard mode is set and config has not been saved this session."""
    return _wizard_config_path is not None and not _wizard_saved


def _purge_expired_tokens() -> None:
    now = time.time()
    expired = [k for k, (_, ts) in _wizard_frames.items() if now - ts > _WIZARD_TOKEN_TTL]
    for k in expired:
        del _wizard_frames[k]


class AppState:
    """Thread-safe shared state between the processing loop and the web server."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._frames: dict[str, bytes] = {}  # latest JPEG per stream
        self._event_queues: list[queue.Queue] = []
        self._pending_resets: set[str] = set()
        self._pending_scores: list[tuple[str, int]] = []  # [(goal_name, n_balls), ...]
        self._buffers: dict = {}   # goal_name -> RollingBuffer
        self._clips_dir: "Path | None" = None
        # Live capture sessions: goal_name -> session dict
        self._capture_sessions: dict = {}

    def request_reset(self, name: str) -> None:
        with self._lock:
            self._pending_resets.add(name)

    def pop_resets(self) -> set[str]:
        """Return and clear any pending reset requests."""
        with self._lock:
            resets = self._pending_resets
            self._pending_resets = set()
            return resets

    def inject_score(self, name: str, n_balls: int = 1) -> None:
        """Queue a manual score injection for the main processing loop."""
        with self._lock:
            self._pending_scores.append((name, n_balls))

    def pop_scores(self) -> list[tuple[str, int]]:
        """Return and clear any pending manual score injections."""
        with self._lock:
            scores = self._pending_scores
            self._pending_scores = []
            return scores

    def update_count(self, name: str, count: int) -> None:
        with self._lock:
            self._counts[name] = count

    def update_frame(self, name: str, jpeg: bytes) -> None:
        with self._lock:
            self._frames[name] = jpeg

    def emit_event(self, name: str, n_balls: int, total: int, timestamp: str) -> None:
        event = {"type": "score", "stream": name, "n_balls": n_balls, "total": total, "time": timestamp}
        with self._lock:
            for q in self._event_queues:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def emit_reset(self, name: str) -> None:
        event = {"type": "reset", "stream": name, "total": 0}
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

    def register_buffer(self, name: str, buf) -> None:
        with self._lock:
            self._buffers[name] = buf

    def set_clips_dir(self, path) -> None:
        with self._lock:
            self._clips_dir = path

    def get_buffer(self, name: str):
        with self._lock:
            return self._buffers.get(name)

    def get_clips_dir(self):
        with self._lock:
            return self._clips_dir

    def press_capture(self, goal_name: str, fps_hint: float = 30.0) -> str:
        """Record a capture button press. Returns 'new', 'grouped', or 'error'."""
        import time as _time
        with self._lock:
            buf = self._buffers.get(goal_name)
            if buf is None:
                return "error"
            latest = buf.latest()
            if latest is None:
                return "error"
            press_frame = latest.frame_idx
            press_ts = latest.timestamp
            session = self._capture_sessions.get(goal_name)
            now = _time.monotonic()
            if session is not None and now - session["last_press_time"] < 5.0:
                session["press_frame_idxs"].append(press_frame)
                session["press_timestamps"].append(press_ts)
                session["last_press_time"] = now
                session["timer"].cancel()
                t = threading.Timer(4.0, self._flush_capture, args=(goal_name, fps_hint))
                t.daemon = True
                t.start()
                session["timer"] = t
                return "grouped"
            else:
                if session is not None:
                    session["timer"].cancel()
                t = threading.Timer(4.0, self._flush_capture, args=(goal_name, fps_hint))
                t.daemon = True
                t.start()
                self._capture_sessions[goal_name] = {
                    "press_frame_idxs": [press_frame],
                    "press_timestamps": [press_ts],
                    "first_press_time": now,
                    "last_press_time": now,
                    "timer": t,
                }
                return "new"

    def _flush_capture(self, goal_name: str, fps: float) -> None:
        """Called 4 s after last press; slices buffer and saves clip."""
        import json as _json
        from pathlib import Path as _Path
        with self._lock:
            session = self._capture_sessions.pop(goal_name, None)
            buf = self._buffers.get(goal_name)
            clips_dir = self._clips_dir
        if session is None or buf is None:
            return
        frames_per_sec = fps
        first_idx = min(session["press_frame_idxs"])
        last_idx = max(session["press_frame_idxs"])
        start_idx = max(0, first_idx - int(frames_per_sec * 1))
        end_idx = last_idx + int(frames_per_sec * 4)
        clip_frames = buf.slice_by_index(start_idx, end_idx)
        if not clip_frames:
            print(f"capture  - {goal_name}: no frames in window, skipping")
            return
        if clips_dir is None:
            clips_dir = _Path("clips")
        from ball_counter.clips import save_clip
        try:
            mp4, jsn = save_clip(clip_frames, goal_name, clips_dir, fps=fps)
            with open(jsn) as f:
                data = _json.load(f)
            data["captures"] = [
                {"frame_idx": fi, "timestamp": ts}
                for fi, ts in zip(session["press_frame_idxs"], session["press_timestamps"])
            ]
            with open(jsn, "w") as f:
                _json.dump(data, f, indent=2)
            print(f"capture  - {goal_name}: saved {mp4.name} ({len(clip_frames)} frames, "
                  f"{len(session['press_frame_idxs'])} press(es))")
        except Exception as e:
            print(f"capture  - {goal_name}: save failed: {e}")

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


def _load_reviewers(clips_dir: Path) -> dict:
    rf = clips_dir / "reviewers.json"
    return json.loads(rf.read_text()) if rf.exists() else {}


def _save_reviewers(clips_dir: Path, reviewers: dict) -> None:
    rf = clips_dir / "reviewers.json"
    rf.write_text(json.dumps(reviewers, indent=2))


_WIZARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Ball Counter — Setup Wizard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:sans-serif;padding:1rem;min-height:100vh}
h1{text-align:center;font-size:1.3rem;color:#aaa;margin-bottom:1rem}
#progress{display:flex;justify-content:center;gap:0.5rem;margin-bottom:1.5rem;flex-wrap:wrap}
.step-dot{padding:0.3rem 0.9rem;border-radius:12px;font-size:0.8rem;background:#222;color:#555;border:1px solid #333}
.step-dot.active{background:#1a3a5c;color:#7bf;border-color:#4af}
.step-dot.done{background:#1a3a1a;color:#5d5;border-color:#3a3}
.panel{display:none;max-width:960px;margin:0 auto}
.panel.active{display:block}
label{display:block;margin-bottom:0.3rem;font-size:0.85rem;color:#aaa}
input[type=text],input[type=number],select{width:100%;padding:0.5rem;background:#1e1e1e;border:1px solid #444;border-radius:4px;color:#eee;font-size:0.95rem}
input[type=number]{width:auto}
.row{display:flex;gap:0.5rem;align-items:flex-end;margin-bottom:1rem}
.row label{margin-bottom:0}
button{padding:0.45rem 1.1rem;border-radius:4px;border:1px solid #555;background:#2a2a2a;color:#ccc;cursor:pointer;font-size:0.9rem}
button:hover{background:#3a3a3a}
button.primary{background:#1a4a7a;border-color:#3af;color:#eef}
button.primary:hover{background:#1e5a94}
.msg{margin:0.5rem 0;padding:0.4rem 0.7rem;border-radius:4px;font-size:0.85rem}
.msg.ok{background:#1a3a1a;border:1px solid #3a3;color:#8f8}
.msg.err{background:#3a1a1a;border:1px solid #a33;color:#f88}
.msg.warn{background:#3a2a00;border:1px solid #a83;color:#fc8}
.msg.info{background:#1a2a3a;border:1px solid #38a;color:#8cf}
#canvas-wrap{background:#000;border-radius:4px;overflow:hidden;margin-bottom:0.7rem}
#canvas-wrap canvas{display:block;max-width:100%}
#toolbar{display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;margin-bottom:0.5rem;padding:0.5rem;background:#1a1a1a;border-radius:4px}
.tool-btn{padding:0.3rem 0.7rem;font-size:0.8rem}
.tool-btn.active{background:#2a4a1a;border-color:#5a3;color:#af8}
.color-btn{width:26px;height:26px;border-radius:50%;border:2px solid #555;cursor:pointer;display:inline-block;flex-shrink:0}
.color-btn.active{border-color:#fff;box-shadow:0 0 4px #fff8}
.goal-item{background:#1e1e1e;border-radius:6px;padding:0.7rem;margin-bottom:0.5rem;border:1px solid #333}
.goal-item summary{cursor:pointer;font-size:0.9rem;padding:0.2rem 0}
.goal-fields{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.7rem}
.goal-fields .full{grid-column:1/-1}
.field-group{margin-bottom:0.3rem}
.field-group label{font-size:0.78rem;color:#888;margin-bottom:0.15rem}
.field-group input,.field-group select{padding:0.3rem 0.5rem;font-size:0.85rem}
#json-preview{background:#0a0a0a;border:1px solid #333;border-radius:4px;padding:0.8rem;font-family:monospace;font-size:0.8rem;white-space:pre;overflow:auto;max-height:400px;color:#9f9}
.nav-row{display:flex;gap:0.5rem;margin-top:1rem;flex-wrap:wrap}
#ram-summary{font-size:0.82rem;color:#aaa;margin:0.4rem 0}
.ds-row{display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;margin-top:0.4rem}
.ds-btn{padding:0.2rem 0.6rem;font-size:0.8rem}
.ds-btn.active{background:#2a4060;border-color:#4af;color:#8df}
.crop-section{margin-bottom:1.2rem;border:1px solid #333;border-radius:6px;padding:0.7rem}
.crop-section h3{font-size:0.9rem;margin-bottom:0.5rem}
.crop-canvas-wrap{background:#000;border-radius:4px;overflow:hidden}
.crop-canvas-wrap canvas{display:block}
</style>
</head>
<body>
<h1>Ball Counter Setup Wizard</h1>
<div id="progress">
  <div class="step-dot active" id="dot-1">1 · Stream</div>
  <div class="step-dot" id="dot-2">2 · Crop Windows</div>
  <div class="step-dot" id="dot-3">3 · Count Lines</div>
  <div class="step-dot" id="dot-4">4 · Tune</div>
  <div class="step-dot" id="dot-5">5 · Save</div>
</div>

<!-- Step 1 -->
<div class="panel active" id="panel-1">
  <div class="row">
    <div style="flex:1">
      <label>Stream URL (RTSP or file path)</label>
      <input type="text" id="url-input" placeholder="rtsp://10.0.0.1:554/stream"/>
    </div>
    <button class="primary" onclick="connectStream()">Connect</button>
  </div>
  <div id="snap-msg"></div>
</div>

<!-- Step 2: Draw Crop Windows -->
<div class="panel" id="panel-2">
  <div id="toolbar">
    <span style="font-size:0.8rem;color:#888">Color:</span>
    <div class="color-btn active" id="color-red" style="background:#e33" title="Red" onclick="setBoxColor('red')"></div>
    <div class="color-btn" id="color-blue" style="background:#36f" title="Blue" onclick="setBoxColor('blue')"></div>
    <span style="font-size:0.8rem;color:#555;margin:0 0.3rem">|</span>
    <button class="tool-btn active" id="tool-draw" onclick="setBoxTool('draw')">Draw Box</button>
    <button class="tool-btn" id="tool-edit" onclick="setBoxTool('edit')">Edit</button>
    <button class="tool-btn" id="tool-delete" onclick="setBoxTool('delete')">Delete</button>
    <span style="margin-left:auto;font-size:0.8rem;color:#aaa" id="canvas-hint">Click &amp; drag to draw a crop window</span>
  </div>
  <div id="canvas-wrap"><canvas id="canvas"></canvas></div>
  <div id="draw-msgs"></div>
  <div id="ram-est" style="margin:0.3rem 0"></div>
  <div class="nav-row">
    <button onclick="goStep(1)">← Back</button>
    <button onclick="addAnotherCamera()">+ Add another camera</button>
    <button class="primary" onclick="goStep(3)">Next: Draw Lines →</button>
  </div>
</div>

<!-- Step 3: Draw Counting Lines -->
<div class="panel" id="panel-3">
  <p style="font-size:0.85rem;color:#888;margin-bottom:0.8rem">Draw a counting line in each crop window. Click and drag to draw; drag again to replace.</p>
  <div id="line-panels"></div>
  <div id="line-msgs"></div>
  <div class="nav-row">
    <button onclick="goStep(2)">← Back</button>
    <button onclick="addAnotherCamera()">+ Add another camera</button>
    <button class="primary" onclick="goStep(4)">Next: Tune →</button>
  </div>
</div>

<!-- Step 4: Tune -->
<div class="panel" id="panel-4">
  <div id="goal-list"></div>
  <div id="ram-summary"></div>
  <details style="margin-top:1rem;border:1px solid #333;border-radius:6px;padding:0.6rem 0.8rem">
    <summary style="cursor:pointer;font-weight:bold;color:#aaa">PFMS Integration <small style="font-weight:normal;color:#666">(optional)</small></summary>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem 1rem;margin-top:0.6rem">
      <div class="field-group full"><label>PFMS Server URL</label><input type="text" id="pfms-url" placeholder="http://pfms.tsl" value=""/></div>
      <div class="field-group"><label>API Key</label><input type="password" id="pfms-key" placeholder="(optional)" value=""/></div>
      <div class="field-group"><label>Source ID</label><input type="text" id="pfms-source" placeholder="ball-counter" value=""/></div>
    </div>
  </details>
  <div class="nav-row">
    <button onclick="goStep(3)">← Back</button>
    <button class="primary" onclick="goStep(5)">Review →</button>
  </div>
</div>

<!-- Step 5: Save -->
<div class="panel" id="panel-5">
  <div id="json-preview"></div>
  <div id="save-msgs" style="margin-top:0.5rem"></div>
  <div class="nav-row">
    <button onclick="goStep(4)">← Back</button>
    <button class="primary" onclick="saveConfig()">Save Config</button>
  </div>
</div>

<script>
// ── global state ──────────────────────────────────────────────────────────────
const state = {
  streams: [],     // banked [{url, goals:[goal]}]
  currentUrl: '',
  token: '',
  frameW: 0, frameH: 0,
  frameScale: 1,
  currentStep: 1,
};

// ── step 2 state ──────────────────────────────────────────────────────────────
// boxes: {id, color, cx, cy, cw, ch, name, mode, ball_area, band_width,
//          fall_ratio, min_peak, cooldown, downsample}  (cx/cy/cw/ch = canvas coords)
let boxes = [];
let boxDrag = null;   // {type:'draw'|'resize'|'move', ...}
let boxTool = 'draw';
let boxColor = 'red';
let nextBoxId = 0;
const HANDLE_HIT = 14;  // px hit radius (larger than rendered)
const HANDLE_R   = 7;   // px rendered radius
const MIN_BOX    = 40;  // min canvas-coord box dimension
const HANDLE_CURSORS = ['nw-resize','n-resize','ne-resize','e-resize',
                         'se-resize','s-resize','sw-resize','w-resize'];
const DS_OPTS = [0.25, 0.5, 0.75, 1.0];

// ── step 3 state ──────────────────────────────────────────────────────────────
let linesByBox = {};    // {boxId: {p1:[cx,cy], p2:[cx,cy]}} zoomed canvas coords
let boxZooms   = {};    // {boxId: number}
let lineDrags  = {};    // {boxId: {x0,y0,x1,y1}} in-progress draws

// ── preload ───────────────────────────────────────────────────────────────────
let _existingConfig = null;
let _configPath = '';
let existingLinesFull = {};  // {boxId: {p1:[fx,fy], p2:[fx,fy]}}

// ── navigation ────────────────────────────────────────────────────────────────
function goStep(n) {
  if (n === 3) {
    if (boxes.length === 0) { showMsg('draw-msgs','err','Draw at least one crop window first.'); return; }
    buildLinePanels();
  }
  if (n === 4) {
    if (!boxes.some(b => linesByBox[b.id])) { showMsg('line-msgs','err','Draw a counting line in at least one crop window.'); return; }
    buildTunePanel();
  }
  if (n === 5) { syncTuneToState(); buildJsonPreview(); }
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-'+n).classList.add('active');
  document.querySelectorAll('.step-dot').forEach((d,i) => {
    d.classList.remove('active','done');
    if (i+1 < n) d.classList.add('done');
    if (i+1 === n) d.classList.add('active');
  });
  state.currentStep = n;
}

function addAnotherCamera() {
  if (boxes.length === 0) { showMsg('draw-msgs','err','Draw at least one crop window for this stream first.'); return; }
  bankCurrentStream();
  boxes = []; linesByBox = {}; existingLinesFull = {}; boxZooms = {}; lineDrags = {};
  showMsg('draw-msgs',''); showMsg('line-msgs','');
  goStep(1);
}

function bankCurrentStream() {
  if (!state.currentUrl) return;
  const goals = buildCurrentGoals();
  const existing = state.streams.find(s => s.url === state.currentUrl);
  if (existing) existing.goals.push(...goals);
  else if (goals.length) state.streams.push({url: state.currentUrl, goals});
}

// ── step 1 ────────────────────────────────────────────────────────────────────
async function connectStream() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;
  showMsg('snap-msg','info','Connecting…');
  try {
    const r = await fetch('/api/wizard/snapshot', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url})
    });
    const data = await r.json();
    if (data.error) { showMsg('snap-msg','err', data.error); return; }
    state.currentUrl = url; state.token = data.token;
    state.frameW = data.width; state.frameH = data.height;
    showMsg('snap-msg','ok', `Got frame ${data.width}×${data.height} — draw crop windows around each goal zone.`);
    setTimeout(() => goStep(2), 500);
    initBoxCanvas();
  } catch(e) { showMsg('snap-msg','err','Request failed: '+e.message); }
}

// ── step 2: box canvas ────────────────────────────────────────────────────────
function initBoxCanvas() {
  const wrap   = document.getElementById('canvas-wrap');
  const canvas = document.getElementById('canvas');
  const maxW   = Math.min(wrap.clientWidth || (window.innerWidth - 32), state.frameW);
  state.frameScale = maxW / state.frameW;
  canvas.width  = Math.round(state.frameW * state.frameScale);
  canvas.height = Math.round(state.frameH * state.frameScale);
  // max-width:100% CSS constrains display; canvasXY() corrects for that scaling
  const img = new Image();
  img.onload = () => { canvas._img = img; loadExistingBoxes(state.currentUrl); redrawBoxCanvas(); };
  img.src = '/api/wizard/frame/' + state.token + '.jpg';
  canvas.onmousedown  = onBoxDown;
  canvas.onmousemove  = onBoxMove;
  canvas.onmouseup    = onBoxUp;
  canvas.onmouseleave = () => { if (boxDrag) { onBoxUp({}); } };
}

function canvasXY(e) {
  const c = document.getElementById('canvas');
  const r = c.getBoundingClientRect();
  // Correct for any CSS scaling (r.width may differ from c.width)
  return [(e.clientX - r.left) * (c.width / r.width),
          (e.clientY - r.top)  * (c.height / r.height)];
}

function getBoxHandles(b) {
  const {cx,cy,cw,ch} = b;
  return [[cx,cy],[cx+cw/2,cy],[cx+cw,cy],[cx+cw,cy+ch/2],
          [cx+cw,cy+ch],[cx+cw/2,cy+ch],[cx,cy+ch],[cx,cy+ch/2]];
}

function findHandle(mx,my) {
  for (let bi = boxes.length-1; bi >= 0; bi--) {
    const hs = getBoxHandles(boxes[bi]);
    for (let hi = 0; hi < hs.length; hi++)
      if (Math.hypot(mx-hs[hi][0], my-hs[hi][1]) < HANDLE_HIT) return {bi, hi};
  }
  return null;
}

function findBoxAt(mx,my) {
  for (let bi = boxes.length-1; bi >= 0; bi--) {
    const {cx,cy,cw,ch} = boxes[bi];
    if (mx>=cx && mx<=cx+cw && my>=cy && my<=cy+ch) return bi;
  }
  return -1;
}

function onBoxDown(e) {
  const [mx,my] = canvasXY(e);
  if (boxTool === 'draw') {
    boxDrag = {type:'draw', x0:mx, y0:my, x1:mx, y1:my};
  } else if (boxTool === 'edit') {
    const h = findHandle(mx,my);
    if (h) { boxDrag = {type:'resize', bi:h.bi, hi:h.hi}; }
    else {
      const bi = findBoxAt(mx,my);
      if (bi>=0) boxDrag = {type:'move', bi, ox:mx-boxes[bi].cx, oy:my-boxes[bi].cy};
    }
  } else if (boxTool === 'delete') {
    const bi = findBoxAt(mx,my);
    if (bi >= 0) {
      const id = boxes[bi].id;
      boxes.splice(bi,1);
      delete linesByBox[id]; delete existingLinesFull[id]; delete boxZooms[id];
      redrawBoxCanvas(); updateRamEst();
    }
  }
}

function onBoxMove(e) {
  const [mx,my] = canvasXY(e);
  const c = document.getElementById('canvas');
  if (!boxDrag) {
    if (boxTool === 'edit') {
      const h = findHandle(mx,my);
      if (h) { c.style.cursor = HANDLE_CURSORS[h.hi]; }
      else { c.style.cursor = findBoxAt(mx,my) >= 0 ? 'move' : 'crosshair'; }
    }
    return;
  }
  const cw = c.width, ch = c.height;
  if (boxDrag.type === 'draw') {
    boxDrag.x1 = mx; boxDrag.y1 = my; redrawBoxCanvas(boxDrag);
  } else if (boxDrag.type === 'resize') {
    applyBoxHandle(boxes[boxDrag.bi], boxDrag.hi, mx, my, cw, ch);
    redrawBoxCanvas(); updateRamEst();
  } else if (boxDrag.type === 'move') {
    const b = boxes[boxDrag.bi];
    b.cx = Math.max(0, Math.min(cw - b.cw, mx - boxDrag.ox));
    b.cy = Math.max(0, Math.min(ch - b.ch, my - boxDrag.oy));
    redrawBoxCanvas();
  }
}

function onBoxUp(e) {
  if (!boxDrag) return;
  if (boxDrag.type === 'draw') {
    const x = Math.min(boxDrag.x0,boxDrag.x1), y = Math.min(boxDrag.y0,boxDrag.y1);
    const w = Math.abs(boxDrag.x1-boxDrag.x0), h = Math.abs(boxDrag.y1-boxDrag.y0);
    if (w > MIN_BOX && h > MIN_BOX) {
      if (boxes.length >= 2) {
        showMsg('draw-msgs','warn','Max 2 crop windows per stream. Delete one before adding another.');
      } else {
        boxes.push({id:nextBoxId++, color:boxColor, cx:x, cy:y, cw:w, ch:h,
          name:boxColor+'-goal', mode:'outlet', ball_area:1500, band_width:10,
          fall_ratio:0.7, min_peak:0, cooldown:0, downsample:1.0});
        // Auto-switch color for second box
        if (boxes.length === 1) setBoxColor(boxColor === 'red' ? 'blue' : 'red');
        redrawBoxCanvas(); updateRamEst();
      }
    }
  }
  boxDrag = null;
}

function applyBoxHandle(b, hi, mx, my, maxW, maxH) {
  const mn = MIN_BOX, r = b.cx+b.cw, bot = b.cy+b.ch;
  switch(hi) {
    case 0:{const nx=Math.max(0,Math.min(mx,r-mn)),ny=Math.max(0,Math.min(my,bot-mn));b.cx=nx;b.cy=ny;b.cw=r-nx;b.ch=bot-ny;break;}
    case 1:{const ny=Math.max(0,Math.min(my,bot-mn));b.cy=ny;b.ch=bot-ny;break;}
    case 2:{const ny=Math.max(0,Math.min(my,bot-mn));b.cy=ny;b.ch=bot-ny;b.cw=Math.min(Math.max(mx-b.cx,mn),maxW-b.cx);break;}
    case 3:{b.cw=Math.min(Math.max(mx-b.cx,mn),maxW-b.cx);break;}
    case 4:{b.cw=Math.min(Math.max(mx-b.cx,mn),maxW-b.cx);b.ch=Math.min(Math.max(my-b.cy,mn),maxH-b.cy);break;}
    case 5:{b.ch=Math.min(Math.max(my-b.cy,mn),maxH-b.cy);break;}
    case 6:{const nx=Math.max(0,Math.min(mx,r-mn));b.cx=nx;b.cw=r-nx;b.ch=Math.min(Math.max(my-b.cy,mn),maxH-b.cy);break;}
    case 7:{const nx=Math.max(0,Math.min(mx,r-mn));b.cx=nx;b.cw=r-nx;break;}
  }
}

function redrawBoxCanvas(drawing) {
  const c = document.getElementById('canvas');
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if (c._img && c._img.complete) ctx.drawImage(c._img,0,0,c.width,c.height);
  const sc = state.frameScale;
  for (const b of boxes) {
    const clr = b.color==='red' ? '#e44' : '#44f';
    const fw = Math.round(b.cw/sc), fh = Math.round(b.ch/sc);
    ctx.save(); ctx.globalAlpha=0.12; ctx.fillStyle=clr; ctx.fillRect(b.cx,b.cy,b.cw,b.ch); ctx.restore();
    ctx.strokeStyle=clr; ctx.lineWidth=2; ctx.setLineDash([]); ctx.strokeRect(b.cx,b.cy,b.cw,b.ch);
    ctx.fillStyle=clr; ctx.font='bold 12px sans-serif';
    ctx.fillText(b.name+' '+fw+'×'+fh+'px', b.cx+5, b.cy+16);
    if (linesByBox[b.id]) {
      ctx.fillStyle='#4f4'; ctx.font='11px sans-serif';
      ctx.fillText('✓ line set', b.cx+5, b.cy+b.ch-6);
    }
    if (boxTool==='edit') {
      getBoxHandles(b).forEach(([hx,hy]) => {
        ctx.fillStyle='#fff'; ctx.strokeStyle='#333'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.arc(hx,hy,HANDLE_R,0,Math.PI*2); ctx.fill(); ctx.stroke();
      });
    }
    if (b.downsample && b.downsample < 1.0) {
      ctx.fillStyle='rgba(0,0,0,0.65)'; ctx.fillRect(b.cx+b.cw-44,b.cy+2,42,16);
      ctx.fillStyle='#8df'; ctx.font='11px sans-serif'; ctx.fillText(b.downsample+'×',b.cx+b.cw-41,b.cy+14);
    }
  }
  if (drawing) {
    const x=Math.min(drawing.x0,drawing.x1), y=Math.min(drawing.y0,drawing.y1);
    const w=Math.abs(drawing.x1-drawing.x0), h=Math.abs(drawing.y1-drawing.y0);
    const clr = boxColor==='red' ? '#e44' : '#44f';
    ctx.save(); ctx.globalAlpha=0.15; ctx.fillStyle=clr; ctx.fillRect(x,y,w,h); ctx.restore();
    ctx.strokeStyle=clr; ctx.lineWidth=2; ctx.setLineDash([6,4]); ctx.strokeRect(x,y,w,h); ctx.setLineDash([]);
  }
}

function setBoxTool(t) {
  boxTool = t;
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tool-'+t).classList.add('active');
  const hints = {draw:'Click & drag to draw a crop window', edit:'Drag handles to resize · drag interior to move', delete:'Click a box to delete it'};
  document.getElementById('canvas-hint').textContent = hints[t] || '';
  document.getElementById('canvas').style.cursor = t==='draw' ? 'crosshair' : t==='delete' ? 'not-allowed' : 'default';
  redrawBoxCanvas();
}

function setBoxColor(c) {
  boxColor = c;
  document.getElementById('color-red').classList.toggle('active', c==='red');
  document.getElementById('color-blue').classList.toggle('active', c==='blue');
}

function updateRamEst() {
  const sc = state.frameScale;
  let total = 0;
  for (const b of boxes) {
    const fw = Math.round(b.cw/sc) * (b.downsample||1.0);
    const fh = Math.round(b.ch/sc) * (b.downsample||1.0);
    total += fw * fh * 0.00002 * 1800;
  }
  for (const s of state.streams) for (const g of s.goals)
    total += g.crop.w * g.crop.h * (g.downsample||1.0) * (g.downsample||1.0) * 0.00002 * 1800;
  const mb = total.toFixed(0);
  const cls = total>2000 ? 'err' : total>500 ? 'warn' : 'info';
  document.getElementById('ram-est').innerHTML =
    `<span class="msg ${cls}" style="display:inline-block">~${mb} MB estimated RAM for 60s buffer</span>`;
}

// ── step 3: zoomed line drawing ───────────────────────────────────────────────
function buildLinePanels() {
  const container = document.getElementById('line-panels');
  container.innerHTML = '';
  for (const b of boxes) {
    const sc = state.frameScale;
    const fx=Math.round(b.cx/sc), fy=Math.round(b.cy/sc);
    const fw=Math.round(b.cw/sc), fh=Math.round(b.ch/sc);
    const targetW = Math.min(800, window.innerWidth - 48);
    const zoom = targetW / fw;
    boxZooms[b.id] = zoom;
    const cvW = Math.round(fw*zoom), cvH = Math.round(fh*zoom);

    // Convert full-frame preloaded line to zoomed coords (once)
    if (existingLinesFull[b.id] && !linesByBox[b.id]) {
      const ef = existingLinesFull[b.id];
      linesByBox[b.id] = {
        p1: [(ef.p1[0]-fx)*zoom, (ef.p1[1]-fy)*zoom],
        p2: [(ef.p2[0]-fx)*zoom, (ef.p2[1]-fy)*zoom],
      };
    }

    const clrHex = b.color==='red' ? '#e66' : '#66f';
    const section = document.createElement('div');
    section.className = 'crop-section';
    section.innerHTML = `<h3 style="color:${clrHex}">${b.name} &mdash; ${fw}×${fh}px (shown ${cvW}×${cvH})</h3>`;

    const wrap = document.createElement('div');
    wrap.className = 'crop-canvas-wrap';
    const cvs = document.createElement('canvas');
    cvs.width = cvW; cvs.height = cvH;
    cvs.style.cursor = 'crosshair';
    cvs._boxId = b.id; cvs._zoom = zoom; cvs._origin = {fx, fy, fw, fh};

    const img = new Image();
    img.onload = () => { cvs._img = img; redrawLineCanvas(cvs); };
    img.src = `/api/wizard/frame/${state.token}/crop.jpg?x=${fx}&y=${fy}&w=${fw}&h=${fh}`;

    function cvsXY(e, canvas) {
      const r = canvas.getBoundingClientRect();
      return [(e.clientX-r.left)*(canvas.width/r.width), (e.clientY-r.top)*(canvas.height/r.height)];
    }
    cvs.onmousedown = e => {
      const [x,y] = cvsXY(e, cvs);
      lineDrags[b.id] = {x0:x, y0:y, x1:x, y1:y};
    };
    cvs.onmousemove = e => {
      const ld = lineDrags[b.id]; if (!ld) return;
      const [x,y] = cvsXY(e, cvs);
      ld.x1 = x; ld.y1 = y; redrawLineCanvas(cvs, ld);
    };
    const finishLine = () => {
      const ld = lineDrags[b.id]; if (!ld) return;
      if (Math.hypot(ld.x1-ld.x0, ld.y1-ld.y0) > 5) {
        linesByBox[b.id] = {p1:[ld.x0,ld.y0], p2:[ld.x1,ld.y1]};
        redrawBoxCanvas();  // update ✓ badge on step-2 canvas
      }
      delete lineDrags[b.id];
      redrawLineCanvas(cvs);
    };
    cvs.onmouseup = finishLine;
    cvs.onmouseleave = finishLine;

    wrap.appendChild(cvs);
    section.appendChild(wrap);
    container.appendChild(section);
  }
}

function bandPoly(p1,p2,bw) {
  const dx=p2[0]-p1[0], dy=p2[1]-p1[1], len=Math.hypot(dx,dy);
  if (len < 1) return null;
  const nx=-dy/len*bw/2, ny=dx/len*bw/2;
  return [[p1[0]+nx,p1[1]+ny],[p2[0]+nx,p2[1]+ny],[p2[0]-nx,p2[1]-ny],[p1[0]-nx,p1[1]-ny]];
}

function redrawLineCanvas(cvs, drawing) {
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0,0,cvs.width,cvs.height);
  if (cvs._img && cvs._img.complete) ctx.drawImage(cvs._img,0,0,cvs.width,cvs.height);
  const b = boxes.find(x => x.id === cvs._boxId);
  const clr = b?.color==='red' ? '#e44' : '#44f';
  const bw  = (b?.band_width ?? 10) * cvs._zoom;

  function drawOneLine(p1, p2, alpha) {
    const poly = bandPoly(p1,p2,bw);
    if (poly) {
      ctx.save(); ctx.globalAlpha=(alpha||1)*0.25; ctx.fillStyle=clr;
      ctx.beginPath(); ctx.moveTo(poly[0][0],poly[0][1]);
      for (let i=1;i<poly.length;i++) ctx.lineTo(poly[i][0],poly[i][1]);
      ctx.closePath(); ctx.fill(); ctx.restore();
    }
    ctx.save(); ctx.globalAlpha=alpha||1; ctx.strokeStyle=clr; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]); ctx.stroke();
    [p1,p2].forEach(pt => { ctx.fillStyle=clr; ctx.beginPath(); ctx.arc(pt[0],pt[1],4,0,Math.PI*2); ctx.fill(); });
    ctx.restore();
  }

  const line = linesByBox[cvs._boxId];
  if (line) drawOneLine(line.p1, line.p2, 1);
  if (drawing) drawOneLine([drawing.x0,drawing.y0],[drawing.x1,drawing.y1], 0.5);
}

// ── step 4: tune ──────────────────────────────────────────────────────────────
function buildTunePanel() {
  const list = document.getElementById('goal-list');
  list.innerHTML = '';
  boxes.forEach((b, i) => {
    const hasLine = !!linesByBox[b.id];
    const clr = b.color==='red' ? '#e66' : '#66f';
    const det = document.createElement('details');
    det.className='goal-item'; det.open=true; det.dataset.idx=i;
    det.innerHTML = `<summary><span style="color:${clr}">■</span> ${b.name}` +
      (hasLine ? '' : ' <small style="color:#a63">(no line — skipped in save)</small>') +
      `</summary><div class="goal-fields">
  <div class="field-group full"><label>Name</label><input type="text" class="g-name" value="${b.name}"/></div>
  <div class="field-group"><label>Mode</label><select class="g-mode">
    <option value="outlet"${b.mode==='outlet'?' selected':''}>outlet</option>
    <option value="inlet"${b.mode==='inlet'?' selected':''}>inlet</option>
  </select></div>
  <div class="field-group"><label>ball_area</label><input type="number" class="g-ball-area" min="100" step="100" value="${b.ball_area}"/></div>
  <div class="field-group"><label>band_width</label><input type="number" class="g-band-width" min="1" step="1" value="${b.band_width}"/></div>
  <div class="field-group"><label>fall_ratio (0.1–1.0)</label><input type="number" class="g-fall-ratio" min="0.1" max="1.0" step="0.05" value="${b.fall_ratio}"/></div>
  <div class="field-group"><label>min_peak</label><input type="number" class="g-min-peak" min="0" step="1" value="${b.min_peak}"/></div>
  <div class="field-group"><label>cooldown (frames)</label><input type="number" class="g-cooldown" min="0" step="1" value="${b.cooldown}"/></div>
  <div class="field-group full"><label>Downsample</label>
    <div class="ds-row" id="ds-row-${i}"></div>
    <div style="font-size:0.78rem;color:#777;margin-top:0.2rem" id="ds-info-${i}"></div>
  </div>
  <div class="field-group full"><label>PFMS element ID <small style="color:#666">(leave blank to skip forwarding)</small></label><input type="text" class="g-pfms-element" placeholder="e.g. speaker" value="${b.pfms_element||''}"/></div>
  </div>`;
    list.appendChild(det);
    buildDsButtons(i, b);
  });
  updateRamSummary();
}

function buildDsButtons(i, b) {
  const row = document.getElementById('ds-row-'+i);
  DS_OPTS.forEach(v => {
    const btn = document.createElement('button');
    btn.className='ds-btn tool-btn'+(b.downsample===v?' active':'');
    btn.textContent = v===1.0 ? '1× (full)' : v+'×';
    btn.onclick = () => {
      b.downsample = v;
      row.querySelectorAll('.ds-btn').forEach(x=>x.classList.remove('active'));
      btn.classList.add('active');
      updateDsInfo(i,b); updateRamSummary();
    };
    row.appendChild(btn);
  });
  updateDsInfo(i,b);
}

function updateDsInfo(i, b) {
  const sc = state.frameScale;
  const fw = Math.round(Math.round(b.cw/sc)*b.downsample);
  const fh = Math.round(Math.round(b.ch/sc)*b.downsample);
  document.getElementById('ds-info-'+i).textContent =
    `→ ${fw}×${fh} pixels per frame (~${(fw*fh*0.00002*1800).toFixed(0)} MB)`;
}

function syncTuneToState() {
  document.querySelectorAll('.goal-item').forEach((det, i) => {
    const b = boxes[i]; if (!b) return;
    b.name      = det.querySelector('.g-name').value.trim() || b.name;
    b.mode      = det.querySelector('.g-mode').value;
    b.ball_area = parseInt(det.querySelector('.g-ball-area').value)  || 1500;
    b.band_width= parseInt(det.querySelector('.g-band-width').value) || 10;
    b.fall_ratio= parseFloat(det.querySelector('.g-fall-ratio').value)|| 0.7;
    b.min_peak     = parseInt(det.querySelector('.g-min-peak').value)   || 0;
    b.cooldown     = parseInt(det.querySelector('.g-cooldown').value)   || 0;
    b.pfms_element = det.querySelector('.g-pfms-element')?.value.trim() || null;
  });
}

function updateRamSummary() {
  let total = 0, n = 0;
  const sc = state.frameScale;
  for (const b of boxes) {
    const fw = Math.round(b.cw/sc)*(b.downsample||1.0), fh = Math.round(b.ch/sc)*(b.downsample||1.0);
    total += fw*fh*0.00002*1800; n++;
  }
  for (const s of state.streams) for (const g of s.goals) {
    total += g.crop.w*g.crop.h*(g.downsample||1.0)*(g.downsample||1.0)*0.00002*1800; n++;
  }
  const mb = total.toFixed(0), cls = total>2000?'err':total>500?'warn':'info';
  document.getElementById('ram-summary').innerHTML =
    `<div class="msg ${cls}">~${mb} MB estimated RAM for 60s buffer (${n} goal${n!==1?'s':''})</div>`;
}

// ── step 5: save ──────────────────────────────────────────────────────────────
function buildCurrentGoals() {
  const sc = state.frameScale;
  return boxes.filter(b => linesByBox[b.id]).map(b => {
    const fx=Math.round(b.cx/sc), fy=Math.round(b.cy/sc);
    const fw=Math.round(b.cw/sc), fh=Math.round(b.ch/sc);
    const line=linesByBox[b.id], zoom=boxZooms[b.id]||1;
    return {
      name:b.name, color:b.color, mode:b.mode,
      p1:[fx+Math.round(line.p1[0]/zoom), fy+Math.round(line.p1[1]/zoom)],
      p2:[fx+Math.round(line.p2[0]/zoom), fy+Math.round(line.p2[1]/zoom)],
      crop:{x:fx,y:fy,w:fw,h:fh},
      ball_area:b.ball_area, band_width:b.band_width, fall_ratio:b.fall_ratio,
      min_peak:b.min_peak, cooldown:b.cooldown, downsample:b.downsample,
    };
  });
}

function buildSavePayload() {
  const byUrl = {};
  for (const s of state.streams) { if (!byUrl[s.url]) byUrl[s.url]=[]; byUrl[s.url].push(...s.goals); }
  if (state.currentUrl) {
    const cur = buildCurrentGoals();
    if (cur.length) { if (!byUrl[state.currentUrl]) byUrl[state.currentUrl]=[]; byUrl[state.currentUrl].push(...cur); }
  }
  const streams = Object.entries(byUrl).map(([url,goals]) => ({
    source: url,
    goals: goals.map(g => {
      const obj = {
        name:g.name, mode:g.mode, draw_color:g.color==='red'?[0,0,255]:[255,0,0],
        line:[g.p1,g.p2], crop_override:[g.crop.x,g.crop.y,g.crop.x+g.crop.w,g.crop.y+g.crop.h],
        ball_area:g.ball_area, band_width:g.band_width, fall_ratio:g.fall_ratio,
        min_peak:g.min_peak, cooldown:g.cooldown,
      };
      if (g.downsample && g.downsample!==1.0) obj.downsample = g.downsample;
      if (g.pfms_element) obj.pfms_element = g.pfms_element;
      return obj;
    }),
  }));
  const pfmsUrl = document.getElementById('pfms-url')?.value.trim();
  const pfmsKey = document.getElementById('pfms-key')?.value.trim();
  const pfmsSrc = document.getElementById('pfms-source')?.value.trim();
  const payload = {config_path: _configPath||'', streams};
  if (pfmsUrl) {
    payload.pfms_url = pfmsUrl;
    if (pfmsKey) payload.pfms_key = pfmsKey;
    if (pfmsSrc) payload.pfms_source = pfmsSrc;
  }
  return payload;
}

function buildJsonPreview() {
  document.getElementById('json-preview').textContent = JSON.stringify(buildSavePayload(), null, 2);
}

async function saveConfig() {
  const payload = buildSavePayload();
  if (!payload.config_path) {
    const p = prompt('Config file path (e.g. configs/my-field.json):');
    if (!p) return;
    payload.config_path = p; _configPath = p;
  }
  showMsg('save-msgs','info','Saving…');
  try {
    const r = await fetch('/api/wizard/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await r.json();
    if (data.ok) showMsg('save-msgs','ok',`Config saved to <b>${payload.config_path}</b>. Restart the backend then <a href="/" style="color:#8f8">open the dashboard</a>.`);
    else showMsg('save-msgs','err', data.error||'Unknown error');
  } catch(e) { showMsg('save-msgs','err','Save failed: '+e.message); }
}

// ── utils ─────────────────────────────────────────────────────────────────────
function showMsg(elId,cls,html) {
  const el = document.getElementById(elId);
  if (el) el.innerHTML = cls ? `<div class="msg ${cls}">${html}</div>` : '';
}

function _colorFromBGR(bgr) { return bgr && bgr[0]>bgr[2] ? 'blue' : 'red'; }

function loadExistingBoxes(sourceUrl) {
  if (!_existingConfig) return;
  const src = _existingConfig.streams.find(s => s.source === sourceUrl);
  if (!src || !src.goals.length) return;
  const sc = state.frameScale;
  const canvas = document.getElementById('canvas');
  for (const g of src.goals) {
    const id = nextBoxId++;
    let cx=0, cy=0, cw=200, ch=200;
    if (g.crop_override) {
      const [x1,y1,x2,y2] = g.crop_override;
      cx=x1*sc; cy=y1*sc; cw=(x2-x1)*sc; ch=(y2-y1)*sc;
    } else if (g.line) {
      const pad=150*sc;
      const lx=[g.line[0][0]*sc, g.line[1][0]*sc], ly=[g.line[0][1]*sc, g.line[1][1]*sc];
      const mnX=Math.min(...lx), mxX=Math.max(...lx), mnY=Math.min(...ly), mxY=Math.max(...ly);
      cx=Math.max(0,mnX-pad); cy=Math.max(0,mnY-pad);
      cw=Math.min(canvas.width, mxX+pad)-cx; ch=Math.min(canvas.height, mxY+pad)-cy;
    }
    boxes.push({id, color:_colorFromBGR(g.draw_color), cx, cy, cw, ch,
      name:g.name||'goal', mode:g.mode||'outlet',
      ball_area:g.ball_area??1500, band_width:g.band_width??10,
      fall_ratio:g.fall_ratio??0.7, min_peak:g.min_peak??0,
      cooldown:g.cooldown??0, downsample:g.downsample??1.0,
      pfms_element:g.pfms_element||null});
    if (g.line) existingLinesFull[id] = {p1:g.line[0], p2:g.line[1]};
  }
  redrawBoxCanvas(); updateRamEst();
}

// ── init ──────────────────────────────────────────────────────────────────────
fetch('/api/wizard/current-config').then(r=>r.json()).then(data => {
  if (data.pfms_url) {
    const el = document.getElementById('pfms-url'); if (el) el.value = data.pfms_url;
    const ek = document.getElementById('pfms-key'); if (ek && data.pfms_key) ek.value = data.pfms_key;
    const es = document.getElementById('pfms-source'); if (es && data.pfms_source) es.value = data.pfms_source;
  }
  if (!data.streams||!data.streams.length) return;
  _existingConfig = data;
  const urlInput = document.getElementById('url-input');
  if (urlInput && !urlInput.value) urlInput.value = data.streams[0].source;
  const nGoals = data.streams.reduce((n,s)=>n+s.goals.length,0);
  showMsg('snap-msg','info',`Existing config loaded: ${nGoals} goal${nGoals!==1?'s':''} across ${data.streams.length} stream${data.streams.length!==1?'s':''}. Connect to reload.`);
}).catch(()=>{});

fetch('/api/wizard/config-path').then(r=>r.json()).then(d => {
  if (d.path) _configPath = d.path;
}).catch(()=>{});
</script>
</body>
</html>"""


_REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Ball Counter — Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
a{color:#7bf;text-decoration:none}
a:hover{text-decoration:underline}
/* nav */
#nav{display:flex;align-items:center;gap:0;background:#1a1a1a;border-bottom:1px solid #333;padding:0 1rem;height:42px;flex-shrink:0}
#nav .brand{font-size:0.95rem;font-weight:bold;color:#aaa;margin-right:1.2rem}
#nav .tab{padding:0 0.9rem;height:42px;display:flex;align-items:center;font-size:0.88rem;color:#888;cursor:pointer;border-bottom:2px solid transparent}
#nav .tab:hover{color:#ccc}
#nav .tab.active{color:#7bf;border-bottom-color:#7bf}
#nav .reviewer-bar{margin-left:auto;display:flex;align-items:center;gap:0.5rem;font-size:0.82rem}
#reviewer-select{background:#222;border:1px solid #444;color:#ccc;border-radius:4px;padding:0.25rem 0.5rem;font-size:0.82rem}
/* layout */
#main{display:flex;flex:1;overflow:hidden}
/* clip list */
#clip-list-panel{width:270px;flex-shrink:0;display:flex;flex-direction:column;border-right:1px solid #333;background:#161616}
#clip-list-header{padding:0.5rem;border-bottom:1px solid #2a2a2a;display:flex;flex-direction:column;gap:0.4rem}
#clip-search{width:100%;background:#1e1e1e;border:1px solid #333;color:#ccc;border-radius:4px;padding:0.3rem 0.5rem;font-size:0.82rem}
#clip-filter{width:100%;background:#1e1e1e;border:1px solid #333;color:#ccc;border-radius:4px;padding:0.3rem 0.5rem;font-size:0.82rem}
#clip-list{flex:1;overflow-y:auto}
.clip-item{padding:0.45rem 0.6rem;border-bottom:1px solid #222;cursor:pointer;display:flex;flex-direction:column;gap:0.15rem}
.clip-item:hover{background:#1e1e1e}
.clip-item.active{background:#1a2a3a;border-left:3px solid #7bf}
.clip-goal{font-size:0.85rem;font-weight:bold}
.clip-meta{font-size:0.75rem;color:#777;display:flex;gap:0.5rem;flex-wrap:wrap}
.clip-badge{font-size:0.7rem;padding:0.1rem 0.35rem;border-radius:3px;margin-top:0.1rem;width:fit-content}
.badge-annotated{background:#1a3a1a;color:#5d5;border:1px solid #3a3}
.badge-unannotated{background:#2a2a2a;color:#888;border:1px solid #444}
/* player */
#player-panel{flex:1;overflow-y:auto;padding:0.8rem 1rem;display:flex;flex-direction:column}
#clip-header{margin-bottom:0.5rem;display:flex;align-items:baseline;gap:0.8rem;flex-wrap:wrap}
#clip-title{font-size:1rem;font-weight:bold}
#clip-subtitle{font-size:0.8rem;color:#888}
video{width:100%;max-height:50vh;background:#000;display:block;border-radius:4px}
/* timeline */
#timeline-wrap{position:relative;margin:0.4rem 0}
#timeline{width:100%;height:60px;display:block;cursor:crosshair;background:#1a1a1a;border-radius:4px}
#speed-btns{display:contents}
#speed-btns span{font-size:0.72rem;color:#555;margin-right:0.1rem}
.speed-btn{background:#1e1e1e;border:1px solid #333;color:#888}
.speed-btn:hover{background:#2a2a2a;color:#ccc}
.speed-btn.active{background:#1a2a3a;border-color:#38a;color:#8cf}
/* events row */
/* annotations */
#anno-section{margin-top:0.6rem;border-top:1px solid #2a2a2a;padding-top:0.6rem}
.anno-controls{display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.5rem}
#n-balls-input{width:60px;background:#1e1e1e;border:1px solid #444;color:#eee;border-radius:4px;padding:0.3rem 0.5rem;font-size:0.9rem}
.anno-btn,.trim-btn,.toggle-btn,.speed-btn{padding:0.3rem 0.7rem;font-size:0.82rem;border-radius:4px;cursor:pointer;touch-action:manipulation}
.anno-btn{background:#2a2a3a;border:1px solid #64a;color:#a8f}
.anno-btn:hover{background:#3a3a4a}
.anno-btn.danger{background:#3a1a1a;border-color:#a33;color:#f88}
.anno-btn.danger:hover{background:#4a2020}
.toggle-btn{display:flex;align-items:center;gap:0.3rem;background:#2a2a2a;border:1px solid #444;color:#888;cursor:pointer;user-select:none}
.toggle-btn input{display:none}
.toggle-btn.on{background:#1a2a3a;border-color:#38a;color:#8cf}
.save-btn{background:#1a4a1a;border-color:#3a3;color:#8f8}
.save-btn:hover{background:#1e5a1e}
#my-marks{margin-top:0.4rem}
#my-marks-list{list-style:none;display:flex;flex-direction:column;gap:0.25rem;max-height:8rem;overflow-y:auto}
.mark-item{display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;padding:0.2rem 0.4rem;background:#1a1a1a;border-radius:3px;border-left:2px solid transparent;transition:background 0.1s}
.mark-item.near{background:#1a2a3a;border-left-color:#7bf}
.mark-time{color:#7bf;cursor:pointer;min-width:50px}
.mark-time:hover{text-decoration:underline}
.mark-n{color:#fc8;min-width:20px}
.mark-ts{color:#555;font-size:0.72rem;flex:1}
.mark-del{background:none;border:none;color:#633;cursor:pointer;font-size:0.85rem;padding:0 0.2rem}
.mark-del:hover{color:#f44}
#other-reviewers{margin-top:0.6rem}
#other-reviewers h4{font-size:0.78rem;color:#666;margin-bottom:0.3rem}
.other-mark-item{display:flex;align-items:center;gap:0.5rem;font-size:0.78rem;padding:0.15rem 0.4rem;background:#141414;border-radius:3px;margin-bottom:0.2rem}
.other-mark-time{color:#8af;cursor:pointer;min-width:50px}
.other-mark-time:hover{text-decoration:underline}
/* agreement */
#agreement-section{margin-top:0.6rem;border-top:1px solid #2a2a2a;padding-top:0.6rem;display:none}
#agreement-section h3{font-size:0.82rem;color:#888;margin-bottom:0.4rem}
#agreement-table{border-collapse:collapse;font-size:0.78rem;width:100%}
#agreement-table th,#agreement-table td{padding:0.2rem 0.5rem;border:1px solid #2a2a2a;text-align:left}
#agreement-table th{background:#1a1a1a;color:#888}
/* download row */
#footer-bar{display:flex;align-items:center;gap:0.8rem;font-size:0.8rem;color:#888;padding:0.4rem 0;margin-top:auto;border-top:1px solid #333;flex-shrink:0}
/* empty state */
#empty-state{display:flex;align-items:center;justify-content:center;height:100%;color:#555;font-size:1rem}
/* trim mode */
#trim-toolbar{display:none;margin:0.4rem 0;padding:0.5rem;background:#1a1a1a;border:1px solid #333;border-radius:4px}
#trim-toolbar.active{display:block}
.trim-actions{display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;font-size:0.82rem}
.trim-btn{background:#2a2a2a;border:1px solid #444;color:#ccc}
.trim-btn:hover{background:#3a3a3a}
.trim-btn.primary{background:#1a3a1a;border-color:#3a3;color:#8f8}
.trim-btn.primary:hover{background:#2a5a2a}
.trim-btn.cancel{background:#3a1a1a;border-color:#a33;color:#f88}
.trim-btn.cancel:hover{background:#4a2020}
#trim-segments-list{margin-top:0.4rem;font-size:0.78rem}
.trim-seg-item{display:flex;align-items:center;gap:0.5rem;padding:0.2rem 0.4rem;background:#1a2a1a;border-radius:3px;margin-bottom:0.2rem;border-left:2px solid #3a3}
.trim-seg-remove{background:none;border:none;color:#633;cursor:pointer;font-size:0.85rem;padding:0 0.2rem}
.trim-seg-remove:hover{color:#f44}
.trim-hint{color:#555;font-size:0.75rem;font-style:italic}
/* menu button (mobile only) */
#menu-btn{display:none;background:none;border:none;color:#aaa;font-size:1.3rem;cursor:pointer;padding:0 0.5rem}
#menu-btn:hover{color:#fff}
/* mobile */
@media(max-width:768px){
  #menu-btn{display:block}
  #clip-list-panel{display:none;position:fixed;inset:42px 0 0 0;z-index:50;width:100%;border-right:none}
  #clip-list-panel.open{display:flex}
  #player-panel{padding:0.5rem}
  #clip-header{font-size:0.85rem;gap:0.4rem}
  video{max-height:35vh}
  #timeline{height:44px}
  #primary-controls{position:fixed;bottom:0;left:0;right:0;z-index:40;background:#1a1a1a;border-top:1px solid #333;padding:0.5rem;justify-content:center;gap:0.8rem;margin:0}
  #primary-controls .anno-btn,.save-btn{padding:0.5rem 1.2rem;font-size:1rem;touch-action:manipulation}
  #secondary-controls{font-size:0.78rem}
  #footer-bar{display:none}
  #anno-section{padding-bottom:3.5rem}
  .reviewer-bar span:first-child{display:none}
}
/* modal */
#modal-overlay{display:none;position:fixed;inset:0;background:#0009;z-index:100;align-items:center;justify-content:center}
#modal-overlay.show{display:flex}
#modal-box{background:#1e1e1e;border:1px solid #444;border-radius:8px;padding:1.4rem;min-width:300px;max-width:400px}
#modal-box h2{font-size:1rem;margin-bottom:0.8rem;color:#ccc}
#reviewer-name-input{width:100%;background:#141414;border:1px solid #444;color:#eee;border-radius:4px;padding:0.4rem 0.6rem;font-size:0.95rem;margin-bottom:0.6rem}
.modal-row{display:flex;gap:0.5rem;justify-content:flex-end}
/* toast */
#toast-container{position:fixed;bottom:1.2rem;right:1.2rem;z-index:200;display:flex;flex-direction:column;gap:0.5rem;pointer-events:none}
.toast{padding:0.55rem 1rem;border-radius:6px;font-size:0.85rem;color:#eee;opacity:0;transform:translateY(8px);transition:opacity 0.2s,transform 0.2s;pointer-events:none;max-width:320px}
.toast.show{opacity:1;transform:translateY(0)}
.toast.info{background:#1e3a5c;border:1px solid #38a}
.toast.ok{background:#1a3a1a;border:1px solid #3a3;color:#8f8}
.toast.err{background:#3a1a1a;border:1px solid #a33;color:#f88}
</style>
</head>
<body>

<div id="toast-container"></div>

<!-- Nav -->
<div id="nav">
  <button id="menu-btn" onclick="toggleClipPanel()">&#9776;</button>
  <span class="brand">Ball Counter</span>
  <div class="reviewer-bar">
    <span style="color:#666">Reviewer:</span>
    <span id="reviewer-label" style="color:#ccc;font-size:0.82rem"></span>
  </div>
  <a href="https://github.com/cinderblock/balls-counter" target="_blank" style="margin-left:auto;color:#444;font-size:0.78rem;text-decoration:none" title="GitHub">&#9135; cinderblock/balls-counter</a>
</div>

<!-- Main -->
<div id="main">

  <!-- Clip List -->
  <div id="clip-list-panel">
    <div id="clip-list-header">
      <input id="clip-search" type="text" placeholder="Search clips..." oninput="renderClipList()"/>
      <select id="clip-filter" onchange="renderClipList()">
        <option value="all">All clips</option>
        <option value="annotated">Annotated by me</option>
        <option value="unannotated">Unannotated by me</option>
      </select>
      <select id="clip-sort" onchange="renderClipList()">
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="fewest-reviews">Fewest reviews</option>
        <option value="most-events">Most events</option>
      </select>
      <div id="clip-count" style="font-size:0.72rem;color:#555;text-align:center"></div>
    </div>
    <div id="clip-list"></div>
  </div>

  <!-- Player Panel -->
  <div id="player-panel">
    <div id="empty-state">Select a clip to review</div>
    <div id="player-content" style="display:none">
      <div id="clip-header">
        <span id="clip-title"></span>
        <span id="clip-subtitle"></span>
        <span style="flex:1"></span>
        <a id="download-link" href="#">Download zip</a>
      </div>
      <video id="video" controls playsinline></video>
      <div id="timeline-wrap">
        <canvas id="timeline"></canvas>
      </div>
      <div id="trim-toolbar">
        <div class="trim-actions">
          <span style="color:#888">Trim / Split</span>
          <button class="trim-btn" onclick="autoDetectSegments()">Auto-detect</button>
          <span class="trim-hint">or click+drag on timeline to add segments</span>
          <span style="flex:1"></span>
          <label style="color:#888;display:flex;align-items:center;gap:0.3rem;cursor:pointer;font-size:0.78rem"><input type="checkbox" id="trim-delete-original" checked/>Delete original</label>
          <button class="trim-btn primary" onclick="applyTrim()">Apply</button>
          <button class="trim-btn cancel" onclick="exitTrimMode()">Cancel</button>
        </div>
        <div id="trim-segments-list"></div>
      </div>
      <div id="anno-section">
        <div id="primary-controls" class="anno-controls">
          <button class="anno-btn save-btn" onclick="saveAnnotations()">Save</button>
          <button class="anno-btn danger" onclick="undoMark()">Undo</button>
          <button class="anno-btn" onclick="markScore()">Mark</button>
        </div>
        <div id="secondary-controls" class="anno-controls">
          <label class="toggle-btn" id="auto-play-btn"><input type="checkbox" id="auto-play" onchange="localStorage.setItem('pref_autoplay',this.checked);this.parentElement.classList.toggle('on',this.checked)"/>Autoplay</label>
          <label class="toggle-btn" id="auto-advance-btn"><input type="checkbox" id="auto-advance" onchange="localStorage.setItem('pref_autoadvance',this.checked);this.parentElement.classList.toggle('on',this.checked)"/>Auto-next</label>
          <button class="anno-btn danger" onclick="clearAllMarks()">Clear all</button>
          <span style="flex:1"></span>
          <span id="speed-btns">
          <button class="speed-btn" data-rate="0.25" onclick="setSpeed(0.25)">¼×</button>
          <button class="speed-btn" data-rate="0.5"  onclick="setSpeed(0.5)">½×</button>
          <button class="speed-btn" data-rate="1"    onclick="setSpeed(1)">1×</button>
          <button class="speed-btn" data-rate="2"    onclick="setSpeed(2)">2×</button>
          <button class="speed-btn" data-rate="4"    onclick="setSpeed(4)">4×</button>
          </span>
          <label class="toggle-btn on" id="autospeed-btn" onclick="toggleAutoSpeed()">Auto-speed</label>
          <button class="trim-btn" id="trim-enter-btn" onclick="enterTrimMode()">Trim / Split</button>
        </div>
        <div id="my-marks">
          <div style="font-size:0.78rem;color:#666;margin-bottom:0.2rem">My marks: <span id="my-marks-count" style="color:#fc8;font-weight:bold"></span></div>
          <ul id="my-marks-list"></ul>
        </div>
        <div id="other-reviewers"></div>
      </div>
      <div id="agreement-section">
        <h3>Agreement</h3>
        <table id="agreement-table"></table>
      </div>
    </div>
    <div id="footer-bar">
      <span style="color:#555">Shortcuts: Space=mark &nbsp; Ctrl+Z=undo &nbsp; Ctrl+S=save &nbsp; ←/→=seek 2s &nbsp; ,/.=speed</span>
    </div>
  </div>
</div>

<!-- Reviewer modal -->
<div id="modal-overlay">
  <div id="modal-box">
    <h2>Who are you?</h2>
    <div style="display:flex;gap:0.5rem;margin-bottom:0.6rem">
      <input id="reviewer-team-input" type="text" inputmode="numeric" placeholder="Team #" maxlength="10" style="width:80px;background:#141414;border:1px solid #444;color:#eee;border-radius:4px;padding:0.4rem 0.6rem;font-size:0.95rem" oninput="this.value=this.value.replace(/[^0-9*]/g,'')"/>
      <input id="reviewer-name-input" type="text" placeholder="Your name" maxlength="64" style="flex:1;background:#141414;border:1px solid #444;color:#eee;border-radius:4px;padding:0.4rem 0.6rem;font-size:0.95rem"/>
    </div>
    <div class="modal-row">
      <button class="anno-btn" onclick="createReviewer()">Start reviewing</button>
    </div>
  </div>
</div>

<script>
// ── state ─────────────────────────────────────────────────────────────────────
let allClips = [];
let reviewers = {};
let currentClip = null;   // sidecar JSON
let myToken = null;
let myMarks = [];         // [{video_time, frame_idx, timestamp, n_balls}]
let lastSave = null;      // {clipId, marks, videoTime} for undo-after-save
let autoSpeed = localStorage.getItem('pref_autospeed') !== 'false';
let timelineRAF = null;
let speedMap = null;      // Uint8Array: 0=fast(2x), 1=near-motion(1x), 2=in-motion(0.25x)

// trim mode
let trimMode = false;
let trimSegments = [];    // [{startFrame, endFrame}, ...]
let trimDragHandle = null; // {segIdx, edge: 'start'|'end'}
let trimAdding = false;
let trimAddStart = null;
let trimAddEnd = null;

function buildSpeedMap(signal, fps) {
  const buf = new Uint8Array(signal.length);
  const window = Math.round(3 * fps);
  for (let i = 0; i < signal.length; i++) {
    if (signal[i] > 0) {
      buf[i] = 2; // in-motion
      const lo = Math.max(0, i - window);
      const hi = Math.min(signal.length - 1, i + window);
      for (let j = lo; j <= hi; j++) if (buf[j] < 1) buf[j] = 1; // near-motion
    }
  }
  return buf;
}

// ── toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = 'info', durationMs = 3500) {
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  const container = document.getElementById('toast-container');
  container.appendChild(el);
  requestAnimationFrame(() => { requestAnimationFrame(() => { el.classList.add('show'); }); });
  setTimeout(() => {
    el.classList.remove('show');
    el.addEventListener('transitionend', () => el.remove(), {once: true});
  }, durationMs);
}

// ── init ──────────────────────────────────────────────────────────────────────
async function init() {
  myToken = getCookie('reviewer_token') || localStorage.getItem('reviewer_token') || null;
  const cb = document.getElementById('auto-advance');
  if (cb) { cb.checked = localStorage.getItem('pref_autoadvance') !== 'false'; cb.parentElement.classList.toggle('on', cb.checked); }
  const ap = document.getElementById('auto-play');
  if (ap) { ap.checked = localStorage.getItem('pref_autoplay') !== 'false'; ap.parentElement.classList.toggle('on', ap.checked); }
  updateAutoSpeedBtn();
  updateSpeedBtns(1);
  await loadReviewers();
  await loadClips();
  renderClipList();
  if (location.hash) {
    const id = location.hash.slice(1);
    const clip = allClips.find(c => c.id === id);
    if (clip) openClip(id);
  } else if (myToken) {
    // Load a random unreviewed clip on fresh page load
    nextUnannotated();
  }
  // Require reviewer selection
  if (!myToken || !reviewers[myToken]) showModal();
  // Poll for new clips every 10s
  setInterval(async () => {
    const prev = allClips.length;
    await loadClips();
    if (allClips.length !== prev) { renderClipList(); toast('Clip list updated', 'info', 2000); }
  }, 10000);
}

// ── reviewers ─────────────────────────────────────────────────────────────────
async function loadReviewers() {
  const r = await fetch('/api/reviewers');
  reviewers = await r.json();
  if (myToken && reviewers[myToken]) {
    document.getElementById('reviewer-label').textContent = reviewers[myToken].label;
  }
}

async function createReviewer() {
  const team = (document.getElementById('reviewer-team-input').value || '').trim();
  const name = (document.getElementById('reviewer-name-input').value || '').trim();
  if (!name) { toast('Name is required', 'err'); return; }
  if (!team) { toast('Team # is required', 'err'); return; }
  const label = team === '*' ? name : team + ' ' + name;
  const r = await fetch('/api/reviewer/create', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({label})
  });
  const data = await r.json();
  myToken = data.token;
  setCookie('reviewer_token', myToken, 365);
  localStorage.setItem('reviewer_token', myToken);
  hideModal();
  await loadReviewers();
  if (currentClip) openClip(currentClip.id);
  else nextUnannotated();
}

function showModal() { document.getElementById('modal-overlay').classList.add('show'); }
function hideModal() { document.getElementById('modal-overlay').classList.remove('show'); }
function toggleClipPanel() { document.getElementById('clip-list-panel').classList.toggle('open'); }

// ── clips ──────────────────────────────────────────────────────────────────────
async function loadClips() {
  const r = await fetch('/api/clips');
  allClips = await r.json();
}

function renderClipList() {
  const search = document.getElementById('clip-search').value.toLowerCase();
  const filter = document.getElementById('clip-filter').value;
  const sort = document.getElementById('clip-sort').value;
  const list = document.getElementById('clip-list');
  list.innerHTML = '';
  const filtered = allClips.filter(c => {
    if (search && !c.id.toLowerCase().includes(search) && !c.goal.toLowerCase().includes(search)) return false;
    if (filter === 'annotated' && !(myToken && c.annotators.includes(myToken))) return false;
    if (filter === 'unannotated' && myToken && c.annotators.includes(myToken)) return false;
    return true;
  });
  if (sort === 'oldest') filtered.reverse();
  else if (sort === 'fewest-reviews') filtered.sort((a, b) => a.annotators.length - b.annotators.length);
  else if (sort === 'most-events') filtered.sort((a, b) => b.n_events - a.n_events);
  // Update count
  const myAnnotated = allClips.filter(c => myToken && c.annotators.includes(myToken)).length;
  document.getElementById('clip-count').textContent = filtered.length + ' shown / ' + allClips.length + ' total / ' + myAnnotated + ' reviewed';
  for (const clip of filtered) {
    const div = document.createElement('div');
    div.className = 'clip-item' + (currentClip && currentClip.id === clip.id ? ' active' : '');
    div.dataset.id = clip.id;
    const goalColor = clip.goal.includes('red') ? '#e55' : (clip.goal.includes('blue') ? '#5af' : '#aaa');
    const annotated = myToken && clip.annotators.includes(myToken);
    const dur = clip.duration ? clip.duration.toFixed(1) + 's' : '';
    div.innerHTML =
      '<div class="clip-goal" style="color:' + goalColor + '">' + esc(clip.goal) + '</div>' +
      '<div class="clip-meta">' +
        '<span>' + esc(clip.saved_at || clip.id) + '</span>' +
        (dur ? '<span>' + dur + '</span>' : '') +
        '<span>' + clip.n_events + ' events</span>' +
      '</div>' +
      '<div class="clip-badge ' + (annotated ? 'badge-annotated' : 'badge-unannotated') + '">' +
        (annotated ? '&#x2713; annotated' : '&#x25cb; unannotated') +
      '</div>';
    div.onclick = () => openClip(clip.id);
    list.appendChild(div);
  }
}

// ── open clip ─────────────────────────────────────────────────────────────────
async function openClip(id) {
  document.getElementById('clip-list-panel').classList.remove('open');
  const r = await fetch('/api/clips/' + id);
  if (!r.ok) return;
  currentClip = await r.json();
  location.hash = id;
  if (trimMode) exitTrimMode();
  myMarks = [];
  if (myToken && currentClip.annotations && currentClip.annotations[myToken]) {
    myMarks = currentClip.annotations[myToken].marks || [];
  }

  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('player-content').style.display = 'block';

  // header
  const goalColor = currentClip.goal.includes('red') ? '#e55' : (currentClip.goal.includes('blue') ? '#5af' : '#aaa');
  document.getElementById('clip-title').innerHTML = '<span style="color:' + goalColor + '">' + esc(currentClip.goal) + '</span>';
  const dur = currentClip.duration ? ' &nbsp; ' + currentClip.duration.toFixed(1) + 's' : '';
  const nevt = ' &nbsp; ' + (currentClip.events ? currentClip.events.length : 0) + ' events';
  document.getElementById('clip-subtitle').innerHTML = esc(currentClip.saved_at || id) + dur + nevt;

  // video
  const video = document.getElementById('video');
  video.src = '/api/clips/' + id + '/video';
  video.load();
  if (document.getElementById('auto-play')?.checked) video.play().catch(() => {});

  // timeline
  drawTimeline();
  video.ontimeupdate = () => drawTimeline();

  // autospeed — reset rate when clip changes but keep the toggle state
  video.playbackRate = 1;
  speedMap = currentClip.signal ? buildSpeedMap(currentClip.signal, currentClip.fps || 30) : null;

  // events row

  // marks
  renderMyMarks();
  renderOtherReviewers();

  // agreement
  loadAgreement();

  // download link
  document.getElementById('download-link').href = '/api/clips/' + id + '/download';

  // highlight in list
  document.querySelectorAll('.clip-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id);
  });
}

// ── timeline ──────────────────────────────────────────────────────────────────
function drawTimeline() {
  const canvas = document.getElementById('timeline');
  const video = document.getElementById('video');
  if (!currentClip) return;
  const W = canvas.offsetWidth; const H = 60;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const signal = currentClip.signal || [];
  const fps = currentClip.fps || 30;
  const nFrames = currentClip.n_frames || (signal.length > 0 ? signal.length : 1);
  const dur = nFrames / fps;

  // signal fill
  if (signal.length > 1) {
    const maxSig = Math.max(...signal, 1);
    ctx.fillStyle = '#1a4a1a';
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let i = 0; i < signal.length; i++) {
      const x = (i / (signal.length - 1)) * W;
      const y = H - (signal[i] / maxSig) * H * 0.9;
      if (i === 0) ctx.lineTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fill();
  }

  // trim mode overlays
  if (trimMode) {
    const sorted = [...trimSegments].sort((a, b) => a.startFrame - b.startFrame);
    // shade discard regions
    ctx.fillStyle = 'rgba(60, 15, 15, 0.55)';
    let prevX = 0;
    for (const seg of sorted) {
      const x1 = (seg.startFrame / nFrames) * W;
      const x2 = (seg.endFrame / nFrames) * W;
      if (x1 > prevX) ctx.fillRect(prevX, 0, x1 - prevX, H);
      prevX = x2;
    }
    if (prevX < W) ctx.fillRect(prevX, 0, W - prevX, H);
    // segment edge lines + handles
    ctx.strokeStyle = '#3a3'; ctx.lineWidth = 2;
    ctx.fillStyle = '#5d5';
    for (const seg of sorted) {
      const x1 = (seg.startFrame / nFrames) * W;
      const x2 = (seg.endFrame / nFrames) * W;
      ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, 0); ctx.lineTo(x2, H); ctx.stroke();
      ctx.fillRect(x1 - 3, 0, 6, 8);
      ctx.fillRect(x1 - 3, H - 8, 6, 8);
      ctx.fillRect(x2 - 3, 0, 6, 8);
      ctx.fillRect(x2 - 3, H - 8, 6, 8);
    }
    // pending segment preview
    if (trimAdding && trimAddStart !== null && trimAddEnd !== null) {
      const px1 = (Math.min(trimAddStart, trimAddEnd) / nFrames) * W;
      const px2 = (Math.max(trimAddStart, trimAddEnd) / nFrames) * W;
      ctx.fillStyle = 'rgba(40, 120, 40, 0.3)';
      ctx.fillRect(px1, 0, px2 - px1, H);
    }
  }

  // auto-detected events (red dots)
  if (currentClip.events) {
    ctx.fillStyle = '#e44';
    for (const ev of currentClip.events) {
      const t = ev.frame_idx != null ? ev.frame_idx / fps : (ev.video_time || 0);
      const x = (t / dur) * W;
      ctx.beginPath(); ctx.arc(x, 6, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  // my marks (blue lines)
  ctx.strokeStyle = '#36f';
  ctx.lineWidth = 1.5;
  for (const m of myMarks) {
    const t = m.video_time || 0;
    const x = (t / dur) * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  // current time (yellow)
  if (video.duration) {
    const x = (video.currentTime / video.duration) * W;
    ctx.strokeStyle = '#ff0';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
}

let dragMarkIdx = null;

function timelineFracToTime(frac) {
  if (!currentClip) return 0;
  const fps = currentClip.fps || 30;
  const dur = (currentClip.n_frames || 1) / fps;
  return Math.max(0, Math.min(dur, frac * dur));
}

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('timeline');
  const SNAP_PX = 8;

  canvas.addEventListener('mousedown', e => {
    if (!currentClip) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const W = rect.width;
    const fps = currentClip.fps || 30;
    const nFrames = currentClip.n_frames || 1;
    const dur = nFrames / fps;

    // Trim mode interactions
    if (trimMode) {
      const frame = Math.round((px / W) * nFrames);
      const HANDLE_PX = 8;
      for (let i = 0; i < trimSegments.length; i++) {
        const seg = trimSegments[i];
        const sx = (seg.startFrame / nFrames) * W;
        const ex = (seg.endFrame / nFrames) * W;
        if (Math.abs(px - sx) < HANDLE_PX) {
          trimDragHandle = {segIdx: i, edge: 'start'};
          e.preventDefault(); return;
        }
        if (Math.abs(px - ex) < HANDLE_PX) {
          trimDragHandle = {segIdx: i, edge: 'end'};
          e.preventDefault(); return;
        }
      }
      trimAdding = true;
      trimAddStart = frame;
      trimAddEnd = frame;
      e.preventDefault();
      return;
    }

    // Check if near a mark — start drag
    for (let i = 0; i < myMarks.length; i++) {
      const mx = (myMarks[i].video_time / dur) * W;
      if (Math.abs(px - mx) < SNAP_PX) {
        dragMarkIdx = i;
        e.preventDefault();
        return;
      }
    }

    // Normal seek
    const video = document.getElementById('video');
    if (!video.duration) return;
    video.currentTime = timelineFracToTime(px / W);
    drawTimeline();
  });

  canvas.addEventListener('mousemove', e => {
    if (trimMode && currentClip) {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const W = rect.width;
      const nFrames = currentClip.n_frames || 1;
      const HANDLE_PX = 8;
      let nearHandle = false;
      for (const seg of trimSegments) {
        const sx = (seg.startFrame / nFrames) * W;
        const ex = (seg.endFrame / nFrames) * W;
        if (Math.abs(px - sx) < HANDLE_PX || Math.abs(px - ex) < HANDLE_PX) { nearHandle = true; break; }
      }
      canvas.style.cursor = nearHandle ? 'ew-resize' : 'crosshair';
      return;
    }
    if (dragMarkIdx !== null) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const W = rect.width;
    const fps = currentClip ? currentClip.fps || 30 : 30;
    const dur = currentClip ? (currentClip.n_frames || 1) / fps : 1;
    const nearMark = myMarks.some(m => Math.abs((m.video_time / dur) * W - px) < SNAP_PX);
    canvas.style.cursor = nearMark ? 'ew-resize' : 'crosshair';
  });
});

document.addEventListener('mousemove', e => {
  // Trim mode dragging
  if (trimMode && currentClip) {
    const canvas = document.getElementById('timeline');
    const rect = canvas.getBoundingClientRect();
    const px = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const nFrames = currentClip.n_frames || 1;
    const frame = Math.round((px / rect.width) * nFrames);
    if (trimDragHandle !== null) {
      const seg = trimSegments[trimDragHandle.segIdx];
      if (trimDragHandle.edge === 'start') seg.startFrame = Math.max(0, Math.min(frame, seg.endFrame - 1));
      else seg.endFrame = Math.min(nFrames, Math.max(frame, seg.startFrame + 1));
      drawTimeline();
      return;
    }
    if (trimAdding) {
      trimAddEnd = frame;
      drawTimeline();
      return;
    }
  }
  if (dragMarkIdx === null || !currentClip) return;
  const canvas = document.getElementById('timeline');
  const rect = canvas.getBoundingClientRect();
  const px = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
  const fps = currentClip.fps || 30;
  const dur = (currentClip.n_frames || 1) / fps;
  const newTime = (px / rect.width) * dur;
  myMarks[dragMarkIdx].video_time = newTime;
  myMarks[dragMarkIdx].frame_idx = Math.round(newTime * fps);
  document.getElementById('video').currentTime = newTime;
  drawTimeline();
});

document.addEventListener('mouseup', () => {
  // Trim mode
  if (trimMode) {
    if (trimDragHandle !== null) {
      trimDragHandle = null;
      renderTrimSegments();
      drawTimeline();
    }
    if (trimAdding && trimAddStart !== null && trimAddEnd !== null) {
      const sf = Math.min(trimAddStart, trimAddEnd);
      const ef = Math.max(trimAddStart, trimAddEnd);
      const nFrames = currentClip ? currentClip.n_frames || 1 : 1;
      if (ef - sf > Math.round(nFrames * 0.005)) {
        trimSegments.push({startFrame: sf, endFrame: ef});
        trimSegments.sort((a, b) => a.startFrame - b.startFrame);
        // merge overlapping
        let merged = [trimSegments[0]];
        for (let i = 1; i < trimSegments.length; i++) {
          const prev = merged[merged.length - 1];
          if (trimSegments[i].startFrame <= prev.endFrame) {
            prev.endFrame = Math.max(prev.endFrame, trimSegments[i].endFrame);
          } else {
            merged.push(trimSegments[i]);
          }
        }
        trimSegments = merged;
        renderTrimSegments();
      }
    }
    trimAdding = false;
    trimAddStart = null;
    trimAddEnd = null;
    drawTimeline();
    return;
  }
  if (dragMarkIdx !== null) {
    dragMarkIdx = null;
    renderMyMarks();
  }
});

// ── auto-speed ────────────────────────────────────────────────────────────────
const SPEEDS = [0.25, 0.5, 1, 2, 4];

function setSpeed(rate) {
  autoSpeed = false;
  localStorage.setItem('pref_autospeed', 'false');
  document.getElementById('video').playbackRate = rate;
  updateAutoSpeedBtn();
  updateSpeedBtns(rate);
}

function stepSpeed(delta) {
  const video = document.getElementById('video');
  const cur = video.playbackRate;
  const idx = SPEEDS.indexOf(cur);
  const next = idx < 0
    ? SPEEDS[delta > 0 ? SPEEDS.length - 1 : 0]
    : SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, idx + delta))];
  setSpeed(next);
}

function updateSpeedBtns(rate) {
  const row = document.getElementById('speed-btns');
  if (!row) return;
  row.querySelectorAll('.speed-btn').forEach(btn => {
    btn.style.display = autoSpeed ? 'none' : '';
    btn.classList.toggle('active', parseFloat(btn.dataset.rate) === rate);
  });
  // Also hide the "Speed:" label when auto-speed is on
  const label = row.querySelector('span');
  if (label) label.style.display = autoSpeed ? 'none' : '';
}

function toggleAutoSpeed() {
  autoSpeed = !autoSpeed;
  localStorage.setItem('pref_autospeed', autoSpeed);
  updateAutoSpeedBtn();
  if (!autoSpeed) {
    const video = document.getElementById('video');
    video.playbackRate = 1;
    updateSpeedBtns(1);
  } else {
    updateSpeedBtns(0);
  }
}

function updateAutoSpeedBtn() {
  document.getElementById('autospeed-btn').classList.toggle('on', autoSpeed);
}

// Poll every 100ms: auto-speed + highlight closest mark
setInterval(() => {
  if (!currentClip) return;
  const video = document.getElementById('video');
  const fps = currentClip.fps || 30;

  // auto-speed
  if (autoSpeed && speedMap && video.duration && !video.paused) {
    const frameIdx = Math.min(Math.round(video.currentTime * fps), speedMap.length - 1);
    // Look ahead by the distance we'll travel before the next poll (plus a small buffer)
    // so we never overshoot a zone boundary at high speed.
    const lookaheadSec = Math.max(video.playbackRate * 0.12, 0.3);
    const lookaheadFrames = Math.round(lookaheadSec * fps);
    let worstZone = 0;
    for (let f = frameIdx; f <= Math.min(frameIdx + lookaheadFrames, speedMap.length - 1); f++) {
      if (speedMap[f] > worstZone) worstZone = speedMap[f];
    }
    if (worstZone === 2) video.playbackRate = 0.25;
    else if (worstZone === 1) video.playbackRate = 1;
    else video.playbackRate = 2;
  }

  // highlight closest mark row
  const cur = video.currentTime;
  let closestIdx = -1, closestDist = 3.0;
  for (let i = 0; i < myMarks.length; i++) {
    const d = Math.abs(myMarks[i].video_time - cur);
    if (d < closestDist) { closestDist = d; closestIdx = i; }
  }
  document.querySelectorAll('.mark-item').forEach(el => {
    const near = parseInt(el.dataset.idx) === closestIdx;
    if (el.classList.contains('near') !== near) {
      el.classList.toggle('near', near);
      if (near) el.scrollIntoView({block: 'nearest'});
    }
  });
}, 100);

// ── events row ────────────────────────────────────────────────────────────────
function renderEventsRow() {
  const container = document.getElementById('events-btns');
  container.innerHTML = '';
  if (!currentClip || !currentClip.events) return;
  const fps = currentClip.fps || 30;
  currentClip.events.forEach((ev, i) => {
    const t = ev.frame_idx != null ? ev.frame_idx / fps : (ev.video_time || 0);
    const btn = document.createElement('button');
    btn.className = 'event-btn';
    btn.textContent = '+ ' + (ev.n_balls || '?') + ' @ ' + t.toFixed(1) + 's';
    btn.onclick = () => { document.getElementById('video').currentTime = Math.max(0, t - 1); };
    container.appendChild(btn);
  });
}

// ── marks ──────────────────────────────────────────────────────────────────────
function markScore() {
  if (!currentClip) return;
  if (!myToken) { toast('Select a reviewer first.', 'err'); showModal(); return; }
  const video = document.getElementById('video');
  const fps = currentClip.fps || 30;
  // 150ms reaction-time compensation only while playing
  const lag = video.paused ? 0 : 0.150 * (video.playbackRate || 1);
  const videoTime = Math.max(0, video.currentTime - lag);
  const frameIdx = Math.round(videoTime * fps);
  // If a mark already exists at this frame, increment its count
  const existing = myMarks.find(m => m.frame_idx === frameIdx);
  if (existing) { existing.n_balls++; }
  else { myMarks.push({video_time: videoTime, frame_idx: frameIdx, timestamp: new Date().toISOString(), n_balls: 1}); }
  renderMyMarks();
  drawTimeline();
}

function clearAllMarks() {
  myMarks = [];
  renderMyMarks();
  drawTimeline();
}

function undoMark() {
  // If no marks to undo but we just saved+advanced, revert to previous clip
  if (myMarks.length === 0 && lastSave) {
    const save = lastSave;
    lastSave = null;
    toast('Undoing save, returning to ' + save.clipId, 'info');
    openClip(save.clipId).then(() => {
      myMarks = save.marks;
      renderMyMarks();
      drawTimeline();
      const video = document.getElementById('video');
      video.currentTime = save.videoTime;
    });
    return;
  }
  const removed = myMarks.pop();
  if (removed) {
    const video = document.getElementById('video');
    video.currentTime = Math.max(0, removed.video_time - 1);
  }
  renderMyMarks();
  drawTimeline();
}

function renderMyMarks() {
  myMarks.sort((a, b) => a.frame_idx - b.frame_idx);
  const ul = document.getElementById('my-marks-list');
  ul.innerHTML = '';
  const total = myMarks.reduce((s, m) => s + (m.n_balls || 1), 0);
  const countEl = document.getElementById('my-marks-count');
  if (countEl) countEl.textContent = myMarks.length ? total + ' balls' : '';
  myMarks.forEach((m, i) => {
    const li = document.createElement('li');
    li.className = 'mark-item';
    li.dataset.idx = i;
    li.innerHTML =
      '<button class="mark-del" onclick="deleteMark(' + i + ')">&#x2715;</button>' +
      '<span class="mark-time" onclick="seekTo(' + m.video_time + ')">frame ' + m.frame_idx + '</span>' +
      '<span class="mark-n">' + m.n_balls + 'b</span>';
    ul.appendChild(li);
  });
}

function deleteMark(i) {
  myMarks.splice(i, 1);
  renderMyMarks();
  drawTimeline();
}

function renderOtherReviewers() {
  const container = document.getElementById('other-reviewers');
  container.innerHTML = '';
  if (!currentClip || !currentClip.annotations) return;
  for (const [token, anno] of Object.entries(currentClip.annotations)) {
    if (token === myToken) continue;
    const label = (reviewers[token] && reviewers[token].label) || token;
    const marks = anno.marks || [];
    if (marks.length === 0) continue;
    const h = document.createElement('div');
    h.innerHTML = '<h4>' + esc(label) + ' (' + marks.length + ' marks)</h4>';
    container.appendChild(h);
    for (const m of marks) {
      const div = document.createElement('div');
      div.className = 'other-mark-item';
      div.innerHTML =
        '<span class="other-mark-time" onclick="seekTo(' + (m.video_time || 0) + ')">frame ' + (m.frame_idx || 0) + '</span>' +
        '<span style="color:#fc8;min-width:20px">' + m.n_balls + 'b</span>' +
        '<span style="color:#555;font-size:0.72rem">' + esc(m.timestamp || '') + '</span>';
      container.appendChild(div);
    }
  }
}

async function saveAnnotations() {
  if (!currentClip || !myToken) { toast('Select a reviewer first.', 'err'); return; }
  const label = (reviewers[myToken] && reviewers[myToken].label) || myToken;
  const video = document.getElementById('video');
  // Snapshot for undo
  lastSave = {clipId: currentClip.id, marks: JSON.parse(JSON.stringify(myMarks)), videoTime: video.currentTime};
  const r = await fetch('/api/clips/' + currentClip.id + '/annotations', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({token: myToken, label, marks: myMarks})
  });
  if (r.ok) {
    toast('Annotations saved.', 'ok');
    // refresh clip list to update annotators
    await loadClips();
    const idx = allClips.findIndex(c => c.id === currentClip.id);
    if (idx >= 0 && !allClips[idx].annotators.includes(myToken)) allClips[idx].annotators.push(myToken);
    renderClipList();
    if (document.getElementById('auto-advance')?.checked) {
      nextUnannotated();
    }
  } else {
    lastSave = null;
    const detail = await r.json().then(d => d.detail || JSON.stringify(d)).catch(() => r.statusText);
    toast('Save failed: ' + detail, 'err', 5000);
  }
}

// ── agreement ─────────────────────────────────────────────────────────────────
async function loadAgreement() {
  if (!currentClip) return;
  const r = await fetch('/api/clips/' + currentClip.id + '/agreement');
  if (!r.ok) return;
  const data = await r.json();
  if (!data.reviewers || data.reviewers.length < 2 || !data.events || data.events.length === 0) {
    document.getElementById('agreement-section').style.display = 'none';
    return;
  }
  document.getElementById('agreement-section').style.display = 'block';
  const table = document.getElementById('agreement-table');
  const rLabels = data.reviewers.map(t => (reviewers[t] && reviewers[t].label) || t);
  let html = '<thead><tr><th>Event</th>' + rLabels.map(l => '<th>' + esc(l) + '</th>').join('') + '</tr></thead><tbody>';
  data.events.forEach((ev, i) => {
    html += '<tr><td>' + (ev.time != null ? ev.time.toFixed(1) + 's' : '?') + '</td>';
    for (const t of data.reviewers) {
      const hit = ev.reviewers && ev.reviewers[t];
      html += '<td style="color:' + (hit ? '#5d5' : '#555') + '">' + (hit ? '&#x2713;' : '&#x25cb;') + '</td>';
    }
    html += '</tr>';
  });
  html += '</tbody>';
  table.innerHTML = html;
}

// ── next unannotated ──────────────────────────────────────────────────────────
async function nextUnannotated() {
  const r = await fetch('/api/clips/random?annotated=false&reviewer=' + (myToken || ''));
  if (!r.ok) return;
  const data = await r.json();
  if (data.id) openClip(data.id);
}

// ── helpers ───────────────────────────────────────────────────────────────────
function seekTo(t) { document.getElementById('video').currentTime = t; }
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function getCookie(name) {
  const m = document.cookie.match('(?:^|;)\\s*' + name + '=([^;]*)');
  return m ? decodeURIComponent(m[1]) : null;
}
function setCookie(name, val, days) {
  const d = new Date(); d.setTime(d.getTime() + days*86400000);
  document.cookie = name + '=' + encodeURIComponent(val) + ';expires=' + d.toUTCString() + ';path=/';
}

// ── trim / split ──────────────────────────────────────────────────────────────
function enterTrimMode() {
  if (!currentClip) return;
  trimMode = true;
  trimSegments = [];
  trimDragHandle = null;
  trimAdding = false;
  document.getElementById('trim-toolbar').classList.add('active');
  document.getElementById('trim-enter-btn').style.display = 'none';
  document.getElementById('anno-section').style.display = 'none';
  drawTimeline();
}

function exitTrimMode() {
  trimMode = false;
  trimSegments = [];
  trimDragHandle = null;
  trimAdding = false;
  trimAddStart = null;
  trimAddEnd = null;
  document.getElementById('trim-toolbar').classList.remove('active');
  document.getElementById('trim-enter-btn').style.display = '';
  document.getElementById('anno-section').style.display = '';
  document.getElementById('trim-segments-list').innerHTML = '';
  drawTimeline();
}

function autoDetectSegments() {
  if (!currentClip || !currentClip.signal) return;
  const signal = currentClip.signal;
  const fps = currentClip.fps || 30;
  const padFrames = Math.round(2 * fps);

  // find contiguous regions where signal > 0
  let regions = [];
  let inRegion = false, regionStart = 0;
  for (let i = 0; i < signal.length; i++) {
    if (signal[i] > 0 && !inRegion) { regionStart = i; inRegion = true; }
    else if (signal[i] === 0 && inRegion) { regions.push({s: regionStart, e: i - 1}); inRegion = false; }
  }
  if (inRegion) regions.push({s: regionStart, e: signal.length - 1});

  if (regions.length === 0) { toast('No activity detected in signal', 'info'); return; }

  // expand by padding for quiet borders
  regions = regions.map(r => ({s: Math.max(0, r.s - padFrames), e: Math.min(signal.length - 1, r.e + padFrames)}));

  // merge overlapping
  regions.sort((a, b) => a.s - b.s);
  let merged = [regions[0]];
  for (let i = 1; i < regions.length; i++) {
    const prev = merged[merged.length - 1];
    if (regions[i].s <= prev.e) prev.e = Math.max(prev.e, regions[i].e);
    else merged.push(regions[i]);
  }

  trimSegments = merged.map(r => ({startFrame: r.s, endFrame: r.e}));
  renderTrimSegments();
  drawTimeline();
  toast('Detected ' + trimSegments.length + ' segment(s)', 'info');
}

function renderTrimSegments() {
  const el = document.getElementById('trim-segments-list');
  if (!currentClip) { el.innerHTML = ''; return; }
  const fps = currentClip.fps || 30;
  el.innerHTML = trimSegments.map((seg, i) => {
    const s = (seg.startFrame / fps).toFixed(1);
    const e = (seg.endFrame / fps).toFixed(1);
    const d = ((seg.endFrame - seg.startFrame) / fps).toFixed(1);
    return '<div class="trim-seg-item">' +
      '<button class="trim-seg-remove" onclick="removeTrimSegment(' + i + ')">&#x2715;</button>' +
      '<span>Part ' + (i + 1) + ': ' + s + 's &#x2013; ' + e + 's (' + d + 's)</span>' +
    '</div>';
  }).join('');
}

function removeTrimSegment(i) {
  trimSegments.splice(i, 1);
  renderTrimSegments();
  drawTimeline();
}

async function applyTrim() {
  if (!currentClip || trimSegments.length === 0) { toast('No segments defined', 'err'); return; }
  const deleteOriginal = document.getElementById('trim-delete-original').checked;
  const btn = document.querySelector('#trim-toolbar .primary');
  btn.disabled = true;
  btn.textContent = 'Processing\u2026';
  try {
    const r = await fetch('/api/clips/' + currentClip.id + '/trim', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        segments: trimSegments.map(s => ({start_frame: s.startFrame, end_frame: s.endFrame})),
        delete_original: deleteOriginal,
      }),
    });
    if (!r.ok) {
      const detail = await r.json().then(d => d.detail || JSON.stringify(d)).catch(() => r.statusText);
      toast('Trim failed: ' + detail, 'err', 5000);
      return;
    }
    const data = await r.json();
    toast('Created ' + data.new_ids.length + ' clip(s)', 'ok');
    exitTrimMode();
    await loadClips();
    renderClipList();
    if (data.new_ids.length > 0) openClip(data.new_ids[0]);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Apply';
  }
}

// ── keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.code === 'Escape' && trimMode) { e.preventDefault(); exitTrimMode(); return; }
  if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') { e.preventDefault(); undoMark(); return; }
  if ((e.ctrlKey || e.metaKey) && e.code === 'KeyS') { e.preventDefault(); saveAnnotations(); return; }
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  const video = document.getElementById('video');
  if (e.code === 'Space') { e.preventDefault(); markScore(); }
  else if (e.code === 'ArrowLeft') { e.preventDefault(); video.currentTime = Math.max(0, video.currentTime - 2); }
  else if (e.code === 'ArrowRight') { e.preventDefault(); video.currentTime = Math.min(video.duration||0, video.currentTime + 2); }
  else if (e.code === 'Comma') { e.preventDefault(); stepSpeed(-1); }
  else if (e.code === 'Period') { e.preventDefault(); stepSpeed(1); }
});

init();
</script>
</body>
</html>"""


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Ball Counter API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/live", response_class=HTMLResponse)
    def dashboard():
        if _wizard_active():
            from fastapi.responses import RedirectResponse
            return RedirectResponse("/wizard")
        streams = state.get_stream_names()
        stream_cards = ""
        for name in streams:
            color = "red" if "red" in name else ("blue" if "blue" in name else "#888")
            stream_cards += f"""
            <div class="card">
              <div class="card-top">
                <button class="clear-btn" onclick="resetGoal('{name}')">Clear</button>
              </div>
              <div class="goal-label" style="color:{color}">{name}</div>
              <div class="count" id="count-{name}" onclick="injectScore('{name}')" title="Tap to +1">0</div>
              <img src="/api/stream/{name}.mjpeg" onerror="this.style.opacity='0.3'" />
              <div class="clip-row">
                <button class="clip-btn" id="clip-{name}" onclick="saveClip('{name}')">Save last 60s</button>
                <button class="capture-btn" id="cap-{name}" onclick="captureScore('{name}')">Capture score</button>
              </div>
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
    .card {{ background: #1e1e1e; border-radius: 8px; padding: 1rem; text-align: center; min-width: 280px; position: relative; }}
    .card-top {{ display: flex; justify-content: flex-end; margin-bottom: 0.2rem; }}
    .goal-label {{ font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.4rem; }}
    .count {{ font-size: 4rem; font-weight: bold; line-height: 1; margin-bottom: 0.6rem; cursor: pointer; user-select: none; transition: text-shadow 0.15s; }}
    .count:hover {{ text-shadow: 0 0 20px rgba(255, 255, 200, 0.4); }}
    .count:active {{ text-shadow: 0 0 30px rgba(255, 255, 100, 0.7); }}
    .card img {{ width: 100%; border-radius: 4px; background: #000; display: block; }}
    .clip-row {{ display: flex; gap: 0.5rem; justify-content: center; margin-top: 0.5rem; flex-wrap: wrap; }}
    #log {{ margin-top: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto; }}
    #log h2 {{ font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; }}
    #events {{ list-style: none; max-height: 200px; overflow-y: auto; }}
    #events li {{ padding: 0.3rem 0.5rem; border-bottom: 1px solid #2a2a2a; font-size: 0.85rem; color: #ccc; }}
    #events li span.flash {{ color: #0f0; font-weight: bold; }}
    #status {{ text-align: center; font-size: 0.75rem; color: #444; margin-top: 1rem; }}
    .clear-btn {{ padding: 0.2rem 0.6rem; background: #222; color: #666; border: 1px solid #444; border-radius: 3px; cursor: pointer; font-size: 0.7rem; }}
    .clear-btn:hover {{ background: #500; color: #fff; border-color: #a00; }}
    .clip-btn {{ padding: 0.3rem 1rem; background: #333; color: #aaa; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }}
    .clip-btn:hover {{ background: #135; color: #9cf; border-color: #47a; }}
    .clip-btn:disabled {{ opacity: 0.5; cursor: default; }}
    .capture-btn {{ padding: 0.3rem 1rem; background: #1a3a1a; color: #8f8; border: 1px solid #383; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }}
    .capture-btn:hover {{ background: #2a5a2a; color: #afa; border-color: #5a5; }}
    .capture-btn.flash {{ background: #4a8a2a; color: #fff; border-color: #8f8; }}
  </style>
</head>
<body>
  <h1>Ball Counter <a href="/" style="font-size:0.7rem;color:#7bf;font-weight:normal;vertical-align:middle;margin-left:1rem">Review</a> <a href="https://github.com/cinderblock/balls-counter" target="_blank" style="font-size:0.6rem;color:#444;font-weight:normal;vertical-align:middle;text-decoration:none" title="GitHub">&#9135; cinderblock/balls-counter</a></h1>
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

    function injectScore(name) {{
      const el = document.getElementById('count-' + name);
      fetch('/api/score/' + name, {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{n_balls: 1}})}})
        .then(() => {{
          el.style.textShadow = '0 0 30px rgba(255, 255, 100, 0.8)';
          setTimeout(() => el.style.textShadow = '', 300);
        }})
        .catch(() => {{}});
    }}

    function captureScore(name) {{
      const btn = document.getElementById('cap-' + name);
      fetch('/api/capture', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{goal: name}})}})
        .then(r => r.json())
        .then(d => {{
          btn.textContent = d.grouped ? '+ grouped' : '✓ captured';
          btn.classList.add('flash');
          setTimeout(() => {{ btn.textContent = 'Capture score'; btn.classList.remove('flash'); }}, 1500);
        }})
        .catch(() => {{ btn.textContent = 'Error'; setTimeout(() => btn.textContent = 'Capture score', 2000); }});
    }}

    function saveClip(name) {{
      const btn = document.getElementById('clip-' + name);
      btn.disabled = true;
      btn.textContent = 'Saving…';
      fetch('/api/clip/save', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{goal: name}})}})
        .then(r => r.json())
        .then(d => {{
          btn.textContent = d.ok ? `Saved ${{d.seconds}}s` : 'Error';
          setTimeout(() => {{ btn.disabled = false; btn.textContent = 'Save last 60s'; }}, 3000);
        }})
        .catch(() => {{
          btn.textContent = 'Error';
          setTimeout(() => {{ btn.disabled = false; btn.textContent = 'Save last 60s'; }}, 3000);
        }});
    }}

    function resetGoal(name) {{
      fetch('/api/reset/' + name, {{method: 'POST'}});
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
      if (ev.type === 'score') {{
        const li = document.createElement('li');
        li.innerHTML = `<span class="flash">+${{ev.n_balls}}</span> ${{ev.stream}} &mdash; total ${{ev.total}} &nbsp;<small style="color:#555">${{ev.time}}</small>`;
        evtList.prepend(li);
        if (evtList.children.length > 50) evtList.lastChild.remove();
      }}
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

    @app.post("/api/score/{name}")
    def inject_score(name: str, body: dict | None = None):
        if name not in state.get_stream_names():
            raise HTTPException(status_code=404, detail=f"Stream '{name}' not found")
        n_balls = (body or {}).get("n_balls", 1)
        state.inject_score(name, n_balls)
        return {"ok": True}

    @app.post("/api/clip/save")
    def clip_save(req: dict):
        from pathlib import Path as _Path
        from fastapi import HTTPException
        goal = req.get("goal")
        if not goal:
            raise HTTPException(status_code=400, detail="Missing 'goal' field")
        buf = state.get_buffer(goal)
        if buf is None:
            raise HTTPException(status_code=404, detail=f"Goal '{goal}' not found")
        frames = buf.snapshot()
        if not frames:
            raise HTTPException(status_code=400, detail="Buffer is empty — no frames captured yet")
        clips_dir = state.get_clips_dir() or _Path("clips")
        from ball_counter.clips import save_clip
        mp4, jsn = save_clip(frames, goal, clips_dir)
        seconds = round(len(frames) / 30.0)
        return {"ok": True, "mp4": str(mp4), "json": str(jsn), "seconds": seconds}

    @app.post("/api/capture")
    def capture(body: dict):
        from fastapi import HTTPException
        goal = body.get("goal")
        if not goal:
            raise HTTPException(status_code=400, detail="Missing 'goal' field")
        result = state.press_capture(goal)
        if result == "error":
            raise HTTPException(status_code=404, detail=f"Goal '{goal}' not found or buffer empty")
        return {"ok": True, "grouped": result == "grouped"}

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

    # ------------------------------------------------------------------ review

    @app.get("/", response_class=HTMLResponse)
    def review():
        if _wizard_active():
            from fastapi.responses import RedirectResponse
            return RedirectResponse("/wizard")
        return HTMLResponse(_REVIEW_HTML)

    @app.get("/review", response_class=HTMLResponse)
    def review_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/")

    @app.get("/api/reviewers")
    def api_reviewers():
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            return {}
        return _load_reviewers(clips_dir)

    @app.post("/api/reviewer/create")
    def api_reviewer_create(body: dict):
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        label = (body.get("label") or "").strip()
        if not label:
            raise HTTPException(status_code=400, detail="label is required")
        reviewers = _load_reviewers(clips_dir)
        # Return existing token if label matches
        for token, info in reviewers.items():
            if info.get("label") == label:
                return {"token": token, "label": label}
        token = str(uuid.uuid4()).replace("-", "")
        from datetime import datetime as _datetime
        reviewers[token] = {"label": label, "created_at": _datetime.utcnow().isoformat() + "Z"}
        _save_reviewers(clips_dir, reviewers)
        return {"token": token, "label": label}

    @app.get("/api/clips")
    def api_clips():
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            return []
        result = []
        seen_stems = set()
        for mp4 in sorted(clips_dir.glob("*.mp4"), key=lambda p: p.name, reverse=True):
            stem = mp4.stem
            if stem == "reviewers":
                continue
            jsn = clips_dir / (stem + ".json")
            if not jsn.exists():
                continue
            seen_stems.add(stem)
            try:
                data = json.loads(jsn.read_text())
            except Exception:
                data = {}
            n_frames = data.get("n_frames") or 0
            fps = data.get("fps") or 30.0
            duration = n_frames / fps if n_frames and fps else None
            frames = data.get("frames") or []
            events = [f for f in frames if f.get("event") is not None]
            captures = data.get("captures") or []
            annotations = data.get("annotations") or {}
            annotators = list(annotations.keys())
            result.append({
                "id": stem,
                "goal": data.get("goal") or stem,
                "saved_at": data.get("saved_at") or stem,
                "fps": fps,
                "n_frames": n_frames,
                "duration": duration,
                "n_events": len(events),
                "n_captures": len(captures),
                "annotators": annotators,
            })
        return result

    @app.get("/api/clips/random")
    def api_clips_random(reviewer: str = "", annotated: str = "false"):
        import random as _random
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        want_annotated = annotated.lower() == "true"
        candidates: list[tuple[str, int]] = []  # (stem, n_annotations)
        for mp4 in clips_dir.glob("*.mp4"):
            stem = mp4.stem
            jsn = clips_dir / (stem + ".json")
            if not jsn.exists():
                continue
            try:
                data = json.loads(jsn.read_text())
            except Exception:
                data = {}
            annotations = data.get("annotations") or {}
            is_annotated = reviewer in annotations if reviewer else bool(annotations)
            if want_annotated == is_annotated:
                candidates.append((stem, len(annotations)))
        if not candidates:
            raise HTTPException(status_code=404, detail="No matching clips")
        # Weight toward clips with fewer reviews: weight = 1 / (1 + n_annotations)
        weights = [1.0 / (1 + n) for _, n in candidates]
        chosen = _random.choices(candidates, weights=weights, k=1)[0]
        return {"id": chosen[0]}

    @app.get("/api/clips/{clip_id}")
    def api_clip_detail(clip_id: str):
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        jsn = clips_dir / (clip_id + ".json")
        if not jsn.exists():
            raise HTTPException(status_code=404, detail="Clip not found")
        d = json.loads(jsn.read_text())
        d["id"] = clip_id
        # Flatten per-frame signal array and events list for UI consumption
        frames = d.get("frames") or []
        if frames and "signal" not in d:
            d["signal"] = [f.get("signal", 0) for f in frames]
            d["events"] = [
                {"frame_idx": i, **(f["event"] or {})}
                for i, f in enumerate(frames) if f.get("event") is not None
            ]
        return d

    @app.get("/api/clips/{clip_id}/video")
    def api_clip_video(clip_id: str):
        from starlette.responses import FileResponse as _FileResponse
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        mp4 = clips_dir / (clip_id + ".mp4")
        if not mp4.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        return _FileResponse(str(mp4), media_type="video/mp4")

    @app.get("/api/clips/{clip_id}/download")
    def api_clip_download(clip_id: str):
        import io
        import zipfile
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        mp4 = clips_dir / (clip_id + ".mp4")
        jsn = clips_dir / (clip_id + ".json")
        if not mp4.exists():
            raise HTTPException(status_code=404, detail="Clip not found")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(mp4, mp4.name)
            if jsn.exists():
                zf.write(jsn, jsn.name)
        buf.seek(0)

        def _stream():
            yield buf.read()

        return StreamingResponse(
            _stream(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={clip_id}.zip"},
        )

    @app.post("/api/clips/{clip_id}/annotations")
    def api_clip_annotations(clip_id: str, body: dict):
        from datetime import datetime as _datetime
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        jsn = clips_dir / (clip_id + ".json")
        if not jsn.exists():
            raise HTTPException(status_code=404, detail="Clip not found")
        token = body.get("token")
        if not token:
            raise HTTPException(status_code=400, detail="token is required")
        label = body.get("label") or token
        marks = body.get("marks") or []
        data = json.loads(jsn.read_text())
        if "annotations" not in data:
            data["annotations"] = {}
        data["annotations"][token] = {
            "label": label,
            "saved_at": _datetime.utcnow().isoformat() + "Z",
            "marks": marks,
        }
        jsn.write_text(json.dumps(data, indent=2))
        return {"ok": True}

    @app.get("/api/clips/{clip_id}/agreement")
    def api_clip_agreement(clip_id: str):
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        jsn = clips_dir / (clip_id + ".json")
        if not jsn.exists():
            raise HTTPException(status_code=404, detail="Clip not found")
        data = json.loads(jsn.read_text())
        events = data.get("events") or []
        annotations = data.get("annotations") or {}
        reviewer_tokens = list(annotations.keys())
        fps = data.get("fps") or 30.0
        result_events = []
        for ev in events:
            t = ev.get("frame_idx", 0) / fps if "frame_idx" in ev else ev.get("video_time", 0)
            rv = {}
            for token, anno in annotations.items():
                marks = anno.get("marks") or []
                rv[token] = any(abs((m.get("video_time") or 0) - t) <= 2.0 for m in marks)
            result_events.append({"time": t, "reviewers": rv})
        return {"reviewers": reviewer_tokens, "events": result_events}

    @app.post("/api/clips/{clip_id}/trim")
    def api_clip_trim(clip_id: str, body: dict):
        from ball_counter.clips import trim_clip
        clips_dir = state.get_clips_dir()
        if clips_dir is None:
            raise HTTPException(status_code=503, detail="clips_dir not configured")
        jsn = clips_dir / (clip_id + ".json")
        if not jsn.exists():
            raise HTTPException(status_code=404, detail="Clip not found")
        segments = body.get("segments")
        if not segments or not isinstance(segments, list):
            raise HTTPException(status_code=400, detail="segments list is required")
        for seg in segments:
            if "start_frame" not in seg or "end_frame" not in seg:
                raise HTTPException(status_code=400, detail="Each segment needs start_frame and end_frame")
        delete_original = body.get("delete_original", False)
        try:
            new_ids = trim_clip(clip_id, clips_dir, segments, delete_original=delete_original)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"ok": True, "new_ids": new_ids}

    # ------------------------------------------------------------------ wizard

    @app.get("/wizard", response_class=HTMLResponse)
    def wizard():
        return HTMLResponse(_WIZARD_HTML)

    @app.post("/api/wizard/snapshot")
    async def wizard_snapshot(body: dict):
        import cv2
        url = body.get("url", "").strip()
        if not url:
            return JSONResponse({"error": "url is required"}, status_code=400)
        _purge_expired_tokens()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG) if url.startswith("rtsp://") else cv2.VideoCapture(url)
        if not cap.isOpened():
            return JSONResponse({"error": "cannot connect to stream"}, status_code=422)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return JSONResponse({"error": "stream opened but no frame received"}, status_code=422)
        h, w = frame.shape[:2]
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return JSONResponse({"error": "failed to encode frame"}, status_code=500)
        token = str(uuid.uuid4()).replace("-", "")
        _wizard_frames[token] = (bytes(buf), time.time())
        return {"token": token, "width": w, "height": h}

    @app.get("/api/wizard/frame/{token}.jpg")
    def wizard_frame(token: str):
        _purge_expired_tokens()
        entry = _wizard_frames.get(token)
        if entry is None:
            raise HTTPException(status_code=404, detail="Token not found or expired")
        jpeg, _ = entry
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/wizard/frame/{token}/crop.jpg")
    def wizard_frame_crop(token: str, x: int, y: int, w: int, h: int):
        import cv2
        import numpy as np
        _purge_expired_tokens()
        entry = _wizard_frames.get(token)
        if entry is None:
            raise HTTPException(status_code=404, detail="Token not found or expired")
        jpeg, _ = entry
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to decode frame")
        fh, fw = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        crop = img[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode crop")
        return Response(content=bytes(buf), media_type="image/jpeg")

    @app.get("/api/wizard/current-config")
    def wizard_current_config():
        if _wizard_existing_configs is None:
            return {"streams": []}
        from ball_counter.config import save_configs
        import io, tempfile
        # Serialize existing configs to dict form
        streams = []
        for src in _wizard_existing_configs:
            goals = []
            for g in src.goals:
                gd: dict = {
                    "name": g.name,
                    "mode": g.mode,
                    "draw_color": list(g.draw_color),
                    "hsv_low": list(g.hsv_low),
                    "hsv_high": list(g.hsv_high),
                    "ball_area": g.ball_area,
                    "band_width": g.band_width,
                    "min_peak": g.min_peak,
                    "fall_ratio": g.fall_ratio,
                    "cooldown": g.cooldown,
                    "downsample": g.downsample,
                }
                if g.line:
                    gd["line"] = g.line
                if g.roi_points:
                    gd["roi_points"] = g.roi_points
                if g.crop_override:
                    gd["crop_override"] = g.crop_override
                if g.pfms_element:
                    gd["pfms_element"] = g.pfms_element
                goals.append(gd)
            streams.append({"source": src.source, "goals": goals})
        result: dict = {"streams": streams}
        if _wizard_existing_pfms is not None:
            result["pfms_url"] = _wizard_existing_pfms.url
            if _wizard_existing_pfms.key:
                result["pfms_key"] = _wizard_existing_pfms.key
            result["pfms_source"] = _wizard_existing_pfms.source
        return result

    @app.get("/api/wizard/config-path")
    def wizard_config_path_route():
        return {"path": _wizard_config_path or ""}

    @app.post("/api/wizard/save")
    async def wizard_save(body: dict):
        config_path = body.get("config_path") or _wizard_config_path
        if not config_path:
            return JSONResponse({"error": "config_path is required"}, status_code=400)
        streams_data = body.get("streams", [])
        from ball_counter.config import GoalConfig, PfmsConfig, SourceConfig, save_configs
        configs = []
        for s in streams_data:
            goals = []
            for g in s.get("goals", []):
                goals.append(GoalConfig(
                    name=g.get("name", "goal"),
                    mode=g.get("mode", "outlet"),
                    line=g.get("line"),
                    roi_points=g.get("roi_points", []),
                    hsv_low=tuple(g.get("hsv_low", [20, 100, 100])),
                    hsv_high=tuple(g.get("hsv_high", [35, 255, 255])),
                    draw_color=tuple(g.get("draw_color", [0, 0, 255])),
                    ball_area=g.get("ball_area", 1500),
                    band_width=g.get("band_width", 10),
                    min_peak=g.get("min_peak", 0),
                    fall_ratio=g.get("fall_ratio", 0.7),
                    cooldown=g.get("cooldown", 0),
                    downsample=g.get("downsample", 1.0),
                    crop_override=g.get("crop_override"),
                    pfms_element=g.get("pfms_element") or None,
                ))
            configs.append(SourceConfig(source=s["source"], goals=goals))
        pfms_cfg = None
        if body.get("pfms_url"):
            pfms_cfg = PfmsConfig(
                url=body["pfms_url"],
                key=body.get("pfms_key") or None,
                source=body.get("pfms_source") or "ball-counter",
            )
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_configs(configs, path, pfms=pfms_cfg)
        global _wizard_saved, _wizard_done_event
        _wizard_saved = True
        if _wizard_done_event is not None:
            _wizard_done_event.set()
        return {"ok": True}

    return app


def start_server(
    state: AppState,
    port: int | None = None,
    host: str = "0.0.0.0",
    socket_path: str | None = None,
    trusted_proxies: list[str] | None = None,
) -> None:
    """Start the uvicorn server in the current thread (blocks until done)."""
    app = create_app(state)
    kwargs: dict = dict(
        log_level="warning",
        proxy_headers=trusted_proxies is not None,
        forwarded_allow_ips=",".join(trusted_proxies) if trusted_proxies else None,
    )
    if socket_path:
        kwargs["uds"] = socket_path
    else:
        kwargs["host"] = host
        kwargs["port"] = port
    uvicorn.run(app, **kwargs)


def start_server_thread(
    state: AppState,
    port: int | None = None,
    host: str = "0.0.0.0",
    socket_path: str | None = None,
    trusted_proxies: list[str] | None = None,
) -> threading.Thread:
    """Launch the web server in a background daemon thread."""
    t = threading.Thread(
        target=start_server,
        args=(state, port, host, socket_path, trusted_proxies),
        daemon=True,
    )
    t.start()
    return t
