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
_wizard_saved: bool = False  # set True after save so redirect deactivates
_wizard_done_event: "threading.Event | None" = None  # signalled when wizard saves
_WIZARD_TOKEN_TTL = 600  # seconds


def set_wizard_state(
    config_path: str,
    existing_configs: list | None = None,
    done_event: "threading.Event | None" = None,
) -> None:
    global _wizard_config_path, _wizard_existing_configs, _wizard_saved, _wizard_done_event
    _wizard_config_path = config_path
    _wizard_existing_configs = existing_configs
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
            print(f"[{goal_name}] capture: no frames in window, skipping")
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
            print(f"[{goal_name}] capture saved: {mp4.name} ({len(clip_frames)} frames, "
                  f"{len(session['press_frame_idxs'])} press(es))")
        except Exception as e:
            print(f"[{goal_name}] capture save failed: {e}")

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
  </div></div>`;
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
    b.min_peak  = parseInt(det.querySelector('.g-min-peak').value)   || 0;
    b.cooldown  = parseInt(det.querySelector('.g-cooldown').value)   || 0;
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
      return obj;
    }),
  }));
  return {config_path: _configPath||'', streams};
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
      cooldown:g.cooldown??0, downsample:g.downsample??1.0});
    if (g.line) existingLinesFull[id] = {p1:g.line[0], p2:g.line[1]};
  }
  redrawBoxCanvas(); updateRamEst();
}

// ── init ──────────────────────────────────────────────────────────────────────
fetch('/api/wizard/current-config').then(r=>r.json()).then(data => {
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
        if _wizard_active():
            from fastapi.responses import RedirectResponse
            return RedirectResponse("/wizard")
        streams = state.get_stream_names()
        stream_cards = ""
        for name in streams:
            color = "red" if "red" in name else ("blue" if "blue" in name else "#888")
            stream_cards += f"""
            <div class="card">
              <div class="goal-label" style="color:{color}">{name}</div>
              <div class="count" id="count-{name}">0</div>
              <div class="btn-row">
                <button class="clear-btn" onclick="resetGoal('{name}')">Clear</button>
                <button class="clip-btn" id="clip-{name}" onclick="saveClip('{name}')">Save clip</button>
                <button class="capture-btn" id="cap-{name}" onclick="captureScore('{name}')">Capture score</button>
              </div>
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
    .btn-row {{ display: flex; gap: 0.5rem; justify-content: center; margin-bottom: 0.6rem; flex-wrap: wrap; }}
    .clear-btn, .clip-btn {{ padding: 0.3rem 1rem; background: #333; color: #aaa; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }}
    .clear-btn:hover {{ background: #500; color: #fff; border-color: #a00; }}
    .clip-btn:hover {{ background: #135; color: #9cf; border-color: #47a; }}
    .clip-btn:disabled {{ opacity: 0.5; cursor: default; }}
    .capture-btn {{ background: #1a3a1a; color: #8f8; border-color: #383; }}
    .capture-btn:hover {{ background: #2a5a2a; color: #afa; border-color: #5a5; }}
    .capture-btn.flash {{ background: #4a8a2a; color: #fff; border-color: #8f8; }}
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
          btn.textContent = d.ok ? `Saved (${{d.n_frames}}f)` : 'Error';
          setTimeout(() => {{ btn.disabled = false; btn.textContent = 'Save clip'; }}, 3000);
        }})
        .catch(() => {{
          btn.textContent = 'Error';
          setTimeout(() => {{ btn.disabled = false; btn.textContent = 'Save clip'; }}, 3000);
        }});
    }}

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
        return {"ok": True, "mp4": str(mp4), "json": str(jsn), "n_frames": len(frames)}

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
                goals.append(gd)
            streams.append({"source": src.source, "goals": goals})
        return {"streams": streams}

    @app.get("/api/wizard/config-path")
    def wizard_config_path_route():
        return {"path": _wizard_config_path or ""}

    @app.post("/api/wizard/save")
    async def wizard_save(body: dict):
        config_path = body.get("config_path") or _wizard_config_path
        if not config_path:
            return JSONResponse({"error": "config_path is required"}, status_code=400)
        streams_data = body.get("streams", [])
        from ball_counter.config import GoalConfig, SourceConfig, save_configs
        configs = []
        for s in streams_data:
            goals = []
            for g in s.get("goals", []):
                line = g.get("line")
                roi = g.get("roi_points", [])
                goals.append(GoalConfig(
                    name=g.get("name", "goal"),
                    mode=g.get("mode", "outlet"),
                    line=line,
                    roi_points=roi,
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
                ))
            configs.append(SourceConfig(source=s["source"], goals=goals))
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_configs(configs, path)
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
