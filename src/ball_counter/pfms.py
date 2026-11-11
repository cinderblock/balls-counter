"""pFMS score event forwarder.

Scores are queued and posted from one worker thread: a burst of balls goes
out as one request, a failed request is retried with backoff instead of
being lost, and pFMS is told how old each score is.

Why the age matters: pFMS judges every score at the moment the ball
actually scored — auto vs teleop, which shift, whether that goal was on.
The detector sees a ball some time after it crossed the goal (decoder
latency, peak detection, GPU contention), and a retry adds more. Each
event therefore carries ``ageMs`` — milliseconds between the ball scoring
and the request being sent — recomputed at every send attempt from the
frame's monotonic read time. pFMS subtracts it from its own receive time,
so no clock agreement between the two machines is needed and a late
report still lands in the right period.

``capture_latency_ms`` covers the part the frame read time cannot see:
camera → RTSP → decoder → Python. Measure it once for the field and set
``pfms_capture_latency_ms`` in the config.
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Scores older than this are dropped rather than sent: the match they belong
# to is long over and pFMS would only file them as outside any match.
MAX_EVENT_AGE_SEC = 10 * 60
# Wait between failed attempts (indexed by attempt number, last value repeats).
RETRY_BACKOFF_SEC = (1.0, 2.0, 5.0, 10.0, 30.0)
# After the first score of a burst arrives, wait this long for the rest so
# they share one request.
BATCH_SETTLE_SEC = 0.05
HTTP_TIMEOUT_SEC = 5.0


@dataclass
class PendingScore:
    alliance: str
    element: str
    n_balls: int
    occurred_mono: float  # time.monotonic() when the ball scored
    occurred_wall: float  # time.time() at the same instant (for the record only)


class PfmsForwarder:
    """Posts ball-scoring events to the pFMS API from a background worker.

    Never raises into the caller — all errors are logged and retried so
    pFMS trouble cannot affect ball counting.
    """

    def __init__(self, url: str, key: str | None, source: str,
                 capture_latency_ms: int = 0) -> None:
        self._submit_url = url.rstrip("/") + "/api/score"
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["X-API-Key"] = key
        self._source = source
        self._capture_latency_ms = max(0, int(capture_latency_ms))
        self._queue: deque[PendingScore] = deque()
        self._cv = threading.Condition()
        self._attempts = 0  # consecutive failures of the batch at the head
        # Status for logs / UI
        self.last_ok_at: float | None = None
        self.last_error: str | None = None
        self.sent = 0
        self.dropped = 0
        self._worker = threading.Thread(target=self._run, daemon=True, name="pfms-forwarder")
        self._worker.start()

    # ── Public ───────────────────────────────────────────────────────

    def send(self, alliance: str, element: str, n_balls: int = 1,
             occurred_mono: float | None = None, occurred_wall: float | None = None) -> None:
        """Queue a score. ``occurred_mono`` is the monotonic time the ball
        scored (default: now); ``occurred_wall`` the matching wall-clock time.
        Returns immediately."""
        now_mono = time.monotonic()
        if occurred_mono is None:
            occurred_mono = now_mono
        if occurred_wall is None:
            occurred_wall = time.time() - (now_mono - occurred_mono)
        with self._cv:
            self._queue.append(PendingScore(alliance, element, n_balls, occurred_mono, occurred_wall))
            self._cv.notify()

    def status(self) -> dict:
        with self._cv:
            queued = len(self._queue)
        return {
            "url": self._submit_url,
            "queued": queued,
            "sent": self.sent,
            "dropped": self.dropped,
            "last_ok_at": self.last_ok_at,
            "last_error": self.last_error,
            "capture_latency_ms": self._capture_latency_ms,
        }

    # ── Worker ───────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            try:
                self._drain_once()
            except Exception as exc:  # the worker must outlive any surprise
                log.exception("pFMS forwarder error: %s", exc)
                time.sleep(1.0)

    def _drain_once(self) -> None:
        with self._cv:
            while not self._queue:
                self._cv.wait()
        # Let a burst of balls accumulate into one request.
        time.sleep(BATCH_SETTLE_SEC)
        with self._cv:
            batch = list(self._queue)
            self._queue.clear()

        # Anything that has waited longer than a match lasts is not
        # worth reporting any more.
        now_mono = time.monotonic()
        fresh = [p for p in batch if now_mono - p.occurred_mono <= MAX_EVENT_AGE_SEC]
        if len(fresh) != len(batch):
            self.dropped += len(batch) - len(fresh)
            log.warning("pFMS: dropped %d score(s) older than %ds", len(batch) - len(fresh), MAX_EVENT_AGE_SEC)
        if not fresh:
            self._attempts = 0
            return

        if self._post(fresh):
            self._attempts = 0
            return

        # Put the batch back at the head so order is kept, then back off.
        with self._cv:
            self._queue.extendleft(reversed(fresh))
        delay = RETRY_BACKOFF_SEC[min(self._attempts, len(RETRY_BACKOFF_SEC) - 1)]
        self._attempts += 1
        time.sleep(delay)

    def _payload(self, batch: list[PendingScore]) -> list[dict]:
        """Build the request body, computing each score's age right now."""
        now_mono = time.monotonic()
        events = []
        for p in batch:
            age_ms = (now_mono - p.occurred_mono) * 1000.0 + self._capture_latency_ms
            events.append({
                "source": self._source,
                "alliance": p.alliance,
                "element": p.element,
                "count": p.n_balls,
                "ageMs": int(round(max(0.0, age_ms))),
                "timestamp": int(round(p.occurred_wall * 1000.0 - self._capture_latency_ms)),
            })
        return events

    def _post(self, batch: list[PendingScore]) -> bool:
        payload = self._payload(batch)
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(self._submit_url, data=data, headers=self._headers, method="POST")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # 4xx means pFMS understood us and said no (bad key, bad body):
            # retrying the same request will not help. Log it and move on.
            body = ""
            try:
                body = exc.read().decode(errors="replace")[:300]
            except Exception:
                pass
            self.last_error = f"HTTP {exc.code}: {body or exc.reason}"
            if 400 <= exc.code < 500:
                self.dropped += len(batch)
                log.warning("pFMS rejected %d score(s): %s", len(batch), self.last_error)
                print(f"pfms     - rejected {len(batch)} score(s): {self.last_error}")
                return True
            log.warning("pFMS forward failed (%s), will retry", self.last_error)
            return False
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("pFMS forward failed (%s), will retry", exc)
            return False

        self.last_ok_at = time.time()
        self.last_error = None
        self.sent += len(batch)
        self._log_receipts(payload, result)
        return True

    @staticmethod
    def _log_receipts(payload: list[dict], result: dict) -> None:
        """One log line per score saying where pFMS filed it."""
        receipts = result.get("events") if isinstance(result, dict) else None
        if not isinstance(receipts, list):
            log.info("pFMS accepted %d score(s): %s", len(payload), result)
            return
        for ev, rc in zip(payload, receipts):
            if not isinstance(rc, dict):
                continue
            where = rc.get("matchSubPeriod") or rc.get("matchPhase") or "free play"
            age = ev["ageMs"] / 1000.0
            # ASCII only: this runs on the worker thread, and a console that
            # can't encode a character would otherwise kill it.
            if rc.get("status") == "rejected":
                log.warning("pFMS rejected %s %s +%d: %s", ev["alliance"], ev["element"], ev["count"], rc.get("reason"))
                print(f"pfms     - REJECTED {ev['alliance']} +{ev['count']}: {rc.get('reason')}")
            elif rc.get("counted", True):
                log.info("pFMS counted %s %s +%d (%.2fs old -> %s)", ev["alliance"], ev["element"], ev["count"], age, where)
                print(f"pfms     - {ev['alliance']} +{ev['count']} ({age:.2f}s old -> {where})")
            else:
                log.info("pFMS did not count %s %s +%d (%.2fs old -> %s: %s)",
                         ev["alliance"], ev["element"], ev["count"], age, where, rc.get("reason"))
                print(f"pfms     - {ev['alliance']} +{ev['count']} ({age:.2f}s old -> {where}) NOT COUNTED: {rc.get('reason')}")
