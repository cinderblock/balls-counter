"""Smoke test for the pFMS forwarder and the review tally's pause grace.

Runs a tiny fake pFMS on localhost and checks that the forwarder:
  - batches a burst of scores into one request,
  - reports each score's age from the frame time (plus capture latency),
  - retries a failed request and recomputes the ages when it does,
  - does not retry a 4xx rejection,
  - survives a response with per-event receipts.
Then checks tally_marks gives balls in flight at a pause the 3 s grace.

Run: uv run python scripts/smoke_pfms_forwarder.py
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from ball_counter.match import tally_marks
from ball_counter.pfms import PfmsForwarder

PORT = 39873


class FakePfms(BaseHTTPRequestHandler):
    requests: list[list[dict]] = []
    fail_next = 0  # respond 503 to this many requests
    reject_next = 0  # respond 400 to this many requests
    got = threading.Condition()

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        cls = FakePfms
        with cls.got:
            cls.requests.append(body)
            cls.got.notify_all()
        if cls.fail_next > 0:
            cls.fail_next -= 1
            self.send_response(503)
            self.end_headers()
            return
        if cls.reject_next > 0:
            cls.reject_next -= 1
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"nope"}')
            return
        receipts = [
            {"status": "accepted", "counted": i % 2 == 0, "matchSubPeriod": "shift1",
             "reason": None if i % 2 == 0 else "goalInactive", "lagMs": ev["ageMs"]}
            for i, ev in enumerate(body)
        ]
        out = json.dumps({"accepted": len(body), "rejected": 0, "deduplicated": 0,
                          "errors": [], "events": receipts}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_):
        pass


def wait_requests(n: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    with FakePfms.got:
        while len(FakePfms.requests) < n:
            left = deadline - time.monotonic()
            assert left > 0, f"timed out waiting for request #{n}"
            FakePfms.got.wait(left)


def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), FakePfms)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fwd = PfmsForwarder(f"http://127.0.0.1:{PORT}", None, "smoke", capture_latency_ms=100)

    # 1. A burst goes out as one request, ages come from frame times.
    now = time.monotonic()
    fwd.send("red", "goal", 1, occurred_mono=now - 1.5)
    fwd.send("blue", "goal", 2, occurred_mono=now - 0.2)
    fwd.send("red", "goal", 1)  # no frame time: "now"
    wait_requests(1, 5)
    body = FakePfms.requests[0]
    assert len(body) == 3, body
    assert body[0]["alliance"] == "red" and body[0]["count"] == 1
    assert body[1]["alliance"] == "blue" and body[1]["count"] == 2
    assert 1550 <= body[0]["ageMs"] <= 1900, body[0]  # 1.5 s + 100 ms latency + settle
    assert 250 <= body[1]["ageMs"] <= 600, body[1]
    assert 100 <= body[2]["ageMs"] <= 400, body[2]
    assert abs(body[0]["timestamp"] - (time.time() - 1.6) * 1000) < 1500
    print("ok - burst batched, ages from frame time (+capture latency)")

    # 2. A failed request is retried with the ages recomputed.
    FakePfms.fail_next = 1
    now = time.monotonic()
    fwd.send("red", "goal", 1, occurred_mono=now - 0.5)
    wait_requests(2, 5)
    first_age = FakePfms.requests[1][0]["ageMs"]
    wait_requests(3, 5)  # retry after ~1 s backoff
    retry_age = FakePfms.requests[2][0]["ageMs"]
    assert retry_age >= first_age + 900, (first_age, retry_age)
    assert fwd.status()["queued"] == 0
    print(f"ok - retried after failure, age grew {first_age} -> {retry_age} ms")

    # 3. A 4xx rejection is not retried.
    FakePfms.reject_next = 1
    fwd.send("blue", "goal", 1)
    wait_requests(4, 5)
    time.sleep(1.5)
    assert len(FakePfms.requests) == 4, len(FakePfms.requests)
    assert fwd.status()["dropped"] == 1
    print("ok - 4xx not retried")

    # 4. Receipts are handled (logged) without error, sent counter is right.
    assert fwd.sent == 4, fwd.sent
    print("ok - receipts handled")
    server.shutdown()

    # 5. Review tally: balls in flight when a pause starts still count.
    spans = [
        {"phase": "auto", "sub": "auto", "start": 0.0, "end": 10.0, "scoring": True, "active": True},
        {"phase": "paused", "sub": None, "start": 10.0, "end": 20.0, "scoring": False, "active": True},
        {"phase": "teleop", "sub": "transition", "start": 20.0, "end": 25.0, "scoring": True, "active": True},
        {"phase": "teleop", "sub": "shift1", "start": 25.0, "end": 35.0, "scoring": True, "active": False},
        {"phase": "paused", "sub": None, "start": 35.0, "end": 45.0, "scoring": False, "active": False},
    ]
    marks = [
        {"video_time": 5.0},   # auto: counts, auto
        {"video_time": 12.0},  # 2 s into pause after an on-goal span: grace, still auto
        {"video_time": 15.0},  # 5 s into pause: excluded
        {"video_time": 26.0},  # goal off since 25 s: shift grace, counts
        {"video_time": 30.0},  # goal off 5 s: goalInactive
        {"video_time": 36.0},  # pause after an off-goal span: no fresh grace, excluded
    ]
    t = tally_marks(marks, spans)
    assert t == {"score": 3, "autoScore": 2, "excluded": 2, "goalInactive": 1}, t
    print("ok - tally: pause grace matches pFMS")
    print("all ok")


if __name__ == "__main__":
    main()
