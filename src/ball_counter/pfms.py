"""Fire-and-forget PFMS score event forwarder."""

import json
import logging
import threading
import urllib.request

log = logging.getLogger(__name__)


class PfmsForwarder:
    """Posts ball-scoring events to the PFMS API in background threads.

    Never raises — all errors are logged and swallowed so PFMS failures
    cannot affect ball counting.
    """

    def __init__(self, url: str, key: str | None, source: str) -> None:
        self._submit_url = url.rstrip("/") + "/api/score"
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["X-API-Key"] = key
        self._source = source

    def send(self, alliance: str, element: str, n_balls: int = 1) -> None:
        """Dispatch a score event. Returns immediately; posts in a daemon thread."""
        payload = {
            "source": self._source,
            "alliance": alliance,
            "element": element,
            "count": n_balls,
        }
        threading.Thread(target=self._post, args=(payload,), daemon=True).start()

    def _post(self, payload: dict) -> None:
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                self._submit_url, data=data, headers=self._headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
            if result.get("rejected", 0) > 0:
                log.warning("PFMS rejected %s: %s", payload, result.get("errors"))
            else:
                log.info("PFMS accepted %s → %s", payload, result)
        except Exception as exc:
            log.warning("PFMS forward failed: %s", exc)
