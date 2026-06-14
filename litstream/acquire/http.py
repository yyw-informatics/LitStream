"""HTTP helpers + per-source rate limiting for Acquire. Pure stdlib (urllib)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "LitStream/0.1 (autonomous literature agent; mailto:research@example.org)"


class RateLimiter:
    """Enforce a minimum interval between calls (e.g. Semantic Scholar = 1.0s)."""

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()


def http_get(url: str, *, params: dict | None = None, headers: dict | None = None,
             timeout: float = 30.0, retries: int = 2) -> str:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:        # backoff on rate limit
                time.sleep(2 ** attempt + 1)
                last_err = e
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("unreachable")


def http_get_json(url: str, **kw) -> dict:
    return json.loads(http_get(url, **kw))


def http_get_bytes(url: str, *, headers: dict | None = None, timeout: float = 60.0,
                   max_bytes: int = 60_000_000) -> bytes:
    """Download binary content (PDFs). Follows redirects (urllib default), caps size."""
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(max_bytes)
