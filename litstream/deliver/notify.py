"""Best-effort delivery over email (sendmail) and a Slack webhook.

Never raises: a failed notification must not fail the run. The digest file on disk
remains the source of truth regardless of delivery outcome.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request

SENDMAIL = "/usr/sbin/sendmail"


def send_email(to: str, subject: str, body: str) -> bool:
    try:
        msg = f"From: {to}\nTo: {to}\nSubject: {subject}\nContent-Type: text/plain; charset=utf-8\n\n{body}"
        subprocess.run([SENDMAIL, "-f", to, to], input=msg.encode(), timeout=20, check=False)
        return True
    except Exception:
        return False


def send_slack(webhook_url: str, subject: str, body: str, max_chars: int = 3500) -> bool:
    try:
        text = f"*{subject}*\n```\n{body[:max_chars]}\n```"
        req = urllib.request.Request(
            webhook_url, data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200
    except Exception:
        return False


def deliver(channels: list[str], *, subject: str, body: str, config: dict) -> dict:
    """Push a digest to the requested channels (e.g. ['digest_md','email','slack']).

    digest_md is the on-disk file, written elsewhere; this handles the push
    channels. config carries the email and webhook targets. Returns per-channel
    success and never raises.
    """
    out: dict[str, bool] = {}
    if "email" in channels and config.get("email_to"):
        out["email"] = send_email(config["email_to"], subject, body)
    if "slack" in channels and config.get("slack_webhook"):
        out["slack"] = send_slack(config["slack_webhook"], subject, body)
    return out
