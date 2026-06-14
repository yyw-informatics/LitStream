"""Fetch open-access PDFs for triage survivors and symlink them into a project's papers/.

Runs at pipeline time, not acquire time: only papers a project kept (triage
score >= 0.5) get a PDF download. A PDF is stored once in the global library
(library/<safe_id>.pdf) and symlinked into projects/<name>/papers/ so the
kb-skills mining skills find it, avoiding duplication across projects.

Resolution order (open-access only; paywalled PDFs are not fetched):
  1. arXiv  -> https://arxiv.org/pdf/<id>
  2. Europe PMC OA -> the paper's fullTextUrlList PDF entry, covering PMC plus
     bioRxiv/medRxiv preprints that Europe PMC indexes.
Anything else is marked 'unavailable' but stays in the library by metadata/abstract.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .http import RateLimiter, http_get_bytes, http_get_json

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UNPAYWALL = "https://api.unpaywall.org/v2/"
PDF_MAGIC = b"%PDF"
PREPRINT_PREFIXES = ("10.1101/", "10.64898/")   # bioRxiv / medRxiv


def safe_id(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


@dataclass
class FetchSummary:
    considered: int = 0
    fetched: int = 0
    linked: int = 0
    unavailable: int = 0


class PdfFetcher:
    def __init__(self, store, *, library_dir: str | Path, max_bytes: int = 60_000_000,
                 email: str | None = None):
        self.store = store
        self.lib = Path(library_dir)
        self.lib.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.email = email or os.environ.get("UNPAYWALL_EMAIL")   # required for Unpaywall
        self.epmc_rl = RateLimiter(1.0)
        self.upw_rl = RateLimiter(0.15)
        self.biorxiv_rl = RateLimiter(4.0)   # bioRxiv 403s rapid requests (anti-bot)

    # -- OA url resolution (ordered candidates; downloader tries each) --------

    def resolve_urls(self, row) -> list[str]:
        out: list[str] = []
        if row["arxiv_id"]:
            out.append(f"https://arxiv.org/pdf/{row['arxiv_id']}")
        epmc = self._epmc_pdf(row)
        if epmc:
            out.append(epmc)
        if self.email and row["doi"]:
            upw = self._unpaywall_pdf(row["doi"])
            if upw:
                out.append(upw)
        # bioRxiv/medRxiv direct-PDF fallback (host ambiguous → try both)
        if row["doi"] and row["doi"].startswith(PREPRINT_PREFIXES):
            for host in ("www.biorxiv.org", "www.medrxiv.org"):
                out.append(f"https://{host}/content/{row['doi']}v1.full.pdf")
        seen: set[str] = set()
        return [u for u in out if u and not (u in seen or seen.add(u))]

    def _epmc_pdf(self, row) -> str | None:
        q = f'DOI:"{row["doi"]}"' if row["doi"] else (
            f'EXT_ID:{row["pmid"]} AND SRC:MED' if row["pmid"] else None)
        if not q:
            return None
        self.epmc_rl.wait()
        try:
            data = http_get_json(EPMC_SEARCH, params={
                "query": q, "resultType": "core", "format": "json", "pageSize": 1})
        except Exception:
            return None
        res = data.get("resultList", {}).get("result", [])
        if not res:
            return None
        for u in (res[0].get("fullTextUrlList") or {}).get("fullTextUrl", []) or []:
            if u.get("documentStyle") == "pdf" and u.get("availabilityCode") in ("OA", "F"):
                return u.get("url")
        return None

    def _unpaywall_pdf(self, doi: str) -> str | None:
        self.upw_rl.wait()
        try:
            data = http_get_json(UNPAYWALL + doi, params={"email": self.email})
        except Exception:
            return None
        loc = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or None

    # -- download + link -----------------------------------------------------

    def _download(self, url: str) -> bytes | None:
        import time
        import urllib.error
        is_biorxiv = "biorxiv.org" in url or "medrxiv.org" in url
        attempts = 3 if is_biorxiv else 2     # bioRxiv 403s intermittently → more tries
        for attempt in range(attempts):
            if is_biorxiv:
                self.biorxiv_rl.wait()
            try:
                data = http_get_bytes(url, timeout=90, max_bytes=self.max_bytes)
                return data if data[:4] == PDF_MAGIC else None   # reject paywall HTML
            except urllib.error.HTTPError as e:
                if e.code in (403, 429) and attempt < attempts - 1:
                    time.sleep(5 * (attempt + 1))   # increasing backoff
                    continue
                return None
            except Exception:
                return None
        return None

    def fetch_for_project(self, project: str, papers_dir: str | Path,
                          limit: int | None = None) -> FetchSummary:
        rows = self.store._conn.execute(
            """SELECT p.* FROM project_papers pp JOIN papers p ON p.paper_id = pp.paper_id
               WHERE pp.project = ? AND pp.status = 'kept'
                 AND p.pdf_status IN ('none', 'unavailable')""", (project,)).fetchall()
        if limit:
            rows = rows[:limit]
        papers_dir = Path(papers_dir)
        papers_dir.mkdir(parents=True, exist_ok=True)
        summ = FetchSummary(considered=len(rows))

        for row in rows:
            data = None
            for url in self.resolve_urls(row):
                data = self._download(url)
                if data:
                    break
            if not data:
                self.store.set_pdf(row["paper_id"], None, "unavailable")
                summ.unavailable += 1
                continue
            path = self.lib / f"{safe_id(row['paper_id'])}.pdf"
            path.write_bytes(data)
            self.store.set_pdf(row["paper_id"], str(path), "fetched")
            summ.fetched += 1
            # symlink into the project's papers/ so kb-skills mining finds it
            link = papers_dir / f"{safe_id(row['paper_id'])}.pdf"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(path.resolve(), link)
            summ.linked += 1
        return summ


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _P
    from litstream.library.store import PaperStore

    ROOT = _P(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Fetch OA PDFs for triage survivors")
    ap.add_argument("--project", required=True)
    ap.add_argument("--papers-dir", required=True, help="projects/<name>/papers/")
    ap.add_argument("--db", default=str(ROOT / "litstream.db"))
    ap.add_argument("--library", default=str(ROOT / "library"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    store = PaperStore(args.db, library_dir=args.library)
    summ = PdfFetcher(store, library_dir=args.library).fetch_for_project(
        args.project, args.papers_dir, limit=args.limit)
    print(f"\n  PDF fetch · {args.project}: {summ.considered} survivors → "
          f"{summ.fetched} fetched, {summ.linked} linked, {summ.unavailable} unavailable\n")
