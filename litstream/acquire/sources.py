"""Source clients: arXiv, Europe PMC (published plus bioRxiv/medRxiv preprints),
Semantic Scholar. Each .search() yields PaperRecords for the library.

Europe PMC indexes bioRxiv/medRxiv preprints (SRC:PPR) alongside PubMed-published
work with full keyword, date, and journal search, so it covers the preprint servers
without the bulk bioRxiv date-dump API. arXiv is searched natively for the ML side.
"""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET

from litstream.library.store import PaperRecord
from .http import RateLimiter, http_get, http_get_json


def _year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


# ---------------------------------------------------------------- arXiv -----
class ArxivSource:
    name = "arxiv"
    API = "http://export.arxiv.org/api/query"
    NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    # Restrict to q-bio. Including the large stat.ML / cs.LG categories floods loose
    # keyword matches with unrelated ML papers; the single-cell/CITE-seq ML methods
    # that matter are cross-listed into q-bio.GN/QM anyway. arXiv is a minor source
    # here (most bio signal lives in Europe PMC), and triage filters the rest.
    DEFAULT_CATS = ["q-bio.GN", "q-bio.QM", "q-bio.MN", "q-bio.TO", "q-bio.CB", "q-bio.OT"]

    def __init__(self, rate: float = 3.0, categories: list[str] | None = None):
        self.rl = RateLimiter(rate)   # arXiv asks for ~1 call / 3s
        self.cats = categories or self.DEFAULT_CATS

    def search(self, query: str, *, since: dt.date | None = None, limit: int = 20) -> list[PaperRecord]:
        self.rl.wait()
        if ":" not in query:                     # plain keywords → search the all: field
            query = f"all:{query}"
        cat_clause = " OR ".join(f"cat:{c}" for c in self.cats)
        query = f"({query}) AND ({cat_clause})"
        xml = http_get(self.API, params={
            "search_query": query, "start": 0, "max_results": limit,
            "sortBy": "submittedDate", "sortOrder": "descending"}, timeout=40)
        root = ET.fromstring(xml)
        out: list[PaperRecord] = []
        for e in root.findall("a:entry", self.NS):
            published = (e.findtext("a:published", default="", namespaces=self.NS) or "")[:10]
            if since and published and published < since.isoformat():
                continue
            raw_id = e.findtext("a:id", default="", namespaces=self.NS)        # http://arxiv.org/abs/2406.12345v1
            arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else None
            out.append(PaperRecord(
                title=" ".join((e.findtext("a:title", "", self.NS) or "").split()),
                abstract=" ".join((e.findtext("a:summary", "", self.NS) or "").split()),
                arxiv_id=arxiv_id,
                doi=e.findtext("arxiv:doi", default=None, namespaces=self.NS),
                year=_year(published),
                journal=e.findtext("arxiv:journal_ref", default=None, namespaces=self.NS),
                source=self.name))
        return out


# ----------------------------------------------------------- Europe PMC -----
class EuropePMCSource:
    name = "europepmc"
    API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, rate: float = 1.0):
        self.rl = RateLimiter(rate)

    def _query(self, query: str, since: dt.date | None, preprint: bool,
               journals: list[dict] | None) -> str:
        parts = [f"({query})"]
        parts.append("SRC:PPR" if preprint else "SRC:MED")
        if since:
            parts.append(f"FIRST_PDATE:[{since.isoformat()} TO {dt.date(2100,1,1).isoformat()}]")
        if journals and not preprint:
            terms: list[str] = []
            for j in journals:
                terms.append(f'JOURNAL:"{j["name"]}"')
                if j.get("issn"):
                    terms.append(f'ISSN:"{j["issn"]}"')
            parts.append("(" + " OR ".join(terms) + ")")
        return " AND ".join(parts)

    def search(self, query: str, *, since: dt.date | None = None, limit: int = 25,
               preprint: bool = False, journals: list[dict] | None = None) -> list[PaperRecord]:
        self.rl.wait()
        data = http_get_json(self.API, params={
            "query": self._query(query, since, preprint, journals),
            "format": "json", "pageSize": min(limit, 100), "resultType": "core"}, timeout=40)
        out: list[PaperRecord] = []
        for r in data.get("resultList", {}).get("result", []):
            ji = (r.get("journalInfo") or {}).get("journal") or {}
            out.append(PaperRecord(
                title=(r.get("title") or "").strip().rstrip("."),
                abstract=(r.get("abstractText") or "").strip(),
                doi=r.get("doi"),
                pmid=r.get("pmid"),
                year=_year(r.get("firstPublicationDate") or str(r.get("pubYear", ""))),
                journal=ji.get("title") or r.get("bookOrReportDetails", {}).get("publisher"),
                source=self.name + ("_ppr" if preprint else ""),
            ))
        return out


# ------------------------------------------------------- Semantic Scholar ---
class SemanticScholarSource:
    name = "semantic_scholar"
    API = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,externalIds,year,venue,publicationDate"

    def __init__(self, api_key: str | None = None, rate: float = 1.6):
        self.api_key = api_key
        self.rl = RateLimiter(rate)   # authed limit ~1 req/s; padded, since the search endpoint is stricter

    def search(self, query: str, *, since: dt.date | None = None, limit: int = 15) -> list[PaperRecord]:
        self.rl.wait()
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        data = http_get_json(self.API, params={
            "query": query, "limit": min(limit, 100), "fields": self.FIELDS},
            headers=headers, timeout=40)
        out: list[PaperRecord] = []
        for r in data.get("data", []) or []:
            pubdate = r.get("publicationDate")
            if since and pubdate and pubdate < since.isoformat():
                continue
            ext = r.get("externalIds") or {}
            out.append(PaperRecord(
                title=(r.get("title") or "").strip(),
                abstract=(r.get("abstract") or "").strip(),
                doi=ext.get("DOI"),
                arxiv_id=ext.get("ArXiv"),
                pmid=str(ext["PubMed"]) if ext.get("PubMed") else None,
                s2_id=r.get("paperId"),
                year=r.get("year"),
                journal=r.get("venue") or None,
                source=self.name))
        return out
