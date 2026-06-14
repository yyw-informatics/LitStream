"""Entity normalization + biological-context compatibility.

Turns raw record strings into typed :class:`Entity` objects, keeping modality distinctions the rest
of the pipeline depends on (CD25 protein != IL2RA RNA; Treg != all CD4 T cells; signature != gene).
Cell-type compatibility is decided from the curated ``immune_cell_aliases.yml`` is_a tree; marker<->gene
bridges from ``marker_gene_aliases.yml``.

No network, no model — only ``yaml`` (stdlib otherwise). Resources ship in ``resources/``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import HypothesisConfig
from .schema import BioContext, Entity, make_entity_id, slug

_RES = Path(__file__).resolve().parent / "resources"

_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu",
    "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi",
    "ψ": "psi", "ω": "omega",
    "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Κ": "kappa", "Λ": "lambda",
    "Μ": "mu", "Σ": "sigma", "Ω": "omega",
}


def fold_greek(text: str) -> str:
    return "".join(_GREEK.get(ch, ch) for ch in (text or ""))


def normalize_text(text: str) -> str:
    """Lowercase, squeeze whitespace, strip edge punctuation. Mirrors the shared eval normalizer so
    'CD8.' == 'cd8'; folds Greek first so 'IFN-β' == 'ifn-beta'."""
    t = re.sub(r"\s+", " ", fold_greek(str(text or ""))).strip().casefold()
    return t.strip(" .,:;()[]{}")


def _canon_key(text: str) -> str:
    """Alias-lookup key: Greek-folded, lowercased, whitespace-squeezed; internal punctuation kept
    (so 'anti-CD3/CD28' stays distinct from 'anti CD3 CD28')."""
    return re.sub(r"\s+", " ", fold_greek(str(text or ""))).strip().casefold()


_DISEASE_ALIASES = {
    "covid-19": "COVID-19", "covid 19": "COVID-19", "covid19": "COVID-19", "covid": "COVID-19",
    "sars-cov-2 infection": "COVID-19", "sars-cov-2": "COVID-19", "sars cov 2": "COVID-19",
    "type 1 diabetes": "type 1 diabetes", "t1d": "type 1 diabetes",
    "systemic lupus erythematosus": "systemic lupus erythematosus", "sle": "systemic lupus erythematosus",
    "rheumatoid arthritis": "rheumatoid arthritis", "ra": "rheumatoid arthritis",
}

_PERTURBATION_ALIASES = {
    "anti-cd3/cd28": "anti-CD3/CD28", "cd3/cd28 stimulation": "anti-CD3/CD28",
    "anti-cd3/anti-cd28": "anti-CD3/CD28", "cd3/cd28": "anti-CD3/CD28", "anti-cd3 anti-cd28": "anti-CD3/CD28",
    "ifn-beta": "IFN-beta", "ifn beta": "IFN-beta", "interferon beta": "IFN-beta",
    "interferon-beta": "IFN-beta", "ifnb": "IFN-beta", "ifn-b": "IFN-beta", "ifnbeta": "IFN-beta",
    "ifn-gamma": "IFN-gamma", "interferon gamma": "IFN-gamma", "ifng": "IFN-gamma",
    "ifn-alpha": "IFN-alpha", "interferon alpha": "IFN-alpha",
    "lps": "LPS", "lipopolysaccharide": "LPS",
    "anti-pd-1": "anti-PD-1", "anti pd 1": "anti-PD-1", "anti-pd1": "anti-PD-1", "pembrolizumab": "anti-PD-1",
    "il-2": "IL-2", "il2": "IL-2", "tgf-beta": "TGF-beta", "tgf beta": "TGF-beta",
}


@lru_cache(maxsize=1)
def _load_yaml(name: str) -> dict[str, Any]:
    import yaml
    path = _RES / name
    return yaml.safe_load(path.read_text()) or {}


@lru_cache(maxsize=1)
def _cell_index() -> tuple[dict[str, str], dict[str, str | None]]:
    """Build (alias->canonical, canonical->parent) from immune_cell_aliases.yml."""
    raw = _load_yaml("immune_cell_aliases.yml")
    alias2canon: dict[str, str] = {}
    parent: dict[str, str | None] = {}
    for canon, body in raw.items():
        body = body or {}
        parent[canon] = body.get("parent")
        alias2canon[normalize_text(canon)] = canon
        for al in body.get("aliases", []) or []:
            alias2canon[normalize_text(al)] = canon
    return alias2canon, parent


@lru_cache(maxsize=1)
def _marker_index() -> tuple[dict[str, str], dict[str, str], list[frozenset[str]]]:
    """(marker_norm->gene, gene_norm->marker, equivalence groups) from marker_gene_aliases.yml."""
    raw = _load_yaml("marker_gene_aliases.yml")
    m2g: dict[str, str] = {}
    g2m: dict[str, str] = {}
    for marker, gene in (raw.get("markers") or {}).items():
        m2g[normalize_text(marker)] = gene
        g2m.setdefault(normalize_text(gene), marker)
    groups = [frozenset(normalize_text(x) for x in grp) for grp in (raw.get("groups") or [])]
    return m2g, g2m, groups


class Normalizer:
    """Stateless façade over the resource tables (resources are module-cached)."""

    def aliases_for(self, symbol: str) -> set[str]:
        """Accepted spellings for a gene/marker symbol, for entity-presence matching only — NEVER for
        sign composition."""
        s = normalize_text(symbol)
        if not s:
            return set()
        out = {s}
        for grp in _marker_index()[2]:
            if s in grp:
                out |= set(grp)
        try:
            from litstream.eval.extraction_score import gene_aliases
            out |= {normalize_text(a) for a in gene_aliases(symbol, "")}
        except Exception:
            pass
        return out

    def gene(self, symbol: str, species: str = "", raw: str | None = None) -> Entity:
        name = (symbol or "").strip()
        sp = self._species(species)
        conf, warns = 1.0, {}
        if sp == "mouse":
            conf = 0.7
        ent_sp = sp or "human" if sp else None
        marker = _marker_index()[1].get(normalize_text(name))
        attrs: dict[str, Any] = {}
        if marker:
            attrs["measured_as_marker"] = marker
        return Entity(
            entity_id=make_entity_id("GENE_RNA", ent_sp, name),
            type="GENE_RNA", canonical_name=name,
            raw_names=tuple(filter(None, {raw, symbol})), species=ent_sp,
            normalizer="rule", normalization_confidence=conf, attrs=attrs,
        )

    def surface_marker(self, marker: str, maps_to_gene: str = "", species: str = "") -> Entity:
        name = (marker or "").strip()
        sp = self._species(species)
        ent_sp = sp or "human" if sp else None
        gene = (maps_to_gene or "").strip() or _marker_index()[0].get(normalize_text(name), "")
        attrs: dict[str, Any] = {"bridge_type": "surface_marker_to_gene"}
        if gene:
            attrs["maps_to_gene"] = gene
        return Entity(
            entity_id=make_entity_id("SURFACE_PROTEIN", ent_sp, name),
            type="SURFACE_PROTEIN", canonical_name=name,
            raw_names=(marker,), species=ent_sp, attrs=attrs,
        )

    def canonical_cell_type(self, name: str) -> tuple[str, bool]:
        """(canonical name, known?). Unknown names pass through cleaned, flagged not-in-ontology."""
        alias2canon, _ = _cell_index()
        key = normalize_text(name)
        if key in alias2canon:
            return alias2canon[key], True
        return (name or "").strip(), False

    def cell_type(self, name: str, species: str = "") -> Entity:
        canon, known = self.canonical_cell_type(name)
        sp = self._species(species)
        ent_sp = sp or "human" if sp else None
        attrs: dict[str, Any] = {}
        parent = _cell_index()[1].get(canon)
        if parent:
            attrs["parent"] = parent
        return Entity(
            entity_id=make_entity_id("CELL_TYPE", ent_sp, canon),
            type="CELL_TYPE", canonical_name=canon, raw_names=(name,),
            species=ent_sp, normalization_confidence=1.0 if known else 0.6, attrs=attrs,
        )

    def _ancestors(self, canon: str) -> set[str]:
        _, parent = _cell_index()
        out: set[str] = set()
        cur = parent.get(canon)
        seen = {canon}
        while cur and cur not in seen:
            out.add(cur)
            seen.add(cur)
            cur = parent.get(cur)
        return out

    def cell_types_compatible(self, a: str, b: str) -> tuple[bool, float, list[str]]:
        """Exact -> 1.0; parent/child -> 0.6; one/both side unknown -> 0.8/0.4 (can't refute);
        both known but no is_a relation -> blocked (different cell types)."""
        if not (a or "").strip() or not (b or "").strip():
            return (True, 0.8, ["cell_type_unknown"])
        ca, ka = self.canonical_cell_type(a)
        cb, kb = self.canonical_cell_type(b)
        if normalize_text(ca) == normalize_text(cb):
            return (True, 1.0, [])
        if ka and kb:
            if cb in self._ancestors(ca) or ca in self._ancestors(cb):
                return (True, 0.6, ["cell_type_parent_child"])
            return (False, 0.0, ["cell_type_lineage_mismatch"])
        return (True, 0.8 if (ka or kb) else 0.4, ["cell_type_unknown"])

    def disease(self, name: str) -> Entity:
        canon = _DISEASE_ALIASES.get(_canon_key(name), fold_greek(name).strip())
        return Entity(
            entity_id=make_entity_id("DISEASE", None, canon),
            type="DISEASE", canonical_name=canon, raw_names=(name,), species=None,
        )

    def perturbation(self, name: str) -> Entity:
        canon = _PERTURBATION_ALIASES.get(_canon_key(name), fold_greek(name).strip())
        return Entity(
            entity_id=make_entity_id("PERTURBATION", None, canon),
            type="PERTURBATION", canonical_name=canon, raw_names=(name,), species=None,
        )

    def signature(self, name: str, genes: list[str] | None = None, species: str = "") -> Entity:
        sp = self._species(species)
        ent_sp = sp or "human" if sp else None
        return Entity(
            entity_id=make_entity_id("SIGNATURE", ent_sp, name),
            type="SIGNATURE", canonical_name=(name or "").strip(), raw_names=(name,),
            species=ent_sp, attrs={"genes": list(genes or [])},
        )

    def marker_for_gene(self, gene_symbol: str) -> str | None:
        return _marker_index()[1].get(normalize_text(gene_symbol))

    def gene_for_marker(self, marker: str) -> str | None:
        return _marker_index()[0].get(normalize_text(marker))

    @staticmethod
    def _species(species: str) -> str:
        t = (species or "").casefold()
        if "human" in t or "homo" in t or "patient" in t:
            return "human"
        if "mouse" in t or "murine" in t or t.strip() in {"mus", "mus musculus"}:
            return "mouse"
        return ""


def _multiset(values: tuple[str, ...]) -> set[str]:
    return {normalize_text(v) for v in values if v and v.strip()}


def contexts_compatible(
    a: BioContext, b: BioContext, config: HypothesisConfig, norm: Normalizer
) -> tuple[bool, float, list[str]]:
    """Decide whether two finding contexts may legitimately compose, and a [0,1] match score.
    Conservative: block on a hard conflict (species, named-disease, cell-lineage) rather than
    penalize. Returns (compatible, score, notes)."""
    notes: list[str] = []
    score = 1.0

    sa, sb = _multiset(a.species), _multiset(b.species)
    if sa and sb and not (sa & sb):
        if not config.allow_cross_species:
            return (False, 0.0, ["species_mismatch"])
        notes.append("cross_species")
        score *= 0.4
    elif not sa or not sb:
        notes.append("species_unknown")
        score *= 0.8

    ca = " ".join(a.cell_type)
    cb = " ".join(b.cell_type)
    ok, cscore, cnotes = norm.cell_types_compatible(ca, cb)
    if not ok:
        return (False, 0.0, cnotes)
    notes += cnotes
    score *= cscore

    da, db = _multiset(a.disease), _multiset(b.disease)
    if da and db and not (da & db):
        return (False, 0.0, ["disease_mismatch"])
    if (da and not db) or (db and not da):
        notes.append("disease_unknown")
        score *= 0.9

    ta, tb = _multiset(a.tissue), _multiset(b.tissue)
    if ta and tb and not (ta & tb):
        if not config.allow_context_transfer:
            return (False, 0.0, ["tissue_mismatch"])
        notes.append("tissue_mismatch")
        score *= 0.5
    elif not ta or not tb:
        score *= 0.95

    return (True, round(min(score, 1.0), 3), notes)
