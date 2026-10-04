"""Phase 4 - query understanding."""
from __future__ import annotations

import re

from src import config
from src.config import FILLER_PREFIXES, SCHEME_ALIASES, OTHER_AMC_ALIASES, FACT_TYPES
from src.types import Intent, QueryPlan


__all__ = [
    "normalize",
    "expand_shorthand",
    "resolve_scheme",
    "mentions_other_amc",
    "classify_intent",
    "build_plan",
]


def normalize(raw: str) -> str:
    if not raw:
        return ""
    t = raw.lower()
    for p in FILLER_PREFIXES:
        t = re.sub(r"^" + re.escape(p) + r"\b\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def expand_shorthand(text: str) -> tuple[str, list[str]]:
    t = text
    audit = []
    for phrase, expansions in getattr(config, "SHORTHAND", {}).items():
        try:
            if re.search(phrase, t, flags=re.IGNORECASE):
                exps = expansions
                if isinstance(exps, str):
                    exps = [exps]
                for e in exps:
                    if e.lower() not in t.lower():
                        t = t + " " + e
                        audit.append(f"expanded:{phrase}->{e}")
        except Exception:
            if str(phrase).lower() in t.lower():
                exps = expansions
                if isinstance(exps, str):
                    exps = [exps]
                for e in exps:
                    if e.lower() not in t.lower():
                        t = t + " " + e
                        audit.append(f"expanded:{phrase}->{e}")
    return t, audit


def resolve_scheme(text: str) -> str | None:
    t = text.lower()
    best = None
    best_len = -1
    for key, aliases in SCHEME_ALIASES.items():
        for a in aliases:
            a_low = a.lower()
            if re.search(r"\b" + re.escape(a_low) + r"\b", t):
                if len(a_low) > best_len:
                    best = key
                    best_len = len(a_low)
    return best


def mentions_other_amc(text: str) -> str | None:
    t = text.lower()
    for amc in OTHER_AMC_ALIASES:
        if re.search(r"\b" + re.escape(amc.lower()) + r"\b", t):
            return amc
    return None


def classify_intent(text: str) -> Intent:
    t = text.lower()
    best = ("unknown", 0)
    for intent, spec in FACT_TYPES.items():
        score = 0
        for k in spec.get("keywords", []):
            k_low = k.lower()
            if re.search(r"\b" + re.escape(k_low) + r"\b", t):
                score += 1
        if score > best[1]:
            best = (intent, score)
    if best[1] == 0:
        return "unknown"
    return best[0]  # type: ignore


def build_plan(raw: str) -> QueryPlan:
    norm = normalize(raw)
    expanded, expansions = expand_shorthand(norm)
    scheme = resolve_scheme(expanded)
    intent = classify_intent(expanded)
    return QueryPlan(
        raw=raw,
        normalized=norm,
        expanded=expanded,
        scheme_key=scheme,
        intent=intent,
        expansions=expansions,
    )