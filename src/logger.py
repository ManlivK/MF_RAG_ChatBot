"""Redacted audit log (ARCHITECTURE.md section 5.3, PRD FR13/NFR7).

One JSONL line per query, append-only, under ``logs/query_log.jsonl``.

Two rules, and the second one is why this module exists at all:

1. A query that tripped a PII rule is written as ``"<pii:detected>"`` - the rule id
   only, never the text. Checking the guardrail *after* the fact is too late, so the
   decision is made here from the plan's raw text and the raw text is dropped.
2. Everything else that gets logged passes through :func:`redact` first. That is a
   second, independent layer: if a PII pattern is ever added to a place the guardrail
   does not look, the log still cannot contain it.

The log is the data used to tune the similarity floor later, so a non-PII line records
the normalised query, the top score, the sources consulted and the latencies.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.guardrails import detect_pii
from src.types import AssistantMessage, QueryPlan, RetrievalResult

__all__ = ["log_path", "redact", "log_query", "read_log"]


def log_path() -> Path:
    """Path of the JSONL audit log."""
    return config.PATHS["LOG_FILE"]


def redact(text: str) -> str:
    """Replace every PII match in ``text`` with ``<pii:kind>``.

    Args:
        text: Any string that may be about to be written to disk.

    Returns:
        The string with each detected PII span replaced by a kind marker - the rule
        id without its ``pii.`` prefix, so a reader can still tell a PAN from an
        email address. Unrecognised text is returned unchanged.
    """
    if not text:
        return text
    redacted = text
    for rule, pattern in config.PII_PATTERNS.items():
        kind = rule.split(".")[-1]
        redacted = re.sub(
            pattern, config.PII_REDACT_TEMPLATE.format(kind=kind), redacted, flags=re.IGNORECASE
        )
    return redacted


def _refusal_rules(guardrails: list[dict[str, Any]]) -> list[str]:
    """Rule ids from a trace's guardrail list that fired."""
    rules: list[str] = []
    for entry in guardrails or []:
        if not isinstance(entry, dict):
            continue
        rule = str(entry.get("rule", ""))
        if not rule:
            continue
        if entry.get("hit") and not entry.get("rules"):
            rules.append(rule)
        elif entry.get("hit"):
            rules.extend(str(item) for item in entry.get("rules") or [])
    return rules


def log_query(
    plan: QueryPlan,
    result: RetrievalResult | None,
    guardrails: list[dict[str, Any]],
    msg: AssistantMessage,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one redacted line to the audit log.

    Args:
        plan: The Stage 5 plan. Its ``raw`` is what the PII check runs on.
        result: The retrieval outcome, or ``None`` when retrieval never ran.
        guardrails: The Phase A and Phase B trace lists. A rule that fired is recorded
            in ``refusal``; that is what lets a reader see *why* a query was refused
            without seeing the query.
        msg: The final message. Only its kind, cached flag and timings are logged.
        timings: Optional ``{"llm_ms", "retrieval_ms", "total_ms"}`` figures.

    Returns:
        The line that was written, so a caller (or a test) can assert on it without
        reading the file back.
    """
    timings = timings or {}
    pii_rules = detect_pii(plan.raw) if plan is not None else []
    pii_detected = bool(pii_rules)

    if pii_detected:
        query_text = config.PII_LOG_PLACEHOLDER
    else:
        query_text = redact(plan.normalized if plan is not None else "")

    sources = [retrieved.source_url for retrieved in (result.chunks if result else [])]
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": "anon",
        "query_redacted": query_text,
        "pii_detected": pii_detected,
        "pii_rules": pii_rules,
        "intent": plan.intent if plan is not None else "unknown",
        "scheme_key": plan.scheme_key if plan is not None else None,
        "refusal": _refusal_rules(guardrails),
        "kind": msg.kind,
        "top_score": result.confidence if result else None,
        "sources": sources,
        "chunks_used": len(sources),
        "cached": msg.cached,
        "llm_ms": int(timings.get("llm_ms", 0) or 0),
        "retrieval_ms": int(timings.get("retrieval_ms", 0) or 0),
        "total_ms": int(timings.get("total_ms", 0) or 0),
    }

    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_log(limit: int | None = None) -> list[dict[str, Any]]:
    """Read the audit log back, newest last.

    Args:
        limit: Return only the last ``limit`` records. ``None`` returns all.

    Returns:
        The parsed records. A missing or unparseable log yields an empty list rather
        than an exception - the sidebar must not crash because the log is absent.
    """
    path = log_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records[-limit:] if limit else records