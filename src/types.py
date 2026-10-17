"""Shared dataclass contracts for every stage of the pipeline.

These are defined in Phase 0 so that later phases *import* contracts instead of
inventing their own. Shapes follow ARCHITECTURE.md sections 4.5, 4.6, 4.7 and
4.10; do not change a field without updating ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# Literal vocabularies
# --------------------------------------------------------------------------- #

#: A recognised question type, or ``"unknown"`` when nothing matched.
Intent = Literal[
    "expense_ratio",
    "exit_load",
    "min_sip",
    "min_lump_sum",
    "lock_in",
    "riskometer",
    "benchmark",
    "nav",
    "statement_download",
    "tax",
    "inception",
    "fund_manager",
    "unknown",
]

#: Why the assistant declined to answer. Doubles as the key into
#: :data:`src.config.REFUSAL_TEMPLATES`.
RefusalKind = Literal[
    "advice",
    "performance",
    "pii",
    "out_of_scope",
    "low_confidence",
    "nav",
    "unknown_intent",
]

#: Whether a turn produced a grounded answer or a refusal.
MessageKind = Literal["answer", "refusal"]


# --------------------------------------------------------------------------- #
# Corpus / ingestion
# --------------------------------------------------------------------------- #


#: ``scheme_key`` value for a document that is not scheme-specific. Such chunks
#: must match *any* resolved scheme during retrieval (Phase 4), so the retriever
#: ORs this value into its metadata filter rather than treating it as a scheme.
CROSS_SCHEME_KEY: str = "all-schemes"


@dataclass(frozen=True)
class SourceSpec:
    """One allow-listed public scheme page, fetched live over HTTP.

    ``url`` is the only URL this project is ever permitted to fetch or cite.
    """

    scheme_key: str
    scheme_name: str
    category: str
    url: str

    def __post_init__(self) -> None:
        for field_name in ("scheme_key", "scheme_name", "category", "url"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"SourceSpec.{field_name} must not be empty")


@dataclass(frozen=True)
class AuthoredSource:
    """A hand-transcribed official document, used where the publisher's own page
    cannot be fetched programmatically.

    Needed because the official AMC domain (``hdfcfund.com``) answers every
    automated request with ``403 Access Denied``, and Groww's scheme pages do
    not publish the ELSS lock-in or the capital-gains-statement procedure at
    all. The content below is transcribed from the cited official page and the
    document says so in its own metadata header - nothing here pretends to be a
    scrape. ``live_fetch_status`` records the status that prompted transcription,
    so ``sources.csv`` never implies we fetched what we did not.

    Attributes:
        doc_id: Filename stem under ``data/corpus/``; also the chunk namespace.
        scheme_key: Owning scheme, or ``None`` when the document applies to
            every scheme (e.g. the RTA statement procedure).
        title: H1 written into the document.
        category: Recorded in the document header and in ``sources.csv``.
        url: The official page the content was transcribed from. Added to the
            allow-list so it can be cited like any other source.
        publisher: Display name for ``sources.csv``.
        markdown_body: The transcribed content, already in Markdown.
        transcribed_on: ISO date of transcription, used as ``fetched_at`` so
            the freshness stamp stays truthful.
        live_fetch_status: HTTP status observed when we tried to fetch ``url``.
    """

    doc_id: str
    scheme_key: str | None
    title: str
    category: str
    url: str
    publisher: str
    markdown_body: str
    transcribed_on: str
    live_fetch_status: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("doc_id", "title", "category", "url", "publisher", "markdown_body"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"AuthoredSource.{field_name} must not be empty")

    @property
    def chunk_scheme_key(self) -> str:
        """``scheme_key`` as it is stored in chunk metadata."""
        return self.scheme_key or CROSS_SCHEME_KEY


# --------------------------------------------------------------------------- #
# Stage 5 - query rewrite
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QueryPlan:
    """Result of rewriting a raw user question.

    ``expanded`` is what gets embedded (and what is shown to the generator);
    ``raw`` is only ever used for guardrails and logging.
    """

    raw: str
    normalized: str
    expanded: str
    scheme_key: str | None
    intent: Intent
    #: Human-readable audit trail, e.g. ``["min sip -> minimum SIP amount"]``.
    #: Rendered in the UI trace so the rewrite is explainable.
    expansions: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stages 6-8 - retrieval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrievedChunk:
    """One scored chunk returned by the vector store."""

    #: The full chunk record as stored in ``data/chunks/chunks.jsonl``.
    chunk: dict[str, Any]
    #: Cosine similarity, higher is better, 1.0 is best.
    score: float
    rank: int

    @property
    def source_url(self) -> str:
        return str(self.chunk.get("source_url", ""))

    @property
    def scheme_key(self) -> str:
        return str(self.chunk.get("scheme_key", ""))

    @property
    def label(self) -> str:
        return str(self.chunk.get("label", ""))

    @property
    def section(self) -> str:
        return str(self.chunk.get("section", ""))

    @property
    def text(self) -> str:
        return str(self.chunk.get("text", ""))

    @property
    def fetched_at(self) -> str:
        return str(self.chunk.get("fetched_at", ""))


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of stages 6-8, including what was thrown away and why."""

    chunks: list[RetrievedChunk]
    dropped_low_score: int
    dropped_scheme_mismatch: int
    #: Top surviving similarity, or 0.0 when nothing survived. Compared against
    #: :data:`src.config.SIMILARITY_FLOOR` by guardrail phase B.
    confidence: float
    #: True when a ``scheme_key`` metadata filter was applied.
    used_filter: bool

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def schemes(self) -> list[str]:
        """Distinct scheme keys present, in rank order."""
        seen: list[str] = []
        for rc in self.chunks:
            if rc.scheme_key and rc.scheme_key not in seen:
                seen.append(rc.scheme_key)
        return seen

    def label_texts(self) -> list[str]:
        """Lower-cased label + section of every chunk, for intent grounding."""
        return [f"{rc.label} {rc.section}".lower() for rc in self.chunks]


# --------------------------------------------------------------------------- #
# Stage 9 - refusals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RefusalResult:
    """A decision to not answer, plus one educational link.

    ``message`` is the body only: no link (kept in :attr:`link`) and no digits,
    so the "no digits in a refusal" invariant holds unconditionally.
    """

    kind: RefusalKind
    message: str
    link: str
    #: Rule id for the trace, e.g. ``"pii.pan"`` or ``"confidence.low"``.
    rule: str


# --------------------------------------------------------------------------- #
# Stage 10 - generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LLMResponse:
    """Parsed, validated output of the generator."""

    answer: str
    #: 1-based index into the numbered context blocks. Clamped on parse.
    source_index: int
    insufficient_context: bool
    #: The untouched completion, kept for the UI trace.
    raw: str
    #: True when the sentence cap cut the answer down.
    truncated: bool = False


# --------------------------------------------------------------------------- #
# Stage 11 - final message
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssistantMessage:
    """Exactly what the UI renders. Assembled by ``src.citation.finalise``."""

    kind: MessageKind
    #: Rendered body: <= 3 sentences, contains no URL.
    text: str
    #: Exactly one allow-listed URL, or "" for a refusal with no link.
    source_url: str
    #: ``"27 Sep 2026"`` style, taken from chunk metadata - never from the LLM.
    last_updated: str
    disclaimer: str
    truncated: bool = False
    #: True when the LLM's URL was discarded and replaced with a chunk's.
    link_substituted: bool = False
    cached: bool = False


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineResult:
    """What ``src.pipeline.answer`` returns."""

    message: AssistantMessage
    #: Keys: ``stage_5_rewrite``, ``guardrails_a``, ``stage_6_8_retrieval``,
    #: ``guardrails_b``, ``stage_10_generation``, ``stage_11_citation``.
    #: Rendered verbatim by the UI's "Show pipeline" expander.
    trace: dict[str, Any] = field(default_factory=dict)
