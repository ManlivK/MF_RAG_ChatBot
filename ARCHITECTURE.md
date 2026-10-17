# Architecture — MF Facts RAG FAQ Assistant

| Field | Value |
| --- | --- |
| Document type | Technical Architecture Document (TAD) |
| Version | 1.0 |
| Status | Draft for team review |
| Parent spec | [PRD.md](./PRD.md) v1.0 |
| Owner | NextLeap RAG Chatbot Team |
| Last updated | 2026-09-27 |

> This document is the technical realisation of **PRD §6 (System Architecture)**. It
> fixes the component boundaries, data contracts, algorithms, failure behaviour, and
> latency budget for each RAG stage. Product requirements and constraints live in the
> PRD; nothing here may contradict it.

---

## 1. Architecture Drivers

Design decisions below are driven by these five forces, in priority order.

| # | Driver | Architectural consequence |
| --- | --- | --- |
| D1 | **Hallucination of a financial fact is the worst possible failure** (PRD R3, C1–C3) | Multiple independent gates: allow-listed corpus, similarity floor, context-only prompt, temperature ~0, URL allow-list verification, deterministic pre-LLM refusal |
| D2 | **Must be a real RAG system, not a lookup table** (PRD G5, C8) | One module per stage, a persisted artifact per stage, and a full per-query trace rendered in the UI |
| D3 | **Live demo, free-tier APIs, laptop hardware** (PRD R4, NFR1) | Ingestion is offline and cached; index is built once and persisted; response cache; cold-start budget enforced |
| D4 | **Tiny corpus (5 pages, ~1–2k words each)** | No reranker model needed — score filtering + metadata filtering is sufficient and keeps latency low |
| D5 | **No PII, ever** (PRD C2, NFR7) | PII detection is the *first* thing that touches user input; PII text is never persisted |

### 1.1 Resolved ambiguity in the PRD

The PRD diagram places guardrails at **stage 9 (after retrieval)**, while PRD §6.5
requires out-of-scope detection to "short-circuit to refusal *before* retrieval". This
architecture resolves it with a **two-phase guardrail**:

- **Phase A (pre-retrieval, stage 5):** PII, advice, performance, out-of-scope scheme.
  Pure string/regex checks, ~0 ms, no embeddings, no LLM.
- **Phase B (post-retrieval, stage 9):** retrieval confidence and fact-type grounding.
  Requires scores, therefore after stage 8.

Refusals are returned from either phase through the same `RefusalResult` shape, so the UI
cannot tell the difference except by the trace.

---

## 2. System Context

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             USER (browser)                                │
│              Retail investor  /  Support-content team                     │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ HTTP (Streamlit websocket)
┌───────────────────────────────────▼────────────────────────────────────────┐
│                     PRESENTATION  —  app.py (Streamlit)                   │
│  welcome line · 3 example questions · chat thread · citation link           │
│  disclaimer · sidebar index stats · "Show pipeline" expander (trace)        │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ in-process function calls
┌───────────────────────────────────▼────────────────────────────────────────┐
│                      ORCHESTRATION  —  src/pipeline.py                    │
│  runs stages 5→11 in order, short-circuits on refusal, emits TraceEvent    │
│  owns cache + logger + response shaping                                     │
└──┬──────────────┬──────────────┬──────────────┬──────────────┬─────────────┘
   │              │              │              │              │
┌──▼─────────┐ ┌──▼──────────┐ ┌▼────────────┐ │              │
│ QUERY      │ │ RETRIEVER  │ │ GUARDRAILS  │ │              │
│ src/query  │ │ src/        │ │ src/        │ │              │
│ rewrite +  │ │ retriever   │ │ guardrails  │ │              │
│ scope      │ │ (stages     │ │ (phase A+B) │ │              │
│ resolve    │ │  6-8)       │ │             │ │              │
└──┬─────────┘ └──┬──────────┘ └─────────────┘ │              │
   │              │                            │              │
   │        ┌─────▼────────────────────────────▼──┐           │
   │        │ VECTOR STORE  src/vectorstore.py    │           │
   │        │ ChromaDB (persistent, local disk)   │           │
   │        │ collection: mf_faq_hdfc             │           │
   │        └─────┬───────────────────────────────┘           │
   │              │ reads                                        │
   │        ┌─────▼────────────────────────────────────┐       │
   │        │ EMBEDDER  src/embedding.py                │       │
   │        │ sentence-transformers/all-MiniLM-L6-v2   │       │
   │        └─────▲────────────────────────────────────┘       │
   │              │ same model, always                         │
   │        ┌─────┴────────────────────────────────────┐       │
   │        │ GENERATOR  src/generator.py              │       │
   │        │ Groq  llama-3.3-70b-versatile            │───────▼──────┐
   │        │ grounded prompt, JSON output              │   ┌──────────────┐
   │        └─────┬────────────────────────────────────┘   │  CITATION    │
   │              │                                        │ src/citation │
   │        ┌─────▼────────────────────────────────────┐   │ .py         │
   │        │ LOGGER  src/logger.py                     │◄──┤ verify+stamp│
   │        │ logs/query_log.jsonl (PII-redacted)       │   └──────────────┘
   │        └───────────────────────────────────────────┘
   │
┌──▼─────────────────────────── OFFLINE INDEX BUILD (button / CLI) ──────────┐
│  src/ingest.py   →  src/chunking.py  →  src/embedding.py  →  vectorstore.py │
│  5 URLs          →  corpus/*.md       →  chunks.jsonl     →  ChromaDB      │
└────────────────────────────────────────────────────────────────────────────┘

External:  (a) 5 public scheme pages — fetched at index time only
           (b) Groq API — generation only
           (c) Hugging Face — model download on first run, then cached locally
```

**Trust boundaries:** only two network egress points — the 5 allow-listed URLs (index
time) and the Groq API (query time). No user text reaches Groq unless it has already
passed the PII guardrail. No user text is written to disk at all.

---

## 3. Component Map

| Module | Stage(s) | Responsibility | Does **not** do |
| --- | --- | --- | --- |
| `src/config.py` | — | All tunable constants, the 5-URL allow-list, scheme keys, disclaimer text, threshold defaults | Any I/O |
| `src/ingest.py` | 1 | Fetch + clean HTML → Markdown snapshot; write `sources.csv`; fall back to committed snapshot | Chunking, embedding |
| `src/chunking.py` | 2 | Structure-aware split → `chunks.jsonl`; label detection; label-carrying merge | Network calls |
| `src/embedding.py` | 3 | Load MiniLM once, embed documents and queries with identical settings, model fingerprint | Persistence |
| `src/vectorstore.py` | 4 | Open/create Chroma collection, upsert chunks idempotently, similarity search with metadata filters | Embedding model loading |
| `src/query.py` | 5 | Normalise, expand shorthand, resolve scheme key, detect out-of-scope | Guardrail policy decisions |
| `src/retriever.py` | 6–8 | Query embedding, top-k search, score floor, scheme filter, context assembly | Calling the LLM |
| `src/guardrails.py` | 9 (A+B) | Phase A regex/refusal policy; Phase B confidence + fact-type grounding; refusal templates | Retrieval |
| `src/generator.py` | 10 | Build grounded prompt, call Groq, parse/validate JSON response | Deciding the final link |
| `src/citation.py` | 11 | Verify link against allow-list, apply fallback, enforce sentence cap, append last-updated stamp + disclaimer | Network calls |
| `src/pipeline.py` | 5–11 | Orchestration, cache, trace emission, response shaping for the UI | Business vocabulary |
| `src/logger.py` | — | Redacted JSONL audit log | Any decision logic |
| `app.py` | — | Streamlit rendering only | Any retrieval/generation logic |

**Layering rule:** `app.py` → `pipeline.py` → stage modules → `config.py`. Stage modules
never import `app.py` or `pipeline.py`. `guardrails.py` and `citation.py` must not import
`generator.py` or `retriever.py` (keeps them pure and unit-testable without a network or
an index).

---

## 4. Stage Design

Each stage below specifies: input, algorithm, output contract, and failure behaviour.

### 4.1 Stage 1 — Document Loading (`src/ingest.py`)

**Input:** the 5 URLs in `config.SOURCES` (allow-list; any URL not in this list is never
fetched), plus the hand-authored documents in `config.AUTHORED_SOURCES`.
**Output:** `data/corpus/<doc-id>.md` (committed), `data/sources.csv`.

**Algorithm**

1. For each `SourceSpec` in `config.SOURCES` (idempotency key = scheme key):
   - `GET url` with `timeout=20`, `headers` containing a descriptive `User-Agent`, and
     `requests.adapters.HTTPAdapter(max_retries=Retry(3, backoff_factor=1.5, status_forcelist=[429,500,502,503,504]))`.
   - **Rate limit:** ≥2 s between requests; serial loop, never parallel. Five pages is
     not worth the politeness risk.
   - On non-200 or exception → log warning, keep the existing snapshot file, mark this
     spec `stale=True` in `sources.csv`. **Never raise** (NFR4).
2. Clean HTML → Markdown with `markdownify` after a `BeautifulSoup4` pass:
   - **Remove** by selector and by tag: `nav`, `footer`, `aside`, `script`, `style`,
     `noscript`, `iframe`, `form`, cookie/consent banners, "related funds" / "compare"
     widgets, and app-store badges.
   - **Do not remove `header`.** The mirror page puts the scheme name, the ELSS `3Y
     Lock-in` tag, the riskometer, NAV and minimum SIP inside `<header>`, and dropping it
     silently deletes required facts. Its non-fact part (nav links) is removed by the
     markers below instead.
   - **Keep** `main` / `[role=main]` / `article`; if absent, fall back through `div#__next`
     and `div#root` to the body.
   - **Narrow to the content element.** `#root` wraps the whole React app, mega-menu
     included, so selecting it is not enough: descend into a child while *exactly one*
     child contains ≥2 of `SCRAP_CONTENT_MARKERS` ("Fund size (AUM)", "Expense ratio",
     "Min. for SIP", "Fund benchmark", "Minimum investments"). Requiring the choice to be
     unique means an ambiguous node stops the walk instead of guessing, and unlike matching
     a hashed CSS class the rule survives a redesign.
   - **Separate inline siblings.** `markdownify` concatenates sibling `<span>`/`<a>` items
     with nothing between them, fusing each label onto its own value
     (`Fund benchmarkBSE 250 SmallCap Total Return Index`). A newline is inserted between
     adjacent short inline siblings, except inside `p`, `a`, table cells, list items and
     headings, so running prose is not split mid-clause. Unit fragments (`%`, `₹`, `Cr`) stay
     welded to the item before them.
   - **Truncate at the footer.** The site-wide mega-menu is rendered as plain `<div>`s, not a
     `<footer>`, so no selector removes it; left in place it ran to two thirds of each file
     and named *other* schemes (Kotak Small Cap Fund, Mirae Asset ELSS Tax Saver Fund).
     Everything from the first footer marker line onward is discarded.
   - **Strip returns and rankings** (PRD C3): the "Return calculator" / "Historic returns" /
     "Returns and rankings" / "Compare" sections, standalone signed return figures, standalone
     return-period labels, and `Rank`-labelled value pairs. A benchmark *name* containing
     "Total Return" is kept. Unsigned percentages (expense ratio, stamp duty, holding weights)
     are kept.
   - **Preserve** every heading (`h1`–`h4`), every list, and every table — tables are
     where holdings live. Markdownify is configured with
     `heading_style=ATX, strip=['script','style']`.
   - Collapse >2 blank lines; strip zero-width and NBSP characters.
3. Prepend an H1 title block (`{scheme_name} — {category}, Direct Growth`) and a metadata
   header so a chunk can recover the scheme without a lookup:
   ```markdown
   # HDFC Large Cap Fund — Direct Growth
   > category: Large Cap | scheme_key: hdfc-large-cap | source_url: https://... | fetched_at: 2026-09-27
   ```
4. Write file, append a row to `data/sources.csv`
   (`scheme_key, scheme_name, category, url, publisher, fetched_at, http_status, stale`).

**Authored sources**

`hdfcfund.com` answers `403 Access Denied` to automated requests, so facts that only the AMC
publishes cannot be fetched. Those are hand-transcribed into `config.AUTHORED_SOURCES` and
written with a header that cannot be confused with a scrape:

```markdown
# HDFC ELSS Tax Saver — Lock-In Period and Section 80C
> category: ELSS | scheme_key: hdfc-elss-tax-saver | source_url: https://www.hdfcfund.com/... | fetched_at: 2026-09-27 | authored: true | live_fetch_status: 403 | publisher: HDFC Asset Management (official AMC page; hand-authored)
```

A document that is not about one scheme carries `scheme_key: all-schemes`
(`types.CROSS_SCHEME_KEY`) rather than borrowing a scheme key, so the Phase 2 scheme filter
cannot pull cross-scheme guidance into a single-scheme answer.

**Failure modes**

| Failure | Behaviour |
| --- | --- |
| Network down / 403 / timeout | Use committed snapshot, `stale=True`, warn in UI sidebar. Demo continues. |
| Layout changed, `main` missing | Body fallback; if extracted text < 500 chars, treat as failure and keep snapshot |
| Encoding issues | `response.encoding = response.apparent_encoding` fallback before parse |

**Design note (for the demo):** the ingest→clean step is deliberately isolated so that
when Groww redesigns, only `ingest.py` changes. The rest of the pipeline consumes
Markdown and is unaware of HTML.

---

### 4.2 Stage 2 — Chunking (`src/chunking.py`)

**Decision record.** The brief requires deciding the strategy *from the data*, so the
decision is stated as a hypothesis, tested against the corpus, and recorded here.

*Observation:* after Stage 1 the corpus is short (est. 1–2k words per scheme), but it is
**label-dense**: facts appear as `label: value` lines, bullet lists, and small tables
(e.g. exit-load slabs, minimum SIP/lump-sum, riskometer, benchmark, "how to download"
steps).

*Amendment from the Stage 1 build:* the live pages lead with a form this section did not
list — a **stacked** label and value, one per line, with no colon:

```markdown
Min. for SIP
₹100
Fund size (AUM)
₹39,933.37 Cr
Expense ratio
1.03%
```

Measured over the committed corpus, 67–98 fact units per scraped scheme are of this shape
against 4–7 inline `label: value` pairs, so it is the *dominant* form, not a rare one. The
corpus is still label-dense, which is what the hypothesis claimed, but a label extractor
that only recognises `key: value` would leave almost every fact unlabelled. Stage 2
therefore has to treat **a label line followed by a value line** as one fact unit, and
`ingest.label_density` reports `pairs` (inline) and `stacked` separately so the two forms
stay visible.

*What this breaks:*

- Fixed 1000-token chunks merge unrelated facts (exit load + benchmark) → dilution, the
  embedding no longer discriminates, and the LLM sees noise.
- Fixed 200-token chunks can split a table, orphaning a number from its column header and
  producing confidently wrong answers (R2).

*Decision:* **structure-aware chunking with a character-splitter safety net.**

**Algorithm**

1. Parse the Markdown with `MarkdownHeaderTextSplitter` (`headers_to_split_on=["#","##","###","####"]`) → a tree of sections per document.
2. For each section, extract a **`label`** for every fact block: the nearest preceding bolded term, table column header, `key: value` key, or — for the stacked form above — the label line immediately before the value line. Labels are stored as chunk metadata so the UI and the generator can reason about them.
3. Split the section body into **fact units**:
   - a stacked label line plus the value line beneath it = one unit;
   - one table row (with its header row attached) = one unit;
   - one bullet list = one unit;
   - one paragraph = one unit.
4. Greedily pack units into chunks until `count_tokens(unit) > CHUNK_SIZE_TOKENS (400)`.
   Never emit a chunk without ≥1 labelled fact — unlabelled residue is merged forward into the previous chunk.
5. If a single unit still exceeds the budget (a very long paragraph), fall back to `RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS, separators=["\n\n","\n",". "," "])`. Table and list units are exempt — a long table is split between **rows**, never inside one.
6. `chunk_overlap = 50` tokens, implemented by carrying the last unit of the previous chunk into the next (unit-level overlap, safer than character overlap because it never duplicates a partial fact).
7. Attach metadata, drop empty chunks, and write `data/chunks/chunks.jsonl` (committed, so chunking experiments are diffable).

**Output contract** (`chunks.jsonl`, one JSON object per line):

```json
{
  "chunk_id": "hdfc-large-cap::0007",
  "text": "### Exit Load\n- 0.00% if redeemed after 12 months from the date of allotment\n- 1.00% if redeemed between 6 and 12 months",
  "scheme_key": "hdfc-large-cap",
  "scheme_name": "HDFC Large Cap Fund — Direct Growth",
  "category": "Large Cap",
  "section": "Exit Load",
  "label": "Exit Load",
  "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "source_title": "HDFC Large Cap Fund — Direct Growth",
  "fetched_at": "2026-09-27",
  "chunk_index": 7,
  "content_hash": "sha256:9f2c…"
}
```

**Parameters** (`config.py`): `CHUNK_SIZE_TOKENS = 400`, `CHUNK_OVERLAP_TOKENS = 50`,
token counting via `tiktoken.get_encoding("cl100k_base")` (deterministic, no model
download).

**Verification before accepting the strategy** (part of the build): a small
`tests/test_chunking.py` asserts that for each of the 6 canonical fact types, the chunk
containing the value also contains its label — i.e. no orphaned facts. This is the
measurable form of R2's mitigation.

---

### 4.3 Stage 3 — Embedding (`src/embedding.py`)

**Input:** list of chunk texts.
**Output:** `list[list[float]]`, 384-dim, L2-normalised.

**Configuration**

- `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True}, model_kwargs={"device": "cpu"})`
- `encode_batch_size = 32`, CPU by default (`config.EMBED_DEVICE`, overridable via env for a GPU machine).
- **Query/document prefix.** The all-MiniLM-L6-v2 model card recommends a prefix for
  *asymmetric* retrieval tasks. Documents are embedded as
  `"Represent this mutual fund scheme document for retrieval: " + text`;
  queries are embedded **unprefixed** (the prefix is for the corpus side only). The
  prefix is a constant in `config.DOC_EMBED_PREFIX` so it can be A/B'd.
- **Model fingerprint.** `embedding_fingerprint = sha256(model_name + "|" + prefix + "|" + normalize)[:12]` is written into a `data/index_meta.json` next to the collection. On startup, if the fingerprint differs from the stored one, the app flags the index stale and offers a rebuild (FR16).

**Failure modes:** model download blocked/offline → first run requires internet once;
afterwards HF cache makes it offline-safe. Mismatched fingerprint → refuse to search a
stale index.

---

### 4.4 Stage 4 — Vector Store (`src/vectorstore.py`)

**Input:** chunks + embeddings.
**Output:** persistent ChromaDB collection.

**Collection contract**

| Property | Value |
| --- | --- |
| Client | `chromadb.PersistentClient(path="data/chroma")` |
| Collection | `mf_faq_hdfc` |
| Distance | `hnsw:space = cosine` (set at creation; changing it requires recreating the collection) |
| Embedding function | `None` (we insert pre-computed vectors, so Chroma never loads a second embedding model) |
| Document id | `chunk_id` |
| Metadata | `scheme_key, scheme_name, category, section, label, source_url, source_title, fetched_at, chunk_index, content_hash` |

**Idempotency:** `upsert` on `chunk_id`, so re-running ingestion never duplicates. Since
chunk ids are positional (`::0007`), a re-ingest that changes earlier content shifts ids;
therefore the rebuild path is `delete_collection()` → recreate → upsert all, and
`content_hash` is used to skip unchanged documents during a partial refresh.

**Lifecycle:** Streamlit reruns the script on every interaction, so the client and
collection are wrapped in `@st.cache_resource` in `app.py` (constructed once per process).
Rebuild is a sidebar button that clears that resource and re-runs the pipeline.

---

### 4.5 Stage 5 — Query Rewrite & Scope Resolution (`src/query.py`)

**Input:** raw user string.
**Output:** `QueryPlan` dataclass.

```python
@dataclass(frozen=True)
class QueryPlan:
    raw: str
    normalized: str            # lowercased, filler + punctuation stripped
    expanded: str              # shorthand expanded, scheme key folded in
    scheme_key: str | None     # resolved canonical key, or None
    intent: str                # one of FACT_TYPES or "unknown"
    expansions: list[str]      # audit trail for the trace
```

**Algorithm (ordered, cheap → expensive)**

1. `normalize()`: lowercase, collapse whitespace, strip trailing punctuation, drop
   greeting/filler prefixes (`hi`, `hello`, `please tell me`, `can you tell me`).
2. `expand_shorthand()`: regex-phrase map in `config.SHORTHAND`, e.g. `\bER\b` →
   `expense ratio`, `min sip` → `minimum SIP amount`, `exit load` →
   `exit load redemption charge`, `elss` → `ELSS tax saver`, `lock in` → `lock-in period`,
   `riskometer` → `riskometer risk level`, `benchmark` → `benchmark index`.
3. `resolve_scheme()`: match the longest alias from `config.SCHEME_ALIASES` (e.g.
   `{"hdfc large cap", "large cap fund", "hdfc large cap direct"}` →
   `hdfc-large-cap`). Longest-match wins so "hdfc balanced advantage" is not shadowed by
   a generic "hdfc". A known AMC name outside the allow-list ("SBI", "Axis", "ICICI") is
   reported by `mentions_other_amc()` rather than encoded in `intent` — the `Intent`
   vocabulary in `types.py` has no out-of-scope member, so scope is decided by the Phase 5
   `scope.other_amc` rule against `plan.raw`. Keeping it out of `QueryPlan` means the
   rewrite cannot be the thing that silently widens or narrows scope.
4. `classify_intent()`: keyword scoring against `config.FACT_TYPES`
   (`expense_ratio, exit_load, min_sip, min_lump_sum, lock_in, riskometer, benchmark,
   nav, statement_download, tax`). Highest score wins; ties → `unknown`. `nav` is
   classified but never answered from the snapshot (see §4.6).

**Determinism:** no LLM is used in rewriting. That keeps stage 5 at ~0 ms, makes it unit
testable, and removes a whole class of failure.

---

### 4.6 Stages 6–8 — Retrieval (`src/retriever.py`)

**Input:** `QueryPlan`. **Output:** `RetrievalResult`.

```python
@dataclass(frozen=True)
class RetrievedChunk:
    chunk: dict          # full chunk record
    score: float         # cosine similarity, 1.0 best
    rank: int

@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    dropped_low_score: int
    dropped_scheme_mismatch: int
    confidence: float    # top score after filtering
    used_filter: bool
```

**Algorithm**

1. Embed `plan.expanded` with the identical embedder + fingerprint (Stage 3). Log a hard error if the query embedder fingerprint ≠ the index fingerprint — never silently compare vectors from different models.
2. `collection.query(query_embeddings=[v], n_results=RETRIEVE_K (15), where=where)` where `where = {"scheme_key": plan.scheme_key}` when the scheme resolved. Rationale: prevents a Small Cap exit-load answer from bleeding into a Large Cap question (D4).
3. Convert distances to similarity (`1 - cosine_distance` as reported by Chroma) and keep `TOP_K = 5` after sorting.
4. **Score floor:** drop chunks with `score < SIMILARITY_FLOOR (0.35)`.
5. **Confidence:** `confidence = chunks[0].score if chunks else 0.0`.
6. **Assemble context** as numbered blocks with the source URL inline:
   ```
   [1] (source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
        scheme: HDFC Large Cap Fund — Direct Growth | section: Exit Load | score: 0.71)
   Exit Load
   - 0.00% if redeemed after 12 months…
   ```
7. **Intent-specific override:** if `plan.intent == "nav"`, do not answer from the
   snapshot at all — return a redirect message with the official scheme page link (NAV is
   live data, excluded by PRD §3.2). Handled as a refusal with a specific template.

**Threshold tuning (D4 rationale).** With 384-dim MiniLM cosine similarity, 0.35 is a
conservative floor for a 5-document corpus. Measured on the PRD §12 evaluation set
(7 documents, 243 chunks) by `python run_index.py --stage eval-retrieval`:

| Intent | Observed top-1 score | Floor | Notes |
| --- | --- | --- | --- |
| expense_ratio | 0.660 | 0.35 | numeric + label, high separability |
| exit_load | 0.647 | 0.35 | table rows |
| min_sip | 0.686 | 0.35 | authored ELSS minimum-SIP chunk ranks first |
| lock_in | 0.622 | 0.35 | authored ELSS lock-in document; phrasing is unambiguous |
| riskometer | 0.570 | 0.35 | see note on ties below |
| min_lump_sum | 0.641 | 0.35 | numeric + label |
| statement_download | 0.578 | 0.35 | cross-scheme RTA document, no scheme filter |
| nav | 0.529 | 0.35 | retrieved, then redirected by guardrail B (`intent.nav`) |
| unknown (in scope tie) | 0.570 | 0.45 | `"riskometer and benchmark"` ties → `unknown` |
| unknown (nonsense) | 0.000 | 0.45 | refuses via `confidence.low` |

In-scope top-1 similarity spans **0.529 – 0.686**; the nonsense control scores
**0.000**. Any floor in `(0.000, 0.529)` separates them, so `SIMILARITY_FLOOR = 0.35`
sits with ~0.18 of headroom above the weakest genuine question and no risk of
admitting an unrelated chunk. `UNKNOWN_INTENT_FLOOR = 0.45` is used when Stage 5
could not classify the question; it clears the 0.570 tie case above while still
refusing the nonsense control.

**Why the lock_in floor is 0.35 and not 0.30.** The pre-build table guessed 0.30 for
`lock_in` on the assumption that ELSS phrasing varies. The measured 0.622 is the
second-highest of all in-scope intents, because the lock-in fact exists only in the
hand-authored ELSS document and nothing else in the corpus competes for it. The
guess was unnecessary; keeping the single global 0.35 also removes a per-intent
threshold table that nothing downstream reads.

**Ties resolve to `unknown`.** `"riskometer and benchmark"` scores 1 for both
intents, so Stage 5 returns `unknown` and the stricter 0.45 floor applies. It still
retrieves 2 chunks, so the question is answered; but guardrail B's `intent.grounded`
check is skipped, because there is no single intent to ground against.

**Retrieval-only latency: 42 ms mean** over the in-scope queries (NFR2 budget
1500 ms). The *first* query in a fresh process additionally pays a one-time MiniLM
model load of roughly 60 s; `app.py` holds it behind `@st.cache_resource` so that
cost is paid once per process, not per question.

No re-ranking model: at 5 documents the top-5 filter plus scheme metadata filter is
already at ceiling quality, and a cross-encoder would add 200–400 ms per query for no
measurable gain (D4, NFR2).

---

### 4.7 Stage 9 — Guardrails (`src/guardrails.py`)

Two phases, one output type. No LLM involved — every rule is a regex or a score, so
guardrail behaviour is deterministic and demonstrable live.

```python
@dataclass(frozen=True)
class RefusalResult:
    kind: str            # advice | performance | pii | out_of_scope | low_confidence | nav | unknown_intent
    message: str         # template-rendered, <= 3 sentences
    link: str            # one educational link, always allow-listed
    rule: str            # rule id for the trace
```

**Phase A — pre-retrieval (runs on `plan.raw` before any embedding)**

| Rule | Pattern (regex, case-insensitive) | Link to |
| --- | --- | --- |
| `pii.pan` | `\b[A-Z]{5}\d{4}[A-Z]\b` | scheme page |
| `pii.aadhaar` | `\b\d{4}\s?\d{4}\s?\d{4}\b` | scheme page |
| `pii.account` | `\b\d{9,18}\b` | scheme page |
| `pii.email` | `[\w.\-]+@[\w.\-]+\.\w+` | scheme page |
| `pii.phone` | `\+?\d[\d\s\-]{8,14}\d` | scheme page |
| `pii.otp` | `\b(otp|one time password)\b` | scheme page |
| `advice.recommend` | `\b(should i|shall i|can i|which (should|do) i)\b` | AMC fee/disclosure page |
| `advice.trade` | `\b(buy|sell|switch|redeem|invest in|allocate|exit my|sip into)\b` | AMC fee/disclosure page |
| `advice.advice_seek` | `\b(recommend|suggest|advise|best (fund|scheme) for|good for me|worth)\b` | AMC fee/disclosure page |
| `performance.compare` | `\b(returns?|performance|yield|ytd|1 ?year return|3 ?year return|best performing|outperform|growth of)\b` | official factsheet link |
| `scope.other_amc` | alias of any non-allow-listed AMC (SBI, Axis, ICICI Princap, Kotak, …) | allow-listed scheme index |

Ordering matters: `pii.*` is evaluated **first and unconditionally** — if PII is
present, the query is refused and the raw text is **dropped without being logged**
(NFR7). Then `advice.*`, then `performance.*`, then `scope.*`.

**Phase B — post-retrieval (runs on `RetrievalResult` + `plan.intent`)**

| Rule | Condition | Outcome |
| --- | --- | --- |
| `confidence.low` | `confidence < SIMILARITY_FLOOR` (no chunks survive) | `RefusalResult(kind="low_confidence")` → *"I don't have that in my sources."* + scheme page link |
| `intent.grounded` | `plan.intent != "unknown"` and no retrieved chunk's `section`/`label` matches the intent's expected label set | `RefusalResult(kind="unknown_intent")` → *"I couldn't find that fact for this scheme."* + link |
| `intent.nav` | `plan.intent == "nav"` | redirect to the official page (live data) |
| `intent.multi_scheme` | ≥2 distinct `scheme_key`s in top-k with near-equal scores | keep both, allow the generator to answer facts-only; the *prompt* forbids comparison |

**Refusal templates** (constants in `config.py`, so wording is identical everywhere and
testable as strings):

```text
ADVICE     : "I can share published facts about these schemes, but I can't give investment
              advice or suggest whether to buy or sell. Please read the scheme's official
              document or consult a SEBI-registered investment adviser."
PERFORMANCE: "I don't calculate or compare returns, since past performance doesn't predict
              future returns. You can check the scheme's official factsheet for published
              figures."
PII        : "I can't process personal details. Please don't share PAN, Aadhaar, account
              numbers, OTPs, email addresses, or phone numbers here."
SCOPE      : "I only have facts for these five HDFC schemes. For other funds, please check
              the fund's official page."
LOWCONF    : "I couldn't find that in the five scheme pages I was built from, so I don't want
              to guess. Here's the scheme page so you can check directly."
```

All templates are ≤3 sentences, contain no numbers, and are followed by the disclaimer.

---

### 4.8 Stage 10 — Grounded Generation (`src/generator.py`)

**Input:** `QueryPlan`, `RetrievalResult`.
**Output:** `LLMResponse` (parsed, validated).

**Model call**

- `ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=400, top_p=0.9, response_format={"type": "json_object"})`
- `max_retries=2` on the `groq` client, exponential backoff; on exhaustion → degrade to a
  no-LLM answer that still cites the top chunk (see §7 degradation matrix).
- `seed` is not exposed by Groq; temperature 0.1 is the closest available determinism.

**Prompt template** (`config.PROMPT_TEMPLATE`, the single most important artefact in the
project — it is also shown verbatim in the UI trace):

```text
SYSTEM
You are a facts-only assistant for five HDFC Asset Management mutual fund schemes.
You do not give investment advice. You do not calculate, estimate, rank, or compare
returns or performance. You never ask for or repeat personal information (PAN, Aadhaar,
account numbers, OTP, email, phone).

Answer ONLY from the <context> below. Treat the context as the complete and only
source of truth. If the context does not contain the answer, reply with the exact
sentence: "I couldn't find that in the sources I was given."

Rules for your reply:
1. At most 3 sentences. No preamble, no pleasantries, no sign-off.
2. State only facts present in the context. Copy numbers, rates, and dates exactly as
   written. Never round, convert units, or infer a value that is not in the context.
3. Never state or imply a return, yield, ranking, comparison, or recommendation.
4. Do not say "as an AI", and do not restate the question.
5. Reply with a single JSON object and nothing else:
   {"answer": "<1-3 sentences>", "source_index": <int 1..N, the context block you used>,
    "insufficient_context": <true|false>}

<context>
{blocks}
</context>

USER
{query}
```

**Output parsing & validation (`LLMResponse`)**

1. `json.loads` on the raw completion, with a fallback regex for the worst case.
2. Validate types; clamp `source_index` into `[1, len(blocks)]`.
3. Sentence-count check on `answer` — if >3, **truncate to the first 3 sentences**
   deterministically (`config.truncate_to_sentences`) and mark `truncated=True` in the
   trace. Truncation is preferred over a retry: it is faster and cannot fail.
4. `insufficient_context == true` → convert to a `RefusalResult` (do not render the
   model's prose as an answer).

**Why JSON mode:** the UI needs the answer text, the used block index, and the refusal
flag as separate values. Parsing free text with a regex is where citation bugs live;
structured output removes that class of bug (and the PRD C5 guarantee is enforced
downstream in `citation.py` regardless).

---

### 4.9 Stage 11 — Citation, Cap & Freshness (`src/citation.py`)

**Input:** `LLMResponse`, `RetrievalResult`, chunk metadata.
**Output:** `AssistantMessage` — exactly what the UI renders.

**Algorithm**

1. **Resolve the link.** Candidate = the `source_url` of block `LLMResponse.source_index`. If no valid index, fall back to `chunks[0]`.
2. **Allow-list verification (C1/C5, the hard gate).** `verify_url(candidate)`:
   - normalise (strip scheme-less prefixes, trailing `/`, tracking params);
   - exact match against `config.ALLOWED_URLS`, or match against the allow-list base
     `https://groww.in/mutual-funds/`;
   - if it fails → **discard the LLM's link entirely** and substitute the top retrieved
     chunk's URL; set `link_substituted=True` and log a warning. A hallucinated link is
     never rendered.
3. **Render the answer block:**
   ```
   <answer text, ≤3 sentences>
   Source: <url>                       ← exactly one link, clickable
   Last updated from sources: <fetched_at from chunk metadata, DD Mon YYYY>
   Facts-only. No investment advice.
   ```
   The `Last updated` date is taken from **chunk metadata**, never from the LLM (an LLM
   will invent a date).
4. **Final invariant check** (asserted in `tests/test_citation.py`): exactly one
   `https://` URL in the rendered body; the URL is in the allow-list; the last-updated
   line is present; the disclaimer is present; body sentences ≤3.
5. If the message is a refusal, `source_index` is irrelevant — the refusal's own
   educational link is used and no citation badge is shown.

---

### 4.10 Orchestration (`src/pipeline.py`)

```python
def answer(query: str) -> PipelineResult:
    trace = Trace(query=query)                       # observability
    plan = phase_a_guards(query) or rewrite(query)    # stage 5 + guardrails A
    if plan is RefusalResult: return shape(plan, trace)

    trace.retrieval = retrieve(plan)                  # stages 6-8
    phase_b = check_guards_b(plan, trace.retrieval)   # stage 9
    if phase_b is not None: return shape(phase_b, trace)

    llm = generate(plan, trace.retrieval)             # stage 10
    if llm.insufficient_context: return shape(low_conf_refusal(), trace)
    msg = citation.finalise(llm, trace.retrieval)     # stage 11
    cache.put(plan.expanded, msg)
    log_query(plan, trace, msg)                       # redacted audit log
    return msg
```

`PipelineResult` carries `message`, `trace`, and `cached: bool` so the UI can show the
"cached" badge and the full trace without a second call.

---

## 5. Data Model

### 5.1 `data/index_meta.json`

```json
{
  "collection": "mf_faq_hdfc",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_fingerprint": "a3f1c9d20b4e",
  "doc_embed_prefix": "Represent this mutual fund scheme document for retrieval: ",
  "chunk_size_tokens": 400,
  "chunk_overlap_tokens": 50,
  "chunk_strategy": "structure-aware-v1",
  "distance": "cosine",
  "doc_count": 5,
  "chunk_count": 148,
  "built_at": "2026-09-27T14:31:02+05:30",
  "generator_model": "llama-3.3-70b-versatile"
}
```

Used for the sidebar stats (PRD FR12), the stale-index check (FR16), and reproducibility
in the README.

### 5.2 `data/sources.csv`

```csv
scheme_key,scheme_name,category,url,publisher,fetched_at,http_status,stale
hdfc-large-cap,"HDFC Large Cap Fund — Direct Growth",Large Cap,https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth,Groww (public mirror of HDFC AMC data),2026-09-27,200,false
```

**7 data rows** (PRD deliverable): 5 scraped scheme pages, then 2 hand-authored documents
from the official AMC whose `http_status` is the `403` that forced the transcription and
whose `publisher` says "hand-authored". A scheme-scoped authored document repeats its
`scheme_key`, and its `scheme_name` holds the *document* title so the two ELSS rows stay
distinguishable; the cross-scheme document uses `scheme_key=all-schemes`.

### 5.3 `logs/query_log.jsonl` (FR13, NFR7)

One line per query. **PII-refusing queries are logged as a rule id only — never the raw
text.**

```json
{"ts":"2026-09-27T14:33:10+05:30","session":"anon","query_redacted":"<pii:pan> what are my returns?","pii_detected":true,"refusal":"pii","top_score":null,"sources":[],"chunks_used":0,"cached":false,"llm_ms":0,"total_ms":41}
```

Non-PII queries are logged with their normalised text, top score, sources used, and
latency — that is the data needed to tune the similarity floor later.

### 5.4 `TraceEvent` (the UI "Show pipeline" payload)

```json
{
  "stage_5_rewrite":   {"raw":"...","normalized":"...","expanded":"...","scheme_key":"hdfc-elss","intent":"lock_in","expansions":["elss -> ELSS tax saver"]},
  "guardrails_a":      [{"rule":"pii.*","hit":false},{"rule":"advice.*","hit":false},{"rule":"performance.*","hit":false},{"rule":"scope.*","hit":false}],
  "stage_6_8_retrieval":{"k":15,"top_k":5,"floor":0.35,"used_filter":true,"chunks":[{"rank":1,"score":0.71,"section":"Lock-in Period","label":"Lock-in","source_url":"...","preview":"..."}]},
  "guardrails_b":      [{"rule":"confidence.low","hit":false},{"rule":"intent.grounded","hit":true}],
  "stage_10_generation":{"model":"llama-3.3-70b-versatile","temperature":0.1,"prompt":"...","raw_response":"...","source_index":1,"truncated":false},
  "stage_11_citation": {"candidate_url":"...","verified":true,"link_substituted":false,"last_updated":"27 Sep 2026","invariants_ok":true}
}
```

This single object is what makes the "we built RAG, not a chatbot" argument in a 3-minute
demo.

---

## 6. Configuration Reference (`src/config.py`)

| Constant | Default | Purpose |
| --- | --- | --- |
| `SOURCES` | 5 `SourceSpec`s | PRD §5; the fetch allow-list |
| `ALLOWED_URLS` | derived from `SOURCES` | Citation verification (C1) |
| `DISCLAIMER` | `"Facts-only. No investment advice."` | Always rendered (C7) |
| `LAST_UPDATED_PREFIX` | `"Last updated from sources: "` | Exact wording (C4) |
| `DOC_EMBED_PREFIX` | `"Represent this mutual fund scheme document for retrieval: "` | Asymmetric retrieval (Stage 3) |
| `CHUNK_SIZE_TOKENS` | `400` | Stage 2 |
| `CHUNK_OVERLAP_TOKENS` | `50` | Stage 2 |
| `RETRIEVE_K` | `15` | Candidates before filtering (D4) |
| `TOP_K` | `5` | Chunks into the context |
| `SIMILARITY_FLOOR` | `0.35` | Anti-hallucination gate (D1) |
| `MAX_ANSWER_SENTENCES` | `3` | Hard cap (C4) |
| `GENERATOR_MODEL` | `"llama-3.3-70b-versatile"` | Stage 10 |
| `GENERATOR_TEMPERATURE` | `0.1` | Stage 10 |
| `SCHEME_ALIASES` | ~4 aliases per scheme | Stage 5 resolution |
| `OTHER_AMC_ALIASES` | SBI, Axis, ICICI Princap, Kotak, … | Scope guard |
| `FACT_TYPES` | 10 intents → expected section labels | Stage 5 + guardrail B |
| `REFUSALS` | 6 templates | Consistent wording |
| `CACHE_TTL_SECONDS` | `86400` | FR15 |
| `EMBED_DEVICE` | `"cpu"` | Override for GPU dev boxes |
| `HTTP_TIMEOUT` / `HTTP_RETRIES` | `20` / `3` | Polite scraping |

Environment: `GROQ_API_KEY` (required, from `.env`), `HF_HOME` (optional),
`EMBED_DEVICE` (optional).

---

## 7. Failure & Degradation Matrix

| Failure | Detection | Response | User sees |
| --- | --- | --- | --- |
| Groww unreachable at index time | non-200 / exception | Use committed snapshot, `stale=True` | Normal answers + "corpus snapshot is stale" sidebar warning |
| Index not built | collection missing | Sidebar prompts "Build index" | Welcome screen, no chat |
| Embedding fingerprint mismatch | `index_meta.json` vs runtime | Refuse to search, prompt rebuild | "Index is out of date — rebuild" |
| Retrieval returns nothing above floor | `confidence < floor` | `low_confidence` refusal | "I couldn't find that in my sources" + link |
| Model says `insufficient_context` | JSON flag | Convert to refusal | "I couldn't find that…" + link |
| Groq 429 / 5xx / timeout | client exception, 2 retries | Degrade to **extractive** answer: top chunk's relevant lines + verified link | Facts with citation, no paraphrase |
| Groq unreachable for the whole demo | repeated failures | `EXTRACTIVE_ONLY=1` mode — retrieval + first matching labelled fact, no LLM | Still cited, still safe (D1) |
| LLM emits a non-allow-listed URL | `verify_url` fails | Discard, substitute top-chunk URL, log | Correct link, trace shows substitution |
| LLM output unparseable | `json.loads` fails | Regex fallback; on failure, extractive mode | Cited extractive answer |
| PII detected | regex phase A | Refuse; **skip logging the text** | Polite PII refusal |

The guiding rule: **every degraded path still returns a verified link and the disclaimer.**
Degradation must never remove the citation guarantee (C4, C5).

---

## 8. Latency Budget (NFR1, NFR2)

| Path | Stage | Budget (ms) | Note |
| --- | --- | --- | --- |
| Index build (once) | 1 Load | 3 000–6 000 | 5 pages, serial with 2 s politeness delay |
| | 2 Chunk | < 200 | Pure Python over ~7k words |
| | 3 Embed | 1 000–2 000 | 5 docs, CPU, first run includes model load |
| | 4 Store | < 300 | ~150 chunks |
| | **Index total** | **~5–9 s** | Once per process/session |
| Query | 5 Rewrite | < 5 | Regex only |
| | 9 Guards A | < 2 | Regex only |
| | 3 Embed query | 10–40 | MiniLM, single sentence |
| | 6–8 Retrieve | 20–80 | 150 vectors, HNSW, trivial |
| | 9 Guards B | < 2 | Score comparisons |
| | 10 Generate | 900–2 500 | Groq 70B; the dominant cost |
| | 11 Cite | < 5 | String ops |
| **Query total (p50)** | | **~1.0–2.6 s** | Meets NFR1 (<10 s cold) and NFR2 (<1.5 s retrieval) |
| **Query total (cached)** | | **< 50 ms** | FR15 |

Retrieval-only p95 must stay under 1.5 s (NFR2) — that is the number to instrument, since
it is the part we own; LLM latency belongs to Groq.

---

## 9. Security & Privacy

| Concern | Control |
| --- | --- |
| API key leakage | `GROQ_API_KEY` in `.env`; `.env` gitignored; `.env.example` with an empty value; key never logged, never echoed in the trace |
| PII ingestion | Phase A regex runs on `plan.raw` before anything else; PII queries are not written to `query_log.jsonl`; no chat persistence anywhere (PRD §3.2) |
| PII in the corpus | Corpus is 5 public pages, no user data; `verify_url` restricts citations to public allow-listed URLs |
| Prompt injection | Corpus is static, committed, and human-reviewed at ingest; retrieved text is placed inside `<context>` and the system prompt forbids following instructions found in context; the generator has no tool access, so injected text cannot cause side effects |
| Data exfiltration | The only outbound call from the query path is Groq, and only after PII screening |
| Dependency risk | Pin exact versions in `requirements.txt`; corpus and index are build artefacts, not executable code |
| Third-party blogs | Structurally impossible: `ingest.py` only reads `config.SOURCES`; `citation.py` only renders allow-listed URLs |

---

## 10. Testing Strategy

Three layers, cheapest first, all runnable without a Groq key except the third.

**Layer 1 — unit (no network, no index).** Fast, and they encode the product rules as
code:

- `test_guardrails.py` — each of the ~14 rules fires on positive and negative samples; PII samples are drawn from the eval set (PAN, Aadhaar, 10-digit, email, phone, OTP); advice and performance phrasings including polite forms ("what would you do?", "is now a good time?").
- `test_citation.py` — `verify_url` accepts the 5 URLs and rejects look-alikes (`groww.in.evil.com/…`, a plausible AMC URL, a shortened link); sentence truncation; last-updated line present and sourced from metadata; the "exactly one URL" invariant on 50 generated samples.
- `test_query.py` — scheme alias resolution (longest match wins), shorthand expansion, intent classification, out-of-scope AMC detection.
- `test_chunking.py` — for each of the 6 canonical fact types, the value and its label land in the same chunk; no chunk exceeds `CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS`; no table row is split.

**Layer 2 — integration (local index, no LLM).** `ingest → chunk → embed → store → retrieve`:
every eval query returns ≥1 chunk above the floor for in-scope intents, and the top chunk's `scheme_key` matches the queried scheme.

**Layer 3 — end-to-end eval harness** (`deliverables/run_eval.py`, produces the sample Q&A file). Runs the 10-query eval set from PRD §12 through the full pipeline with a mocked or live Groq client and **asserts**:

| Assertion | Check |
| --- | --- |
| `A1` cited | exactly one URL present, and in the allow-list |
| `A2` short | ≤3 sentences |
| `A3` fresh | `Last updated from sources:` present |
| `A4` safe | no return/performance number in an answer, no recommendation verbs |
| `A5` refused | queries 9 and 10 produce a refusal with an educational link |
| `A6` grounded | every number in the answer appears verbatim in the retrieved context (string containment on digits) — the mechanical anti-hallucination test for D1 |
| `A7` no PII stored | `query_log.jsonl` contains no PII sample string |

`A6` is the most valuable test in the project: it turns "don't hallucinate" from a
prompt instruction into a pass/fail assertion.

**Fixtures:** a committed `data/eval/` copy of the cleaned corpus so Layer 2 and 3 never
hit the network.

---

## 11. Repository Layout

```text
NextLeap/
├─ PRD.md                        # product requirements (parent doc)
├─ ARCHITECTURE.md               # this document
├─ README.md                     # setup, scope, known limits
├─ app.py                        # Streamlit UI (presentation only)
├─ requirements.txt              # pinned
├─ .env.example                  # GROQ_API_KEY=
├─ .gitignore                    # .env, data/chroma/, logs/, __pycache__/
├─ Makefile                      # or run_index.py CLI for Windows-friendly setup
├─ src/
│  ├─ config.py                  # §6 configuration reference
│  ├─ ingest.py                  # stage 1  -> data/corpus/*.md
│  ├─ chunking.py                # stage 2  -> data/chunks/chunks.jsonl
│  ├─ embedding.py               # stage 3  MiniLM
│  ├─ vectorstore.py             # stage 4  ChromaDB
│  ├─ query.py                   # stage 5  rewrite + scope
│  ├─ retriever.py               # stages 6-8
│  ├─ guardrails.py              # stage 9  phases A + B
│  ├─ generator.py               # stage 10 Groq
│  ├─ citation.py                # stage 11 verify + stamp
│  ├─ pipeline.py                # orchestration + trace
│  ├─ logger.py                  # redacted JSONL
│  └─ types.py                   # QueryPlan, RetrievedChunk, RefusalResult, ...
├─ data/
│  ├─ corpus/*.md                # committed snapshot of the 5 pages
│  ├─ chunks/chunks.jsonl        # committed
│  ├─ index_meta.json            # fingerprint + build stats
│  ├─ sources.csv                # deliverable: 7 rows
│  └─ chroma/                    # gitignored, rebuilt
├─ deliverables/
│  ├─ sample_qa.md               # 10 queries + answers + links
│  ├─ disclaimer.md              # exact UI text
│  └─ run_eval.py                # eval harness (A1..A7)
├─ logs/query_log.jsonl          # gitignored
└─ tests/                        # layers 1 & 2
```

---

## 12. Key Technical Decisions (ADR summary)

| # | Decision | Alternatives rejected | Why |
| --- | --- | --- | --- |
| AD1 | `all-MiniLM-L6-v2` | BGE-small, `text-embedding-3-small` | Mandated by the brief; CPU-fast; no API cost |
| AD2 | ChromaDB, persistent | FAISS, LanceDB, pgvector | Mandated; zero-ops embedded store; survives Streamlit reruns |
| AD3 | Groq `llama-3.3-70b` | OpenAI, Ollama, Gemini | Free tier + lowest latency → demo reliability (R4) |
| AD4 | Structure-aware chunking | Fixed 1000-token, fixed 200-token, whole-document | Whole-doc retrieval can't discriminate; fixed splits orphan facts (R2) |
| AD5 | No re-ranker model | Cross-encoder `bge-reranker` | 5 documents: score floor + metadata filter is at ceiling; saves 200–400 ms (D4, NFR2) |
| AD6 | Guardrails **before** the LLM | Prompt-only refusal | Deterministic, testable, and free; a prompt alone will eventually leak advice (R5) |
| AD7 | Regex query rewrite | LLM query expansion | Latency, cost, non-determinism; the vocabulary here is tiny |
| AD8 | JSON-mode generation | Free-text + regex parse | Removes the whole class of citation-parsing bugs |
| AD9 | Link taken from **chunk metadata**, LLM only picks the index | Let the LLM write the URL | Metadata cannot hallucinate; a hallucinated URL is discarded (C1) |
| AD10 | Streamlit, single process | FastAPI + SPA, Chainlit | Fewest moving parts for a ≤3-min demo; `cache_resource` solves state |
| AD11 | Committed corpus snapshot | Live scrape at query time | Reproducible, offline-safe, and immune to Groww redesigns (R1, NFR4/5) |
| AD12 | Extractive fallback when the LLM is down | Show an error | The citation guarantee must never lapse (§7) |

---

## 13. Build Order

Each step is independently runnable and demoable, so the demo exists from day 1 and
improves rather than appearing at the end.

| Step | Deliverable | Verifies |
| --- | --- | --- |
| 1 | `config.py`, `ingest.py`, committed `data/corpus/*.md`, `data/sources.csv` | FR1, deliverable "source list" |
| 2 | `chunking.py` + `test_chunking.py` + `chunks.jsonl` | FR2, R2 |
| 3 | `embedding.py` + `vectorstore.py` + `index_meta.json` | FR3, C9 |
| 4 | `retriever.py` + a CLI that prints top-5 chunks for an eval query | §4.6 tuning table filled in |
| 5 | `guardrails.py` + `test_guardrails.py` | FR8, C2, C3, C6 |
| 6 | `generator.py` + `citation.py` + `test_citation.py` | FR4–FR7, C4, C5 |
| 7 | `pipeline.py` + `logger.py` | FR13 |
| 8 | `app.py` with chat, examples, disclaimer, sidebar, trace expander | FR10–FR12, C7 |
| 9 | `deliverables/run_eval.py` + `sample_qa.md` | Metrics, assertions A1–A7 |
| 10 | README, demo script, ≤3-min video | All deliverables |

Steps 1–6 need no Streamlit and no UI polish, which is where most of the demo risk lives.
The UI is the last mile, not the load-bearing part.

---

## 14. Requirement Traceability

| PRD ref | Where it is implemented |
| --- | --- |
| FR1 | §4.1 `ingest.py` |
| FR2 | §4.2 `chunking.py` |
| FR3 | §4.3, §4.4 `embedding.py`, `vectorstore.py` |
| FR4 | §4.8 `generator.py` |
| FR5, C5 | §4.9 `citation.py` (allow-list verify) |
| FR6, C4 | §4.8 step 3 (truncation), §4.9 |
| FR7, C4 | §4.9 step 3 (`Last updated from sources:`) |
| FR8, C6, C2, C3 | §4.7 both guardrail phases + templates |
| FR9 | §4.7 `confidence.low` |
| FR10, C7 | §2 presentation layer, `app.py` |
| FR11 | §5.4 `TraceEvent` |
| FR12 | §4.4 lifespan, `app.py` sidebar |
| FR13, NFR7 | §5.3 `logger.py` |
| FR15 | §4.10 cache, §8 |
| FR16 | §4.3 fingerprint |
| NFR1, NFR2 | §8 latency budget |
| NFR3 | §9 |
| NFR4, NFR5 | §4.1 fallback + committed snapshot |
| C1 | §4.1 allow-list, §4.9 verification |
| C8 | §3 component map — one module per stage |
| C9 | §4.3, §4.4 |
| R1–R8 | §4.1, §4.2, §4.6, §7, §9, §12 |
