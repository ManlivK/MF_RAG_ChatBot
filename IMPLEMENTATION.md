# Implementation Guide — MF Facts RAG FAQ Assistant

| Field | Value |
| --- | --- |
| Document type | Phase-wise Implementation Guide (build playbook) |
| Version | 1.0 |
| Status | Ready to build |
| Inputs | [PRD.md](./PRD.md) v1.0 (what), [ARCHITECTURE.md](./ARCHITECTURE.md) v1.0 (how) |
| Owner | NextLeap RAG Chatbot Team |
| Last updated | 2026-09-27 |

---

## 0. How To Use This Document

This is a **build script for Cursor**. Work top to bottom, one phase at a time. Each
phase is independently runnable, has its own tests, and ends with a commit. Do not start
phase *N+1* until phase *N*'s **Definition of Done** is green.

### 0.1 Ground rules for the AI pair

1. **PRD = what, ARCHITECTURE = how, this file = in what order.** When instructions
   conflict, this order wins. Never contradict ARCHITECTURE §12 (ADRs) without writing
   an ADR.
2. **One phase per request.** Do not build ahead. Do not create files belonging to a
   later phase.
3. **No new dependencies** beyond the install list in Phase 0 without asking.
4. **Stage modules stay pure.** `guardrails.py` and `citation.py` must never import
   `generator.py` or `retriever.py` (ARCH §3). `app.py` contains zero retrieval or
   generation logic.
5. **Every function gets a type hint and a docstring.** No `Any` without a comment.
6. **Config is centralised.** No magic numbers, no URLs, no regexes in stage modules —
   they all live in `src/config.py` (ARCH §6).
7. **No secrets in code.** `GROQ_API_KEY` is read from `.env` and never printed.
8. **No PII in logs, ever** (ARCH §5.3).
9. **Ask before deviating.** If the corpus doesn't look like ARCHITECTURE §4.2 assumes
   (e.g. facts are not label-dense), stop and report — do not silently pick a different
   chunking strategy.

### 0.2 Phase map

| Phase | Name | Produces | Depends on |
| --- | --- | --- | --- |
| 0 | Scaffold & Config | repo skeleton, venv, deps, `config.py`, `types.py` | — |
| 1 | Ingestion (Stage 1) | `data/corpus/*.md`, `data/sources.csv` | 0 |
| 2 | Chunking (Stage 2) | `src/chunking.py`, `data/chunks/chunks.jsonl` | 1 |
| 3 | Embedding + Store (Stages 3–4) | `src/embedding.py`, `src/vectorstore.py`, `data/index_meta.json` | 2 |
| 4 | Query + Retrieval (Stages 5–8) | `src/query.py`, `src/retriever.py` | 3 |
| 5 | Guardrails (Stage 9) | `src/guardrails.py` | 4 |
| 6 | Generation + Citation (Stages 10–11) | `src/generator.py`, `src/citation.py` | 5 |
| 7 | Pipeline + Logging | `src/pipeline.py`, `src/logger.py` | 6 |
| 8 | UI | `app.py` | 7 |
| 9 | Eval harness + deliverables | `deliverables/*`, `deliverables/sample_qa.md` | 8 |
| 10 | README + demo script | `README.md`, demo video | 9 |

### 0.3 Environment (Windows / PowerShell — this repo is built on Windows)

```powershell
cd "C:\Users\MANLIV KAUR\NextLeap"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

> If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> (one-time, current user only).

macOS/Linux equivalent: `python3 -m venv .venv && source .venv/bin/activate`

---

## PHASE 0 — Scaffold & Configuration

**Goal:** a runnable skeleton with zero product logic, so every later phase has a place
to land and a place to fail.

**References:** ARCHITECTURE §3, §6, §11.

### 0.1 Create

```text
NextLeap/
├─ .env.example
├─ .gitignore
├─ requirements.txt
├─ requirements.lock.txt        # generated at the end of this phase
├─ run_index.py                # CLI wrapper: python run_index.py
├─ app.py                      # placeholder "hello" for now
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  └─ types.py
├─ data/{corpus,chunks,eval}/.gitkeep
├─ deliverables/.gitkeep
├─ logs/.gitkeep
└─ tests/
   ├─ __init__.py
   ├─ conftest.py
   ├─ test_config.py
   ├─ test_guardrails.py       # stub: collect-only, filled in Phase 5
   ├─ test_citation.py         # stub, filled in Phase 6
   ├─ test_query.py            # stub, filled in Phase 4
   └─ test_chunking.py         # stub, filled in Phase 2
```

### 0.2 Dependencies

```powershell
python -m pip install streamlit requests beautifulsoup4 markdownify lxml `
  chromadb langchain-core langchain-text-splitters langchain-huggingface `
  langchain-chroma langchain-groq sentence-transformers tiktoken python-dotenv pytest
```

Then freeze and commit:

```powershell
python -m pip freeze > requirements.lock.txt
```

> Use `requirements.txt` for the human-readable list above, and
> `requirements.lock.txt` (exact pins) for reproducible installs. Do not hand-edit pins.

### 0.3 `.env.example`

```dotenv
# Copy to .env and fill in. .env is gitignored — never commit a real key.
GROQ_API_KEY=
# Optional overrides
# EMBED_DEVICE=cpu
# HF_HOME=
```

### 0.4 `.gitignore`

```gitignore
.env
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
data/chroma/
logs/*.jsonl
.streamlit/secrets.toml
```

> Note: `data/chroma/` is gitignored (rebuildable) but `data/corpus/` and
> `data/chunks/` are **committed** (PRD NFR5 — the demo must be reproducible offline).

### 0.5 `src/types.py` — the contracts every phase codes against

Dataclasses exactly as in ARCHITECTURE §4.5, §4.6, §4.7, §4.10. Define them **now** so
later phases import rather than invent.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

Intent = Literal[
    "expense_ratio", "exit_load", "min_sip", "min_lump_sum", "lock_in",
    "riskometer", "benchmark", "nav", "statement_download", "tax", "unknown",
]
RefusalKind = Literal[
    "advice", "performance", "pii", "out_of_scope", "low_confidence", "nav", "unknown_intent",
]

@dataclass(frozen=True)
class SourceSpec:
    scheme_key: str
    scheme_name: str
    category: str
    url: str

@dataclass(frozen=True)
class QueryPlan:
    raw: str
    normalized: str
    expanded: str
    scheme_key: str | None
    intent: Intent
    expansions: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: dict[str, Any]
    score: float
    rank: int

@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    dropped_low_score: int
    dropped_scheme_mismatch: int
    confidence: float
    used_filter: bool

@dataclass(frozen=True)
class RefusalResult:
    kind: RefusalKind
    message: str
    link: str
    rule: str

@dataclass(frozen=True)
class LLMResponse:
    answer: str
    source_index: int
    insufficient_context: bool
    raw: str
    truncated: bool = False

@dataclass(frozen=True)
class AssistantMessage:
    kind: Literal["answer", "refusal"]
    text: str                      # final rendered body, ≤3 sentences, no URL
    source_url: str                # exactly one allow-listed URL ("" for refusal with no link)
    last_updated: str              # "27 Sep 2026" or "" 
    disclaimer: str
    truncated: bool = False
    link_substituted: bool = False
    cached: bool = False
```

Add `TraceEvent` as a plain `dict[str, Any]` (documented in ARCH §5.4) plus
`PipelineResult`:

```python
@dataclass(frozen=True)
class PipelineResult:
    message: AssistantMessage
    trace: dict[str, Any]
```

### 0.6 `src/config.py` — every constant from ARCHITECTURE §6

Required contents:

- `SOURCES: list[SourceSpec]` — the 5 schemes, keys `hdfc-large-cap`,
  `hdfc-equity-flexi-cap`, `hdfc-elss-tax-saver`, `hdfc-small-cap`,
  `hdfc-balanced-advantage`, with the exact URLs from PRD §5.
- `ALLOWED_URL_BASE = "https://groww.in/mutual-funds/"` and
  `ALLOWED_URLS: set[str]` derived from `SOURCES`.
- `DISCLAIMER = "Facts-only. No investment advice."`
- `LAST_UPDATED_PREFIX = "Last updated from sources: "`
- `DOC_EMBED_PREFIX = "Represent this mutual fund scheme document for retrieval: "`
- `CHUNK_SIZE_TOKENS = 400`, `CHUNK_OVERLAP_TOKENS = 50`, `CHUNK_STRATEGY = "structure-aware-v1"`
- `RETRIEVE_K = 15`, `TOP_K = 5`, `SIMILARITY_FLOOR = 0.35`
- `MAX_ANSWER_SENTENCES = 3`
- `GENERATOR_MODEL = "llama-3.3-70b-versatile"`, `GENERATOR_TEMPERATURE = 0.1`, `GENERATOR_MAX_TOKENS = 400`
- `SCHEME_ALIASES: dict[str, list[str]]` — ≥4 aliases per scheme, longest-match resolution.
- `OTHER_AMC_ALIASES: list[str]` — at least SBI, Axis, ICICI Princap, Kotak, Kotak Mahindra, Aditya Birla, Sundaram, Nippon, Motilal, Tata.
- `FACT_TYPES: dict[str, dict]` — intent → `{"keywords": [...], "expected_labels": [...]}`.
  *Critical mapping for guardrail B:*
  | intent | expected_labels |
  | --- | --- |
  | expense_ratio | Expense Ratio, TER, Charges, exit load, Direct Plan |
  | exit_load | Exit Load, Redemption, Load |
  | min_sip | Minimum SIP, SIP, How to invest |
  | min_lump_sum | Minimum Lump Sum, Lump Sum, How to invest |
  | lock_in | Lock-in Period, ELSS, Tax Saver, Tax benefit |
  | riskometer | Riskometer, Risk, Risk Level |
  | benchmark | Benchmark, Index |
  | nav | NAV, AUM |
  | statement_download | statement, Statement, Capital Gains, Tax, Download, Portfolio |
  | tax | Tax, Tax benefit, ELSS, 80C |
- `PII_PATTERNS: dict[str, str]` — the six regexes from ARCH §4.7.
- `ADVICE_PATTERNS`, `PERFORMANCE_PATTERNS: dict[str, str]` — the rule → regex maps.
- `REFUSAL_TEMPLATES: dict[str, str]` — the six exact templates from ARCH §4.7, each ≤3 sentences, each with a `{link}` slot.
- `PROMPT_TEMPLATE` — the system prompt from ARCH §4.8, with `{blocks}` and `{query}` placeholders. Keep it a module constant so the UI can print it verbatim.
- `CACHE_TTL_SECONDS = 86400`, `EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")`
- `HTTP_TIMEOUT = 20`, `HTTP_RETRIES = 3`, `HTTP_DELAY_SECONDS = 2.0`
- `PATHS` — `DATA_DIR`, `CORPUS_DIR`, `CHUNKS_DIR`, `CHROMA_DIR`, `LOG_PATH`, `SOURCES_CSV`, `INDEX_META`
- `load_api_key()` — reads `.env` via `python-dotenv`, raises a clear error if `GROQ_API_KEY` is missing. Never logs the value.
- Helper `chunk_count_tokens(text) -> int` using `tiktoken.get_encoding("cl100k_base")`.
- Helper `truncate_to_sentences(text, n) -> str`.

### 0.7 Tests

`tests/conftest.py`: fixtures `config`, `corpus_dir`, `chunks_path`, `sample_chunk`.
`tests/test_config.py`:

- All 5 URLs are unique, non-empty, and start with `ALLOWED_URL_BASE`.
- Every `FACT_TYPES` intent has non-empty `keywords` and `expected_labels`.
- Every `RefusalKind` has a template in `REFUSAL_TEMPLATES`.
- `DISCLAIMER` and `LAST_UPDATED_PREFIX` match PRD wording **exactly**.
- `truncate_to_sentences("A. B. C. D.", 3)` returns `"A. B. C."`; `"A. B."` unchanged.

### 0.8 Definition of Done

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q
```

- [ ] All tests pass
- [ ] `.\.venv\Scripts\python.exe -c "import src.config as c; print(len(c.SOURCES))"` prints `5`
- [ ] `requirements.lock.txt` committed
- [ ] `.env` created locally and gitignored; `git status` shows no `.env`
- [ ] `app.py` runs as a placeholder: `streamlit run app.py`

**Commit:** `chore: scaffold project, config and shared types (phase 0)`

---

## PHASE 1 — Ingestion (RAG Stage 1: Loading)

**Goal:** 5 public scheme pages + 2 hand-authored AMC documents → 7 clean Markdown files +
`sources.csv`. Network-free after the first run.

**References:** ARCHITECTURE §4.1, PRD FR1, §5.

**Files:** `src/ingest.py`, `run_index.py` (add `--stage ingest`).

### 1.1 Function contract

```python
def fetch_page(spec: SourceSpec) -> tuple[str | None, int | None, str | None]:
    """GET spec.url. Returns (html, status_code, error). Never raises."""

def clean_html_to_markdown(html: str) -> str:
    """BeautifulSoup + markdownify per ARCH §4.1 step 2. Raises ValueError
    if extracted text < MIN_CHARS (500) — signals a layout change."""

def build_document(spec: SourceSpec, markdown_body: str, fetched_at: str) -> str:
    """Prepend the H1 title + metadata header block, collapse blank lines."""

def ingest_scheme(spec: SourceSpec) -> dict[str, Any]:
    """fetch -> clean -> write data/corpus/<scheme_key>.md -> return a
    sources.csv row. On any failure, keep the existing snapshot and return a
    row with stale=True. Never raises."""

def write_sources_csv(rows: list[dict[str, Any]]) -> None:
    """Deterministic column order per ARCH §5.2."""

def run_ingest() -> list[dict[str, Any]]:
    """Serial loop over config.SOURCES with >= HTTP_DELAY_SECONDS between requests."""
```

### 1.2 Implementation order inside the file

1. Selectors to strip (put in a module constant, not inline): `nav, footer, aside, script, style, noscript, iframe, form`, plus a regex list for cookie/consent text, "related funds"/"compare" widgets, and app-store badges. **Do not strip `header`** — it holds the scheme name, the ELSS `3Y Lock-in` tag, the riskometer and NAV.
2. Root element preference: `main` → `[role=main]` → `article` → `div#__next` → `div#root` → `body`. Record which was used (useful for debugging later). Then narrow: descend into a child while exactly one child holds ≥2 of `SCRAP_CONTENT_MARKERS`.
3. Insert a newline between adjacent short inline `<span>`/`<a>` siblings (not inside `p`, `a`, table cells, list items or headings) so markdownify cannot fuse a label onto its value, and keep unit fragments (`%`, `₹`, `Cr`) welded to the item before them.
4. Truncate at the first footer marker line: the mega-menu is plain `<div>`s, not a `<footer>`, and left in place it runs to two thirds of each file and names other schemes.
5. Strip returns and rankings (PRD C3): return-calculator / historic-returns / returns-and-rankings / compare sections, standalone signed return figures, return-period labels, and `Rank`-labelled value pairs. Keep benchmark names containing "Total Return"; keep unsigned percentages.
6. `markdownify(html, heading_style="ATX", strip=["script", "style"])`.
7. Cleanup: NBSP → space, zero-width chars removed, >2 blank lines collapsed, trailing whitespace stripped.
8. Header block format (exact — chunking depends on it):
   ```markdown
   # {scheme_name}
   > category: {category} | scheme_key: {scheme_key} | source_url: {url} | fetched_at: {YYYY-MM-DD}

   {body}
   ```
   An authored document adds `| authored: true | live_fetch_status: 403 | publisher: …` to the
   same line, and a cross-scheme document uses `scheme_key: all-schemes`.
9. `fetched_at` = `date.today().isoformat()` at fetch time; if the snapshot is reused, keep the date inside the existing file (so the "last updated" stamp stays truthful).

### 1.3 Inspect before continuing (required)

After the first successful run, print for each file: char count, heading list, label-value
and table-row counts, and the first 40 lines. **Compare against ARCH §4.2's assumption that
the corpus is label-dense.** If facts are not in `key: value` / stacked-label / table / bullet
form, stop and raise it before writing Phase 2 — the chunking strategy depends on it (ground
rule 9). Count stacked pairs separately from inline pairs: the build found the stacked form
to be the dominant one (67–98 units per scheme vs 4–7 inline), which is an amendment to
ARCH §4.2 and a direct instruction for the Stage 2 label extractor.

### 1.4 Definition of Done

```powershell
.\.venv\Scripts\python.exe run_index.py --stage ingest
```

- [ ] `data/corpus/` has 7 `.md` files (5 scraped + 2 authored), each > 500 chars of real content
- [ ] `data/sources.csv` has 7 data rows, correct column order, and a 403 + "hand-authored" publisher on the two transcribed rows
- [ ] No nav/footer/cookie text in any file (`Select-String -Path data\corpus\*.md -Pattern "cookie|Log in|Sign up|All Rights Reserved|SIP Calculator|Top Gainers"` returns nothing)
- [ ] No return figure or ranking survived; benchmark names containing "Total Return" did
- [ ] Re-running is idempotent (git diff shows no changes on a second run, except possibly a changed date)
- [ ] Killing the network and re-running still succeeds from snapshots, with `stale=true` in the CSV

**Commit:** `feat: scrape and clean 5 scheme pages to markdown snapshots (phase 1)`

---

## PHASE 2 — Chunking (RAG Stage 2)

**Goal:** labelled, self-contained chunks that never orphan a value from its label.

**References:** ARCHITECTURE §4.2, PRD FR2, risk R2.

**Files:** `src/chunking.py`, fill `tests/test_chunking.py`.

### 2.1 Function contract

```python
def token_count(text: str) -> int
def extract_label(unit: str, section: str) -> str
    """Nearest bolded term, table header, or 'key: value' key; else the section name."""
def split_into_units(section_body: str) -> list[str]
    """One table row (with its header row attached) = 1 unit; one bullet list = 1 unit;
    one paragraph = 1 unit. Never split inside a table row or a list item."""
def pack_units(units: list[tuple[str, str]], budget: int) -> list[dict]
    """Greedy pack (text, label) pairs up to CHUNK_SIZE_TOKENS. Unlabelled residue
    merges forward. Unit-level overlap of CHUNK_OVERLAP_TOKENS by carrying the last
    unit of the previous chunk."""
def split_oversized_unit(unit: str, label: str) -> list[dict]
    """RecursiveCharacterTextSplitter fallback. Table/list units split between ROWS;
    paragraphs split on the separator list."""
def chunk_document(scheme_key, markdown, meta) -> list[dict]
def run_chunking() -> list[dict]
    """All 5 docs -> data/chunks/chunks.jsonl. Print stats: docs, chunks, mean/median
    token count, chunks with no label (must be 0)."""
```

### 2.2 Chunk record schema (exact — ChromaDB and the citation stage depend on it)

```json
{"chunk_id":"hdfc-large-cap::0007","text":"...","scheme_key":"hdfc-large-cap",
 "scheme_name":"HDFC Large Cap Fund — Direct Growth","category":"Large Cap",
 "section":"Exit Load","label":"Exit Load","source_url":"https://...",
 "source_title":"HDFC Large Cap Fund — Direct Growth","fetched_at":"2026-09-27",
 "chunk_index":7,"content_hash":"sha256:9f2c…"}
```

`chunk_id` format: `{scheme_key}::{chunk_index:04d}` (must match the Chroma document id).

### 2.3 Tests (this phase's tests are the R2 mitigation — write them properly)

`tests/test_chunking.py` uses the committed corpus (no network):

1. **No orphaned facts (the key test).** For each of the 6 canonical fact types, find the chunk whose text contains the known value and assert the same chunk also contains its label. Parametrise with the real values read from the corpus.
2. No chunk exceeds `CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS`.
3. Every chunk has a non-empty `label` and a non-empty `section`.
4. No table row is split: for each `<scheme>::` group, no chunk starts mid-row.
5. Overlap is real: assert ≥1 shared token between consecutive chunks of the same document (except at document boundaries).
6. `content_hash` is stable across two runs.
7. Every chunk's `source_url` is in `config.ALLOWED_URLS`.

### 2.4 Definition of Done

```powershell
.\.venv\Scripts\python.exe run_index.py --stage chunk
.\.venv\Scripts\python.exe -m pytest tests/test_chunking.py -q
```

- [ ] Tests 1–7 pass
- [ ] Stats print: 5 docs, mean chunk ~200–400 tokens, **0 unlabelled chunks**
- [ ] `data/chunks/chunks.jsonl` committed (one JSON object per line, no trailing comma, UTF-8)
- [ ] Manual read of 3 random chunks: each reads as a complete, self-contained fact

**Commit:** `feat: structure-aware chunking with label preservation (phase 2)`

---

## PHASE 3 — Embedding + Vector Store (RAG Stages 3–4)

**Goal:** MiniLM vectors in a persistent ChromaDB collection with a fingerprint that
prevents model/index mismatch.

**References:** ARCHITECTURE §4.3, §4.4, PRD FR3, C9.

**Files:** `src/embedding.py`, `src/vectorstore.py`, `run_index.py --stage index`.

### 3.1 `src/embedding.py`

```python
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """model_name=config.EMBED_CONFIG_MODEL, encode_kwargs={'normalize_embeddings': True},
    model_kwargs={'device': config.EMBED_DEVICE}, embed_batch_size=32."""

def embedding_fingerprint() -> str:
    """sha256(model_name|doc_prefix|normalize|device_class)[:12] — device excluded from
    the fingerprint (CPU and GPU produce the same vectors) but recorded separately."""

def embed_documents(texts: list[str]) -> list[list[float]]:
    """Applies DOC_EMBED_PREFIX to each text before embedding. 384-dim."""

def embed_query(text: str) -> list[float]:
    """NO prefix — asymmetry with the document side is intentional (ARCH §4.3)."""
```

### 3.2 `src/vectorstore.py`

```python
COLLECTION_NAME = "mf_faq_hdfc"   # in config.py, not here

@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """PersistentClient(path=config.PATHS.CHROMA_DIR);
    get_or_create_collection(name, metadata={'hnsw:space': 'cosine'},
                             embedding_function=None)  # we insert precomputed vectors"""

def rebuild_index(chunks: list[dict]) -> int
    """delete_collection -> recreate -> upsert(ids=chunk_id, documents=text,
    embeddings=embed_documents(text), metadatas=metadata) -> returns count."""

def upsert_chunks(chunks: list[dict]) -> int   # idempotent path used by rebuild

def write_index_meta(stats: dict) -> None      # data/index_meta.json, schema ARCH §5.1
def read_index_meta() -> dict | None
def is_index_stale() -> tuple[bool, str]
    """True if meta is missing, or embedding_fingerprint() != meta['embedding_fingerprint']."""

def get_stats() -> dict  # doc_count, chunk_count, built_at, for the sidebar
```

**Critical:** `embedding_function=None`. If Chroma loads its own default embedder, you
silently get a second, different model in the same collection.

### 3.3 Definition of Done

```powershell
.\.venv\Scripts\python.exe run_index.py --stage index --rebuild
.\.venv\Scripts\python.exe -m pytest tests/test_vectorstore.py -q   # add this file in this phase
```

- [ ] `data/index_meta.json` written with the exact schema from ARCH §5.1
- [ ] Collection has the same `chunk_count` as `chunks.jsonl`
- [ ] `is_index_stale()` returns `False` on a second call
- [ ] Round-trip: a hand-typed query vector search returns chunks with plausible `source_url` metadata
- [ ] `data/chroma/` is gitignored; the index rebuilds from scratch after `git clean -xdf`
- [ ] Sanity check: cos-sim between `"exit load of HDFC large cap"` and the Exit Load chunk is the highest among the top 5

**Commit:** `feat: embed with all-MiniLM-L6-v2 and persist to ChromaDB (phase 3)`

---

## PHASE 4 — Query Rewrite + Retrieval (RAG Stages 5–8)

**Goal:** deterministic query understanding and high-precision retrieval with a
similarity floor.

**References:** ARCHITECTURE §4.5, §4.6, PRD FR9, NFR2.

**Files:** `src/query.py`, `src/retriever.py`, fill `tests/test_query.py`,
add `tests/test_retriever.py`.

### 4.1 `src/query.py`

```python
FILLER_PREFIXES = ("hi", "hello", "hey", "please tell me", "can you tell me", "i want to know")

def normalize(raw: str) -> str
def expand_shorthand(text: str) -> tuple[str, list[str]]
    """config.SHORTHAND phrase map. Returns (expanded_text, audit list)."""
def resolve_scheme(text: str) -> str | None
    """Longest alias match wins, so 'hdfc balanced advantage' is not shadowed by 'hdfc'."""
def mentions_other_amc(text: str) -> str | None
    """Returns the AMC name if an out-of-scope AMC is mentioned."""
def classify_intent(text: str) -> Intent
    """Keyword scoring over config.FACT_TYPES; ties -> 'unknown'. Deterministic, no LLM."""
def build_plan(raw: str) -> QueryPlan
```

### 4.2 `src/retriever.py`

```python
def retrieve(plan: QueryPlan) -> RetrievalResult
    """1. assert embedding_fingerprint matches index_meta (hard error, never silent)
       2. embed_query(plan.expanded)
       3. collection.query(n_results=RETRIEVE_K, where={'scheme_key': plan.scheme_key} if resolved)
       4. similarity = 1 - distance; sort desc; take TOP_K
       5. drop score < SIMILARITY_FLOOR
       6. confidence = chunks[0].score if chunks else 0.0
       7. build numbered context blocks"""

def format_context(result: RetrievalResult) -> str
    """Exact format from ARCH §4.6 step 6:
    [1] (source: <url> | scheme: <name> | section: <s> | score: 0.71)
    <chunk text>
    """
```

**Rules:** if the fingerprint check fails → raise `IndexStaleError`. If a scheme filter is
applied, set `used_filter=True` and count `dropped_scheme_mismatch`. Never drop below 1
context block when chunks exist — the generator needs at least one.

### 4.3 Tests

`tests/test_query.py`:

- Alias resolution: `"large cap fund"` → `hdfc-large-cap`; `"hdfc balanced advantage"` → `hdfc-balanced-advantage` (longest match, not `hdfc-large-cap`).
- Shorthand: `"ER of ELSS"` expands to include `expense ratio` and `ELSS tax saver`.
- Intent: each of the 10 intents classifies correctly from a paraphrased query; a nonsense query → `unknown`.
- Out-of-scope AMC: `"SBI Bluechip"` → `mentions_other_amc` is truthy.

`tests/test_retriever.py` (needs a built index; skip if the index is missing):

- Each of the 8 in-scope eval queries returns ≥1 chunk above the floor.
- Top chunk's `scheme_key` equals the queried scheme for all 5 schemes.
- A nonsense query ("what is the atmospheric pressure on Jupiter") returns `confidence == 0.0`.

### 4.4 Threshold tuning (do this now, record the numbers)

Run all 10 eval queries from PRD §12, record top-1 similarity, then set
`SIMILARITY_FLOOR` in `config.py` and copy the table into ARCH §4.6. If a legitimate
question falls below the floor, fix it by improving `expanded` (query rewrite), **not** by
lowering the floor globally.

```powershell
.\.venv\Scripts\python.exe run_index.py --stage eval-retrieval
```

### 4.5 Definition of Done

- [ ] `tests/test_query.py` and `tests/test_retriever.py` pass
- [ ] Retrieval-only latency measured and < 1.5 s (NFR2)
- [ ] `SIMILARITY_FLOOR` justified by the tuning table, and the table filled into ARCH §4.6
- [ ] Nonsense queries return empty retrieval rather than a weak match

**Commit:** `feat: query rewrite, scope resolution and filtered retrieval (phase 4)`

---

## PHASE 5 — Guardrails (RAG Stage 9, both phases)

**Goal:** deterministic, unit-testable refusal logic that runs **before** the LLM for
PII/advice/performance/scope.

**References:** ARCHITECTURE §4.7, §1.1, PRD FR8, C2, C3, C6, risk R5.

**Files:** `src/guardrails.py`, fill `tests/test_guardrails.py`.

### 5.1 Function contract

```python
# ---- Phase A (pre-retrieval) ----
def detect_pii(text: str) -> list[str]          # rule ids hit
def detect_advice(text: str) -> list[str]
def detect_performance(text: str) -> list[str]
def check_phase_a(raw: str, plan: QueryPlan | None = None) -> RefusalResult | None
    """ORDER IS FIXED: pii -> advice -> performance -> scope. Returns None to continue."""

# ---- Phase B (post-retrieval) ----
def check_phase_b(plan: QueryPlan, result: RetrievalResult) -> RefusalResult | None
    """confidence.low -> intent.nav -> intent.grounded"""

# ---- helpers ----
def educational_link(kind: RefusalKind, plan: QueryPlan | None) -> str
    """Always an allow-listed URL: the matched scheme's page, else the first scheme page.
    For 'performance' use the scheme page's factsheet link if the corpus contains one,
    else the scheme page."""
def render_refusal(kind, link) -> RefusalResult
    """Template + link, sentence count asserted <= 3."""
```

### 5.2 Non-negotiable behaviours

1. **PII is checked first and the raw text is never passed onward.** `pipeline.py` must
   not log a `pii` refusal's query text (ARCH §5.3).
2. **Every refusal carries exactly one allow-listed link.** A refusal with a bad link is
   a test failure, not a warning.
3. **Every refusal message is ≤3 sentences and contains no numbers.** (A number in a
   refusal would be a fact the corpus never gave us.)
4. **No LLM import in this file.** Enforce with a test that greps the module source for
   `generator`/`groq` imports.

### 5.3 Tests (the largest test file in the project)

Parametrised positive/negative cases per rule, all from `config`:

- PII: valid PAN `[A-Z]{5}\d{4}[A-Z]`, spaced Aadhaar `1234 5678 9012`, 10- and 12-digit numbers, `user@example.com`, `+91 98765 43210`, `"my OTP is 123456"`. Negatives: `"expense ratio is 0.52%"` (5 digits, no PAN shape), `"hdfc-elss"`, a 4-digit year.
- Advice: "should I buy", "which fund is best for me", "is now a good time to invest", "can you suggest a portfolio", "should I sell after 3 years", "allocate 60% to small cap".
- Performance: "1 year return", "YTD performance", "which performed best", "returns of ELSS vs Large Cap", "what is the yield".
- Scope: "SBI Bluechip expense ratio", "Axis Large Cap", "ICICI Princap".
- **Priority test:** a query containing *both* PII and advice returns `kind == "pii"`.
- **Bypass test:** "Is HDFC Small Cap good for me? My PAN is ABCDE1234F" → `pii` wins.
- Purity test: module source contains no LLM client import.

### 5.4 Definition of Done

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_guardrails.py -q
```

- [ ] ≥40 parametrised cases pass
- [ ] Priority and bypass tests pass
- [ ] Every refusal renders ≤3 sentences, no digits, one allow-listed link
- [ ] Module is LLM-free (test enforced)

**Commit:** `feat: two-phase deterministic guardrails with refusal templates (phase 5)`

---

## PHASE 6 — Generation + Citation (RAG Stages 10–11)

**Goal:** a grounded Groq call and an unbreakable one-link citation guarantee.

**References:** ARCHITECTURE §4.8, §4.9, PRD FR4–FR7, C1, C4, C5.

**Files:** `src/generator.py`, `src/citation.py`, fill `tests/test_citation.py`.

### 6.1 `src/generator.py`

```python
def build_prompt(plan: QueryPlan, context: str) -> str
    """config.PROMPT_TEMPLATE.format(blocks=context, query=plan.expanded)."""

def call_llm(prompt: str) -> str
    """ChatGroq(model, temperature, max_tokens, top_p=0.9,
    response_format={'type':'json_object'}) + max_retries=2, backoff.
    Raises LLMUnavailable on exhaustion."""

def parse_response(raw: str, n_blocks: int) -> LLMResponse
    """json.loads -> validate -> clamp source_index to [1, n_blocks].
    Regex fallback for malformed JSON. On total failure raise ParseFailure,
    which the pipeline converts to extractive mode."""

def enforce_sentence_cap(answer: str) -> tuple[str, bool]
    """truncate_to_sentences(config.MAX_ANSWER_SENTENCES) -> (text, truncated)"""

def generate(plan: QueryPlan, result: RetrievalResult) -> LLMResponse
    """build_prompt -> call_llm -> parse -> sentence cap.
    insufficient_context=True -> pipeline converts to a low_confidence refusal."""
```

**Do not** add conversational pleasantries, memory, or history. The prompt forbids them.

### 6.2 `src/citation.py`

```python
def normalize_url(url: str) -> str
    """lowercase host, strip scheme, strip trailing '/', drop query/fragment/tracking params."""
def verify_url(url: str) -> bool
    """True only if normalize_url(url) matches an allow-listed source or starts with
    config.ALLOWED_URL_BASE."""
def resolve_link(resp: LLMResponse, result: RetrievalResult) -> tuple[str, bool]
    """(url, link_substituted). Block at resp.source_index -> else top chunk -> else ''.
    If verify_url fails on the LLM-derived candidate, DISCARD it and use the top chunk."""
def format_last_updated(fetched_at_iso: str) -> str   # 'YYYY-MM-DD' -> '27 Sep 2026'
def finalise(resp: LLMResponse, result: RetrievalResult) -> AssistantMessage
    """Builds the final body and runs assert_invariants()."""
def assert_invariants(msg: AssistantMessage) -> None
    """Raise InvariantViolation unless ALL hold:
       (a) exactly one https URL in the rendered text
       (b) that URL passes verify_url
       (c) 'Last updated from sources:' present when kind == 'answer'
       (d) config.DISCLAIMER present
       (e) body <= MAX_ANSWER_SENTENCES sentences
       (f) no digits appear in a refusal message"""
```

**`finalise` is the last code that touches the answer.** Nothing downstream may append
free text.

### 6.3 Tests

`tests/test_citation.py` (no network needed — construct `LLMResponse` fixtures):

- `verify_url` accepts all 5 allow-listed URLs (raw, with trailing `/`, with `?utm_source=x`).
- `verify_url` rejects: `https://groww.in.evil.com/mutual-funds/x`,
  `https://groww.in/mutual-funds/` + unknown slug, a non-Groww AMC URL, `http://` variants, a shortened link.
- A hallucinated `Source:` URL from the LLM is discarded and substituted, with `link_substituted=True`.
- `source_index` out of range is clamped, never crashes.
- Sentence cap: a 6-sentence answer renders as 3 and sets `truncated=True`.
- `assert_invariants` raises for: two URLs, an unverified URL, a missing disclaimer, a missing last-updated line, a refusal containing digits.
- "Exactly one URL" is asserted by regex count over 50 generated messages, not just 2.

### 6.4 Live smoke test (needs `GROQ_API_KEY`)

```powershell
.\.venv\Scripts\python.exe run_index.py --stage smoke
```

Expected: 3 answers, each ≤3 sentences, one allow-listed link, last-updated line, and the
disclaimer. Also verify a deliberately hallucinated URL is substituted.

**Commit:** `feat: grounded Groq generation with verified single-link citation (phase 6)`

---

## PHASE 7 — Pipeline Orchestration + Logging

**Goal:** one function the UI can call, plus a redacted audit log.

**References:** ARCHITECTURE §4.10, §5.3, PRD FR13, FR15, NFR7.

**Files:** `src/pipeline.py`, `src/logger.py`.

### 7.1 `src/logger.py`

```python
def redact(text: str) -> str
    """Replaces any PII regex match with '<pii:kind>'. Applied to EVERYTHING logged."""
def log_query(plan, result | None, guardrails: list[dict], msg: AssistantMessage,
              timings: dict) -> None
    """One JSONL line to logs/query_log.jsonl. If any phase-A PII rule fired, write
    query='<pii:detected>' and DO NOT include the raw text."""
def read_log() -> list[dict]     # for the sidebar/debug
```

### 7.2 `src/pipeline.py`

```python
def answer(query: str) -> PipelineResult
    """Exactly the flow in ARCH §4.10. Emits trace keys:
    'stage_5_rewrite', 'guardrails_a', 'stage_6_8_retrieval', 'guardrails_b',
    'stage_10_generation', 'stage_11_citation' — the UI renders these verbatim."""
```

Order of operations inside `answer`:

1. `trace['stage_5_rewrite']` build
2. `check_phase_a(raw)` → if refusal: shape + log + return (**no log of raw text if PII**)
3. `retrieve(plan)` → `trace['stage_6_8_retrieval']`
4. `check_phase_b(plan, result)` → if refusal: shape + log + return
5. `generate(plan, result)` → on `LLMUnavailable`/`ParseFailure`: `EXTRACTIVE_ONLY` path
6. `insufficient_context` → low-confidence refusal
7. `finalise(resp, result)` → `trace['stage_11_citation']`
8. cache put (key = `plan.expanded`); log; return with `cached=False`

**Extractive path (degradation, must work):** no LLM. Find the top chunk whose
`label`/`section` matches `plan.intent`'s `expected_labels`; return its first 3 sentences
as the answer with the same verified link, last-updated stamp and disclaimer. This is
also the offline demo mode.

```python
def answer_extractive(plan: QueryPlan, result: RetrievalResult) -> AssistantMessage
def clear_cache() -> int
```

### 7.3 Definition of Done

- [ ] All 10 PRD §12 eval queries return a valid `PipelineResult` with a complete trace
- [ ] `GROQ_API_KEY=""` (or the network blocked) → all 10 still return cited answers via extractive mode
- [ ] A PII query produces a log line containing `<pii:detected>` and **not** the PAN string:
      `.\.venv\Scripts\python.exe -c "import pathlib;print([l for l in pathlib.Path('logs/query_log.jsonl').read_text().splitlines() if 'ABCDE1234F' in l])"` prints `[]`
- [ ] Repeating an identical query returns `cached=True` in <50 ms

**Commit:** `feat: pipeline orchestration, caching and redacted audit log (phase 7)`

---

## PHASE 8 — Streamlit UI

**Goal:** the demo surface. Presentation only — no retrieval or generation logic.

**References:** ARCHITECTURE §2, PRD §7, FR10–FR12, C7.

**Files:** `app.py`.

### 8.1 Structure

```python
@st.cache_resource
def get_resources():      # collection, embeddings, embedder  (built once per process)
    ...
def render_message(msg: AssistantMessage, trace: dict) -> None: ...
def main() -> None:
    st.set_page_config(page_title="MF Facts — HDFC Scheme FAQ", page_icon="📊", layout="centered")
    # ---- sidebar ----
    #   corpus stats: doc_count, chunk_count, embedding model, vector DB, last indexed
    #   Build / Rebuild Index button -> clears get_resources, calls run_index
    #   stale-index warning banner if is_index_stale()
    #   sources list (from data/sources.csv)
    # ---- main ----
    #   st.title / welcome line (exact text below)
    #   disclaimer note: st.info(config.DISCLAIMER)  — always visible
    #   3 example-question buttons (exact text below)
    #   chat history via st.session_state.messages
    #   st.chat_input("Ask a factual question about these 5 HDFC schemes")
    #   per assistant message: text, one citation link, last-updated, sources expander,
    #                         "Show pipeline" expander with st.json(trace)
```

**Welcome line (exact, PRD §7):** `Ask factual questions about 5 HDFC mutual fund schemes.`

**The 3 example questions (exact):**
1. `What is the expense ratio of HDFC Large Cap Direct Growth?`
2. `What is the lock-in period in HDFC ELSS Tax Saver Fund?`
3. `How do I download my capital gains statement?`

### 8.2 Rendering rules

| Element | Rule |
| --- | --- |
| Answer body | ≤3 sentences, plain text |
| Citation | exactly one `[Source](url)` — never more |
| Last updated | rendered from `msg.last_updated` with the `Last updated from sources: ` prefix |
| Disclaimer | on every assistant turn, always |
| Refusal | `st.info(...)` block, educational link, **no** citation badge |
| Cached turn | a small "cached" badge |
| Sources expander | top-3 chunks with label, section, score, and a 200-char preview |
| Show pipeline | `st.json(trace, expanded=False)` — the raw object from ARCH §5.4 |
| No index yet | friendly blocking message pointing at the sidebar button |

### 8.3 Definition of Done

```powershell
streamlit run app.py
```

- [ ] Welcome line, 3 clickable examples, and `Facts-only. No investment advice.` visible on first load
- [ ] Clicking an example sends it and returns a ≤3-sentence answer with one link
- [ ] "Should I buy HDFC Small Cap?" → refusal block with a link, no citation badge
- [ ] "My PAN is ABCDE1234F, what are my returns?" → PII refusal
- [ ] "How to download capital gains statement?" → answer (procedural, from corpus)
- [ ] "Show pipeline" expander shows rewrite → chunks+scores → guardrails → prompt → raw response → citation checks
- [ ] Reloading the page does not re-embed (no index rebuild) and chat state resets cleanly
- [ ] No PAN string anywhere in the UI, the log, or `data/chroma/`

**Commit:** `feat: streamlit chat UI with citations, disclaimers and pipeline trace (phase 8)`

---

## PHASE 9 — Eval Harness & Deliverables

**Goal:** the numbers that prove the PRD's success metrics, and the files the milestone
asks for.

**References:** ARCHITECTURE §10, PRD §12, §16.

**Files:** `deliverables/run_eval.py`, `deliverables/sample_qa.md`,
`deliverables/disclaimer.md`, `tests/test_eval.py`.

### 9.1 `deliverables/run_eval.py`

```python
EVAL_QUERIES: list[dict]   # the 10 from PRD §12 verbatim, with expected_kind
    # each: {"q": ..., "expect": "answer"|"refusal_advice"|"refusal_pii"|"refusal_nav", ...}

def check(cond: bool, code: str) -> bool     # A1..A7
def run_case(case: dict, use_llm: bool) -> dict
def main(use_llm: bool = True) -> int       # prints a table, writes sample_qa.md, exit 1 on failure
```

**Assertions (ARCH §10 Layer 3):**

| Code | Check |
| --- | --- |
| A1 cited | exactly one URL in the body, and it passes `verify_url` |
| A2 short | ≤3 sentences |
| A3 fresh | `Last updated from sources:` present |
| A4 safe | no return/performance phrasing, no recommendation verbs |
| A5 refused | the advice, performance, and PII cases produce refusals with an educational link |
| A6 grounded | **every number in the answer appears verbatim in the retrieved context** (digit-substring containment) |
| A7 no PII stored | no PII sample string anywhere in `logs/query_log.jsonl` |

> A6 is the project-defining test. It converts "do not hallucinate" from a prompt
> instruction into a pass/fail check. Implement it first, and let it fail honestly at
> least once before you tune anything.

### 9.2 Generated deliverables

- `deliverables/sample_qa.md` — the 10 queries with the assistant's actual answers, the one source link each, the last-updated stamp, and the assertion results table.
- `deliverables/disclaimer.md` — the exact UI strings: the disclaimer, the welcome line, the 3 examples, the last-updated prefix, and the refusal templates.
- `data/sources.csv` — already produced in Phase 1; verify 7 rows.

### 9.3 Definition of Done

```powershell
.\.venv\Scripts\python.exe deliverables\run_eval.py
```

- [ ] 10/10 cases run; A1–A7 all pass; exit code 0
- [ ] `sample_qa.md` contains 10 real answers with real links
- [ ] `pytest -q` fully green

**Commit:** `feat: eval harness with A1-A7 assertions and sample Q&A deliverable (phase 9)`

---

## PHASE 10 — README & Demo Script

**Goal:** the submission. PRD §16's checklist is the acceptance gate.

**References:** PRD §7, §11, §12, §16.

**Files:** `README.md`, and a ≤3-minute demo video (or a hosted app link).

### 10.1 `README.md` required sections

1. **What it is** — one paragraph, facts-only RAG assistant for 5 HDFC schemes.
2. **Scope** — AMC + the 5 schemes table with links (copied from `data/sources.csv`).
3. **Architecture in one picture** — the stage diagram from ARCH §2, plus a link to `ARCHITECTURE.md`.
4. **Setup** — for Windows PowerShell **and** macOS/Linux, ≤5 commands, including where to get a free `GROQ_API_KEY`.
5. **How to run** — `run_index.py --stage index` then `streamlit run app.py`.
6. **Example questions** — the 3 UI examples + the refusal examples.
7. **How it works** — loading → chunking → embedding → store → retrieval → generation, 2 lines each, with the specific parameters used (chunk 400/50, MiniLM, Chroma `mf_faq_hdfc`, k=15/5, floor 0.35, temp 0.1).
8. **Guardrails** — the rules table and what gets refused.
9. **Known limits** — required by the brief. Minimum: static snapshot (no live NAV), single AMC, no cross-scheme comparison, English-only phrasing tolerance, similarity-floor false negatives, free-tier rate limits, scrape depends on Groww's layout.
10. **Testing** — `pytest -q` and `deliverables/run_eval.py`, with the current pass numbers.
11. **Disclaimer** — the exact text, and the refusal policy in one line.
12. **Project layout** — the tree from ARCH §11.

### 10.2 Demo video script (≤3 min, 6 shots)

| Shot | Time | Show |
| --- | --- | --- |
| 1 | 0:00–0:20 | Welcome screen, disclaimer, 3 examples; sidebar corpus stats |
| 2 | 0:20–0:55 | Expense-ratio question → answer, click the citation, point at `Last updated from sources:` |
| 3 | 0:55–1:25 | **"Show pipeline"** expander: rewrite → 5 chunks with scores → guardrails → prompt → raw response |
| 4 | 1:25–1:50 | "Should I buy HDFC Small Cap?" → refusal; "best performer?" → refusal |
| 5 | 1:50–2:10 | PII refusal, and show the redacted log line |
| 6 | 2:10–3:00 | `run_eval.py` output: 10 cases, A1–A7 green |

> Shot 3 is the one that proves it is RAG. Rehearse it; the trace is already rendered.

### 10.3 Definition of Done (PRD §16)

- [ ] Working prototype link or a ≤3-min video
- [ ] `data/sources.csv` — all 7 URLs (5 scraped + 2 authored), publisher, fetch date
- [ ] `deliverables/sample_qa.md` — 5–10 queries with answers + links
- [ ] `deliverables/disclaimer.md` — exact UI disclaimer
- [ ] `README.md` — setup, scope, known limits
- [ ] Every answer: ≤3 sentences, one link, last-updated stamp, disclaimer
- [ ] Advice/performance/PII refused politely with an educational link
- [ ] Pipeline expander functional
- [ ] Clean clone → running app in ≤5 commands (verify on a fresh clone, not your working tree)
- [ ] No secrets committed; no PII in logs

**Commit:** `docs: README with setup, scope and known limits (phase 10)`

---

## Appendix A — Phase Dependency Graph

```text
Phase 0  config, types, deps
   │
Phase 1  ingest  ─────────────► data/corpus/*.md, sources.csv
   │
Phase 2  chunking ────────────► chunks.jsonl          (needs the real corpus)
   │
Phase 3  embedding + store ───► chroma/, index_meta.json
   │
Phase 4  query + retriever ───► SIMILARITY_FLOOR tuned
   │
Phase 5  guardrails ──────────► refusals, no LLM needed
   │
Phase 6  generator + citation► grounded answer + verified link
   │
Phase 7  pipeline + logger ───► answer()  (first end-to-end moment)
   │
Phase 8  UI  ────────────────► the demo
   │
Phase 9  eval + deliverables ─► proof (A1-A7)
   │
Phase 10 README + video ─────► submission
```

Phases 1–7 have **no UI dependency**. If time is short, cut the UI polish before cutting
the eval harness — the harness is what makes the claims checkable.

## Appendix B — Copy-Paste Verification Suite (run before every commit)

```powershell
.\.venv\Scripts\python.exe -m pytest -q                                  # all unit tests green
.\.venv\Scripts\python.exe run_index.py --stage all                      # rebuild end to end
.\.venv\Scripts\python.exe deliverables\run_eval.py                      # A1-A7, exit 0
.\.venv\Scripts\python.exe -c "import pathlib;t=pathlib.Path('logs/query_log.jsonl').read_text();print('PII LEAK' if 'ABCDE1234F' in t else 'log clean')"
Select-String -Path data\corpus\*.md -Pattern "cookie|Log in|All Rights Reserved"  # expect no output
git status --porcelain .env                                            # expect no output
```

## Appendix C — Cursor Prompt Starter (Phase 1 example)

> Use this as the shape of the per-phase prompt. Replace the bracketed parts.

```text
Implement Phase 1 of IMPLEMENTATION.md in this repo, following ARCHITECTURE.md §4.1.

Context: PRD.md (what) and ARCHITECTURE.md (how) are in the repo root. IMPLEMENTATION.md
§PHASE 1 lists the exact function contracts and Definition of Done.

Rules:
- Do not start any later phase.
- All constants, URLs and selectors live in src/config.py or as module constants in
  src/ingest.py — no magic values inline.
- ingest.py must never raise: on network failure it keeps the existing snapshot and
  marks the row stale=true in data/sources.csv.
- Type hints + docstrings on every function. No `Any` without a comment explaining why.
- After writing the code, run `.\.venv\Scripts\python.exe run_index.py --stage ingest`
  and show me: the 5 output filenames, char counts, each file's heading list, and the
  first 40 lines of one file. Then stop and wait — I need to confirm the corpus is
  label-dense before Phase 2, because the chunking strategy depends on it.

Do not write tests for this phase yet; do not modify ARCHITECTURE.md.
```

## Appendix D — Common Failure Modes to Pre-empt

| Symptom | Cause | Fix |
| --- | --- | --- |
| Answers cite the wrong scheme | `where` filter missing, or alias mis-resolved | Check `resolve_scheme`; enable the scheme filter whenever `scheme_key` is not None |
| Hallucinated link rendered | `verify_url` not called before rendering | Route every link through `resolve_link`; the invariant test must fail if bypassed |
| "I couldn't find that" on a valid question | `SIMILARITY_FLOOR` too high, or the chunk lacks the label | Improve `expand_shorthand`, re-check the label-preservation test — do not just lower the floor |
| Answers 5+ sentences | JSON parse succeeded but cap not applied | `enforce_sentence_cap` must run on every path, including the extractive path |
| Index rebuilds on every page reload | `get_collection` not wrapped in `st.cache_resource` | Wrap it |
| Second embedding model appears | Chroma default embedder not disabled | `embedding_function=None` at collection creation |
| Stale figures in the demo | Corpus snapshot is old | Re-run `--stage ingest`; the last-updated stamp must reflect the new `fetched_at` |
| Log contains a PAN | Logged before the PII guard | Log only after `check_phase_a`; redact with `logger.redact` as a second layer |
| Demo dies on a 429 | Groq free tier | Cache is on; keep the recorded video as backup; add `max_retries=2` |
