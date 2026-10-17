# MF Facts — A Facts-Only RAG Assistant for HDFC Mutual Fund Schemes

## What it is

MF Facts is a facts-only Retrieval-Augmented Generation (RAG) assistant for five HDFC mutual fund schemes. It answers verifiable, point-in-time facts (e.g., expense ratio, exit load, minimum SIP/lump sum, ELSS lock-in, riskometer/benchmark, how to download a capital gains statement) from a committed corpus of the schemes' public pages and hand-transcribed official documents. It refuses advice, performance comparisons/rankings, and personal data, never fabricates numbers, and always attaches exactly one verified link plus a freshness stamp.

## Scope

**AMC:** HDFC Asset Management (five schemes). The UI and guardrails are HDFC-only by design.

| Scheme | Category | Source link |
| --- | --- | --- |
| HDFC Large Cap Fund — Direct Growth | Large Cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| HDFC Equity (Flexi Cap) Fund — Direct Growth | Flexi Cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| HDFC ELSS Tax Saver — Direct Growth | ELSS | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| HDFC Small Cap Fund — Direct Growth | Small Cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| HDFC Balanced Advantage Fund — Direct Growth | Hybrid | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

The committed corpus also includes two authored documents (ELSS lock-in + tax details and a step-by-step capital-gains statement guide) and per-scheme inception details, all transcribed from official HDFC pages (HTTP 403 on automated access) and recorded in `data/sources.csv`.

## Architecture (at a glance)

The end-to-end flow matches the design in `ARCHITECTURE.md` (Section 2). The "Show pipeline" expander in the UI renders this trace for each query.

```text
User -> Streamlit (app.py) -> pipeline (stages 5→11)
  → query rewrite/resolve → retriever (6–8, ChromaDB) → guardrails (A,B)
  → generator (Groq openai/gpt-oss-120b) or extractive → citation (verify link, stamp)
  → logger (redacted) → UI (answer + 1 link + "Last updated from sources:" + disclaimer)
```

Stages: Loading → Chunking → Embedding → Vector store → Query rewrite & scope → Retrieval → Guardrails → Generation → Citation → Orchestration/Logging.

## Setup

### Windows (PowerShell)

```powershell
# 1. Clone and change to the repo root
cd "C:\path\to\NextLeap"

# 2. Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key (free at https://console.groq.com/keys)
copy .env.example .env
# Edit .env and set GROQ_API_KEY=<your-key>
```

**Note:** If `Activate.ps1` fails due to policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window.

### macOS/Linux (bash/zsh)

```bash
cd NextLeap
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set GROQ_API_KEY in .env
```

## How to run

1. **Build the index** (offline, one-time or after any corpus change):

```powershell
.\.venv\Scripts\python.exe run_index.py --stage index
```

2. **Launch the UI**:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the printed URL in your browser. The app will warn if the index is stale and allow "Rebuild index" from the sidebar.

## Example questions

**Demo clicks (UI):**
1. What is the expense ratio of HDFC Large Cap Direct Growth?
2. What is the lock-in period in HDFC ELSS Tax Saver Fund?
3. How do I download my capital gains statement?

**Refusal examples (by guardrail):**
- Advice: "Should I buy HDFC Small Cap?" → refusal (advice) with an educational link; no citation badge.
- Performance: "How did HDFC Large Cap perform last year?" → performance refusal.
- PII: "My PAN is ABCDE1234F, what are my returns?" → PII refusal, input never logged.

## How it works (specific parameters)

- **Stage 1 Loading:** fetch only the 5 allow-listed URLs at index time; authored docs transcribed; `data/sources.csv` records publisher/fetch status (no automated fetch for `403` pages).
- **Stage 2 Chunking:** structure-aware splitter, chunk size 400 tokens with 50 token overlap; fallback 1600/200; metadata and labels preserved. **Final chunks:** 248 (`data/index_meta.json`).
- **Stage 3 Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (384d), docs prefixed for retrieval, queries unprefixed. Device CPU; batch 32. Fingerprint excludes device (stable).
- **Stage 4 Vector Store:** ChromaDB, collection `mf_faq_hdfc`, cosine distance. Index persistent in `data/chroma/`.
- **Stages 6–8 Retrieval:** candidates `RETRIEVE_K=15`, top `TOP_K=5`; similarity floors `SIMILARITY_FLOOR=0.35`, `UNKNOWN_INTENT_FLOOR=0.45`; label-based boost `RETRIEVAL_LABEL_BOOST=0.35`; metadata filter by scheme when resolvable.
- **Stage 9 Guardrails:** Phase A runs PII→advice→performance→scope on raw query; Phase B checks confidence, blocks live NAV, and checks intent grounding. No LLM in guardrails.
- **Stage 10 Generation:** Groq `openai/gpt-oss-120b`, `temperature=0.1`, `top_p=0.9`, `max_tokens=400`; JSON output required; degrades to **extractive** (no model call) if unavailable/malformed (`EXTRACTIVE_ONLY`, no key, errors); answers capped at `MAX_ANSWER_SENTENCES=3`.
- **Stage 11 Citation:** link taken strictly from chunk metadata and must be in the 7-URL allow-list; answer body stripped of any URLs; exactly one link, verified; `Last updated from sources: <date>` from chunk metadata; invariants asserted on every turn.

## Guardrails

| Rule | Outcome | What trips it |
| --- | --- | --- |
| PII first | Refuse (template, educational link) | PAN/Aadhaar/account/OTP/email/phone patterns (raw query never logged). |
| Advice/opinion | Refuse | "should I...", "recommend/suggest/advice", buy/sell/switch/redeem/allocate. |
| Performance | Refuse | returns/performance, compare, last year, CAGR, best performing, etc. |
| Out-of-scope AMC | Refuse | mentions SBI, Axis, ICICI, Kotak, etc. |
| Low confidence | Refuse ("not in sources") | `top_score < floor` (0.35 or 0.45) after filtering. |
| Live NAV | Refuse (redirect) | intent `nav` (current price). |
| Unknown intent ungrounded | Refuse | recognised intent but no expected label in top chunks. |

## Known limits

- **Static snapshot:** no live NAV; NAV/price questions are refused and redirected to the official page.
- **Single AMC (HDFC):** scope is the 5 named schemes. Other AMCs are refused.
- **No cross-scheme comparison:** performance/comparison is explicitly refused.
- **English-only:** query rewrite and regex guardrails assume English phrasing.
- **Layout sensitivity:** ingestion relies on Groww's page structure; a major redesign could require re-ingest.
- **Similarity-floor tradeoff:** could produce false negatives in rare ambiguous queries; guardrails prefer refusing over guessing.
- **Free-tier rate limits:** Groq may rate-limit; the pipeline degrades to extractive with no loss of the citation guarantee.
- **Grounding-determinism:** numbers must be verbatim in retrieved context (A6) — reformatting/rounding is rejected by design.
- **Demo constraints:** ELSS slug required a correction (documented in code/config); index already built for the committed snapshot.

## Testing

Run the unit suite and the eval harness. With the committed index, the full set passes.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe deliverables\run_eval.py
```

Results (from this checkout):
- Unit tests: **686 passed in ~64s** (`tests` include Phase 0–10 unit/integration tests as present).
- Eval (A1–A7): **10/10 cases pass**, exit code 0. See `deliverables/sample_qa.md`.

**Quick verification checks:** PII sample string `ABCDE1234F` does not appear in `logs/query_log.jsonl`; corpus has no cookie/login/footer residuals; all cited URLs pass allow-list.

## Disclaimer

**Facts-only. No investment advice.**

Every answer carries this disclaimer and is grounded only in the committed corpus. Refusals are polite, cite an educational page, and never contain digits. No personal information is stored.

## Project layout

```text
NextLeap/
├─ PRD.md, ARCHITECTURE.md, README.md, app.py
├─ requirements.txt, requirements.lock.txt, .env.example, .gitignore
├─ run_index.py, src/*, tests/*
├─ data/corpus/*.md, data/chunks/chunks.jsonl, data/index_meta.json, data/sources.csv
├─ data/chroma/ (gitignored)
├─ deliverables/run_eval.py, deliverables/sample_qa.md, deliverables/disclaimer.md
└─ logs/query_log.jsonl (gitignored)
```

See `ARCHITECTURE.md` §11 for the canonical tree and `ARCHITECTURE.md` §2 for the stage diagram.
