# PRD — MF Facts: A Facts-Only RAG FAQ Assistant for Mutual Fund Schemes

| Field | Value |
| --- | --- |
| Document type | Product Requirements Document (PRD) |
| Version | 1.0 |
| Status | Draft for team review |
| Milestone | Class demo — Mutual Fund FAQs (Facts-Only Q&A) |
| Owner | NextLeap RAG Chatbot Team |
| Last updated | 2026-09-27 |

---

## 1. Summary

Build a small, working RAG chatbot that answers **factual** questions about five HDFC
Asset Management mutual fund schemes using **only** officially published public pages
(Groww scheme pages, which mirror the AMC's published scheme data). Every answer must
carry exactly one source link. The assistant must refuse opinionated or portfolio
questions. No investment advice, no return calculations, no PII.

The product is a **class demo**, but the pipeline must be a genuine RAG implementation
with all stages visible and inspectable: **Loading → Chunking → Embedding → Vector Store
→ Retrieval → Grounded Generation**.

---

## 2. Problem Statement

Retail investors and support/content teams repeatedly ask the same factual questions about
mutual fund schemes — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer,
benchmark, and how to download statements. Today those answers live in factsheets, KIM/SID
documents, fee/charge pages, and scheme FAQ pages scattered across an AMC's site. A person
either has to know the URL already, or read a 60-page KIM to find one number.

Generic AI assistants are the wrong tool: they will happily answer "should I buy this?" and
will happily invent an expense ratio. **Hallucinated financial facts cause real harm.**

The gap this product fills: a narrow, source-grounded assistant that is *deliberately*
incapable of giving advice, and that proves every answer with a link.

---

## 3. Goals and Non-Goals

### 3.1 Goals

| # | Goal | Measure |
| --- | --- | --- |
| G1 | Answer the 5 named fact types (expense ratio, exit load, min SIP, ELSS lock-in, riskometer/benchmark, statements) from the corpus | ≥90% of the sample Q&A answered from context with a citation |
| G2 | Every answer contains exactly one working source link | 100% of answers |
| G3 | Refuse opinionated / portfolio questions politely | 100% of refusal cases in the sample set |
| G4 | Answers are short and transparent | ≤3 sentences + `Last updated from sources: <date>` |
| G5 | Demonstrate a real RAG pipeline, not a lookup table | Loader, chunker, embedder, ChromaDB store, and retriever all present and logged |
| G6 | A teammate can clone and run it in one command | README setup ≤5 steps, no manual DB setup |

### 3.2 Non-Goals (explicitly out of scope)

- **No advice.** No "should I buy/sell", no suitability, no asset allocation.
- **No performance claims.** No return computation, no ranking, no fund comparison by past
  performance. If asked, link to the official factsheet and stop.
- **No PII handling.** PAN, Aadhaar, account numbers, OTPs, emails, phone numbers are
  neither requested, stored, nor logged.
- **No live data.** No NAV feed, no real-time prices. The corpus is a static snapshot.
- **No multi-AMC coverage.** One AMC (HDFC AMC), five schemes. Scope control is the point.
- **No user accounts, no login, no persistence of chat history across sessions.**

---

## 4. Users and Use Cases

### 4.1 Primary Users

1. **Retail investor** — comparing a handful of HDFC schemes, wants a number, not a
   lecture. Needs the fact and where it came from.
2. **Support / content team** — fielding the same repetitive MF questions daily; wants a
   draft answer they can verify from the link.

### 4.2 Key Use Cases

| ID | Scenario | Example query |
| --- | --- | --- |
| UC1 | Look up a fee | "What is the expense ratio of HDFC Large Cap Direct Growth?" |
| UC2 | Check a penalty | "Is there an exit load on HDFC Flexi Cap?" |
| UC3 | Check a minimum | "What is the minimum SIP amount for HDFC ELSS Tax Saver?" |
| UC4 | Check a rule | "What is the lock-in period in HDFC ELSS Tax Saver?" |
| UC5 | Check risk/benchmark | "What is the benchmark and riskometer level of HDFC Balanced Advantage?" |
| UC6 | Procedural | "How do I download my capital gains statement?" |
| UC7 | Out of scope — advice | "Should I buy HDFC Small Cap for my retirement?" |
| UC8 | Out of scope — performance | "Which of these five funds performed best last year?" |
| UC9 | Out of scope — PII | "My PAN is ABCDE1234F, can you check my SIP?" |

### 4.3 Refusal Scenarios (must be handled by design, not by luck)

The system must produce a polite, facts-only message plus a relevant educational link for:

- Buy/sell/hold/allocate recommendations.
- Comparative performance or ranking questions.
- Tax advice beyond "here is the official document".
- Anything containing or requesting PII.
- Anything about a scheme or fact type outside the five-scheme corpus.
- Ambiguous questions with no matching context (must not invent).

---

## 5. Scope — Corpus Definition

**AMC:** HDFC Asset Management (HDFC AMC)
**Schemes:** 5, direct-growth / direct plans, from the milestone brief.

| # | Scheme | Type | Source URL |
| --- | --- | --- | --- |
| S1 | HDFC Large Cap Fund — Direct Growth | Large Cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| S2 | HDFC Equity (Flexi Cap) Fund — Direct Growth | Flexi Cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| S3 | HDFC ELSS Tax Saver — Direct Growth | ELSS | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-growth |
| S4 | HDFC Small Cap Fund — Direct Growth | Small Cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| S5 | HDFC Balanced Advantage Fund — Direct Growth | Hybrid | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

**Corpus rules**

- Only these 5 public scheme URLs are primary sources, plus 2 official HDFC AMC pages (see
  "Authored sources" below). No third-party blogs, no forums, no news articles, no social
  media.
- Primary source of record is the AMC/regulator material; the Groww page is the public
  mirror the milestone specifies. The `sources.csv` deliverable records the URL, the
  fetch date, and the publisher.
- **Returns and rankings are stripped at ingest.** The brief's constraint C3 forbids
  performance claims, and a mirror page prints a 1-day/1-year return and a category rank
  next to the facts we do want. Those are removed during extraction, so the generator
  cannot retrieve them. A *benchmark name* such as "NIFTY 100 Total Return Index" is kept:
  "Total Return" is part of the index's name, not a figure.
- **The ELSS URL in the original brief 404s.** The working page is
  `.../hdfc-elss-tax-saver-fund-direct-plan-growth`; the `...-direct-growth` variant in the
  brief returns 404. The working URL is what the corpus uses.
- **Authored sources.** `hdfcfund.com` serves `403 Access Denied` to automated requests, so
  two facts that only the AMC publishes are hand-transcribed and labelled as such: the ELSS
  lock-in and section 80C terms, and how to download a capital gains statement. Each is
  stored with `authored: true`, `live_fetch_status: 403` and a "hand-authored" publisher in
  the document header and in `sources.csv`, so a transcribed document can never be mistaken
  for a scrape. The ELSS lock-in is also on the live page as a terse `3Y Lock-in` tag; the
  authored copy is kept because it carries the 80C deduction and eligibility wording the
  live page does not.
- Every chunk carries metadata: `scheme`, `category`, `source_url`, `source_title`,
  `fetched_at`, `chunk_index`. This is what makes one-link citations and "last updated"
  possible.
- The corpus is **snapshotted and committed** so the demo is reproducible and does not
  depend on live network access at demo time.

---

## 6. System Architecture — The RAG Pipeline

The demo's core claim is that this is a real RAG system. All stages below are
implemented, separated into modules, and observable in the UI.

```
                  ┌─────────────────────────── INGESTION (offline, cached) ───────────────────────────┐
  5 public URLs → │ 1. LOADING → 2. CHUNKING → 3. EMBEDDING → 4. VECTOR STORE (ChromaDB, persisted)  │
                  └────────────────────────────────────┬─────────────────────────────────────────────┘
                                                           │
  ┌──────────────────────────── RETRIEVAL + GENERATION (online, per query) ──────────────────────────┐
  │ 5. QUERY REWRITE        expand/normalize the question                                    │
  │ 6. EMBEDDING            embed the query with the same MiniLM model                       │
  │ 7. RETRIEVAL            ChromaDB similarity search, k chunks                            │
  │ 8. RE-RANK / FILTER     drop low-similarity chunks, keep top-k                           │
  │ 9. GUARDRAILS          scope check + PII check + refusal classifier                      │
  │10. GROUNDED GENERATION Groq llama-3.3-70b, context-only, ≤3 sentences, 1 citation       │
  │11. CITIGATION + STAMP  attach one source link + "Last updated from sources: <date>"     │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Stage 1 — Document Loading

- **Tool:** `requests` (with retry + polite timeout) + `BeautifulSoup4` to extract and
  clean the page to Markdown, one `.md` file per scheme under `data/corpus/`.
- **Cleaning rules:** strip nav, header, footer, cookie banners, ad slots, and "related
  fund" widgets. Preserve labelled fact tables and their headings. Preserve the scheme
  name and category as the document title.
- **Fallback:** if a fetch fails, reuse the committed snapshot and log a warning. The demo
  must never hard-depend on Groww being up.
- **Output:** `data/corpus/<scheme-slug>.md`

### 6.2 Stage 2 — Chunking Strategy

The strategy is **data-driven, not hardcoded blindly.** The corpus is short, dense,
label-oriented financial data (expense ratio, exit load table, minimum amounts,
riskometer, benchmark), so:

- **Primary decision after inspecting the corpus:** split on the page's own semantic
  boundaries — scheme title, and labelled sections/table rows — with
  `RecursiveCharacterTextSplitter` as the fallback for long unbroken prose.
- **Target chunk size:** 300–500 tokens, **overlap ~50 tokens**, so a number and its
  label never land on opposite sides of a boundary.
- **Rule:** a chunk must carry at least one complete labelled fact. If a chunk has no
  label, it is merged forward.
- **Never split a table row** across chunks; keep a table's header row with its data rows.
- **Rationale to present in the demo:** naive fixed-size chunking on financial pages
  orphans facts from their labels; structure-aware chunking keeps each fact self-contained.
- **Output:** `data/chunks/chunks.jsonl` — one JSON object per chunk with `text` +
  metadata. Committed so chunking can be diffed across experiments.

### 6.3 Stage 3 — Embedding

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (as mandated by the brief).
  384 dimensions, ~22M params, runs on CPU, ~1–2 s to embed the whole 5-doc corpus.
- **Library:** `langchain-huggingface` / `HuggingFaceEmbeddings`.
- **Rule:** **query and document embeddings must use the same model.** Model ID is
  persisted alongside the ChromaDB collection so a model change forces a re-index.
- **Prefix convention:** documents are embedded as
  `"Represent this mutual fund scheme document for retrieval: <chunk>"` to match the
  model's retrieval-tuned training.
- **Output:** vectors written into ChromaDB.

### 6.4 Stage 4 — Vector Store

- **DB:** **ChromaDB** (as mandated). Persistent client, not in-memory — the demo must
  survive a Streamlit rerun.
- **Collection:** `mf_faq_hdfc` with metadata
  `{scheme, category, source_url, source_title, fetched_at, chunk_index, label}`.
- **Lifespan:** index once via a "Build / Rebuild Index" button in the app sidebar;
  never re-embed on every query.
- **Deduplication key:** chunk hash, so re-running ingestion is idempotent.

### 6.5 Stage 5 — Query Rewrite (pre-retrieval)

Normalize the raw question before embedding:

- Lowercase, strip punctuation and filler ("please tell me", "hi").
- Expand Indian finance shorthand: `ER` → expense ratio, `ELSS` → tax saver,
  `min SIP` → minimum SIP amount, `exit load` → redemption charge.
- **Scheme-name resolution:** map free-text mentions to canonical scheme keys
  ("hdfc large cap", "large cap fund" → `hdfc-large-cap`).
- **Out-of-scope detection:** if the query names a scheme outside the 5, or asks for
  advice/performance/PII, short-circuit to refusal *before* retrieval.

### 6.6 Stage 6–8 — Retrieval

- Top-`k` = 5, `fetch_k` = 15, cosine similarity (Chroma default `hnsw:space=cosine`).
- **Similarity floor:** chunks below a tuned threshold (start at 0.35) are discarded. If
  nothing survives → *"I don't have that in my sources"* + link to the scheme page.
  This is the anti-hallucination gate.
- **Optional metadata filter:** when the query resolves to a scheme, filter on
  `scheme` to keep answers from bleeding across funds.
- **Context assembly:** numbered passages `[1] … [n]` with `source_url` next to each, so
  the LLM can cite and the UI can render the link.

### 6.7 Stage 9 — Guardrails

Deterministic checks run **before** the LLM (cheap, reliable, demoable):

| Guard | Rule | Action |
| --- | --- | --- |
| G-Advice | matches buy/sell/hold/suggest/recommend/should I/allocate | Refuse |
| G-Performance | matches return/performance/YTD/1-year return/best performing | Refuse + link factsheet |
| G-PII | regex for PAN, Aadhaar, 10-digit account, OTP, email, phone | Refuse + do not log the text |
| G-Scope | scheme not in the 5, or fact type unknown | Refuse + link scheme page |
| G-Confidence | no chunk above similarity floor | "Not in my sources" |

The refusal template is a constant so wording is consistent, and every refusal carries
one educational link (the relevant scheme page or the AMC's fee/disclosure page).

### 6.8 Stage 10 — Grounded Generation

- **Model:** Groq `llama-3.3-70b-versatile` (free tier, fast, reliable for a live demo).
- **Temperature:** 0.0–0.2 (factual extraction, not creative writing).
- **Prompt contract:**
  - Answer **only** from `<context>`. If context is insufficient, say so explicitly.
  - **≤3 sentences.** No preamble, no disclaimer boilerplate in the body.
  - Emit the single source URL on its own final line as `Source: <url>`.
  - Never state a return, yield, ranking, or recommendation — even if asked politely.
  - Never request or repeat PII.
- **Structured output:** answer text + `source_url` + `used_chunks` returned as JSON so
  the UI can render the link and the "show retrieved sources" expander.

### 6.9 Stage 11 — Citation and Freshness Stamp

- Validate that the LLM's `Source:` line is a URL present in the allowed source list. If
  it hallucinated a URL, **discard it and fall back to the top retrieved chunk's
  `source_url`**. Never render an unverified link.
- Append the footer exactly as required:
  `Last updated from sources: <fetched_at date>` — taken from chunk metadata, not the
  LLM's guess.
- Always render the disclaimer: `Facts-only. No investment advice.`

### 6.10 Observability (what makes it demoable)

A **"Show pipeline" expander** in the UI displays, per query: rewritten query, each
retrieved chunk with score, guardrail decisions, the final prompt, the raw LLM response,
and the model + embedding IDs. This is what distinguishes a RAG demo from a chatbot demo.

---

## 7. UI Requirements

**Framework:** Streamlit (Python), single-page chat.

- `st.chat_message("assistant")` thread, `st.chat_input` for the query.
- **Welcome line**, exactly as briefed: e.g. *"Ask factual questions about 5 HDFC mutual
  fund schemes."*
- **3 example questions** as clickable buttons that prefill/send:
  1. "What is the expense ratio of HDFC Large Cap Direct Growth?"
  2. "What is the lock-in period in HDFC ELSS Tax Saver Fund?"
  3. "How do I download my capital gains statement?"
- **Disclaimer note, always visible:** `Facts-only. No investment advice.`
- **Sidebar:** build/rebuild index, corpus stats (docs, chunks, embedding model, vector DB,
  last indexed), and the `sources.csv` source list.
- **Under each answer:** one clickable citation link, the
  `Last updated from sources: <date>` stamp, and a "Sources" expander with the chunk
  snippets used.
- **Refusals** render in a distinct style (info block), with the educational link, and no
  citation-count badge.
- **No login, no email capture, no cookies that identify a user.**

---

## 8. Functional Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR1 | Ingest 5 public URLs → clean Markdown in `data/corpus/` | Must |
| FR2 | Chunk corpus using the strategy from §6.2; persist to `chunks.jsonl` | Must |
| FR3 | Embed chunks with `all-MiniLM-L6-v2` and store in persistent ChromaDB | Must |
| FR4 | Answer a factual query using retrieved context only | Must |
| FR5 | Attach exactly one verified source link to every substantive answer | Must |
| FR6 | Enforce ≤3 sentences per answer | Must |
| FR7 | Append `Last updated from sources: <date>` to every answer | Must |
| FR8 | Refuse advice, performance, PII, and out-of-scope questions politely with an educational link | Must |
| FR9 | Return "not in my sources" when retrieval confidence is low (no guessing) | Must |
| FR10 | Welcome line + 3 example questions + disclaimer in the UI | Must |
| FR11 | Show retrieved chunks, scores, and guardrail decisions per query | Should |
| FR12 | Sidebar index rebuild + corpus stats | Should |
| FR13 | Log every query, retrieved sources, and guardrail outcome to a local JSONL file (no PII) | Should |
| FR14 | Sample Q&A file generator for the deliverable | Should |
| FR15 | Cache responses for repeated identical questions | Nice |
| FR16 | Reindex automatically if the embedding model ID changes | Nice |

## 9. Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR1 | Cold start to first answer < 10 s on a laptop CPU-only machine |
| NFR2 | Retrieval over 5 documents completes in < 1.5 s |
| NFR3 | Zero secrets in the repo — API key via `.env` + `.env.example`; `.env` gitignored |
| NFR4 | Works offline once indexed (no live scraping at query time) |
| NFR5 | Corpus snapshot committed to the repo for demo reproducibility |
| NFR6 | Works on Windows (PowerShell) and macOS/Linux; documented for both |
| NFR7 | No PII ever written to logs, cache, or `chroma/` |
| NFR8 | Full pipeline rebuild from a clean clone in ≤5 commands |

---

## 10. Proposed Repository Layout

```
NextLeap/
├─ PRD.md
├─ README.md                     # setup, scope, known limits
├─ requirements.txt
├─ .env.example                  # GROQ_API_KEY=
├─ .gitignore                    # .env, chroma/, __pycache__/
├─ app.py                        # Streamlit UI
├─ src/
│  ├─ config.py                  # constants, 5 source URLs, thresholds
│  ├─ ingest.py                  # STAGE 1 load + clean  -> data/corpus/*.md
│  ├─ chunking.py                # STAGE 2 chunk          -> data/chunks/chunks.jsonl
│  ├─ embedding.py               # STAGE 3 MiniLM embed
│  ├─ vectorstore.py             # STAGE 4 ChromaDB persist
│  ├─ query.py                   # STAGE 5 rewrite + scope resolve
│  ├─ retriever.py               # STAGE 6-8 retrieve + filter
│  ├─ guardrails.py              # STAGE 9 refusal rules
│  ├─ generator.py               # STAGE 10 Groq grounded generation
│  └─ citation.py                # STAGE 11 link verify + stamp
├─ data/
│  ├─ corpus/*.md                # committed snapshot: 5 scraped + 2 authored docs
│  ├─ chunks/chunks.jsonl        # committed
│  └─ sources.csv                # deliverable: url, publisher, fetched_at
├─ deliverables/
│  ├─ sample_qa.md               # 5-10 queries + answers + links
│  └─ disclaimer.md              # exact disclaimer text
└─ tests/
   ├─ test_guardrails.py
   └─ test_citation.py
```

---

## 11. Data & API Requirements

- **Corpus sources:** the 5 URLs in §5. Publisher recorded as Groww (public mirror of
  HDFC AMC scheme data).
- **Models:**
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, Hugging Face).
  - Generation: `llama-3.3-70b-versatile` via Groq API.
- **External services:** Groq inference API only. ChromaDB runs embedded/local.
- **Secrets:** `GROQ_API_KEY` in `.env`. Never commit. Never log.

---

## 12. Success Metrics

| Metric | Target |
| --- | --- |
| Sample Q&A questions answered from context with a valid citation | 5–10 / 10 |
| Answers with exactly one source link | 100% |
| Answers ≤3 sentences | 100% |
| Advice/performance/PII questions correctly refused | 100% |
| Hallucinated facts (number not present in retrieved context) | 0 |
| Valid source URLs in `sources.csv` | 7 / 7 |
| Clean-clone setup time | < 10 min |
| Demo video length (if used) | ≤ 3 min |

### Evaluation set (minimum 10 queries)

1. Expense ratio — HDFC Large Cap Direct Growth
2. Exit load — HDFC Equity (Flexi Cap) Direct Growth
3. Minimum SIP — HDFC ELSS Tax Saver Direct Growth
4. Lock-in period — HDFC ELSS Tax Saver
5. Riskometer / benchmark — HDFC Balanced Advantage
6. Minimum lump sum — HDFC Small Cap Direct Growth
7. "How do I download my capital gains statement?"
8. NAV / "current price" (should be redirected to official page)
9. "Should I buy HDFC Small Cap?" (refusal)
10. "My PAN is ABCDE1234F, what are my returns?" (PII refusal)

---

## 13. Risks and Mitigations

| # | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| R1 | Groww page layout changes, scraping breaks | Ingestion fails | Committed snapshot; cleaning isolated in `ingest.py`; graceful fallback |
| R2 | Chunks orphan a number from its label | Wrong/hallucinated facts | Structure-aware chunking; never split a table row; sanity-check known facts |
| R3 | Hallucinated facts or links | Trust failure, the one thing that matters here | Context-only prompt, temperature ~0, similarity floor, URL allow-list verification |
| R4 | Groq free-tier rate limit mid-demo | Demo failure | Caching (FR15), response queue, and a recorded backup demo video |
| R5 | Model advises despite instructions | Policy violation | Deterministic pre-LLM guardrails catch advice/PII before generation |
| R6 | PII leaks into logs | Privacy violation | PII guardrail before logging; redaction in the logger; no input persistence |
| R7 | Scope creep beyond 5 schemes | Delays, weaker demo | Scope frozen in `config.py`; out-of-scope guardrail |
| R8 | Financial figures stale vs AMC site | Wrong info | `Last updated from sources:` stamp + link so the user can verify |

---

## 14. Out of Scope / Future Versions

- **v2:** widen to 2–3 AMCs; add KIM/SID PDFs with table-aware parsing.
- **v2:** multilingual (Hindi/Hinglish) queries; voice input.
- **v2:** per-scheme comparison view — *facts only*, never performance.
- **v3:** human-in-the-loop "was this answer correct?" feedback loop feeding a QA eval set.
- **v3:** scheduled re-scrape with a diff report on which facts changed.

---

## 15. Constraints Register (from the brief — must not be violated)

| # | Constraint | How the design satisfies it |
| --- | --- | --- |
| C1 | **Public sources only.** No third-party blogs. | Closed allow-list of 5 URLs in `config.py`; citation verifier rejects any other URL |
| C2 | **No PII.** Never accept/store PAN, Aadhaar, account no., OTP, email, phone. | PII guardrail rejects before logging; no persistence layer; disclaimer in UI |
| C3 | **No performance claims.** No computing/comparing returns. | Performance guardrail refuses; prompt forbids returns; link to factsheet instead |
| C4 | **Clarity & transparency.** ≤3 sentences + last-updated stamp. | Prompt contract + `citation.py` stamp; verified single link per answer |
| C5 | **One source link per answer.** | Generator contract + allow-list verification + single-link UI rendering |
| C6 | **Refuse opinionated/portfolio questions** with a polite message + educational link. | Guardrails table (§6.7) with a fixed refusal template |
| C7 | **Tiny UI:** welcome line + 3 examples + facts-only note. | §7 |
| C8 | **All RAG stages implemented** (load, chunk, embed, store). | §6, one module per stage, all surfaced in the UI expander |
| C9 | **Embeddings = all-MiniLM-L6-v2; VectorDB = ChromaDB.** | §6.3, §6.4 — fixed, not negotiable |

---

## 16. Acceptance Checklist

- [ ] `README.md` with setup steps, scope (AMC + 5 schemes), and known limits
- [ ] `data/sources.csv` lists all 7 source URLs (5 scraped + 2 hand-authored), publisher, fetch date
- [ ] `deliverables/sample_qa.md` has 5–10 queries with answers + links
- [ ] `deliverables/disclaimer.md` has the exact UI disclaimer text
- [ ] Working prototype link, or a ≤3-min demo video
- [ ] Every answer: ≤3 sentences, one link, last-updated stamp, disclaimer
- [ ] Advice / performance / PII queries refused politely with an educational link
- [ ] Pipeline expander shows rewrite → chunks + scores → guardrails → prompt → response
- [ ] Clean clone → running app in ≤5 commands
- [ ] No secrets committed; no PII in logs
