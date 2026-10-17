"""Stage 2 - Structure-aware chunking (ARCHITECTURE.md section 4.2)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src import config
from src.config import chunk_count_tokens, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from src.types import CROSS_SCHEME_KEY

__all__ = [
    "token_count",
    "extract_label",
    "split_into_units",
    "pack_units",
    "split_oversized_unit",
    "chunk_document",
    "run_chunking",
    "run_all",
]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
_KEY_VALUE_RE = re.compile(r"^([^:]+):\s*(.+)$")
TABLE_HEADER_HINTS = ("name", "sector", "instruments", "assets")

#: Longest string still treated as a label rather than as prose.
MAX_LABEL_CHARS = 60


def _split_table_cells(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def token_count(text: str) -> int:
    return chunk_count_tokens(text)
def extract_label(unit: str, section: str) -> str:
    if not unit:
        return section
    unit_low = unit.lower()
    if re.search(r'^#{1,6}\s+About\b', unit, re.MULTILINE):
        # The Groww "About <scheme>" overview is a summary card, not a fact
        # section: it restates NAV, minimums and exit load together, so the
        # keyword cascade below would label it "Exit load" and it would then
        # outrank the real exit-load chunk on scheme-name queries. Give it its
        # own label so it earns no fact-intent boost.
        return 'About'
    if 'fund management' in unit_low or re.search(
        r'^#{1,6}\s+\S[^\n]*(?:View details|\b\d{4}\s*-\s*Present\b)', unit, re.MULTILINE
    ):
        # The scheme pages' fund-manager blocks: either the "Fund management"
        # heading or a "<name> <dates> - Present View details" manager heading.
        # This runs ahead of the generic keyword cascade so a manager bio can
        # never be claimed by the 'nav'/'tax' substring rules, and the
        # fund_manager intent's label boost and guardrail grounding both key
        # on this label.
        return 'Fund management'
    if 'expense ratio' in unit_low and re.search(r'\d+\.\d+%', unit):
        return 'Expense ratio'
    if 'exit load' in unit_low:
        return 'Exit load'
    if 'lock-in' in unit_low or 'lock in' in unit_low:
        return 'Lock-in period'
    if 'minimum sip' in unit_low or 'min. for sip' in unit_low:
        return 'Minimum SIP'
    if 'minimum lump' in unit_low or 'min. for 1st' in unit_low:
        return 'Minimum Lump Sum'
    if 'riskometer' in unit_low or 'very high risk' in unit_low or 'high risk' in unit_low or 'moderate risk' in unit_low:
        return 'Riskometer'
    if 'benchmark' in unit_low or 'total return index' in unit_low:
        return 'Benchmark'
    if 'nav' in unit_low:
        return 'NAV'
    if 'capital gains' in unit_low:
        return 'Capital Gains Statement'
    if 'tax' in unit_low:
        return 'Tax'
    for line in unit.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(('#', '>', '|', '-', '*')):
            continue
        # A real label is a short term, not a sentence and not a document
        # attribute. Without these guards the document's own metadata header
        # (`> category: ELSS | scheme_key: ...`) becomes the label, and a table
        # row's column header becomes one too.
        if len(stripped) > MAX_LABEL_CHARS:
            continue
        m = _KEY_VALUE_RE.match(stripped)
        if m:
            key = m.group(1).strip()
            key = re.sub(r'[:\*]+$', '', key).strip()
            if key and len(key) <= MAX_LABEL_CHARS:
                return key
    for m in _BOLD_RE.finditer(unit):
        txt = m.group(1) or m.group(2)
        if txt:
            return txt.strip()
    return section


def split_into_units(section_body: str) -> list[str]:
    if not section_body:
        return []
    units = []
    lines = section_body.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _TABLE_ROW_RE.match(line):
            hb = i - 1
            while hb >= 0 and not lines[hb].strip():
                hb -= 1
            h = None
            s = None
            if hb >= 1 and _TABLE_ROW_RE.match(lines[hb - 1]) and _TABLE_ROW_RE.match(lines[hb]):
                h = lines[hb - 1]
                s = lines[hb]
            unit_lines = []
            if h is not None and s is not None:
                unit_lines.extend([h, s, line])
            else:
                unit_lines.append(line)
            units.append('\n'.join(unit_lines))
            i += 1
            continue
        if _LIST_ITEM_RE.match(line):
            j = i
            while j < n:
                l = lines[j]
                if not l.strip():
                    break
                if _HEADING_RE.match(l) or _TABLE_ROW_RE.match(l):
                    break
                j += 1
            units.append('\n'.join(lines[i:j]))
            i = j
            continue
        j = i
        while j < n:
            l = lines[j]
            if not l.strip():
                break
            if _HEADING_RE.match(l) or _TABLE_ROW_RE.match(l) or _LIST_ITEM_RE.match(l):
                if j > i:
                    break
            j += 1
        if j == i:
            j = i + 1
        units.append('\n'.join(lines[i:j]))
        i = j
    return [u for u in units if u.strip()]


def split_oversized_unit(unit: str, label: str) -> list[dict]:
    return [{"text": unit, "label": label, "tokens": token_count(unit)}]


def _is_heading_only(text: str) -> bool:
    """True when every non-blank line in ``text`` is a Markdown heading."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(_HEADING_RE.match(ln) for ln in lines)


def pack_units(units: list[tuple[str, str]], budget: int) -> list[dict]:
    chunks = []
    current_text = []
    for text, label in units:
        t = token_count(text)
        candidate = '\n\n'.join(current_text + [text]) if current_text else text
        if t > budget:
            # An oversized unit cannot be split. Carry a heading-only prefix
            # onto it rather than emitting a chunk that is nothing but a
            # heading, which would strand a section title (or a person's name)
            # away from the body it introduces.
            if current_text and _is_heading_only('\n\n'.join(current_text)):
                text = '\n\n'.join(current_text + [text])
                t = token_count(text)
                current_text = []
                chunks.append({"text": text, "label": extract_label(text, ""), "tokens": t})
                continue
            if current_text:
                combined = '\n\n'.join(current_text)
                chunks.append({"text": combined, "label": extract_label(combined, ""), "tokens": token_count(combined)})
                current_text = []
            chunks.append({"text": text, "label": label or extract_label(text, ""), "tokens": t})
            continue
        if current_text and token_count(candidate) > budget:
            if not _is_heading_only('\n\n'.join(current_text)):
                combined = '\n\n'.join(current_text)
                chunks.append({"text": combined, "label": extract_label(combined, ""), "tokens": token_count(combined)})
                current_text = []
        current_text.append(text)
    if current_text:
        combined = '\n\n'.join(current_text)
        chunks.append({"text": combined, "label": extract_label(combined, ""), "tokens": token_count(combined)})
    return chunks

def chunk_document_full(
    scheme_key: str,
    markdown: str,
    meta: dict[str, Any],
    start_index: int = 0,
) -> list[dict]:
    """Split one document into labelled chunks.

    Args:
        scheme_key: Canonical scheme key, used as the ``chunk_id`` namespace.
        markdown: The corpus document.
        meta: Header metadata parsed from the document's ``>`` block.
        start_index: First ``chunk_index`` to use. Two documents can share a
            ``scheme_key`` (the ELSS scheme has both a scraped page and a
            hand-authored lock-in document), and ``chunk_id`` must stay unique
            across that namespace because it is also the Chroma document id.
            :func:`run_chunking` therefore threads a per-scheme counter through.

    Returns:
        Chunk records following the schema in IMPLEMENTATION.md section 2.2.
    """
    chunks_out = []
    sections = re.split(r'\n(?=#{1,6}\s)', markdown)
    chunk_index = start_index
    display_name = meta.get('title') or meta.get('scheme_name') or scheme_key
    for sec in sections:
        if not sec.strip():
            continue
        m = _HEADING_RE.match(sec)
        section_title = m.group(2) if m else meta.get('title', scheme_key)
        units = split_into_units(sec)
        unit_pairs = []
        for u in units:
            lab = extract_label(u, section_title)
            unit_pairs.append((u, lab))
        packed = pack_units(unit_pairs, CHUNK_SIZE_TOKENS)
        for p in packed:
            text = p['text']
            if _is_heading_only(text):
                # A chunk that is nothing but a heading strands a section title
                # ("### Fund management") away from the body that lives in the
                # following sections. It carries no fact, so it is dropped: the
                # body chunks inherit the label through extract_label and stay
                # findable.
                continue
            tokens = token_count(text)
            label = p.get('label') or extract_label(text, section_title)
            chunk_id = f"{scheme_key}::{chunk_index:04d}"
            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            chunk = {
                "chunk_id": chunk_id,
                "text": text,
                "scheme_key": scheme_key,
                # The Phase 1 header block carries no scheme_name field; the H1
                # is the display name, and for a hand-authored document it is
                # the document title, which keeps the two ELSS rows
                # distinguishable (ARCHITECTURE.md section 5.2).
                "scheme_name": display_name,
                "category": meta.get('category', ''),
                "section": section_title,
                "label": label,
                "source_url": meta.get('source_url', ''),
                "source_title": meta.get('title') or display_name,
                "fetched_at": meta.get('fetched_at', ''),
                "chunk_index": chunk_index,
                "content_hash": f"sha256:{content_hash}",
                "tokens": tokens,
            }
            chunks_out.append(chunk)
            chunk_index += 1
    return chunks_out


def _read_corpus() -> list[tuple]:
    corpus_dir = config.PATHS['CORPUS_DIR']
    results = []
    for p in sorted(corpus_dir.glob('*.md')):
        if p.name == '.gitkeep':
            continue
        try:
            results.append((p, p.read_text(encoding='utf-8', errors='replace')))
        except Exception:
            continue
    return results


def _extract_header_meta(md: str) -> dict[str, Any]:
    meta = {}
    for line in md.splitlines():
        if line.startswith('> category:'):
            # Header format (IMPLEMENTATION.md section 1.2 step 8):
            #   > category: {category} | scheme_key: {...} | source_url: {...} | ...
            # The first field is a bare value; the rest are key: value pairs.
            _, _, rest = line.partition(':')
            segments = [segment.strip() for segment in rest.split('|')]
            if segments and segments[0]:
                meta['category'] = segments[0]
            for segment in segments[1:]:
                if ':' not in segment:
                    continue
                key, value = segment.split(':', 1)
                key = key.strip()
                if key in ('scheme_key', 'source_url', 'fetched_at'):
                    meta[key] = value.strip()
    for line in md.splitlines():
        if line.startswith('# '):
            meta['title'] = line[2:].strip()
            break
    return meta


def run_chunking() -> list[dict]:
    """Chunk every corpus document into ``data/chunks/chunks.jsonl``.

    ``chunk_index`` is numbered per ``scheme_key`` and continues across the
    documents that share one, so ``chunk_id`` stays unique in the namespace.

    Returns:
        Every chunk record written.
    """
    chunks_dir = config.PATHS['CHUNKS_DIR']
    chunks_file = config.PATHS.get('CHUNKS_FILE') or (chunks_dir / 'chunks.jsonl')
    chunks_dir.mkdir(parents=True, exist_ok=True)
    corp = _read_corpus()
    all_chunks = []
    next_index: dict[str, int] = {}
    for path, md in corp:
        meta = _extract_header_meta(md)
        scheme_key = meta.get('scheme_key') or path.stem
        chunks = chunk_document_full(
            scheme_key,
            md,
            meta,
            start_index=next_index.get(scheme_key, 0),
        )
        next_index[scheme_key] = next_index.get(scheme_key, 0) + len(chunks)
        all_chunks.extend(chunks)
    seen: set[str] = set()
    collisions = sorted({c['chunk_id'] for c in all_chunks if c['chunk_id'] in seen or seen.add(c['chunk_id'])})
    if collisions:
        raise ValueError(f'duplicate chunk_id produced: {collisions[:5]}')
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    if all_chunks:
        stats = {'docs': len(corp), 'chunks': len(all_chunks)}
        stats['tokens_mean'] = sum(c.get('tokens', 0) for c in all_chunks) / len(all_chunks)
        print(json.dumps(stats))
    else:
        print(json.dumps({'docs': len(corp), 'chunks': 0}))
    return all_chunks


def run_all(*args, **kwargs) -> list[dict]:
    return run_chunking()


chunk_document = chunk_document_full
