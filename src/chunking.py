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
        if not stripped or stripped.startswith(('#',)):
            continue
        m = _KEY_VALUE_RE.match(stripped)
        if m:
            key = m.group(1).strip()
            key = re.sub(r'[:\*]+$', '', key).strip()
            if key:
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


def pack_units(units: list[tuple[str, str]], budget: int) -> list[dict]:
    chunks = []
    current_text = []
    for text, label in units:
        t = token_count(text)
        candidate = '\n\n'.join(current_text + [text]) if current_text else text
        if t > budget:
            if current_text:
                combined = '\n\n'.join(current_text)
                chunks.append({"text": combined, "label": extract_label(combined, ""), "tokens": token_count(combined)})
                current_text = []
            chunks.append({"text": text, "label": label or extract_label(text, ""), "tokens": t})
            continue
        if current_text and token_count(candidate) > budget:
            combined = '\n\n'.join(current_text)
            chunks.append({"text": combined, "label": extract_label(combined, ""), "tokens": token_count(combined)})
            current_text = []
        current_text.append(text)
    if current_text:
        combined = '\n\n'.join(current_text)
        chunks.append({"text": combined, "label": extract_label(combined, ""), "tokens": token_count(combined)})
    return chunks

def chunk_document_full(scheme_key: str, markdown: str, meta: dict[str, Any]) -> list[dict]:
    chunks_out = []
    sections = re.split(r'\n(?=#{1,6}\s)', markdown)
    chunk_index = 0
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
            tokens = token_count(text)
            label = p.get('label') or extract_label(text, section_title)
            chunk_id = f"{scheme_key}::{chunk_index:04d}"
            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            chunk = {
                "chunk_id": chunk_id,
                "text": text,
                "scheme_key": scheme_key,
                "scheme_name": meta.get('scheme_name', ''),
                "category": meta.get('category', ''),
                "section": section_title,
                "label": label,
                "source_url": meta.get('source_url', ''),
                "source_title": meta.get('title', meta.get('scheme_name', '')),
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
            try:
                rest = line[10:]
                for pr in rest.split('|'):
                    pr = pr.strip()
                    if ':' in pr:
                        k, v = pr.split(':', 1)
                        k = k.strip()
                        v = v.strip()
                        if k == 'category':
                            meta['category'] = v
                        if k == 'scheme_key':
                            meta['scheme_key'] = v
                        if k == 'source_url':
                            meta['source_url'] = v
                        if k == 'fetched_at':
                            meta['fetched_at'] = v
            except Exception:
                pass
    for line in md.splitlines():
        if line.startswith('# '):
            meta['title'] = line[2:].strip()
            break
    return meta


def run_chunking() -> list[dict]:
    chunks_dir = config.PATHS['CHUNKS_DIR']
    chunks_file = config.PATHS.get('CHUNKS_FILE') or (chunks_dir / 'chunks.jsonl')
    chunks_dir.mkdir(parents=True, exist_ok=True)
    corp = _read_corpus()
    all_chunks = []
    for path, md in corp:
        meta = _extract_header_meta(md)
        scheme_key = meta.get('scheme_key') or path.stem
        chunks = chunk_document_full(scheme_key, md, meta)
        all_chunks.extend(chunks)
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
