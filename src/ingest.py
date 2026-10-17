"""Stage 1 - Document loading (ARCHITECTURE.md section 4.1).

Turns the allow-listed public pages into committed Markdown snapshots plus
``data/sources.csv``. Network-free after the first run: every page is fetched
once, cleaned, and written to ``data/corpus/``. A later run reuses the committed
snapshot if the network is unavailable, and says so via ``stale=true``.

Two design points are worth knowing before changing anything here:

1. **The corpus is deliberately stripped of performance data.** Every Groww page
   ships a live return calculator and a "Returns and rankings" table. PRD C3
   forbids performance claims, so those blocks are deleted at ingest rather than
   being left for the generator to avoid. Data that is never chunked cannot leak
   into an answer. The rules are in ``config.SCRAP_DROP_*`` and are deliberately
   narrow - the word "return" is never matched on its own, because the benchmark
   is legitimately "NIFTY 100 Total Return Index".

2. **Two documents are hand-transcribed, not scraped.** Groww does not publish
   the ELSS lock-in or the capital-gains-statement procedure at all, and the
   official AMC domain answers automated requests with 403. Those live in
   ``config.AUTHORED_SOURCES``; they bypass the cleaner entirely and carry an
   ``authored: true`` header so a reader is never misled about their provenance.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from markdownify import markdownify
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config
from src.types import AuthoredSource, SourceSpec

__all__ = [
    "fetch_page",
    "clean_html_to_markdown",
    "clean_html_to_markdown_detailed",
    "build_document",
    "build_authored_document",
    "ingest_scheme",
    "ingest_authored",
    "write_sources_csv",
    "run_ingest",
    "run_all",
    "inspect_corpus",
]


# --------------------------------------------------------------------------- #
# Block model
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_TABLE_DELIM_RE = re.compile(r"^\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?$")
_ZERO_WIDTH_RE = re.compile(r"[‌‍﻿]")
_LINK_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^>]+)>")
_FETCHED_AT_RE = re.compile(r"^fetched_at:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


@dataclass
class _Block:
    """One blank-line-delimited chunk of Markdown plus what we know about it."""

    text: str
    heading_level: int | None = None
    heading_text: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def is_heading(self) -> bool:
        return self.heading_level is not None

    @property
    def is_table(self) -> bool:
        return self._table_header_cells is not None

    @property
    def _table_header_cells(self) -> list[str] | None:
        """Header-row cells when this block is a Markdown table, else None."""
        if len(self.lines) < 2:
            return None
        if not self.lines[0].lstrip().startswith("|"):
            return None
        if not _TABLE_DELIM_RE.match(self.lines[1].strip()):
            return None
        return _table_cells(self.lines[0])

    @property
    def normalised_heading(self) -> str:
        """Heading text lower-cased, punctuation removed, whitespace collapsed."""
        return re.sub(r"[^\w\s]+", " ", self.heading_text.lower()).strip()


def _table_cells(line: str) -> list[str]:
    """Split a Markdown table row into trimmed, non-empty cell strings."""
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _build_session() -> requests.Session:
    """Session with the retry/backoff policy from ARCHITECTURE section 4.1."""
    retry = Retry(
        total=config.HTTP_RETRIES,
        backoff_factor=config.HTTP_BACKOFF_FACTOR,
        status_forcelist=list(config.HTTP_RETRY_STATUSES),
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": config.HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-IN,en;q=0.9",
        }
    )
    return session


def fetch_page(spec: SourceSpec) -> tuple[str | None, int | None, str | None]:
    """GET ``spec.url``.

    Returns:
        ``(html, status_code, error)``. Never raises - a dead network must degrade
        to the committed snapshot, not abort the build (PRD NFR4). ``html`` is
        None unless the response was a 200 with a non-empty body.

    Raises:
        Nothing. Unexpected exceptions are returned in ``error``.
    """
    if spec.url not in config.ALLOWED_URLS:
        return None, None, f"refused: URL not in allow-list ({spec.url})"
    session: requests.Session | None = None
    try:
        session = _build_session()
        response = session.get(spec.url, timeout=config.HTTP_TIMEOUT)
        status = int(response.status_code)
        if status != 200:
            return None, status, f"HTTP {status}"
        # response.encoding can be wrong for pages served without a charset;
        # apparent_encoding is the ARCHITECTURE 4.1 fallback.
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
        if not response.text.strip():
            return None, status, "empty response body"
        return response.text, status, None
    except requests.RequestException as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive, fetch must never raise
        return None, None, f"unexpected {type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            session.close()


# --------------------------------------------------------------------------- #
# HTML -> Markdown
# --------------------------------------------------------------------------- #


def _select_root(soup: BeautifulSoup) -> tuple[Tag, str]:
    """First element matching :data:`config.SCRAP_ROOT_SELECTORS`.

    Returns:
        ``(element, selector_used)``. The selector is recorded for debugging,
        because a layout change on the publisher's side is the usual cause of a
        suddenly thin corpus.

    Raises:
        RuntimeError: if even ``body`` is missing, i.e. the response was not HTML.
    """
    for selector in config.SCRAP_ROOT_SELECTORS:
        try:
            found = soup.select_one(selector)
        except Exception:  # a malformed selector in config must not kill ingest
            continue
        if isinstance(found, Tag):
            return found, selector
    raise RuntimeError("no root element found (response was not an HTML document)")


def _marker_count(element: Tag) -> int:
    """How many of :data:`config.SCRAP_CONTENT_MARKERS` this element's text holds."""
    text = element.get_text(" ", strip=True)
    return sum(1 for marker in config.SCRAP_CONTENT_MARKERS if marker in text)


def _narrow_to_content(root: Tag, depth: int = 0) -> Tag:
    """Descend to the descendant that actually holds the fund facts.

    ``#__next`` contains the whole React app, mega-menu included, so selecting it
    is not enough. Keep stepping into a child while *exactly one* child carries
    at least :data:`config.SCRAP_CONTENT_MIN_MARKERS` of
    :data:`config.SCRAP_CONTENT_MARKERS`:

        #__next  -> [loader (0 markers), #root (5)]   -> descend into #root
        #root    -> [menu (0), scheme page (5)]       -> descend into the page
        page     -> [no child has 2+ markers]         -> stop

    Stopping as soon as a node qualifies would be wrong: ``#root`` qualifies and
    still wraps the menu. Requiring the choice to be *unique* means an ambiguous
    node stops the walk instead of guessing, and unlike matching the menu's
    hashed CSS class the whole rule survives a redesign.

    Returns:
        The content element, or ``root`` unchanged when nothing qualifies -
        narrowing is an optimisation, and refusing to guess is always correct.
    """
    if depth >= config.SCRAP_CONTENT_MAX_DEPTH:
        return root
    children = root.find_all(recursive=False)
    if len(children) < 2:
        # A single-child chain is just a wrapper; keep descending through it.
        if len(children) == 1:
            return _narrow_to_content(children[0], depth + 1)
        return root
    qualifying = [child for child in children if _marker_count(child) >= config.SCRAP_CONTENT_MIN_MARKERS]
    if len(qualifying) == 1:
        return _narrow_to_content(qualifying[0], depth + 1)
    return root


def _separate_sibling_items(root: Tag) -> int:
    """Insert a newline between adjacent inline item siblings, in place.

    Groww renders the identity and key-fact rows as flex rows of sibling
    ``<span>``/``<a>`` elements, and markdownify joins siblings with nothing in
    between, fusing each label onto its own value::

        Fund benchmarkBSE 250 SmallCap Total Return Index
        EquityLarge CapVery High Risk
        CustodianDeutsche Bank

    A fused pair is worse than untidy: an answer would quote the benchmark
    run-together, and the guardrail that matches a chunk's expected label would
    not find "benchmark". Restoring the break returns the document to the
    ``label`` / ``value`` shape Phase 2 is designed around.

    Sentence-bearing parents are excluded (``SCRAP_NEVER_SPLIT_PARENTS``), and so
    is running prose in general, so no fact is split mid-clause.

    Returns:
        How many separators were inserted, for the run log.
    """
    split_tags = set(config.SCRAP_INLINE_SPLIT_TAGS)
    skip_parents = set(config.SCRAP_NEVER_SPLIT_PARENTS)
    skip_texts = set(config.SCRAP_INLINE_SPLIT_SKIP_TEXTS)
    inserted = 0
    for parent in list(root.find_all(True)):
        if parent.name in skip_parents:
            continue
        children = [child for child in parent.find_all(recursive=False) if isinstance(child, Tag)]
        if len(children) < 2:
            continue
        for child in children[:-1]:
            if child.name not in split_tags:
                continue
            text = child.get_text(" ", strip=True)
            if not text or text in skip_texts:
                # A unit fragment belongs to the item before it: splitting
                # "+8.71" from "%" would defeat the performance filter.
                continue
            if len(text) > config.SCRAP_INLINE_SPLIT_MAX_CHARS:
                continue
            child.insert_after("\n")
            inserted += 1
    return inserted


def _strip_by_selector(root: Tag) -> None:
    """Remove chrome and non-content elements, in place."""
    for selector in config.SCRAP_STRIP_SELECTORS:
        for tag in root.select(selector):
            tag.decompose()


def _split_blocks(markdown: str) -> list[_Block]:
    """Split Markdown into blank-line-delimited blocks, forcing a break at every
    heading *and* immediately after one.

    Both breaks matter. A break at a heading gives every section a predictable
    extent, which is what :func:`_drop_performance_sections` relies on. The break
    *after* a heading matters just as much: markdownify emits a page ``<h1>``
    followed directly by the scheme widget with no blank line between them, and
    without the break the two arrive as one block whose first line is a heading -
    so the whole widget is then classified as a heading block and skipped by the
    line-level filters. A heading is structure; its content belongs to a body
    block.
    """
    groups: list[list[str]] = [[]]
    pending_break = False
    for line in markdown.splitlines():
        is_heading = _HEADING_RE.match(line) is not None
        has_content = any(ln.strip() for group in groups for ln in group)
        if line.strip() and has_content and (is_heading or pending_break):
            groups.append([line])
            # A heading that *starts* a group must still arm the break that
            # follows it, which is the whole point of this loop.
            pending_break = is_heading
            continue
        if is_heading:
            pending_break = True
        groups[-1].append(line)

    blocks: list[_Block] = []
    for group in groups:
        text = "\n".join(group).strip("\n")
        if not text.strip():
            continue
        body_lines = [ln for ln in text.splitlines() if ln.strip()]
        first = body_lines[0]
        heading = _HEADING_RE.match(first)
        if heading:
            blocks.append(
                _Block(
                    text=text,
                    heading_level=len(heading.group(1)),
                    heading_text=heading.group(2),
                    lines=body_lines,
                )
            )
        else:
            blocks.append(_Block(text=text, lines=body_lines))
    return blocks


def _drop_performance_sections(blocks: list[_Block]) -> list[_Block]:
    """Remove a section from its heading up to the next same-or-higher heading."""
    wanted = {re.sub(r"[^\w\s]+", " ", h.lower()).strip() for h in config.SCRAP_DROP_SECTION_HEADINGS}
    kept: list[_Block] = []
    drop_at_level: int | None = None
    for block in blocks:
        if block.is_heading:
            assert block.heading_level is not None
            if drop_at_level is not None and block.heading_level <= drop_at_level:
                drop_at_level = None
            if block.normalised_heading in wanted:
                drop_at_level = block.heading_level
                continue
        elif drop_at_level is not None:
            continue
        kept.append(block)
    return kept


def _is_performance_table(block: _Block) -> bool:
    """True when a table's header row identifies it as returns/rankings.

    Requires :data:`config.SCRAP_DROP_TABLE_MIN_TOKENS` performance-ish header
    tokens and no :data:`config.SCRAP_KEEP_TABLE_MARKERS`, so exit-load and fee
    tables survive even if they gain a stray "1 year" column.
    """
    cells = block._table_header_cells
    if cells is None:
        return False
    header = " | ".join(cells).lower()
    if any(marker.lower() in header for marker in config.SCRAP_DROP_TABLE_MARKERS):
        return True
    if any(marker.lower() in header for marker in config.SCRAP_KEEP_TABLE_MARKERS):
        return False
    hits = sum(1 for token in config.SCRAP_DROP_TABLE_TOKENS if token.lower() in header)
    return hits >= config.SCRAP_DROP_TABLE_MIN_TOKENS


def _cut_at_drop_text(block: _Block) -> _Block | None:
    """Truncate a block at the first "Return calculator"-style marker.

    Groww packs the page header into a single pipe-separated line, so the
    calculator's own controls ("Return calculator | Monthly SIP | One time |
    Monthly investment") sit on the same line as the facts we want to keep
    (NAV, AUM, expense ratio, minimum SIP). Dropping the whole block would lose
    those, so only the tail is cut. Returns None when nothing useful remains.

    Headings and tables pass through untouched: a heading is structure, and a
    table has already had its own performance check in
    :func:`_is_performance_table`.
    """
    if block.is_heading or block.is_table:
        return block
    cut = len(block.text)
    for pattern in config.SCRAP_CUT_BLOCK_AT_TEXT:
        match = re.search(pattern, block.text, re.IGNORECASE)
        if match and match.start() < cut:
            cut = match.start()
    if cut >= len(block.text):
        return block
    kept = block.text[:cut].rstrip()
    if len(kept) < 20:
        return None
    return _Block(text=kept, lines=kept.splitlines())


_PERFORMANCE_PATTERNS: list[re.Pattern[str]] | None = None


def _performance_patterns() -> list[re.Pattern[str]]:
    """Compiled :data:`config.SCRAP_RETURN_FIGURE_LINE_PATTERNS`, built once."""
    global _PERFORMANCE_PATTERNS
    if _PERFORMANCE_PATTERNS is None:
        _PERFORMANCE_PATTERNS = [
            re.compile(pattern, re.IGNORECASE) for pattern in config.SCRAP_RETURN_FIGURE_LINE_PATTERNS
        ]
    return _PERFORMANCE_PATTERNS


def _is_return_figure_line(line: str) -> bool:
    """True when a whole line is part of the header return widget."""
    return any(pattern.match(line) for pattern in _performance_patterns())


def _drop_return_figure_lines(block: _Block, audit: list[str] | None = None) -> _Block | None:
    """Delete standalone return-widget lines from a block, keeping the rest.

    The header widget interleaves performance with the facts we need, one item
    per line, so it cannot be dropped as a unit. Sign is the discriminator: Groww
    writes performance as ``+8.71%`` while the expense ratio is ``1.03%`` on its
    own line. Tables and headings are left alone - a holdings row of unsigned
    percentages is data, and a heading is structure.

    Args:
        block: The block to filter.
        audit: When given, every removed line is appended, so the run can print
            exactly what the performance filter deleted. A rule this blunt
            deserves to be reviewable.
    """
    if block.is_table or block.is_heading:
        return block
    patterns = _performance_patterns()
    kept: list[str] = []
    for line in block.lines:
        if any(pattern.match(line) for pattern in patterns):
            if audit is not None:
                audit.append(line.strip())
            continue
        kept.append(line)
    if len(kept) == len(block.lines):
        return block
    text = "\n".join(kept).strip()
    if not text:
        return None
    return _Block(
        text=text,
        heading_level=block.heading_level,
        heading_text=block.heading_text,
        lines=kept,
    )


def _stop_at_marker(blocks: list[_Block]) -> list[_Block]:
    """Truncate the corpus at the first footer marker line, discarding the rest.

    :func:`_cut_at_drop_text` trims the tail of the block it matches, which is
    right for a mid-line marker like "Return calculator" but wrong here. Groww
    emits the whole disclosure-plus-footer run as *one* block: the same block
    holds the fields worth keeping (custodian, RTA, registered office) and then
    several hundred lines of footer. Truncating at the first marker *line* keeps
    the disclosure fields and drops the footer, and because the footer is
    trailing, everything from the marker onward goes.
    """
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config.SCRAP_STOP_AT_BLOCK_TEXT]
    if not patterns:
        return blocks
    kept: list[_Block] = []
    for block in blocks:
        if block.is_table:
            kept.append(block)
            continue
        cut_index: int | None = None
        for index, line in enumerate(block.lines):
            stripped = line.strip()
            if any(pattern.match(stripped) for pattern in patterns):
                cut_index = index
                break
        if cut_index is None:
            kept.append(block)
            continue
        if cut_index > 0:
            lines = block.lines[:cut_index]
            text = "\n".join(lines).strip()
            if len(text) >= 20:
                kept.append(_Block(text=text, lines=lines))
        break
    return kept


def _reattach_unit_fragments(block: _Block) -> _Block:
    """Weld a trailing unit fragment back onto the figure it belongs to.

    Groww splits a figure across sibling *block* elements, so markdownify emits
    ``+8.71`` and ``%`` as two lines. Left alone, the line-level performance
    filter no longer recognises ``+8.71%`` as a return figure and the number
    survives into the corpus. Reattaching first restores the real token, which
    both fixes the leak and improves the kept facts (``1.03`` + ``%`` becomes the
    proper ``1.03%``).
    """
    if block.is_table or block.is_heading:
        return block
    fragments = set(config.SCRAP_INLINE_SPLIT_SKIP_TEXTS)
    lines: list[str] = []
    for line in block.lines:
        stripped = line.strip()
        previous = lines[-1].strip() if lines else ""
        if (
            stripped in fragments
            and previous
            and not previous.startswith(("|", "#", ">"))
            and any(char.isdigit() for char in previous)
        ):
            lines[-1] = f"{lines[-1].rstrip()}{stripped}"
            continue
        lines.append(line)
    if len(lines) == len(block.lines):
        return block
    text = "\n".join(lines).strip()
    if not text:
        return block
    return _Block(
        text=text,
        heading_level=block.heading_level,
        heading_text=block.heading_text,
        lines=lines,
    )


def _drop_labelled_pairs(blocks: list[_Block]) -> list[_Block]:
    """Remove a ``label`` / ``value`` pair whose label is a ranking.

    Groww publishes ``Rank (total assets)`` / ``#2 in India`` in the disclosure
    block. It is a published fact, but the decision taken for this corpus was to
    keep rankings out alongside returns.

    The pair is dropped across the block list rather than inside one block, because
    the two halves are not always adjacent in the same block: on the live pages the
    disclosure run arrives as a single block, while a label and value in separate
    divs arrive as two blocks with a blank line between them. A per-block rule
    would delete the label in the first shape and leave the orphaned value in the
    second.

    The label patterns are anchored, which is what keeps them from matching a value
    or a sentence that merely contains the word.
    """
    patterns = [re.compile(p, re.IGNORECASE) for p in config.SCRAP_DROP_STACKED_LABELS]
    if not patterns:
        return blocks
    kept: list[_Block] = []
    skip_value = False
    for block in blocks:
        if block.is_table or block.is_heading:
            kept.append(block)
            continue
        if skip_value:
            skip_value = False
            if _looks_like_value(block.text.strip()):
                continue
        remaining: list[str] = []
        changed = False
        dropped_label = False
        for line in block.lines:
            stripped = line.strip()
            if dropped_label:
                if not stripped:
                    # Keep the blank separator but stay armed: markdownify puts one
                    # between the label and the value, and a lone "#2 in India" left
                    # in the corpus is a ranking the decision was to exclude.
                    remaining.append(line)
                    continue
                dropped_label = False
                if _looks_like_value(stripped):
                    changed = True
                    continue
                remaining.append(line)
                continue
            if any(pattern.match(stripped) for pattern in patterns):
                dropped_label = True
                changed = True
                continue
            remaining.append(line)
        skip_value = dropped_label
        if not changed:
            kept.append(block)
            continue
        if remaining:
            kept.append(_Block(text="\n".join(remaining).strip(), lines=remaining))
    return kept


def _drop_trailing_runs(blocks: list[_Block]) -> list[_Block]:
    """Cut a block at a labelled run that is always its tail.

    The fund-manager blocks end with "Also manages these schemes" followed by the
    names of every other scheme the manager runs. Two problems with keeping it:
    the run is long enough to push a manager chunk past the token budget, and the
    other-scheme names it contains make one scheme's manager block match another
    scheme's query. The label is always trailing within its block, so the block is
    cut at the label line and everything after it is discarded.
    """
    labels = [
        re.compile(rf"^\s*{re.escape(label)}\s*$", re.IGNORECASE)
        for label in config.SCRAP_DROP_TRAILING_RUN_LABELS
    ]
    if not labels:
        return blocks
    kept: list[_Block] = []
    for block in blocks:
        if block.is_table or block.is_heading:
            kept.append(block)
            continue
        cut: int | None = None
        for index, line in enumerate(block.lines):
            if any(pattern.match(line.strip()) for pattern in labels):
                cut = index
                break
        if cut is None:
            kept.append(block)
            continue
        lines = block.lines[:cut]
        text = "\n".join(lines).strip()
        if text:
            kept.append(
                _Block(
                    text=text,
                    heading_level=block.heading_level,
                    heading_text=block.heading_text,
                    lines=lines,
                )
            )
    return kept


def _drop_widget_blocks(blocks: list[_Block]) -> list[_Block]:
    """Remove short blocks that are pure site chrome (cookie banners, badges)."""
    patterns = [re.compile(p, re.IGNORECASE) for p in config.SCRAP_DROP_TEXT_PATTERNS]
    line_patterns = [re.compile(p, re.IGNORECASE) for p in config.SCRAP_DROP_LINE_PATTERNS]
    max_chars = config.SCRAP_DROP_BLOCK_MAX_CHARS

    kept: list[_Block] = []
    for block in blocks:
        if block.is_table:
            kept.append(block)
            continue
        if len(block.text) <= max_chars and any(p.search(block.text) for p in patterns):
            continue
        lines = [ln for ln in block.lines if not any(p.search(ln) for p in line_patterns)]
        if not lines:
            continue
        text = "\n".join(lines).strip()
        if not text:
            continue
        kept.append(
            _Block(
                text=text,
                heading_level=block.heading_level,
                heading_text=block.heading_text,
                lines=lines,
            )
        )
    return kept


def _scrub_residual_lines(markdown: str) -> str:
    """Last-resort line deletion for unambiguous site chrome.

    Runs after the structural passes. Each pattern is chrome that must never
    appear in a facts corpus; the exit-load, fee and benchmark blocks cannot
    contain them, so the false-positive risk is nil.
    """
    patterns = [re.compile(p, re.IGNORECASE) for p in config.SCRAP_SCRUB_LINE_PATTERNS]
    return "\n".join(ln for ln in markdown.splitlines() if not any(p.search(ln) for p in patterns))


def _scrub_about_blurb(markdown: str) -> str:
    """Remove the fund-house values the "About <scheme>" summary mislabels.

    The publisher's summary reuses HDFC Mutual Fund's own launch date and total AUM
    for every scheme, and names a single fund manager even when the manager section
    lists two, so those sentences contradict the scheme-level structured sections.
    The remaining summary facts (NAV, risk, minimums, exit load) agree with the
    structured sections and are kept, which is why this removes sentences and one
    clause rather than a whole block. Sentences are split with
    :func:`config.split_sentences`, which protects the decimals and abbreviations in
    the kept facts.
    """
    phrases = [p.lower() for p in config.SCRAP_DROP_ABOUT_SENTENCE_PHRASES]
    rewrites = [(re.compile(p), r) for p, r in config.SCRAP_ABOUT_CLAUSE_REWRITES]
    lines: list[str] = []
    for line in markdown.splitlines():
        if any(phrase in line.lower() for phrase in phrases):
            line = " ".join(
                sentence
                for sentence in config.split_sentences(line)
                if not any(phrase in sentence.lower() for phrase in phrases)
            )
        for pattern, replacement in rewrites:
            line = pattern.sub(replacement, line)
        lines.append(line)
    return "\n".join(lines)


def _derelink(markdown: str) -> str:
    """Reduce Markdown links and images to their text.

    Groww emits relative hrefs to ``/stocks/...`` and to
    ``/mutual-funds/<other-amc>/...``. An href inside a chunk is a
    non-allow-listed URL one copy-paste away from the citation, and the
    cross-fund names in those hrefs would trip the out-of-scope AMC aliases
    during retrieval. The visible label is what carries the fact, so the label
    is kept.
    """
    if not config.SCRAP_DEREFERENCE_LINKS:
        return markdown
    markdown = _LINK_RE.sub("", markdown)  # images: drop entirely
    markdown = _MD_LINK_RE.sub(r"\1", markdown)
    return _AUTOLINK_RE.sub(r"\1", markdown)


def _normalise_whitespace(markdown: str) -> str:
    """NBSP and zero-width removal, blank-line collapsing, trailing-space strip."""
    text = markdown.replace("\u00a0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_html_to_markdown_detailed(html: str) -> tuple[str, str]:
    """Clean HTML and report which root selector was used.

    Returns:
        ``(markdown, root_selector)``.

    Raises:
        ValueError: if fewer than :data:`config.MIN_DOC_CHARS` characters survive.
            That is the layout-change signal from ARCHITECTURE 4.1 - a page that
            used to yield 17k characters and now yields 200 has almost certainly
            stopped being the content page.
    """
    soup = BeautifulSoup(html, "lxml")
    root, selector = _select_root(soup)
    root = _narrow_to_content(root)
    _separate_sibling_items(root)
    _strip_by_selector(root)
    markdown = markdownify(str(root), heading_style="ATX", strip=["script", "style"])

    blocks = _split_blocks(markdown)
    blocks = _drop_performance_sections(blocks)
    blocks = [b for b in blocks if not _is_performance_table(b)]
    cut_blocks = [_cut_at_drop_text(b) for b in blocks]
    blocks = [b for b in cut_blocks if b is not None]
    blocks = _stop_at_marker(blocks)
    blocks = [b for b in (_reattach_unit_fragments(b) for b in blocks) if b is not None]
    blocks = [b for b in (_drop_return_figure_lines(b) for b in blocks) if b is not None]
    blocks = _drop_labelled_pairs(blocks)
    blocks = _drop_trailing_runs(blocks)
    blocks = _drop_widget_blocks(blocks)
    markdown = "\n\n".join(b.text for b in blocks)

    markdown = _derelink(markdown)
    markdown = _scrub_residual_lines(markdown)
    markdown = _scrub_about_blurb(markdown)
    markdown = _normalise_whitespace(markdown)

    if len(markdown) < config.MIN_DOC_CHARS:
        raise ValueError(
            f"extracted only {len(markdown)} chars from {selector} "
            f"(minimum {config.MIN_DOC_CHARS}); treating as a layout change"
        )
    return markdown, selector


def clean_html_to_markdown(html: str) -> str:
    """Clean one page of HTML to Markdown.

    Raises:
        ValueError: if the extracted text is shorter than
            :data:`config.MIN_DOC_CHARS` - see :func:`clean_html_to_markdown_detailed`.
    """
    return clean_html_to_markdown_detailed(html)[0]


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #


def build_document(spec: SourceSpec, markdown_body: str, fetched_at: str) -> str:
    """Prepend the H1 title and metadata header, collapse blank lines.

    The header format is fixed by IMPLEMENTATION 1.2 because Phase 2 chunking
    relies on the scheme being recoverable from the text alone, without a
    metadata lookup.
    """
    header = (
        f"# {spec.scheme_name}\n"
        f"> category: {spec.category} | scheme_key: {spec.scheme_key} | "
        f"source_url: {spec.url} | fetched_at: {fetched_at}"
    )
    document = f"{header}\n\n{markdown_body.strip()}\n"
    return _normalise_whitespace(document) + "\n"


def build_authored_document(doc: AuthoredSource) -> str:
    """Assemble a transcribed document, stating its provenance in the header.

    ``authored: true`` and ``live_fetch_status`` are part of the document itself,
    not just of ``sources.csv``, so that anyone opening the file - or any chunk
    quoted in the demo - can see that it was transcribed from the cited page
    rather than fetched.
    """
    header = (
        f"# {doc.title}\n"
        f"> category: {doc.category} | scheme_key: {doc.chunk_scheme_key} | "
        f"source_url: {doc.url} | fetched_at: {doc.transcribed_on} | "
        f"authored: true | live_fetch_status: {doc.live_fetch_status} | "
        f"publisher: {doc.publisher}"
    )
    body = doc.markdown_body.strip()
    notice = (
        "This document was transcribed by hand from the official HDFC Asset "
        "Management page cited above, which refuses automated requests "
        f"(HTTP {doc.live_fetch_status}). It was not scraped, and it carries no "
        "performance figures."
    )
    document = f"{header}\n\n> note: {notice}\n\n{body}\n"
    return _normalise_whitespace(document) + "\n"


def corpus_path(doc_id: str) -> Path:
    """Absolute path of a corpus file for ``doc_id``."""
    return config.PATHS["CORPUS_DIR"] / f"{doc_id}.md"


def snapshot_fetched_at(path: Path) -> str | None:
    """``fetched_at`` recorded in an existing snapshot, or None if unreadable.

    Reusing a snapshot must keep the date *inside* the file, otherwise the
    "Last updated from sources" stamp in the UI starts claiming freshness the
    content does not have (IMPLEMENTATION 1.2 step 6).
    """
    try:
        match = _FETCHED_AT_RE.search(path.read_text(encoding=config.CORPUS_ENCODING))
    except OSError:
        return None
    return match.group(1) if match else None


def residual_noise(markdown: str) -> list[str]:
    """Site-chrome phrases still present in a document.

    The Phase 1 Definition of Done asserts none survive. Exposed so the check
    runs inside the pipeline rather than only in a shell one-liner.
    """
    found: list[str] = []
    lowered = markdown.lower()
    for pattern in config.RESIDUAL_NOISE_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            found.append(pattern)
    return found


def performance_leaks(markdown: str) -> list[str]:
    """Performance phrases still present in a document.

    Like :func:`residual_noise`, this turns a Phase 1 exit criterion into
    something the code itself verifies.
    """
    found: list[str] = []
    lowered = markdown.lower()
    for pattern in config.PERFORMANCE_LEAK_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            found.append(pattern)
    return found


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def _row(
    *,
    scheme_key: str,
    scheme_name: str,
    category: str,
    url: str,
    publisher: str,
    fetched_at: str,
    http_status: int | str,
    stale: bool,
) -> dict[str, Any]:
    """One ``sources.csv`` record, in the column order of ARCHITECTURE 5.2."""
    return {
        "scheme_key": scheme_key,
        "scheme_name": scheme_name,
        "category": category,
        "url": url,
        "publisher": publisher,
        "fetched_at": fetched_at,
        "http_status": http_status,
        "stale": stale,
    }


def _write_corpus(doc_id: str, document: str) -> None:
    """Write a snapshot, creating ``data/corpus/`` on first use."""
    config.PATHS["CORPUS_DIR"].mkdir(parents=True, exist_ok=True)
    path = corpus_path(doc_id)
    with path.open("w", encoding=config.CORPUS_ENCODING, newline=config.CORPUS_NEWLINE) as handle:
        handle.write(document)


def ingest_scheme(spec: SourceSpec) -> dict[str, Any]:
    """Fetch, clean and store one scheme page, or reuse the committed snapshot.

    Returns:
        A ``sources.csv`` row. ``stale`` is True whenever the committed snapshot
        was kept - because the network failed, or because cleaning produced too
        little text to be the real page. Never raises.
    """
    path = corpus_path(spec.scheme_key)
    html, status, error = fetch_page(spec)

    if html is not None:
        try:
            body, _root = clean_html_to_markdown_detailed(html)
        except (ValueError, RuntimeError) as exc:
            error = f"clean failed: {exc}"
        else:
            document = build_document(spec, body, date.today().isoformat())
            try:
                _write_corpus(spec.scheme_key, document)
            except OSError as exc:
                error = f"write failed: {exc}"
            else:
                return _row(
                    scheme_key=spec.scheme_key,
                    scheme_name=spec.scheme_name,
                    category=spec.category,
                    url=spec.url,
                    publisher=config.SOURCE_PUBLISHER,
                    fetched_at=date.today().isoformat(),
                    http_status=status if status is not None else "",
                    stale=False,
                )

    # Degraded path: keep whatever snapshot is already committed.
    existing = snapshot_fetched_at(path)
    if existing is None:
        print(f"  [WARN] {spec.scheme_key}: {error} and no snapshot to fall back on")
        return _row(
            scheme_key=spec.scheme_key,
            scheme_name=spec.scheme_name,
            category=spec.category,
            url=spec.url,
            publisher=config.SOURCE_PUBLISHER,
            fetched_at="",
            http_status=status if status is not None else "",
            stale=True,
        )
    print(f"  [WARN] {spec.scheme_key}: {error}; using committed snapshot from {existing}")
    return _row(
        scheme_key=spec.scheme_key,
        scheme_name=spec.scheme_name,
        category=spec.category,
        url=spec.url,
        publisher=config.SOURCE_PUBLISHER,
        fetched_at=existing,
        http_status=status if status is not None else "",
        stale=True,
    )


def ingest_authored(doc: AuthoredSource) -> dict[str, Any]:
    """Store a transcribed document.

    No network call: the content is fixed in ``config.AUTHORED_SOURCES`` and the
    official page is known to be unfetchable, so re-requesting it on every build
    would only produce a 403 and a confusing warning. ``http_status`` records the
    403 that prompted the transcription, so the CSV never implies a successful
    fetch.
    """
    document = build_authored_document(doc)
    _write_corpus(doc.doc_id, document)
    return _row(
        scheme_key=doc.chunk_scheme_key,
        scheme_name=doc.title,
        category=doc.category,
        url=doc.url,
        publisher=doc.publisher,
        fetched_at=doc.transcribed_on,
        http_status=doc.live_fetch_status if doc.live_fetch_status is not None else "",
        stale=False,
    )


def write_sources_csv(rows: list[dict[str, Any]]) -> None:
    """Write ``data/sources.csv`` with the column order of ARCHITECTURE 5.2.

    ``lineterminator`` is pinned so the file is byte-identical on Windows and
    Linux, which is what makes the "re-running changes nothing" exit criterion
    checkable with ``git diff``.
    """
    columns = list(config.SOURCES_CSV_COLUMNS)
    path = config.PATHS["SOURCES_CSV"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=config.CORPUS_ENCODING, newline=config.CORPUS_NEWLINE) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator=config.CORPUS_NEWLINE)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def run_ingest() -> list[dict[str, Any]]:
    """Build the corpus: 5 scraped scheme pages then 2 transcribed documents.

    Requests are serial with at least :data:`config.HTTP_DELAY_SECONDS` between
    them - five pages is not worth the politeness risk (ARCHITECTURE 4.1).

    Returns:
        The ``sources.csv`` rows, in a deterministic order.
    """
    config.PATHS["CORPUS_DIR"].mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, spec in enumerate(config.SOURCES):
        if index:
            time.sleep(config.HTTP_DELAY_SECONDS)
        print(f"  fetching {spec.scheme_key} ...")
        rows.append(ingest_scheme(spec))

    for doc in config.AUTHORED_SOURCES:
        print(f"  writing transcribed {doc.doc_id} ...")
        rows.append(ingest_authored(doc))

    write_sources_csv(rows)
    print(f"  wrote {config.PATHS['SOURCES_CSV'].relative_to(config.PATHS['ROOT'])}")
    return rows


def run_all() -> int:
    """Entry point for ``run_index.py --stage ingest``."""
    rows = run_ingest()
    failures = [row for row in rows if row["stale"]]
    inspect_corpus()
    if failures:
        print(f"  {len(failures)} of {len(rows)} sources are running on stale snapshots")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Required inspection (IMPLEMENTATION 1.3)
# --------------------------------------------------------------------------- #


def inspect_corpus(files: Iterable[str] | None = None) -> None:
    """Print the Phase 1 exit-criteria report for every corpus file.

    IMPLEMENTATION 1.3 requires inspecting char count, headings and the opening
    lines before Phase 2, because the chunking strategy assumes the corpus is
    label-dense. The label-density question itself is answered by
    :func:`label_density`, whose output follows this report.
    """
    doc_ids = list(files) if files is not None else _corpus_doc_ids()
    print()
    print("=" * 78)
    print("Phase 1 inspection - char counts, headings, and opening lines")
    print("=" * 78)

    for doc_id in doc_ids:
        path = corpus_path(doc_id)
        if not path.exists():
            print(f"\n--- {doc_id}: MISSING ({path.name}) ---")
            continue
        text = path.read_text(encoding=config.CORPUS_ENCODING)
        lines = text.splitlines()
        headings = [ln for ln in lines if _HEADING_RE.match(ln)]
        noise = residual_noise(text)
        leaks = performance_leaks(text)
        density = label_density(text)

        print(f"\n--- {doc_id} ---")
        print(f"  chars: {len(text)}   lines: {len(lines)}   headings: {len(headings)}")
        print(
            f"  fact units: {density.labelled} labelled "
            f"({density.pairs} same-line, {density.stacked} stacked) / "
            f"{density.table_rows} table rows / {density.bullets} bullets / "
            f"{density.paragraphs} prose  -> label ratio {density.ratio:.0%}"
        )
        print(f"  headings: {headings[:14]}{' ...' if len(headings) > 14 else ''}")
        print(f"  site chrome: {noise if noise else 'none'}")
        print(f"  performance leaks: {leaks if leaks else 'none'}")

        print("  first 40 lines:")
        for line in lines[:40]:
            print(f"    | {line[:150]}")


def _corpus_doc_ids() -> list[str]:
    """Every corpus file stem, in the pipeline's own order then alphabetically."""
    ordered = [spec.scheme_key for spec in config.SOURCES]
    ordered += [doc.doc_id for doc in config.AUTHORED_SOURCES]
    if not config.PATHS["CORPUS_DIR"].exists():
        return ordered
    on_disk = {path.stem for path in config.PATHS["CORPUS_DIR"].glob("*.md")}
    return [doc_id for doc_id in ordered if doc_id in on_disk] + sorted(on_disk - set(ordered))


@dataclass(frozen=True)
class LabelDensity:
    """How label-dense a document is, the property Phase 2 depends on.

    Attributes:
        pairs: ``label: value`` on one line.
        stacked: ``label`` on one line with ``value`` on the next. This is the
            form the real pages actually use, and it is not in ARCHITECTURE
            4.2's list - see :func:`label_density`.
        table_rows: Data rows in Markdown tables.
        bullets: List items.
        paragraphs: Prose with no label at all.
    """

    pairs: int
    stacked: int
    table_rows: int
    bullets: int
    paragraphs: int

    @property
    def labelled(self) -> int:
        return self.pairs + self.stacked

    @property
    def units(self) -> int:
        return self.labelled + self.table_rows + self.bullets + self.paragraphs

    @property
    def ratio(self) -> float:
        """Share of fact units that carry a label, 0.0-1.0."""
        return self.labelled / self.units if self.units else 0.0


_LABEL_VALUE_LINE_RE = re.compile(r"(?:\||:)\s*\S")
_VALUE_ONLY_RE = re.compile(r"^[₹$]?[\d][\d,.\s]*\s*(?:%|Cr)?$|^₹[\d,.]+$|^[A-Za-z][\w\s.()-]{0,38}$")


def _looks_like_value(line: str) -> bool:
    """True when a line is a bare value waiting for the label above it."""
    stripped = line.strip()
    if not stripped or len(stripped) > 40 or "|" in stripped:
        return False
    has_number = any(ch.isdigit() for ch in stripped)
    has_money = "₹" in stripped or "Cr" in stripped or "%" in stripped
    return has_number or has_money or _VALUE_ONLY_RE.match(stripped) is not None


def label_density(markdown: str) -> LabelDensity:
    """Count the fact units in a document.

    This is the measurable form of the ARCHITECTURE 4.2 assumption that the
    corpus is label-dense. The architecture lists ``key: value`` pairs, tables and
    bullets; the live pages turn out to use a fourth form that it does not list -
    a **stacked** label and value, one per line, as in::

        Min. for SIP
        ₹100
        Fund size (AUM)
        ₹39,933.37 Cr
        Expense ratio
        1.03%

    so ``stacked`` is counted separately rather than being misfiled as prose. If
    ``stacked`` dominates, the corpus *is* label-dense but Phase 2 has to learn to
    pair a label line with the value line beneath it, which is reported at the
    Phase 1 gate rather than discovered later.
    """
    pairs = stacked = table_rows = bullets = paragraphs = 0
    in_table = False
    lines = [ln for ln in markdown.splitlines()]

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            in_table = False
            continue
        if stripped.startswith("|"):
            if not in_table:
                in_table = True
            if not _TABLE_DELIM_RE.match(stripped):
                table_rows += 1
            continue
        in_table = False
        if _HEADING_RE.match(stripped) or stripped.startswith(">"):
            continue
        if stripped.startswith(("- ", "* ", "+ ")):
            bullets += 1
            continue
        if _LABEL_VALUE_LINE_RE.search(stripped):
            pairs += 1
            continue
        nxt = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if nxt and not nxt.startswith(("|", "#", ">")) and _looks_like_value(nxt):
            stacked += 1
            continue
        paragraphs += 1

    return LabelDensity(
        pairs=pairs,
        stacked=stacked,
        table_rows=table_rows,
        bullets=bullets,
        paragraphs=paragraphs,
    )


def load_corpus() -> list[dict[str, Any]]:
    """Read every committed snapshot back, with its header metadata parsed.

    Phase 2 consumes this. Kept here so the header format written by
    :func:`build_document` is parsed by exactly one function.

    Returns:
        One record per file: ``doc_id``, ``title``, ``category``, ``scheme_key``,
        ``source_url``, ``fetched_at``, ``authored``, ``text``.
    """
    records: list[dict[str, Any]] = []
    for doc_id in _corpus_doc_ids():
        path = corpus_path(doc_id)
        if not path.exists():
            continue
        text = path.read_text(encoding=config.CORPUS_ENCODING)
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else doc_id
        fields: dict[str, str] = {}
        for line in lines[1:6]:
            if line.startswith(">"):
                for pair in line.lstrip("> ").split(" | "):
                    if ":" in pair:
                        key, _, value = pair.partition(":")
                        fields[key.strip()] = value.strip()
        body = text.split("\n\n", 1)[1] if "\n\n" in text else text
        records.append(
            {
                "doc_id": doc_id,
                "title": title,
                "category": fields.get("category", ""),
                "scheme_key": fields.get("scheme_key", doc_id),
                "source_url": fields.get("source_url", ""),
                "fetched_at": fields.get("fetched_at", ""),
                "authored": fields.get("authored") == "true",
                "text": body,
            }
        )
    return records
