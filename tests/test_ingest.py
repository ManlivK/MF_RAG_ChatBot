"""Tests for Phase 1 ingestion.

The unit tests here are deliberately HTML-in / Markdown-out with no network: the
cleaning rules are the part of Stage 1 most likely to regress silently, because a
rule that stops matching a changed page only shows up as a quietly wrong fact
several phases later. ``run_ingest`` itself is exercised by running the real
ingest stage and inspecting the committed corpus.
"""

from __future__ import annotations

import csv
import dataclasses

import pytest

from src import config, ingest
from src.types import CROSS_SCHEME_KEY

# A miniature page that reproduces every structural quirk the real Groww page has,
# so the rules below are tested against the shapes that actually caused bugs: a
# React-style root wrapper, an identity row of sibling inline elements, a figure
# whose unit sits in its own block, a return widget, a performance section, a
# holdings table, and the site footer mega-menu.
#
# The content markers are deliberately spread one per sibling div. Content
# narrowing descends into a child only while exactly one child holds two or more
# markers, so clustering them in one div would - correctly - make the extractor
# treat that div as the page. The real page spreads them, and so does this.
HOLDING_ROWS = "".join(
    f"<tr><td>Company Number {n} Limited</td><td>Financial Services</td><td>{n}.05%</td></tr>"
    for n in range(1, 13)
)

PAGE = f"""
<html><body>
<div id="__next">
  <div id="root">
    <nav>Home Mutual Funds Stocks IPO</nav>
    <div class="content">
      <div class="header">
        <span>Equity</span><span>Large Cap</span><span>Very High Risk</span>
        <span>+8.71</span><span>%</span>
      </div>
      <div class="aum">Fund size (AUM)</div>
      <div class="aumvalue">&#8377;39,933.37 Cr</div>
      <div class="navvalue">NAV: 25 Sep '26</div>
      <div class="navamount">&#8377;1,189.08</div>
      <div class="siplabel">Min. for SIP</div>
      <div class="sipvalue">&#8377;100</div>
      <div class="ratiolabel">Expense ratio</div>
      <div class="ratiovalue">1.03%</div>
      <div class="ranklabel">Rank (total assets)</div>
      <div class="rankvalue">#2 in India</div>
      <div class="benchlabel">Fund benchmark</div>
      <div class="benchvalue">NIFTY 100 Total Return Index</div>
      <div class="minlabel">Minimum investments</div>
      <div>Return calculator | Monthly SIP | One time | +12.47% 1Y annualised</div>
      <h2>Historic returns</h2>
      <p>1Y annualised returns were +18.22% and 3Y annualised returns were +21.05%.</p>
      <h2>Holdings (12)</h2>
      <table>
        <thead><tr><th>Name</th><th>Sector</th><th>Assets</th></tr></thead>
        <tbody>
          <tr><td>ICICI Bank Ltd</td><td>Financial</td><td>10.05%</td></tr>
          <tr><td>Reliance Industries Ltd</td><td>Energy</td><td>4.78%</td></tr>
          {HOLDING_ROWS}
        </tbody>
      </table>
      <h2>Exit load</h2>
      <p>Exit load is 1% if the investment is redeemed within 12 months of purchase.</p>
      <div class="footer">
        <p>GROWW</p>
        <p>Contact Us</p>
        <p>PRODUCTS</p>
        <p>Kotak Small Cap Fund</p>
        <p>SIP Calculator</p>
        <p>Home Loan EMI Calculator</p>
        <p>Top Gainers Stocks</p>
        <p>Groww Large Cap Fund</p>
        <p>Z</p>
      </div>
    </div>
  </div>
</div>
</body></html>
"""


@pytest.fixture(scope="module")
def cleaned() -> str:
    """Clean :data:`PAGE` once; the rules are independent of ordering."""
    return ingest.clean_html_to_markdown(PAGE)


def _read_sources() -> list[dict[str, str]]:
    with config.PATHS["SOURCES_CSV"].open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- required facts survive -------------------------------------------------


def test_keeps_the_facts_the_brief_requires(cleaned: str) -> None:
    lowered = cleaned.lower()
    for needle in (
        "nav:",
        "1,189.08",
        "expense ratio",
        "1.03%",
        "min. for sip",
        "very high risk",
        "fund benchmark",
        "nifty 100 total return index",
        "exit load",
        "icici bank ltd",
    ):
        assert needle in lowered, f"lost required fact: {needle}"


def test_splits_fused_label_value_siblings(cleaned: str) -> None:
    """A label and its value must not end up on one line.

    markdownify concatenates sibling inline elements, which fused labels to their
    own values (``Fund benchmarkBSE 250 SmallCap Total Return Index``). A fused
    pair would be quoted run-together by the generator and would break the
    guardrail that matches a chunk's expected label.
    """
    lines = [line.strip() for line in cleaned.splitlines()]
    assert "Fund benchmark" in lines
    assert "NIFTY 100 Total Return Index" in lines
    assert "Very High Risk" in lines
    assert not any("benchmarkNIFTY" in line for line in lines)
    assert not any("Very High Risk" in line and line != "Very High Risk" for line in lines)


def test_keeps_table_with_percentages(cleaned: str) -> None:
    """The holdings table is a fact, not performance data.

    Percentages inside it are unsigned weights, so neither the table filter nor the
    line-level return filter may touch them.
    """
    assert "| Name | Sector | Assets |" in cleaned
    assert "10.05%" in cleaned
    assert "4.78%" in cleaned


# --- performance is removed (PRD C3) ---------------------------------------


def test_removes_return_widget_and_figure(cleaned: str) -> None:
    lowered = cleaned.lower()
    assert "return calculator" not in lowered
    assert "monthly sip | one time" not in lowered
    assert "+12.47" not in cleaned


def test_removes_return_sections_and_signed_figures(cleaned: str) -> None:
    assert "Historic returns" not in cleaned
    assert "+8.71" not in cleaned
    assert "+18.22" not in cleaned
    assert "+21.05" not in cleaned


def test_keeps_benchmark_name_with_return_in_it(cleaned: str) -> None:
    """``Total Return Index`` is part of a benchmark's name, not a return figure.

    Dropping every occurrence of "return" would delete the benchmark fact and
    break the scope guardrail that answers benchmark questions.
    """
    assert "NIFTY 100 Total Return Index" in cleaned


def test_drops_ranking_pairs(cleaned: str) -> None:
    """The decision taken for this corpus keeps rankings out alongside returns."""
    assert "Rank (total assets)" not in cleaned
    assert "#2 in India" not in cleaned


def test_reports_no_performance_leaks(cleaned: str) -> None:
    assert ingest.performance_leaks(cleaned) == []


# --- site chrome is removed -------------------------------------------------


def test_removes_nav_and_footer(cleaned: str) -> None:
    for needle in (
        "PRODUCTS",
        "SIP Calculator",
        "Home Loan EMI",
        "Top Gainers",
        "Groww Large Cap Fund",
        "Kotak Small Cap Fund",
    ):
        assert needle not in cleaned, f"footer chrome survived: {needle}"


def test_keeps_the_facts_that_precede_the_footer(cleaned: str) -> None:
    """The footer shares a block with content worth keeping.

    Groww emits the exit-load section and the whole mega-menu close together, so a
    naive block-level drop would also take the facts that precede the footer.
    """
    assert "Exit load" in cleaned
    assert "redeemed within 12 months" in cleaned


def test_reports_no_residual_noise(cleaned: str) -> None:
    assert ingest.residual_noise(cleaned) == []


# --- document assembly ------------------------------------------------------


def test_build_document_writes_provenance_header() -> None:
    spec = config.SOURCES[0]
    document = ingest.build_document(spec, "# Body\n\ntext", "2026-09-27")
    title, meta = document.splitlines()[:2]
    assert title == f"# {spec.scheme_name}"
    assert f"scheme_key: {spec.scheme_key}" in meta
    assert f"source_url: {spec.url}" in meta
    assert "fetched_at: 2026-09-27" in meta
    assert document.rstrip().endswith("text")


def test_build_authored_document_labels_itself_as_authored() -> None:
    """An authored document must never be mistakable for a live scrape."""
    source = config.AUTHORED_SOURCES[0]
    document = ingest.build_authored_document(source)
    assert f"# {source.title}" in document
    assert "authored: true" in document
    assert f"live_fetch_status: {source.live_fetch_status}" in document
    assert f"source_url: {source.url}" in document
    assert "transcribed by hand" in document


def test_authored_documents_are_not_attributed_to_groww() -> None:
    for source in config.AUTHORED_SOURCES:
        assert "Groww" not in source.publisher


def test_authored_document_stores_a_cross_scheme_key_as_all_schemes() -> None:
    """A document about every scheme cannot claim one scheme's key.

    Stamping ``hdfc-large-cap`` on it would let the Phase 2 scheme filter pull
    cross-scheme guidance into a single-scheme answer.
    """
    cross_scheme = [d for d in config.AUTHORED_SOURCES if d.scheme_key is None]
    assert cross_scheme, "expected a cross-scheme authored document"
    for source in cross_scheme:
        assert source.chunk_scheme_key == CROSS_SCHEME_KEY
        assert source.chunk_scheme_key not in {s.scheme_key for s in config.SOURCES}


# --- label density (ARCHITECTURE 4.2) --------------------------------------


def test_label_density_counts_stacked_and_inline_pairs() -> None:
    """The live pages use a fourth form the architecture does not list.

    ARCHITECTURE 4.2 lists ``key: value`` pairs, tables and bullets. The pages
    actually lead with a *stacked* label and value, one per line, so that form is
    counted separately instead of being misfiled as prose.
    """
    density = ingest.label_density(
        "Expense ratio\n1.03%\n\nMinimum SIP: \u20b9100\n\n"
        "## Holdings (1)\n\n| Name | Assets |\n| --- | --- |\n| ICICI Bank Ltd | 10.05% |\n"
    )
    assert density.stacked >= 1
    assert density.pairs >= 1
    assert density.table_rows >= 1
    assert density.pairs + density.stacked >= 2


def test_scraped_corpus_is_label_dense() -> None:
    """Phase 1 gate: scraped pages must yield label-led chunks, not prose only.

    Authored documents are excluded: the inception and capital-gains documents are
    deliberate prose and are not scraped scheme pages.
    """
    corpus_dir = config.PATHS["CORPUS_DIR"]
    for path in sorted(corpus_dir.glob("hdfc-*.md")):
        if path.stem in config.AUTHORED_BY_ID:
            continue
        density = ingest.label_density(path.read_text(encoding="utf-8"))
        labelled = density.pairs + density.stacked
        assert labelled >= 10, f"{path.name} is not label dense: {density}"


def test_committed_corpus_is_clean() -> None:
    """The Definition of Done for Stage 1, checked against the committed files."""
    corpus_dir = config.PATHS["CORPUS_DIR"]
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert ingest.residual_noise(text) == [], f"{path.name} kept site chrome"
        assert ingest.performance_leaks(text) == [], f"{path.name} kept a return figure"


# --- failure handling (NFR4) ------------------------------------------------


def test_clean_rejects_a_page_that_lost_its_layout() -> None:
    """A layout change must fail loudly instead of writing a near-empty snapshot."""
    with pytest.raises(ValueError):
        ingest.clean_html_to_markdown("<html><body><p>Too short.</p></body></html>")


def test_fetch_page_refuses_a_url_outside_the_allow_list() -> None:
    """A source URL is config, not input: an unlisted URL must not be fetched.

    NFR4 also requires the fetch to degrade rather than raise, so the refusal is
    reported through the error slot of the return tuple.
    """
    rogue = dataclasses.replace(config.SOURCES[0], url="https://example.com/not-allowed")
    html, status, error = ingest.fetch_page(rogue)
    assert html is None
    assert status is None
    assert error is not None and "allow-list" in error


# --- the manifest deliverable ----------------------------------------------


def test_sources_csv_has_a_row_for_every_source() -> None:
    expected = len(config.SOURCES) + len(config.AUTHORED_SOURCES)
    rows = _read_sources()
    assert len(rows) == expected
    assert list(rows[0].keys()) == list(config.SOURCES_CSV_COLUMNS)


def test_sources_csv_urls_are_all_allow_listed() -> None:
    for row in _read_sources():
        assert row["url"] in config.ALLOWED_URLS


def test_sources_csv_marks_authored_rows_with_the_blocked_status() -> None:
    """A 403 must stay visible in the deliverable, not be papered over."""
    authored_urls = {source.url for source in config.AUTHORED_SOURCES}
    for row in _read_sources():
        if row["url"] in authored_urls:
            assert row["http_status"] == str(config.AUTHORED_LIVE_FETCH_STATUS)
            assert "authored" in row["publisher"].lower()
        else:
            assert row["http_status"] == "200"


def test_corpus_has_one_file_per_source() -> None:
    expected = {source.scheme_key for source in config.SOURCES} | {
        source.doc_id for source in config.AUTHORED_SOURCES
    }
    assert set(ingest._corpus_doc_ids()) == expected
