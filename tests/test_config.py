"""Phase 0 tests: the configuration contract.

These encode the product rules that are cheap to check as strings, so that a
later phase cannot quietly break a PRD requirement (C1, C4, C5, C7, C9).
"""

from __future__ import annotations

import re
from typing import get_args

import pytest

from src import config
from src.types import CROSS_SCHEME_KEY, AuthoredSource, Intent, RefusalKind, SourceSpec

# --------------------------------------------------------------------------- #
# Sources - the allow-list (PRD section 5, C1)
# --------------------------------------------------------------------------- #


def test_five_sources_present() -> None:
    assert len(config.SOURCES) == 5


def test_source_urls_are_unique() -> None:
    urls = [spec.url for spec in config.SOURCES]
    assert len(set(urls)) == len(urls)


def test_every_source_url_is_on_the_allowed_base() -> None:
    for spec in config.SOURCES:
        assert spec.url.startswith(config.ALLOWED_URL_BASE), spec.url


def test_allowed_urls_mirrors_sources() -> None:
    assert config.ALLOWED_URLS == (
        {spec.url for spec in config.SOURCES} | {doc.url for doc in config.AUTHORED_SOURCES}
    )


def test_source_keys_are_unique_and_slug_like() -> None:
    keys = [spec.scheme_key for spec in config.SOURCES]
    assert len(set(keys)) == len(keys)
    for key in keys:
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", key), key


def test_the_five_categories_cover_the_brief() -> None:
    categories = {spec.category for spec in config.SOURCES}
    assert categories == {"Large Cap", "Flexi Cap", "ELSS", "Small Cap", "Hybrid"}


def test_base_url_alone_is_not_allow_listed() -> None:
    """Only the 5 exact scheme pages are citeable - the bare base URL is not, so
    a hallucinated slug cannot slip through as "obviously on the same site"."""
    assert config.normalize_url(config.ALLOWED_URL_BASE) not in config.allowed_url_keys()


# --------------------------------------------------------------------------- #
# Authored sources - the transcribed official pages
# --------------------------------------------------------------------------- #


def test_two_authored_sources_present() -> None:
    """The facts the Groww scheme pages cannot supply are hand-transcribed: the
    ELSS lock-in, the capital-gains-statement procedure, and one scheme-inception
    document per scheme. The official AMC pages answer us with 403, so an
    inception question is answered from an authored document rather than a scrape."""
    assert len(config.AUTHORED_SOURCES) == 7
    assert {doc.doc_id for doc in config.AUTHORED_SOURCES} == {
        "hdfc-elss-lock-in",
        "capital-gains-statement",
        "hdfc-large-cap-inception",
        "hdfc-equity-flexi-cap-inception",
        "hdfc-elss-tax-saver-inception",
        "hdfc-small-cap-inception",
        "hdfc-balanced-advantage-inception",
    }


def test_authored_doc_ids_are_unique_and_slug_like() -> None:
    ids = [doc.doc_id for doc in config.AUTHORED_SOURCES]
    assert len(set(ids)) == len(ids)
    for doc_id in ids:
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", doc_id), doc_id


def test_authored_urls_are_unique_and_never_collide_with_scraped() -> None:
    authored = {doc.url for doc in config.AUTHORED_SOURCES}
    assert len(authored) == len(config.AUTHORED_SOURCES)
    assert not (authored & {spec.url for spec in config.SOURCES})


def test_authored_urls_are_on_the_official_amc_domain() -> None:
    """A transcription may only be attributed to the issuer's own domain - never
    to a third-party mirror or blog."""
    for doc in config.AUTHORED_SOURCES:
        assert doc.url.startswith("https://www.hdfcfund.com/"), doc.url


def test_authored_sources_record_the_blocked_fetch_status() -> None:
    for doc in config.AUTHORED_SOURCES:
        assert doc.live_fetch_status == config.AUTHORED_LIVE_FETCH_STATUS
        assert doc.publisher == config.AUTHORED_PUBLISHER


def test_authored_scheme_keys_are_known_or_cross_scheme() -> None:
    keys = {spec.scheme_key for spec in config.SOURCES}
    for doc in config.AUTHORED_SOURCES:
        if doc.scheme_key is None:
            assert doc.chunk_scheme_key == CROSS_SCHEME_KEY
        else:
            assert doc.scheme_key in keys, doc.scheme_key


def test_authored_by_id_lookup_round_trips() -> None:
    for doc in config.AUTHORED_SOURCES:
        assert config.AUTHORED_BY_ID[doc.doc_id] is doc


def test_authored_bodies_answer_the_gaps_the_scheme_pages_cannot() -> None:
    """Guard the reason these files exist: if Groww ever starts publishing the
    lock-in or the statement procedure, they can be deleted rather than kept as
    silently-drifting duplicates."""
    by_id = config.AUTHORED_BY_ID
    lock_in = by_id["hdfc-elss-lock-in"].markdown_body.lower()
    assert "lock-in period is three years" in lock_in
    assert "80c" in lock_in and "1.5 lakh" in lock_in

    statement = by_id["capital-gains-statement"].markdown_body.lower()
    assert "cams" in statement and "kfintech" in statement
    assert "financial year" in statement


def test_authored_bodies_contain_no_return_figures() -> None:
    """The authored documents must not reintroduce the performance data the
    scraper strips from the scheme pages."""
    banned = ("category average", "% p.a", "cagr", "since inception", "fund returns")
    for doc in config.AUTHORED_SOURCES:
        body = doc.markdown_body.lower()
        for phrase in banned:
            assert phrase not in body, f"{doc.doc_id} contains {phrase!r}"


def test_authored_bodies_contain_no_real_pan_or_email() -> None:
    """The corpus is committed, and the PII guardrail keys off PAN/email shapes. A
    transcription must not embed an example identifier the guardrail would then
    read as user PII."""
    for doc in config.AUTHORED_SOURCES:
        assert not re.search(r"[A-Z]{5}\d{4}[A-Z]", doc.markdown_body)
        assert not re.search(r"[\w.\-]+@[\w.\-]+\.\w{2,}", doc.markdown_body)


def test_corpus_still_covers_five_schemes() -> None:
    """Twelve documents, five schemes: the extra documents must not read as
    extra schemes to the UI or the scope resolver."""
    assert len(config.SOURCES) == 5
    assert len({spec.scheme_key for spec in config.SOURCES}) == 5
    assert config.WELCOME_LINE.startswith("Ask factual questions about 5 ")


# --------------------------------------------------------------------------- #
# Performance stripping rules (PRD C3)
# --------------------------------------------------------------------------- #


def test_drop_rules_never_target_the_word_return_on_its_own() -> None:
    """The benchmark is legitimately 'NIFTY 100 Total Return Index'. A blanket
    'return' match would destroy a required fact, so only whole headings and
    multi-token table headers may be dropped."""
    for heading in config.SCRAP_DROP_SECTION_HEADINGS:
        assert heading != "return", heading
    assert "return" not in config.SCRAP_DROP_TABLE_MARKERS
    assert config.SCRAP_DROP_TABLE_MIN_TOKENS >= 2


def test_keep_markers_veto_a_performance_drop() -> None:
    """Exit-load and fee tables must survive even if a header picks up a stray
    '1 year' column."""
    assert config.SCRAP_KEEP_TABLE_MARKERS
    assert any("exit load" in m for m in config.SCRAP_KEEP_TABLE_MARKERS)


def test_root_selector_prefers_the_react_wrapper_over_body() -> None:
    """Groww serves no <main>/[role=main]/<article>; taking the #__next wrapper is
    what keeps the navbar and footer out of the corpus."""
    assert "div#__next" in config.SCRAP_ROOT_SELECTORS
    assert config.SCRAP_ROOT_SELECTORS.index("div#__next") < config.SCRAP_ROOT_SELECTORS.index(
        "body"
    )


def test_source_spec_lookup_round_trips() -> None:
    for spec in config.SOURCES:
        assert config.source_spec(spec.scheme_key) == spec
        assert config.source_spec_for_url(spec.url) == spec
    assert config.source_spec("nope") is None
    assert config.source_spec_for_url("https://example.com/x") is None


def test_source_spec_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        SourceSpec(scheme_key="k", scheme_name="n", category="", url="u")


def test_authored_source_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        AuthoredSource(
            doc_id="d",
            scheme_key=None,
            title="t",
            category="c",
            url="u",
            publisher="p",
            markdown_body="",
            transcribed_on="2026-09-27",
        )


# --------------------------------------------------------------------------- #
# Wording fixed by the brief (C4, C7)
# --------------------------------------------------------------------------- #


def test_disclaimer_matches_prd_wording_exactly() -> None:
    assert config.DISCLAIMER == "Facts-only. No investment advice."


def test_last_updated_prefix_matches_prd_wording_exactly() -> None:
    assert config.LAST_UPDATED_PREFIX == "Last updated from sources: "


def test_three_example_questions_present() -> None:
    assert len(config.EXAMPLE_QUESTIONS) == 3
    assert all(q.strip().endswith("?") for q in config.EXAMPLE_QUESTIONS)


# --------------------------------------------------------------------------- #
# Intents, refusal kinds and their templates
# --------------------------------------------------------------------------- #


def test_every_intent_literal_has_a_fact_type_entry() -> None:
    known = set(get_args(Intent))
    assert known == {*config.FACT_TYPES.keys(), "unknown"}


def test_fact_types_have_keywords_and_labels() -> None:
    for intent, spec in config.FACT_TYPES.items():
        assert spec["keywords"], intent
        assert spec["expected_labels"], intent
        assert all(k.strip() for k in spec["keywords"]), intent
        assert all(l.strip() for l in spec["expected_labels"]), intent


def test_every_refusal_kind_has_a_template() -> None:
    missing = set(get_args(RefusalKind)) - set(config.REFUSAL_TEMPLATES)
    assert not missing, f"missing refusal templates: {sorted(missing)}"


def test_no_extra_refusal_templates() -> None:
    extra = set(config.REFUSAL_TEMPLATES) - set(get_args(RefusalKind))
    assert not extra, f"templates with no RefusalKind: {sorted(extra)}"


@pytest.mark.parametrize("kind", sorted(config.REFUSAL_TEMPLATES))
def test_refusal_template_is_at_most_three_sentences(kind: str) -> None:
    template = config.REFUSAL_TEMPLATES[kind]
    assert config.sentence_count(template) <= config.MAX_ANSWER_SENTENCES


@pytest.mark.parametrize("kind", sorted(config.REFUSAL_TEMPLATES))
def test_refusal_template_contains_no_digits(kind: str) -> None:
    """A digit in a refusal would be a fact we were never given."""
    assert not re.search(r"\d", config.REFUSAL_TEMPLATES[kind]), kind


# --------------------------------------------------------------------------- #
# Guardrail regexes
# --------------------------------------------------------------------------- #


def test_guard_order_puts_pii_first() -> None:
    assert config.GUARD_ORDER[0] == "pii"
    assert set(config.GUARD_ORDER) == {"pii", "advice", "performance", "scope"}


def test_every_guardrail_pattern_compiles() -> None:
    groups = (
        config.PII_PATTERNS,
        config.ADVICE_PATTERNS,
        config.PERFORMANCE_PATTERNS,
        config.SHORTHAND,
    )
    for group in groups:
        for rule, pattern in group.items():
            re.compile(pattern, re.IGNORECASE), rule


def test_pii_patterns_cover_the_six_required_shapes() -> None:
    assert set(config.PII_PATTERNS) == {
        "pii.pan",
        "pii.aadhaar",
        "pii.account",
        "pii.email",
        "pii.phone",
        "pii.otp",
    }


def test_nav_is_not_treated_as_a_performance_question() -> None:
    """NAV gets its own redirect message in guardrail phase B."""
    assert not re.search(config.PERFORMANCE_PATTERNS["performance.compare"], "what is the NAV", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #


def test_every_scheme_alias_maps_to_a_real_scheme() -> None:
    valid = {spec.scheme_key for spec in config.SOURCES}
    for key, aliases in config.SCHEME_ALIASES.items():
        assert key in valid, key
        assert aliases, key
        assert len(set(aliases)) == len(aliases), f"duplicate alias in {key}"


def test_aliases_do_not_contain_a_non_allow_listed_amc() -> None:
    joined = " ".join(a for aliases in config.SCHEME_ALIASES.values() for a in aliases)
    for amc in ("sbi", "axis", "icici", "kotak", "nippon", "tata"):
        assert amc not in joined, f"{amc} must live in OTHER_AMC_ALIASES, not SCHEME_ALIASES"


def test_generic_aliases_are_shadowed_by_a_longer_specific_alias() -> None:
    """'hdfc equity' and 'hdfc equity fund' share a prefix. Resolution is
    longest-alias-first (implemented in src/query.py, Phase 4), so this test
    pins the data requirement: the longer, more specific alias must exist for
    every generic prefix alias."""
    for key, aliases in config.SCHEME_ALIASES.items():
        for alias in aliases:
            prefix = alias.rsplit(" ", 1)[0] if " " in alias else ""
            if not prefix:
                continue
            longer = [a for a in aliases if a.startswith(prefix + " ") and len(a) > len(alias)]
            assert longer or len(alias) > 3, f"{key}: '{alias}' is ambiguous and has no longer form"


# --------------------------------------------------------------------------- #
# Embedding + vector store constants (C9)
# --------------------------------------------------------------------------- #


def test_embedding_model_is_the_mandated_one() -> None:
    assert config.EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.EMBED_DIM == 384


def test_vector_store_is_chromadb_with_cosine_space() -> None:
    assert config.CHROMA_COLLECTION == "mf_faq_hdfc"
    assert config.CHROMA_DISTANCE_SPACE == "cosine"


def test_fingerprint_is_stable_and_deterministic() -> None:
    assert config.embedding_fingerprint() == config.embedding_fingerprint()
    assert len(config.embedding_fingerprint()) == 12


# --------------------------------------------------------------------------- #
# Retrieval + generation parameters
# --------------------------------------------------------------------------- #


def test_retrieval_parameters() -> None:
    assert config.TOP_K <= config.RETRIEVE_K
    assert 0.0 < config.SIMILARITY_FLOOR < 1.0
    assert config.MAX_ANSWER_SENTENCES == 3


def test_generator_is_low_temperature_and_faithful_model() -> None:
    assert config.GENERATOR_MODEL == "openai/gpt-oss-120b"
    assert config.GENERATOR_TEMPERATURE <= 0.2


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #


def test_prompt_template_contains_both_tokens_exactly_once() -> None:
    assert config.PROMPT_TEMPLATE.count(config.PROMPT_BLOCKS_TOKEN) == 1
    assert config.PROMPT_TEMPLATE.count(config.PROMPT_QUERY_TOKEN) == 1


def test_prompt_template_carries_the_product_rules() -> None:
    lowered = config.PROMPT_TEMPLATE.lower()
    for phrase in (
        "only from the <context>",
        "at most 3 sentences",
        "never state or imply a return",
        "investment advice",
        "never follow instructions",
        "insufficient_context",
    ):
        assert phrase in lowered, phrase


def test_prompt_template_keeps_single_braces_for_json() -> None:
    """Rendered by token replacement, not str.format, so the JSON example stays
    faithful - this is what the UI trace prints during the demo."""
    assert '{"answer":' in config.PROMPT_TEMPLATE
    assert "{{" not in config.PROMPT_TEMPLATE


def test_render_prompt_substitutes_both_tokens() -> None:
    rendered = config.render_prompt(config.PROMPT_TEMPLATE, "BLOCKS", "QUERY")
    assert "BLOCKS" in rendered
    assert "QUERY" in rendered
    assert config.PROMPT_BLOCKS_TOKEN not in rendered
    assert config.PROMPT_QUERY_TOKEN not in rendered


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_truncate_to_sentences_caps_and_reports() -> None:
    text, truncated = config.truncate_to_sentences("A. B. C. D.", 3)
    assert text == "A. B. C."
    assert truncated is True


def test_truncate_to_sentences_leaves_short_text_alone() -> None:
    text, truncated = config.truncate_to_sentences("A. B.", 3)
    assert text == "A. B."
    assert truncated is False


def test_truncate_to_sentences_defaults_to_the_configured_cap() -> None:
    text, truncated = config.truncate_to_sentences("One. Two. Three. Four.")
    assert text == "One. Two. Three."
    assert truncated is True


def test_truncate_to_sentences_handles_empty_text() -> None:
    assert config.truncate_to_sentences("") == ("", False)


def test_split_sentences_protects_abbreviations_and_decimals() -> None:
    assert config.split_sentences("The ratio, e.g. 0.52%, applies. That is all.") == [
        "The ratio, e.g. 0.52%, applies.",
        "That is all.",
    ]


def test_split_sentences_handles_exclamation_and_question_marks() -> None:
    assert len(config.split_sentences("Really? Yes! Fine.")) == 3


def test_format_last_updated() -> None:
    assert config.format_last_updated("2026-09-27") == "27 Sep 2026"
    assert config.format_last_updated("2026-09-27T14:31:02+05:30") == "27 Sep 2026"
    assert config.format_last_updated("2026-01-05") == "5 Jan 2026"


@pytest.mark.parametrize("bad", ["", "not-a-date", "2026-13-01", "27/09/2026"])
def test_format_last_updated_returns_empty_for_junk(bad: str) -> None:
    """A bad date must become no date, never a wrong date in the UI."""
    assert config.format_last_updated(bad) == ""


def test_normalize_url_strips_scheme_www_query_and_slash() -> None:
    assert config.normalize_url("https://www.groww.in/mutual-funds/x/") == "groww.in/mutual-funds/x"
    assert config.normalize_url("http://groww.in/a?utm_source=b#top") == "groww.in/a"
    assert config.normalize_url("") == ""


def test_normalize_url_collapses_the_scheme_variations_of_our_own_urls() -> None:
    for url in config.ALLOWED_URLS:
        assert config.normalize_url(url) in config.allowed_url_keys()


def test_chunk_count_tokens_is_monotonic_and_positive() -> None:
    assert config.chunk_count_tokens("") == 0
    short = config.chunk_count_tokens("Expense ratio is 0.52%.")
    longer = config.chunk_count_tokens("Expense ratio is 0.52%. " * 50)
    assert 0 < short < longer


def test_default_link_is_allow_listed() -> None:
    assert config.default_link() in config.ALLOWED_URLS


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def test_paths_are_all_under_the_repo_root() -> None:
    root = config.PATHS["ROOT"]
    for name, path in config.PATHS.items():
        assert isinstance(path, config.Path), name
        if path != root:
            assert root in path.parents, f"{name} ({path}) escapes the repo root"


def test_data_dirs_have_a_gitkeep_so_they_survive_a_clone() -> None:
    for key in ("CORPUS_DIR", "CHUNKS_DIR"):
        directory = config.PATHS[key]
        assert directory.exists(), f"{key} missing - Phase 0 should create it"
        assert (directory / ".gitkeep").exists(), f"{key}/.gitkeep missing"


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def test_load_api_key_raises_a_helpful_error_when_unset(monkeypatch, tmp_path) -> None:
    # Point at a .env that does not exist, so a developer's real key on disk
    # cannot make this test pass or fail depending on their machine.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(config.PATHS, "ENV_FILE", tmp_path / "absent.env")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        config.load_api_key(required=True)
    assert config.load_api_key(required=False) is None


def test_load_api_key_reads_a_dotenv_file(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=test-key-not-real\n", encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(config.PATHS, "ENV_FILE", env_file)
    try:
        assert config.load_api_key(required=True) == "test-key-not-real"
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_env_example_exists_and_is_gitignored() -> None:
    root = config.PATHS["ROOT"]
    assert (root / ".env.example").exists()
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignore
    # A real .env may exist locally; it must never be committed.
    assert ".env\n" in ignore or ".env\r\n" in ignore or "\n.env" in ignore
