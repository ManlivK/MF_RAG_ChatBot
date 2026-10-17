"""Shared pytest fixtures.

Everything here works with a clean checkout: no network, no index, no API key.
Tests that need those are skipped with a clear reason instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src import config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config_module():
    """The configuration module itself, so tests can assert on its contents."""
    return config


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    """``data/corpus`` - populated by Phase 1."""
    return config.PATHS["CORPUS_DIR"]


@pytest.fixture(scope="session")
def chunks_path() -> Path:
    """``data/chunks/chunks.jsonl`` - populated by Phase 2."""
    return config.PATHS["CHUNKS_FILE"]


@pytest.fixture(scope="session")
def chunks() -> list[dict[str, Any]]:
    """Every chunk record, or skip if Phase 2 has not run."""
    path = config.PATHS["CHUNKS_FILE"]
    if not path.exists():
        pytest.skip("chunks.jsonl not built yet - run: python run_index.py --stage chunk")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def sample_chunk() -> dict[str, Any]:
    """A minimal, schema-complete chunk record for tests that don't need the corpus."""
    return {
        "chunk_id": "hdfc-large-cap::0000",
        "text": "### Exit Load\n- 0.00% if redeemed after 12 months from allotment",
        "scheme_key": "hdfc-large-cap",
        "scheme_name": "HDFC Large Cap Fund — Direct Growth",
        "category": "Large Cap",
        "section": "Exit Load",
        "label": "Exit Load",
        "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "source_title": "HDFC Large Cap Fund — Direct Growth",
        "fetched_at": "2026-09-27",
        "chunk_index": 0,
        "content_hash": "sha256:test",
    }


@pytest.fixture()
def api_key() -> str:
    """Groq key, or skip - only the end-to-end harness needs one."""
    key = config.load_api_key(required=False)
    if not key:
        pytest.skip("GROQ_API_KEY not set - copy .env.example to .env")
    return key


@pytest.fixture(scope="session")
def index_ready() -> None:
    """Skip unless a non-stale index is on disk.

    Phase 3 and Phase 4 tests search the real collection; running them against a
    stale or absent index would either raise or silently pass on empty results.
    """
    from src import vectorstore

    stale, reason = vectorstore.is_index_stale()
    if stale:
        pytest.skip(
            f"index is stale ({reason}) - run: "
            "python run_index.py --stage index --rebuild"
        )
