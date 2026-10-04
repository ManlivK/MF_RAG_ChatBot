"""Phase 3 - vectorstore tests."""
from __future__ import annotations

from src import vectorstore, config


def test_index_meta_written():
    meta = vectorstore.read_index_meta()
    assert meta is not None
    assert meta['chunk_count'] > 0


def test_is_index_stale_false_after_build():
    stale, _ = vectorstore.is_index_stale()
    assert not stale