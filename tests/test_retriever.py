"""Phase 4 - retriever tests."""
from __future__ import annotations

from src import retriever, query
from src.vectorstore import is_index_stale


def test_retrieve_returns_results_for_eval_queries():
    if is_index_stale()[0]:
        assert False
    plan = query.build_plan("What is the expense ratio of HDFC Large Cap Direct Growth?")
    res = retriever.retrieve(plan)
    assert len(res.chunks) >= 1