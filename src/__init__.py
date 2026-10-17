"""MF Facts - a facts-only RAG FAQ assistant for five HDFC mutual fund schemes.

Layering (ARCHITECTURE.md section 3):

    app.py  ->  src.pipeline  ->  stage modules  ->  src.config

Stage modules must never import ``app`` or ``pipeline``. ``guardrails`` and
``citation`` must never import ``generator`` or ``retriever`` so they stay pure
and unit-testable without a network or a built index.
"""

__all__ = ["config", "types"]
