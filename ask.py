r"""Interactive Q&A loop - type a question, see the retrieval answer and trace.

Usage:
    .\.venv\Scripts\python.exe ask.py
"""

from __future__ import annotations

import sys

from src import pipeline


def _force_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _show(question: str) -> None:
    result = pipeline.answer(question)
    message = result.message

    print("\n" + "-" * 70)
    print(f"Q: {question}")
    print(f"[{message.kind}] {message.text}")
    if message.source_url:
        print(f"source: {message.source_url}  (updated {message.last_updated})")

    retrieval = result.trace.get("stage_6_8_retrieval")
    if retrieval:
        print(
            f"retrieval: top_score={retrieval['top_score']:.3f} "
            f"floor={retrieval['floor']} chunks={len(retrieval['chunks'])} "
            f"used_filter={retrieval['used_filter']}"
        )
    print("-" * 70)


def main() -> int:
    _force_utf8_console()
    print("Ask about the HDFC schemes. Blank line or 'q' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"q", "quit", "exit"}:
            break
        try:
            _show(question)
        except Exception as exc:  # keep the loop alive on any one failure
            print(f"error: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
