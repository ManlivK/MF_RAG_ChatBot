"""CLI entry point for the offline pipeline.

Phase 0 ships this as a stage dispatcher with every stage unimplemented, so the
team can verify the environment before any real logic lands.

Usage:
    python run_index.py --stage all
    python run_index.py --stage ingest
"""

from __future__ import annotations

import argparse
import sys

#: stage name -> (implementing module, human label). Added as each phase lands.
STAGES: dict[str, tuple[str, str]] = {
    "ingest": ("src.ingest", "Stage 1 - load and clean the 5 scheme pages"),
    "chunk": ("src.chunking", "Stage 2 - structure-aware chunking"),
    "index": ("src.vectorstore", "Stages 3-4 - embed with MiniLM, store in ChromaDB"),
    "eval-retrieval": ("src.retriever", "Stages 6-8 - retrieval quality report"),
    "smoke": ("src.pipeline", "Stages 5-11 - end-to-end smoke test"),
    "all": ("", "Full rebuild: ingest -> chunk -> index"),
}

#: Order used by the ``all`` stage. ``ingest`` and ``chunk`` are skipped when the
#: committed snapshot and chunks are already present, so a routine index rebuild
#: does not re-hit the network.
ALL_STAGES: tuple[str, ...] = ("ingest", "chunk", "index")


def _run_single(stage: str, rebuild: bool) -> int:
    """Import one stage module and call its runner."""
    module_name, label = STAGES[stage]
    print(f"[{stage}] {label}")
    try:
        module = __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError:
        print(f"  not implemented yet - {module_name}.py does not exist")
        return 1
    runner = (
        getattr(module, "run_all", None)
        or getattr(module, "run_ingest", None)
        or getattr(module, "run_chunking", None)
        or getattr(module, "main", None)
    )
    if runner is None:
        print(f"  {module_name} has no run_all()/run_ingest()/main() entry point yet")
        return 1
    try:
        result = runner(rebuild=rebuild) if rebuild else runner()
    except TypeError:
        result = runner()
    if isinstance(result, list):
        return 0
    return 0 if result in (0, None, True) else 1


def _run_stage(stage: str, rebuild: bool) -> int:
    """Dispatch one stage, or every stage in dependency order for ``all``."""
    if stage != "all":
        return _run_single(stage, rebuild)
    print("[all] Full rebuild: ingest -> chunk -> index")
    for name in ALL_STAGES:
        code = _run_single(name, rebuild)
        if code != 0:
            return code
    return 0


def _force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 with replacement, so a corpus containing the rupee
    sign (U+20B9) prints on a default Windows console instead of raising
    ``UnicodeEncodeError`` under cp1252."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - already-detached stream
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    parser = argparse.ArgumentParser(
        prog="run_index.py",
        description="Build and inspect the MF Facts RAG index.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=sorted(STAGES),
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the ChromaDB collection instead of upserting.",
    )
    args = parser.parse_args(argv)
    return _run_stage(args.stage, args.rebuild)


if __name__ == "__main__":
    sys.exit(main())
