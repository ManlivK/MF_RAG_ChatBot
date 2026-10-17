"""MF Facts - Streamlit chat UI.

Presentation only. No retrieval, generation, or policy decisions — those
live in ``src.pipeline``. See ARCHITECTURE.md and IMPLEMENTATION.md (Phase 8).
"""

from __future__ import annotations

import json

import streamlit as st

from src import config, pipeline
from src import vectorstore
from src.types import AssistantMessage


def render_message(msg: AssistantMessage, trace: dict | None = None) -> None:
    """Render one finalised assistant turn.

    An answer gets its body, its single verified link, the freshness stamp and
    the disclaimer. A refusal renders as an info block with its educational link
    and no citation badge (IMPLEMENTATION.md Phase 8.2).
    """
    if msg.kind == "answer":
        st.markdown(msg.text)
        if msg.source_url:
            st.link_button("Source", msg.source_url)
        if msg.last_updated:
            st.caption(f"{config.LAST_UPDATED_PREFIX}{msg.last_updated}")
        if msg.cached:
            st.badge("cached")
    else:
        st.info(msg.text)
        if msg.source_url:
            st.link_button("Learn more", msg.source_url)
    st.caption(config.DISCLAIMER)


def _render_pipeline_expander(trace: dict) -> None:
    """The raw stage trace from ARCHITECTURE.md section 5.4."""
    with st.expander("Show pipeline", expanded=False):
        st.json(trace, expanded=False)


def _render_sources_expander(trace: dict) -> None:
    """Top-3 retrieved chunks: label, section, score and a short preview."""
    chunks = (trace or {}).get("stage_6_8_retrieval", {}).get("chunks", [])
    if not chunks:
        return
    with st.expander("Sources", expanded=False):
        for chunk in chunks[:3]:
            label = chunk.get("label") or chunk.get("section") or ""
            st.write(f"[{chunk['rank']}] {label} · score={chunk.get('score')}")
            st.caption(chunk.get("preview") or "")


def _corpus_sidebar() -> None:
    """Corpus stats, index health, the source list and the index controls."""
    with st.sidebar:
        st.subheader("Corpus")
        stale, reason = vectorstore.is_index_stale()
        stats = vectorstore.get_stats()
        if stale:
            st.warning(f"Index stale: {reason}. Rebuild it below.")
        else:
            st.success("Index built")
        st.caption(f"Docs: {stats.get('doc_count') or 0}")
        st.caption(f"Chunks: {stats.get('chunk_count') or 0}")
        st.caption(f"Embedding: {config.EMBED_MODEL}")
        st.caption(f"Vector DB: ChromaDB / {config.CHROMA_COLLECTION}")
        st.caption(f"Last indexed: {stats.get('built_at') or '—'}")

        st.dataframe(
            [
                {
                    "Scheme": spec.scheme_name,
                    "Category": spec.category,
                    "URL": spec.url,
                }
                for spec in config.SOURCES
            ],
            width="stretch",
        )

        st.subheader("Index")
        if st.button("Clear answer cache"):
            pipeline.clear_cache()
            st.toast("Answer cache cleared")
            st.rerun()
        if st.button("Rebuild index"):
            try:
                from run_index import main as run_main

                run_main(["--stage", "index", "--rebuild"])
                vectorstore.reset_collection_cache()
                st.toast("Index rebuilt")
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - the sidebar must not crash
                st.error(f"Rebuild failed: {exc}")


def _parse_assistant_turn(message: dict) -> dict | None:
    """Turn one stored assistant payload into ``{"final": ..., "trace": ...}``.

    Returns ``None`` when the payload cannot be rendered (corrupt JSON, a
    payload written by an older build, or a message missing required fields).
    Rendering must never raise: one bad turn would blank the whole chat.
    """
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
        final = AssistantMessage(**payload["message"])
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    trace = payload.get("trace") or {}
    if not isinstance(trace, dict):
        trace = {}
    return {"final": final, "trace": trace}


def _run_query(prompt: str) -> None:
    """Send one user prompt through the pipeline and store the turn.

    A pipeline failure must surface as an inline error, never as an unhandled
    exception that blanks the page mid-rerun.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        result = pipeline.answer(prompt)
    except Exception as exc:  # noqa: BLE001 - the UI must survive any pipeline error
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "message": {
                            "kind": "refusal",
                            "text": (
                                "Something went wrong while running that question. "
                                "Please try again."
                            ),
                            "source_url": "",
                            "last_updated": "",
                            "disclaimer": config.DISCLAIMER,
                            "truncated": False,
                            "link_substituted": False,
                            "cached": False,
                        },
                        "trace": {"error": repr(exc)},
                    },
                    ensure_ascii=False,
                ),
            }
        )
        return
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {"message": result.message.__dict__, "trace": result.trace},
                ensure_ascii=False,
            ),
        }
    )


def main() -> None:
    """Render the single-page chat surface."""
    st.set_page_config(
        page_title="MF Facts - HDFC Scheme FAQ",
        page_icon="📊",
        layout="centered",
    )

    st.title("MF Facts")
    st.write(config.WELCOME_LINE)
    st.info(config.DISCLAIMER)

    _corpus_sidebar()

    stats = vectorstore.get_stats()
    stale, _reason = vectorstore.is_index_stale()
    if stale or not stats.get("chunk_count"):
        st.warning(
            "No searchable index found. Click **Rebuild index** in the sidebar "
            "to build it from the committed corpus."
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for question in config.EXAMPLE_QUESTIONS:
        if st.button(question, key=f"example::{question}"):
            _run_query(question)
            st.rerun()

    kept: list[dict] = []
    for message in st.session_state.messages:
        if message.get("role") == "user":
            kept.append(message)
            with st.chat_message("user"):
                st.write(message.get("content", ""))
            continue
        turn = _parse_assistant_turn(message)
        if turn is None:
            # Stale session state from an older build: drop the payload rather
            # than let one KeyError blank the whole chat.
            continue
        kept.append(message)
        with st.chat_message("assistant"):
            render_message(turn["final"], turn["trace"])
            _render_sources_expander(turn["trace"])
    if len(kept) != len(st.session_state.messages):
        st.session_state.messages = kept

    prompt = st.chat_input("Ask a factual question about these 5 HDFC schemes")
    if prompt:
        _run_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
