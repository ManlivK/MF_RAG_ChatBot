"""MF Facts - Streamlit chat UI.

Presentation only. No retrieval, generation, or policy decisions — those
live in ``src.pipeline``. See ARCHITECTURE.md and IMPLEMENTATION.md (Phase 8).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import streamlit as st

from src import config, pipeline
from src import vectorstore
from src.types import AssistantMessage

#: What the secrets bridge found, for the sidebar. Never contains a value.
_SECRETS_NOTE = "secrets: not read yet"


def _publish_secret(name: str, value: object, found: list[str]) -> None:
    """Publish one secret as an environment variable, case-insensitively.

    Accepts ``GROQ_API_KEY``, ``groq_api_key`` and ``[groq] api_key`` alike -
    a key that is present but spelled differently must not silently disable
    the model.

    Args:
        name: The secret's name (section prefix already joined with ``_``).
        value: The secret's value, or a nested mapping for a TOML section.
        found: Accumulates the names that were seen, for the sidebar note.
    """
    if isinstance(value, str):
        os.environ.setdefault(name.upper(), value)
        found.append(name)
    elif isinstance(value, Mapping):
        for sub_name, sub_value in value.items():
            _publish_secret(f"{name}_{sub_name}", sub_value, found)


def _import_streamlit_secrets() -> str:
    """Publish Streamlit secrets as environment variables.

    Streamlit Cloud stores app secrets in a TOML file that only
    ``st.secrets`` can read, while ``config.load_api_key`` reads the
    environment. Without this bridge a key added under Settings → Secrets is
    invisible to the pipeline and every answer silently degrades to the
    extractive path.

    Returns:
        A note describing what was found - key *names* and error types only,
        never a value. It is rendered in the sidebar so a misconfigured
        deployment says which of the three failure modes it is in.
    """
    global _SECRETS_NOTE
    try:
        items = list(st.secrets.items())
    except FileNotFoundError as exc:
        # One exception class covers both "no file" and "file does not parse".
        first_line = (str(exc).strip().splitlines() or [""])[0]
        if "parsing" in first_line.lower():
            _SECRETS_NOTE = f"secrets: invalid TOML — {first_line[:160]}"
        else:
            _SECRETS_NOTE = (
                "secrets: no secrets.toml found for this app "
                "(set GROQ_API_KEY in .env locally, or in Settings → Secrets online)"
            )
        return _SECRETS_NOTE
    except Exception as exc:  # noqa: BLE001 - bad TOML must not take the UI down
        _SECRETS_NOTE = f"secrets: {type(exc).__name__}: {exc}"
        return _SECRETS_NOTE

    found: list[str] = []
    for name, value in items:
        _publish_secret(str(name), value, found)
    if os.environ.get("GROQ_API_KEY"):
        _SECRETS_NOTE = f"secrets: published {', '.join(sorted(set(found)))}"
    else:
        seen = ", ".join(sorted(set(found))) or "nothing usable"
        _SECRETS_NOTE = f"secrets: found {seen} — no GROQ_API_KEY among them"
    return _SECRETS_NOTE


def _git_revision() -> str:
    """Short commit the running app was built from, or ``""`` outside a clone."""
    git_dir = Path(__file__).resolve().parent / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(" ", 1)[1].strip()
        packed = git_dir / "packed-refs"
        if (git_dir / ref).exists():
            return (git_dir / ref).read_text(encoding="ascii").strip()[:7]
        if packed.exists():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line.endswith(f" {ref}") or line.endswith(ref):
                    return line.split()[0][:7]
    except Exception:  # noqa: BLE001 - a missing .git is normal for a tarball deploy
        return ""
    return ""


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
            scheme = chunk.get("scheme_name") or ""
            title = f"{label} · {scheme}" if scheme else label
            st.write(f"[{chunk['rank']}] {title} · score={chunk.get('score')}")
            st.caption(chunk.get("preview") or "")


def _live_chunk_count() -> int:
    """Chunks actually present in the Chroma collection.

    ``data/index_meta.json`` is committed while ``data/chroma/`` is not, so a
    fresh deployment reports a healthy, 228-chunk index while the collection it
    describes does not exist yet - and every query then returns nothing. The
    live count, not the metadata file, is what decides whether retrieval can
    work.
    """
    try:
        return vectorstore.get_collection().count()
    except Exception:  # noqa: BLE001 - a missing/unopenable collection is "0"
        return 0


def _ensure_index() -> tuple[bool, str]:
    """Build the index on first load when no live collection exists.

    Returns:
        ``(searchable, reason)``. Any exception is caught and reported rather
        than raised: the UI must render a warning, never a crash.
    """
    stale, reason = vectorstore.is_index_stale()
    if not stale and _live_chunk_count():
        return True, "ok"
    try:
        from run_index import main as run_main

        with st.spinner(
            "Building the search index (first load only - downloads the embedding "
            "model and embeds the committed corpus)... "
        ):
            if not vectorstore.load_chunks():
                run_main(["--stage", "chunk"])
            run_main(["--stage", "index", "--rebuild"])
            vectorstore.reset_collection_cache()
    except Exception as exc:  # noqa: BLE001 - degrade to a warning, not a crash
        return False, f"index build failed: {exc}"
    count = _live_chunk_count()
    if not count:
        return False, "index build produced no chunks"
    return True, "ok"


def _corpus_sidebar(searchable: bool, reason: str) -> None:
    """Corpus stats, index health, the source list and the index controls.

    Args:
        searchable: Whether the collection can actually answer queries (the
            result of the startup :func:`_ensure_index` attempt).
        reason: ``"ok"`` when searchable, otherwise why it is not.
    """
    with st.sidebar:
        st.subheader("Corpus")
        stale, stale_reason = vectorstore.is_index_stale()
        stats = vectorstore.get_stats()
        if not searchable:
            st.warning(f"No searchable index: {reason}. Use **Rebuild index** below.")
        elif stale:
            st.warning(f"Index stale: {stale_reason}. Rebuild it below.")
        else:
            st.success("Index built")
        st.caption(f"Docs: {stats.get('doc_count') or 0}")
        st.caption(f"Chunks: {stats.get('chunk_count') or 0}")
        st.caption(f"Embedding: {config.EMBED_MODEL}")
        st.caption(f"Vector DB: ChromaDB / {config.CHROMA_COLLECTION}")
        st.caption(f"Last indexed: {stats.get('built_at') or '—'}")
        if pipeline.llm_available():
            st.caption(f"Answers: {config.GENERATOR_MODEL}")
        else:
            st.caption("Answers: corpus text only — the model is not called.")
            if config.EXTRACTIVE_ONLY:
                st.caption("reason: EXTRACTIVE_ONLY=1 is set")
            st.caption(_SECRETS_NOTE)
            revision = _git_revision()
            if revision:
                st.caption(f"build: {revision}")

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
    _import_streamlit_secrets()

    st.title("MF Facts")
    st.write(config.WELCOME_LINE)
    st.info(config.DISCLAIMER)

    searchable, reason = _ensure_index()
    _corpus_sidebar(searchable, reason)
    if not searchable:
        st.warning(
            "No searchable index found and the automatic build did not complete "
            f"({reason}). Click **Rebuild index** in the sidebar to build it from "
            "the committed corpus."
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
