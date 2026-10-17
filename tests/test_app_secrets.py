"""Deployment-configuration tests for the Streamlit UI (``app.py``).

The bridge from ``st.secrets`` to the environment is what makes a key added
on Streamlit Cloud visible to ``config.load_api_key``. A regression here
never fails locally - it silently turns every deployed answer into the
extractive path - so the bridge is covered without needing a deployed app.

Nothing in this module writes to disk or reads a real secrets file: the
``st.secrets`` object is replaced with a stand-in that yields the same
shapes (flat key, lower-case key, TOML section, missing file, broken TOML).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import app
from src import config, pipeline


class _SecretsStandIn:
    """Behaves like ``st.secrets`` for one fixed outcome."""

    def __init__(self, data: dict | None = None, error: BaseException | None = None):
        self._data = data or {}
        self._error = error

    def items(self):
        if self._error is not None:
            raise self._error
        return list(self._data.items())


@pytest.fixture()
def secrets_env(monkeypatch):
    """A controlled key-less environment with the bridge's globals restored."""
    saved_note = app._SECRETS_NOTE
    saved_key = os.environ.pop("GROQ_API_KEY", None)
    monkeypatch.setattr(config, "EXTRACTIVE_ONLY", False)
    monkeypatch.setattr(
        config, "PATHS", {**config.PATHS, "ENV_FILE": Path("no-such.env")}
    )
    yield
    app._SECRETS_NOTE = saved_note
    os.environ.pop("GROQ_API_KEY", None)
    if saved_key is not None:
        os.environ["GROQ_API_KEY"] = saved_key


# --------------------------------------------------------------------------- #
# st.secrets -> environment
# --------------------------------------------------------------------------- #


def test_canonical_key_disables_the_extractive_fallback(secrets_env, monkeypatch):
    monkeypatch.setattr(app.st, "secrets", _SecretsStandIn({"GROQ_API_KEY": "gsk_test"}))
    note = app._import_streamlit_secrets()

    assert os.environ["GROQ_API_KEY"] == "gsk_test"
    assert pipeline.llm_available()
    assert "GROQ_API_KEY" in note


def test_lower_case_key_is_published_anyway(secrets_env, monkeypatch):
    monkeypatch.setattr(app.st, "secrets", _SecretsStandIn({"groq_api_key": "gsk_test"}))
    note = app._import_streamlit_secrets()

    assert os.environ["GROQ_API_KEY"] == "gsk_test"
    assert pipeline.llm_available()
    assert "no GROQ_API_KEY" not in note


def test_key_inside_a_toml_section_is_published(secrets_env, monkeypatch):
    monkeypatch.setattr(
        app.st, "secrets", _SecretsStandIn({"groq": {"api_key": "gsk_test"}})
    )
    app._import_streamlit_secrets()

    assert os.environ["GROQ_API_KEY"] == "gsk_test"
    assert pipeline.llm_available()


def test_the_real_environment_is_not_overwritten(secrets_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
    monkeypatch.setattr(app.st, "secrets", _SecretsStandIn({"GROQ_API_KEY": "gsk_test"}))
    app._import_streamlit_secrets()

    assert os.environ["GROQ_API_KEY"] == "gsk_from_env"


def test_secrets_without_the_key_name_the_ones_found(secrets_env, monkeypatch):
    monkeypatch.setattr(app.st, "secrets", _SecretsStandIn({"OPENAI_BASE_URL": "x"}))
    note = app._import_streamlit_secrets()

    assert os.environ.get("GROQ_API_KEY") is None
    assert not pipeline.llm_available()
    assert "OPENAI_BASE_URL" in note and "no GROQ_API_KEY among them" in note


def test_a_value_is_never_echoed_into_the_note(secrets_env, monkeypatch):
    monkeypatch.setattr(app.st, "secrets", _SecretsStandIn({"GROQ_API_KEY": "gsk_secret"}))
    note = app._import_streamlit_secrets()

    assert "gsk_secret" not in note


# --------------------------------------------------------------------------- #
# failure modes the sidebar must distinguish
# --------------------------------------------------------------------------- #


def test_missing_secrets_file_is_reported_as_missing(secrets_env, monkeypatch):
    monkeypatch.setattr(
        app.st,
        "secrets",
        _SecretsStandIn(
            error=FileNotFoundError("No secrets found. Valid paths for a secrets.toml")
        ),
    )
    note = app._import_streamlit_secrets()

    assert note.startswith("secrets: no secrets.toml found for this app")
    assert os.environ.get("GROQ_API_KEY") is None
    assert not pipeline.llm_available()


def test_broken_toml_is_reported_as_invalid_toml(secrets_env, monkeypatch):
    monkeypatch.setattr(
        app.st,
        "secrets",
        _SecretsStandIn(
            error=FileNotFoundError(
                "Error parsing secrets file at .streamlit/secrets.toml: "
                "Unbalanced quotes (line 1 column 30 char 29)"
            )
        ),
    )
    note = app._import_streamlit_secrets()

    assert note.startswith("secrets: invalid TOML")


def test_an_unexpected_error_never_escapes_into_the_ui(secrets_env, monkeypatch):
    monkeypatch.setattr(
        app.st, "secrets", _SecretsStandIn(error=RuntimeError("boom"))
    )
    note = app._import_streamlit_secrets()

    assert note == "secrets: RuntimeError: boom"
    assert os.environ.get("GROQ_API_KEY") is None


def test_bridge_reports_before_it_has_run():
    """The sidebar must never show an unexplained blank when asked early."""
    assert app._SECRETS_NOTE.startswith("secrets:")


# --------------------------------------------------------------------------- #
# deployed revision
# --------------------------------------------------------------------------- #


def test_git_revision_is_a_short_sha_or_empty():
    """A clone reports its build; a tarball deploy reports nothing."""
    assert re.fullmatch(r"[0-9a-f]{7}|", app._git_revision())
