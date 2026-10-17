"""Test suite for MF Facts.

Layering follows ARCHITECTURE.md section 10:

* Layer 1 (here) - unit tests, no network and no built index required.
* Layer 2 - integration, needs the corpus and a built index.
* Layer 3 - the end-to-end eval harness in ``deliverables/run_eval.py``.
"""
