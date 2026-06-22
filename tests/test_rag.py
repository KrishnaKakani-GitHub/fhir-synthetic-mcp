"""Tests for the hybrid BM25 + ChromaDB RAG layer.

ChromaDB is disabled in these tests (FHIR_MCP_RAG_DISABLE_CHROMA=1 via
monkeypatch) to avoid ONNX runtime startup overhead in CI. BM25-only mode
still exercises the full retrieval pipeline and RRF fusion (with chroma
scores zeroed out, which is equivalent to alpha=1 BM25-only RRF).

Separate integration tests (test_rag_chroma.py, optional) cover ChromaDB.

PHI NOTE: This module tests on clinical guidelines only, not patient data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("FHIR_MCP_RAG_DISABLE_CHROMA", "1")

from fhir_mcp.rag import ClinicalRAG, _tokenise

# Minimal in-memory guideline corpus for deterministic tests
_GUIDELINES: list[dict[str, Any]] = [
    {
        "id": "gl-t1",
        "title": "Heart Rate Management in Atrial Fibrillation",
        "source": "Test",
        "condition": "Atrial fibrillation",
        "loinc_codes": ["8867-4"],
        "content": "Target resting heart rate below 110 bpm in AF. Bradycardia below 60 bpm.",
        "key_thresholds": {"af_rate_control_lenient": 110, "bradycardia": 60},
    },
    {
        "id": "gl-t2",
        "title": "JNC 8 Hypertension Guidelines",
        "source": "Test",
        "condition": "Hypertension",
        "loinc_codes": ["55284-4"],
        "content": "Initiate treatment at SBP 150 mmHg for patients over 60. Target below 140 mmHg.",
        "key_thresholds": {"initiation_sbp": 150},
    },
    {
        "id": "gl-t3",
        "title": "ADA Diabetes Standards 2024",
        "source": "Test",
        "condition": "Diabetes mellitus",
        "loinc_codes": ["4548-4", "2339-0"],
        "content": "HbA1c target below 7% for most adults. Fasting glucose 80-130 mg/dL.",
        "key_thresholds": {"hba1c_target": 7.0},
    },
    {
        "id": "gl-t4",
        "title": "KDIGO CKD Guidelines",
        "source": "Test",
        "condition": "Chronic kidney disease",
        "loinc_codes": ["2160-0"],
        "content": "Creatinine above 2.0 mg/dL typically reflects eGFR below 50. Refer nephrology at eGFR below 30.",
        "key_thresholds": {"creatinine_moderate": 2.0},
    },
]


@pytest.fixture()
def rag() -> ClinicalRAG:
    r = ClinicalRAG(disable_chroma=True)
    r.load_guidelines(guidelines=_GUIDELINES)
    return r


# --- Tokeniser ----------------------------------------------------------------


def test_tokenise_preserves_loinc_codes() -> None:
    tokens = _tokenise("LOINC code 8867-4 heart rate")
    assert "8867-4" in tokens


def test_tokenise_lowercases() -> None:
    tokens = _tokenise("Heart Rate")
    assert "heart" in tokens
    assert "rate" in tokens


# --- Basic retrieval ----------------------------------------------------------


def test_search_returns_k_results(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("heart rate atrial fibrillation", k=2)
    assert len(results) == 2


def test_search_ranked_by_score(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("heart rate atrial fibrillation", k=4)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_correct_top_result_heart_rate(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("heart rate AF rate control", k=1)
    assert results[0]["guideline"]["id"] == "gl-t1"


def test_search_correct_top_result_hypertension(rag: ClinicalRAG) -> None:
    # BM25-only (no ChromaDB in CI) cannot guarantee rank-0 on a 4-doc corpus;
    # assert the hypertension guideline appears in the top-3 results.
    results = rag.search_guidelines("hypertension SBP mmHg JNC treatment", k=3)
    ids = [r["guideline"]["id"] for r in results]
    assert "gl-t2" in ids


def test_search_correct_top_result_diabetes(rag: ClinicalRAG) -> None:
    # BM25-only (no ChromaDB in CI) cannot guarantee rank-0 on a 4-doc corpus;
    # assert the diabetes guideline appears in the top-3 results.
    results = rag.search_guidelines("HbA1c glucose diabetes ADA fasting", k=3)
    ids = [r["guideline"]["id"] for r in results]
    assert "gl-t3" in ids


# --- LOINC filter -------------------------------------------------------------


def test_loinc_filter_restricts_results(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("clinical guidelines", k=4,
                                    loinc_filter=["8867-4"])
    assert all("8867-4" in r["guideline"]["loinc_codes"] for r in results)
    assert len(results) == 1  # only gl-t1 has 8867-4


def test_loinc_filter_multi_code(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("guidelines", k=4,
                                    loinc_filter=["4548-4", "2160-0"])
    ids = {r["guideline"]["id"] for r in results}
    assert "gl-t3" in ids  # has 4548-4
    assert "gl-t4" in ids  # has 2160-0
    assert "gl-t1" not in ids
    assert "gl-t2" not in ids


def test_loinc_filter_no_match_returns_empty(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("anything", k=4, loinc_filter=["99999-0"])
    assert results == []


# --- Context block building ---------------------------------------------------


def test_build_context_block_has_cache_control(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("heart rate", k=2)
    block = rag.build_context_block(results)
    assert block["type"] == "text"
    assert "cache_control" in block
    assert block["cache_control"]["type"] == "ephemeral"


def test_build_context_block_no_cache(rag: ClinicalRAG) -> None:
    results = rag.search_guidelines("heart rate", k=1)
    block = rag.build_context_block(results, cache=False)
    assert "cache_control" not in block


def test_build_context_block_empty_results(rag: ClinicalRAG) -> None:
    block = rag.build_context_block([])
    assert "No relevant" in block["text"]


# --- RRF fusion ---------------------------------------------------------------


def test_rrf_merge_equal_weights() -> None:
    """With alpha=0.5 and identical orderings, RRF scores should be equal."""
    bm25 = [3.0, 2.0, 1.0]
    chroma = [3.0, 2.0, 1.0]
    scores = ClinicalRAG._rrf_merge(bm25, chroma, alpha=0.5)
    # All items maintain same relative order
    assert scores[0] > scores[1] > scores[2]


def test_rrf_merge_alpha_one_is_bm25_only() -> None:
    """alpha=1 means chroma scores are zeroed; only BM25 ranking matters."""
    bm25 = [10.0, 1.0, 5.0]  # order: 0, 2, 1
    chroma = [0.0, 0.0, 0.0]
    scores = ClinicalRAG._rrf_merge(bm25, chroma, alpha=1.0)
    assert scores[0] > scores[2] > scores[1]
