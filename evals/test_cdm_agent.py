"""Tests for the pluggable CDM agent backends (evals/cdm_agent.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cdm_agent import (
    AgentCDMResponse,
    MeditronBackend,
    MockBackend,
    make_backend,
)


def test_mock_backend_echoes_gold():
    gold = {"C1": {"icd": ["I50"], "rxnorm": ["4603"], "loinc": ["718-7"], "cpt": []}}
    be = make_backend("mock", gold_lookup=gold)
    r = be.propose("C1", "heart failure presentation")
    assert r.is_mocked is True
    assert r.backend == "mock"
    assert r.proposed_icd == ["I50"]
    assert r.proposed_loinc == ["718-7"]
    assert r.proposed_cpt == []


def test_mock_backend_unknown_case_returns_empty():
    be = make_backend("mock", gold_lookup={})
    r = be.propose("missing", "x")
    assert r.proposed_icd == [] and r.proposed_rxnorm == []


def test_make_backend_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown CDM backend"):
        make_backend("gpt9")


def test_meditron_parser_extracts_embedded_json():
    raw = 'Here is my answer: {"diagnosis_icd": ["E11"], "labs_loinc": ["4548-4"]} done.'
    parsed = MeditronBackend._parse(raw)
    assert parsed.proposed_icd == ["E11"]
    assert parsed.proposed_loinc == ["4548-4"]
    assert parsed.proposed_rxnorm == []


def test_meditron_parser_coerces_scalar_to_list():
    parsed = MeditronBackend._parse('{"diagnosis_icd": "I21.3"}')
    assert parsed.proposed_icd == ["I21.3"]


def test_meditron_parser_handles_garbage():
    parsed = MeditronBackend._parse("the model rambled with no json at all")
    assert parsed.proposed_icd == []
    assert parsed.backend == "meditron"


def test_meditron_backend_requires_ollama_or_raises():
    """Without the ollama client installed, construction must raise clearly."""
    try:
        import ollama  # noqa: F401
        pytest.skip("ollama installed; construction guard not exercised")
    except ImportError:
        with pytest.raises(RuntimeError, match="ollama"):
            MeditronBackend()


def test_agent_response_defaults():
    r = AgentCDMResponse(case_id="X")
    assert r.proposed_icd == [] and r.backend == "mock" and r.is_mocked is False
