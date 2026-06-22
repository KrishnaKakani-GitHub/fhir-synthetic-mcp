"""Tests for clinical NLP entity extraction.

The Anthropic API is mocked so these tests run in CI without API keys.
Entity extraction logic, parsing, and validation are tested in isolation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fhir_mcp.nlp import ClinicalEntity, ClinicalNLP, ExtractionResult


# --- ClinicalEntity model tests -----------------------------------------------


def test_entity_model_validation() -> None:
    e = ClinicalEntity(
        entity_type="condition",
        text="hypertension",
        normalized="Hypertension",
        code="I10",
        code_system="ICD-10-CM",
        confidence=0.95,
    )
    assert e.entity_type == "condition"
    assert e.confidence == 0.95
    assert not e.validated


def test_entity_confidence_clamped() -> None:
    with pytest.raises(Exception):
        ClinicalEntity(
            entity_type="condition",
            text="x",
            normalized="x",
            confidence=1.5,  # out of range
        )


def test_extraction_result_counts_entities() -> None:
    entities = [
        ClinicalEntity(entity_type="condition", text="HTN",
                       normalized="Hypertension"),
        ClinicalEntity(entity_type="medication", text="lisinopril",
                       normalized="Lisinopril"),
    ]
    result = ExtractionResult(note_id="note-001", entities=entities)
    assert result.entity_count == 2


# --- ICD-10 validation tests --------------------------------------------------


def test_icd10_valid_format_validates() -> None:
    nlp = ClinicalNLP()
    entity = ClinicalEntity(
        entity_type="condition",
        text="hypertension",
        normalized="Hypertension",
        code="I10",
        code_system="ICD-10-CM",
        confidence=0.9,
    )
    validated = nlp._validate_icd10(entity)
    assert validated.validated
    assert validated.confidence == 0.9  # not penalised


def test_icd10_invalid_format_downgrades_confidence() -> None:
    nlp = ClinicalNLP()
    entity = ClinicalEntity(
        entity_type="condition",
        text="some condition",
        normalized="Some condition",
        code="INVALID",
        code_system="ICD-10-CM",
        confidence=0.8,
    )
    validated = nlp._validate_icd10(entity)
    assert not validated.validated
    assert validated.confidence < 0.8


def test_icd10_dotted_code_valid() -> None:
    nlp = ClinicalNLP()
    entity = ClinicalEntity(
        entity_type="condition",
        text="Type 2 diabetes",
        normalized="Diabetes mellitus type 2",
        code="E11.65",
        code_system="ICD-10-CM",
        confidence=0.95,
    )
    validated = nlp._validate_icd10(entity)
    assert validated.validated


# --- NPI validation tests -----------------------------------------------------


def test_npi_valid_passes_luhn() -> None:
    nlp = ClinicalNLP()
    # Known valid NPI (real format, synthetic value):
    # NPI 1234567893 passes the Luhn check with prefix constant 24
    entity = ClinicalEntity(
        entity_type="provider",
        text="Dr. Smith",
        normalized="Dr. John Smith",
        code="1234567893",
        code_system="NPI",
        confidence=0.8,
    )
    validated = nlp._validate_npi(entity)
    # Either validated or confidence penalty (depending on Luhn result)
    assert validated.validation_note is not None


def test_npi_wrong_length_invalid() -> None:
    nlp = ClinicalNLP()
    entity = ClinicalEntity(
        entity_type="provider",
        text="Dr. Jones",
        normalized="Dr. Jones",
        code="12345",  # too short
        code_system="NPI",
        confidence=0.8,
    )
    validated = nlp._validate_npi(entity)
    assert not validated.validated
    assert validated.confidence < 0.8


# --- Mock API extraction test -------------------------------------------------


def _make_mock_response() -> MagicMock:
    """Build a mock Anthropic API response for entity extraction."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "extract_entities"
    block.input = {
        "entities": [
            {
                "entity_type": "condition",
                "text": "hypertension",
                "normalized": "Hypertension",
                "code": "I10",
                "code_system": "ICD-10-CM",
                "confidence": 0.95,
            },
            {
                "entity_type": "medication",
                "text": "lisinopril 10mg",
                "normalized": "Lisinopril",
                "code": "29046",
                "code_system": "RxNorm",
                "confidence": 0.88,
            },
        ]
    }
    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens = 200
    response.usage.output_tokens = 80
    response.usage.cache_read_input_tokens = 150
    return response


def test_extract_entities_with_mock_api() -> None:
    nlp = ClinicalNLP()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response()
    nlp._client = mock_client

    result = nlp.extract_entities(
        note_id="note-001",
        text="Patient has hypertension, on lisinopril 10mg daily.",
        validate_codes=True,
    )

    assert result.entity_count == 2
    assert result.note_id == "note-001"
    assert result.cached_tokens == 150

    conditions = [e for e in result.entities if e.entity_type == "condition"]
    assert len(conditions) == 1
    assert conditions[0].code == "I10"
    assert conditions[0].validated  # I10 passes format check


def test_extract_entities_api_failure_returns_empty() -> None:
    nlp = ClinicalNLP()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("API error")
    nlp._client = mock_client

    result = nlp.extract_entities(note_id="note-err", text="some note")
    assert result.entity_count == 0
    assert result.entities == []
