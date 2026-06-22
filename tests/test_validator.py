"""Tests for the deterministic LOINC validation gate.

The validator is called from store.stage_write() — it is a hard gate,
not a soft warning. These tests confirm the gate behaviour directly
(validator.py) and also through the store integration path.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from fhir_mcp.models import ProposedObservation
from fhir_mcp.validator import (
    ValidationError,
    ValidationResult,
    validate_observation,
    reload_rules,
)

DATA_SRC = Path(__file__).resolve().parents[1] / "data" / "synthetic_patients.json"


def _obs(
    code: str = "8867-4",
    value: float = 75,
    unit: str = "/min",
    display: str = "Heart rate",
) -> ProposedObservation:
    return ProposedObservation(
        patient_id="pat-001",
        code=code,
        display=display,
        value=value,
        unit=unit,
        effective_date=date(2026, 6, 1),
    )


# --- Valid cases --------------------------------------------------------------


def test_valid_heart_rate() -> None:
    result = validate_observation(_obs(code="8867-4", value=75, unit="/min"))
    assert result.ok
    assert not result.violations


def test_valid_sbp() -> None:
    result = validate_observation(_obs(code="55284-4", value=120, unit="mm[Hg]",
                                       display="Systolic blood pressure"))
    assert result.ok


def test_valid_temperature_fahrenheit() -> None:
    result = validate_observation(_obs(code="8310-5", value=98.6, unit="degF",
                                       display="Body temperature"))
    assert result.ok


def test_valid_spo2() -> None:
    result = validate_observation(_obs(code="59408-5", value=98, unit="%",
                                       display="SpO2"))
    assert result.ok


# --- Boundary violations ------------------------------------------------------


def test_rejects_heart_rate_too_high() -> None:
    result = validate_observation(_obs(code="8867-4", value=350, unit="/min"))
    assert not result.ok
    assert any("350" in v for v in result.violations)


def test_rejects_heart_rate_too_low() -> None:
    result = validate_observation(_obs(code="8867-4", value=10, unit="/min"))
    assert not result.ok


def test_rejects_sbp_above_max() -> None:
    result = validate_observation(
        _obs(code="55284-4", value=400, unit="mm[Hg]", display="SBP")
    )
    assert not result.ok


def test_rejects_temperature_below_range() -> None:
    result = validate_observation(_obs(code="8310-5", value=75, unit="degF",
                                       display="Body temperature"))
    assert not result.ok


# --- Unit mismatch ------------------------------------------------------------


def test_rejects_wrong_unit_heart_rate() -> None:
    result = validate_observation(_obs(code="8867-4", value=75, unit="bpm"))
    assert not result.ok
    assert any("unit" in v.lower() or "bpm" in v for v in result.violations)


def test_rejects_wrong_unit_temperature() -> None:
    """Celsius submitted for a rule that expects Fahrenheit."""
    result = validate_observation(_obs(code="8310-5", value=37.0, unit="Cel",
                                       display="Body temperature"))
    assert not result.ok


# --- Unknown code handling ----------------------------------------------------


def test_unknown_code_strict_mode_rejects() -> None:
    result = validate_observation(_obs(code="99999-9", value=10, unit="unit"))
    assert not result.ok
    assert any("not in the clinical registry" in v for v in result.violations)


def test_unknown_code_non_strict_accepts() -> None:
    result = validate_observation(
        _obs(code="99999-9", value=10, unit="unit"),
        strict_unknown=False,
    )
    assert result.ok
    assert result.warnings  # warning is emitted


# --- Clinical flag (warning, not rejection) -----------------------------------


def test_flag_above_threshold_is_warning_not_error() -> None:
    """HR 220 is above flag_above=200 but within max=300 — warning, not rejection."""
    result = validate_observation(_obs(code="8867-4", value=220, unit="/min"))
    assert result.ok
    assert result.warnings
    assert any("220" in w or "flag" in w.lower() or "200" in w
               for w in result.warnings)


# --- Store integration --------------------------------------------------------


def test_store_stage_write_rejects_invalid_loinc(tmp_path: Path) -> None:
    """Validator is wired into store.stage_write() and raises StoreError."""
    from fhir_mcp.store import FhirStore, StoreError

    store = FhirStore(tmp_path / "v.db")
    store.import_from_json(DATA_SRC)

    with pytest.raises(StoreError, match="Proposal rejected"):
        store.stage_write(_obs(code="8867-4", value=500, unit="/min"))


def test_store_stage_write_accepts_valid_observation(tmp_path: Path) -> None:
    from fhir_mcp.store import FhirStore

    store = FhirStore(tmp_path / "v2.db")
    store.import_from_json(DATA_SRC)
    pending = store.stage_write(_obs(code="8867-4", value=72, unit="/min"))
    assert pending.write_id.startswith("pw-")


def test_store_stage_write_carries_warnings(tmp_path: Path) -> None:
    """A flagged-but-valid value should produce a pending write with warnings."""
    from fhir_mcp.store import FhirStore

    store = FhirStore(tmp_path / "v3.db")
    store.import_from_json(DATA_SRC)
    pending = store.stage_write(_obs(code="8867-4", value=220, unit="/min"))
    assert pending.validation_warnings  # clinical flags surfaced to approver
