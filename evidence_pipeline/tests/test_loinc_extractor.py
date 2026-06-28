"""Tests for evidence_pipeline/extraction/loinc_extractor.py"""
from __future__ import annotations

from evidence_pipeline.extraction.loinc_extractor import (
    ExtractionResult,
    ExtractedObservation,
    batch_extract,
    extract_from_note,
)
from evidence_pipeline.datasets.mimic import generate_synthetic_notes


# --- unit tests on known patterns ------------------------------------------

def test_extracts_hemoglobin() -> None:
    result = extract_from_note("N001", "Hemoglobin: 8.2 g/dL")
    codes = result.loinc_codes
    assert "718-7" in codes

def test_extracts_creatinine() -> None:
    result = extract_from_note("N002", "Creatinine: 1.8 mg/dL")
    assert "2160-0" in result.loinc_codes

def test_extracts_hba1c() -> None:
    result = extract_from_note("N003", "HbA1c: 9.4 %")
    assert "4548-4" in result.loinc_codes

def test_extracts_bnp() -> None:
    result = extract_from_note("N004", "BNP: 1840 pg/mL")
    assert "42637-9" in result.loinc_codes

def test_extracts_spo2() -> None:
    result = extract_from_note("N005", "SpO2: 94%")
    assert "59408-5" in result.loinc_codes

def test_extracts_blood_pressure() -> None:
    result = extract_from_note("N006", "Blood pressure: 150/90 mmHg")
    assert "55284-4" in result.loinc_codes

def test_extracts_wbc() -> None:
    result = extract_from_note("N007", "WBC: 14.2 x10^9/L")
    assert "6690-2" in result.loinc_codes

def test_extracts_ldh() -> None:
    result = extract_from_note("N008", "LDH: 1420 U/L")
    assert "2532-0" in result.loinc_codes

def test_empty_note_returns_no_observations() -> None:
    result = extract_from_note("N009", "Patient stable. No labs today.")
    assert result.n == 0

def test_observation_fields() -> None:
    result = extract_from_note("N010", "Hemoglobin: 7.8 g/dL")
    obs = result.observations[0]
    assert isinstance(obs, ExtractedObservation)
    assert obs.note_id == "N010"
    assert obs.loinc_code == "718-7"
    assert obs.raw_value == "7.8"
    assert obs.unit_hint == "g/dL"


# --- batch extraction over synthetic notes ---------------------------------

def test_batch_extract_all_synthetic_notes() -> None:
    notes = generate_synthetic_notes()
    results = batch_extract(notes)
    assert len(results) == len(notes)

def test_batch_extract_total_observations_nonzero() -> None:
    notes = generate_synthetic_notes()
    results = batch_extract(notes)
    total = sum(r.n for r in results)
    assert total >= 20, f"Expected >=20 observations across 10 notes, got {total}"

def test_pnh_note_extracts_haptoglobin_and_ldh() -> None:
    notes = generate_synthetic_notes()
    pnh_note = next(n for n in notes if n.note_id == "SYN001")
    result = extract_from_note(pnh_note.note_id, pnh_note.text)
    codes = result.loinc_codes
    assert "13945-1" in codes, "Expected haptoglobin LOINC 13945-1"
    assert "2532-0" in codes,  "Expected LDH LOINC 2532-0"
    assert "718-7" in codes,   "Expected hemoglobin LOINC 718-7"

def test_hf_note_extracts_bnp() -> None:
    notes = generate_synthetic_notes()
    hf_note = next(n for n in notes if n.note_id == "SYN004")
    result = extract_from_note(hf_note.note_id, hf_note.text)
    assert "42637-9" in result.loinc_codes or "33762-6" in result.loinc_codes
