"""Tests for evidence_pipeline/pipeline/end_to_end.py"""
from __future__ import annotations

from evidence_pipeline.datasets.mimic import generate_synthetic_notes
from evidence_pipeline.pipeline.end_to_end import (
    HumanGate,
    PipelineMetrics,
    ValidationResult,
    run_pipeline,
)


# --- HumanGate unit tests --------------------------------------------------

def test_human_gate_committed_zero_in_automated_mode() -> None:
    gate = HumanGate()
    vr = ValidationResult(
        observation_id="N001:718-7:10",
        loinc_code="718-7",
        note_id="N001",
        display_name="Hemoglobin",
        raw_value="7.8",
        passed=True,
        reason="",
    )
    gate.propose(vr)
    # Nothing committed -- human approval required
    assert gate.committed_count == 0
    assert gate.pending_count == 1

def test_human_gate_approve_increments_committed() -> None:
    gate = HumanGate()
    obs_id = "N001:718-7:10"
    vr = ValidationResult(obs_id, "718-7", "N001", "Hemoglobin", "7.8", True, "")
    gate.propose(vr)
    assert gate.approve(obs_id)
    assert gate.committed_count == 1
    assert gate.pending_count == 0

def test_human_gate_audit_log_populated() -> None:
    gate = HumanGate()
    vr = ValidationResult("N002:2160-0:20", "2160-0", "N002", "Creatinine", "1.5", True, "")
    gate.propose(vr)
    log = gate.audit_log()
    assert len(log) == 1
    assert log[0]["who"] == "pipeline:automated"
    assert log[0]["committed"] is False
    assert "loinc=2160-0" in log[0]["what"]


# --- end-to-end pipeline ---------------------------------------------------

def test_pipeline_runs_on_synthetic_notes() -> None:
    notes = generate_synthetic_notes()
    metrics, gate = run_pipeline(notes)
    assert isinstance(metrics, PipelineMetrics)
    assert isinstance(gate, HumanGate)

def test_pipeline_notes_processed_count() -> None:
    notes = generate_synthetic_notes()
    metrics, _ = run_pipeline(notes)
    assert metrics.notes_processed == len(notes)

def test_pipeline_zero_committed_without_approval() -> None:
    """Core governance invariant: nothing commits without human approval."""
    notes = generate_synthetic_notes()
    metrics, gate = run_pipeline(notes)
    assert metrics.committed == 0
    assert gate.committed_count == 0

def test_pipeline_extracts_nonzero_observations() -> None:
    notes = generate_synthetic_notes()
    metrics, _ = run_pipeline(notes)
    assert metrics.observations_extracted > 0

def test_pipeline_validation_rate_above_90_percent() -> None:
    notes = generate_synthetic_notes()
    metrics, _ = run_pipeline(notes)
    assert metrics.validation_rate >= 0.90, (
        f"Validation rate {metrics.validation_rate:.1%} below 90% threshold"
    )

def test_pipeline_one_liner_contains_key_fields() -> None:
    notes = generate_synthetic_notes()
    metrics, _ = run_pipeline(notes)
    one_liner = metrics.one_liner()
    assert "LOINC" in one_liner
    assert "0 committed" in one_liner
    assert "validated" in one_liner
    assert "rejected" in one_liner

def test_pipeline_is_synthetic_flag() -> None:
    notes = generate_synthetic_notes()
    metrics, _ = run_pipeline(notes)
    assert metrics.is_synthetic is True
