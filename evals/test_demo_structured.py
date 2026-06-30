"""Tests for the MIMIC demo structured loader (evals/demo_structured.py).

Includes the PHI-protection guarantees: refuses in-repo paths, drops
date/PHI-adjacent columns, logs ids only; and the lab label fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.demo_structured import (
    DemoStructuredLoader,
    StructuredCase,
    _FORBIDDEN_COLUMNS,
    _read_csv_safe,
    _reject_in_repo,
)


def _make_demo(tmp_path: Path) -> Path:
    """Build a tiny fake hosp/ tree (synthetic, no real PHI)."""
    hosp = tmp_path / "hosp"
    hosp.mkdir()
    # itemid 51222 has BLANK loinc -> should fall back to label "Hemoglobin"
    (hosp / "d_labitems.csv").write_text(
        "itemid,loinc_code,label\n50931,2345-7,Glucose\n51222,,Hemoglobin\n")
    (hosp / "diagnoses_icd.csv").write_text(
        "subject_id,hadm_id,icd_code,icd_version\n1,100,E11,10\n1,100,I10,10\n2,200,4019,9\n")
    (hosp / "labevents.csv").write_text(
        "subject_id,hadm_id,itemid,charttime\n1,100,50931,2150-01-01\n2,200,51222,2150-02-02\n")
    (hosp / "prescriptions.csv").write_text(
        "subject_id,hadm_id,drug,starttime\n1,100,Metformin,2150-01-01\n")
    (hosp / "procedures_icd.csv").write_text(
        "subject_id,hadm_id,icd_code\n2,200,0016070\n")
    return tmp_path


def test_reject_in_repo_blocks_repo_paths():
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="inside the repo"):
        _reject_in_repo(repo_root / "data" / "mimic")


def test_reject_in_repo_allows_external(tmp_path):
    # tmp_path is outside the repo -> must not raise
    _reject_in_repo(tmp_path)


def test_forbidden_columns_are_dropped(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("hadm_id,itemid,charttime,admittime\n100,5,2150-01-01,2150-01-02\n")
    rows = _read_csv_safe(p)
    assert rows[0] == {"hadm_id": "100", "itemid": "5"}
    assert "charttime" not in rows[0] and "admittime" not in rows[0]


def test_dates_in_forbidden_set():
    for col in ("charttime", "admittime", "dischtime", "starttime", "dob", "dod"):
        assert col in _FORBIDDEN_COLUMNS


def test_loader_missing_hosp_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="hosp/"):
        DemoStructuredLoader(tmp_path)


def test_loader_builds_structured_cases(tmp_path):
    demo = _make_demo(tmp_path)
    cases = DemoStructuredLoader(demo).load()
    by_hadm = {c.hadm_id: c for c in cases}
    assert set(by_hadm) == {"100", "200"}
    assert by_hadm["100"].diagnosis_icd == ["E11", "I10"]
    assert by_hadm["100"].labs == ["2345-7"]          # itemid 50931 -> LOINC
    assert by_hadm["200"].labs == ["Hemoglobin"]      # blank LOINC -> label fallback
    assert by_hadm["100"].drugs == ["Metformin"]
    assert by_hadm["200"].procedures_icd == ["0016070"]


def test_icd_version_recorded(tmp_path):
    demo = _make_demo(tmp_path)
    by_hadm = {c.hadm_id: c for c in DemoStructuredLoader(demo).load()}
    assert by_hadm["100"].icd_version == "10"
    assert by_hadm["200"].icd_version == "9"


def test_loader_respects_limit(tmp_path):
    demo = _make_demo(tmp_path)
    cases = DemoStructuredLoader(demo).load(limit=1)
    assert len(cases) == 1


def test_summary_line_has_counts_not_values(tmp_path):
    demo = _make_demo(tmp_path)
    case = DemoStructuredLoader(demo).load(limit=1)[0]
    line = case.summary_line()
    assert "hadm_id=" in line and "dx=" in line
    # must not leak raw codes/values into the audit line
    assert "E11" not in line and "Metformin" not in line


def test_structured_case_defaults():
    c = StructuredCase(hadm_id="x")
    assert c.diagnosis_icd == [] and c.labs == []
