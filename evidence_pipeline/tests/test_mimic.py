"""Tests for evidence_pipeline/datasets/mimic.py"""
from __future__ import annotations

from evidence_pipeline.datasets.mimic import (
    MIMICNote,
    generate_synthetic_notes,
)


def test_synthetic_notes_count() -> None:
    notes = generate_synthetic_notes()
    assert len(notes) == 10

def test_synthetic_notes_are_flagged() -> None:
    notes = generate_synthetic_notes()
    assert all(n.is_synthetic for n in notes)

def test_synthetic_notes_have_text() -> None:
    notes = generate_synthetic_notes()
    for n in notes:
        assert len(n.text) > 50, f"{n.note_id} text too short"

def test_synthetic_notes_have_unique_ids() -> None:
    notes = generate_synthetic_notes()
    ids = [n.note_id for n in notes]
    assert len(ids) == len(set(ids))

def test_synthetic_note_fields() -> None:
    note = generate_synthetic_notes()[0]
    assert isinstance(note, MIMICNote)
    assert note.note_id
    assert note.subject_id
    assert note.hadm_id
    assert note.chartdate
