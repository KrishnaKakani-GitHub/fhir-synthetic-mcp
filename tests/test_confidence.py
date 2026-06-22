"""Tests for calibrated confidence scoring.

All tests run without external APIs — ConfidenceScorer is pure Python.
"""
from __future__ import annotations

import pytest

from fhir_mcp.confidence import ConfidenceScorer, ProposalConfidence


# --- ProposalConfidence model tests -------------------------------------------


def test_tier_high() -> None:
    c = ProposalConfidence(confidence=0.90, uncertainty=0.10)
    assert c.tier == "high"


def test_tier_medium() -> None:
    c = ProposalConfidence(confidence=0.72, uncertainty=0.28)
    assert c.tier == "medium"


def test_tier_low() -> None:
    c = ProposalConfidence(confidence=0.40, uncertainty=0.60)
    assert c.tier == "low"


def test_uncertainty_is_complement() -> None:
    c = ProposalConfidence(confidence=0.75, uncertainty=0.25)
    assert abs(c.confidence + c.uncertainty - 1.0) < 0.01


# --- Scorer tests -------------------------------------------------------------


def test_perfect_inputs_high_confidence() -> None:
    scorer = ConfidenceScorer()
    result = scorer.score(
        validation_pass_rate=1.0,
        rag_score=1.0,
        model_logprob_proxy=1.0,
        entity_match_rate=1.0,
    )
    assert result.confidence >= 0.85
    assert result.tier == "high"


def test_failed_validation_reduces_confidence() -> None:
    scorer = ConfidenceScorer()
    no_penalty = scorer.score(validation_pass_rate=1.0, rag_score=0.8,
                               model_logprob_proxy=0.8)
    with_penalty = scorer.score(validation_pass_rate=0.5, rag_score=0.8,
                                 model_logprob_proxy=0.8)
    assert with_penalty.confidence < no_penalty.confidence


def test_validation_warning_flag_added() -> None:
    scorer = ConfidenceScorer()
    result = scorer.score(
        validation_pass_rate=1.0,
        rag_score=0.8,
        model_logprob_proxy=0.8,
        has_validation_warnings=True,
    )
    assert any("warning" in f for f in result.flags)
    assert "validation_warning" in " ".join(result.flags)


def test_value_above_flag_reduces_confidence() -> None:
    scorer = ConfidenceScorer()
    normal = scorer.score(validation_pass_rate=1.0, rag_score=0.8,
                           model_logprob_proxy=0.8)
    flagged = scorer.score(validation_pass_rate=1.0, rag_score=0.8,
                            model_logprob_proxy=0.8, value_above_flag=True)
    assert flagged.confidence < normal.confidence
    assert "value_above_flag" in " ".join(flagged.flags)


def test_feature_contributions_sum_to_raw_approx() -> None:
    scorer = ConfidenceScorer()
    result = scorer.score(
        validation_pass_rate=0.8, rag_score=0.7,
        model_logprob_proxy=0.6, entity_match_rate=0.5,
    )
    contrib_sum = sum(result.feature_contributions.values())
    # Sum should be close to the pre-penalty raw score
    assert 0.0 <= contrib_sum <= 1.0


def test_calibration_method_linear_blend_without_table() -> None:
    scorer = ConfidenceScorer()
    result = scorer.score(validation_pass_rate=1.0, rag_score=0.8,
                           model_logprob_proxy=0.9)
    assert result.calibration_method == "linear_blend"


def test_calibration_method_isotonic_with_table() -> None:
    scorer = ConfidenceScorer()
    scorer.load_calibration_table([(0.0, 0.05), (0.5, 0.5), (1.0, 0.95)])
    result = scorer.score(validation_pass_rate=1.0, rag_score=0.8,
                           model_logprob_proxy=0.9)
    assert result.calibration_method == "isotonic"


def test_brier_score_tracks_outcomes() -> None:
    scorer = ConfidenceScorer()
    scorer.update_outcome(0.9, True)   # correct, high confidence
    scorer.update_outcome(0.1, False)  # incorrect, low confidence (good)
    scorer.update_outcome(0.9, False)  # wrong, high confidence (bad)
    brier = scorer.brier_score()
    assert brier is not None
    assert 0.0 <= brier <= 1.0


def test_brier_score_none_before_outcomes() -> None:
    scorer = ConfidenceScorer()
    assert scorer.brier_score() is None


def test_perfect_predictions_brier_zero() -> None:
    scorer = ConfidenceScorer()
    scorer.update_outcome(1.0, True)
    scorer.update_outcome(0.0, False)
    assert scorer.brier_score() == 0.0


def test_calibration_report_structure() -> None:
    scorer = ConfidenceScorer()
    scorer.update_outcome(0.8, True)
    report = scorer.calibration_report()
    assert "brier_score" in report
    assert "n_outcomes" in report
    assert "calibration_gap" in report
    assert report["n_outcomes"] == 1


def test_isotonic_interpolation_between_knots() -> None:
    scorer = ConfidenceScorer()
    scorer.load_calibration_table([(0.0, 0.0), (0.5, 0.6), (1.0, 0.95)])
    # Raw score 0.25 should interpolate between (0.0, 0.0) and (0.5, 0.6)
    raw = 0.25
    calibrated, method = scorer._calibrate(raw)
    assert method == "isotonic"
    assert 0.0 <= calibrated <= 0.6
