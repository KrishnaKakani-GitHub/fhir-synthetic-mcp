"""Calibrated confidence scoring for clinical observation proposals.

A proposal's confidence score is a calibrated float in [0, 1] representing
the probability that the proposal is clinically correct and would be approved
by a human reviewer.

Inputs (features used in scoring):
  1. validation_pass_rate: fraction of deterministic validation checks passed
     (0.0 = all checks failed, 1.0 = all passed including no warnings)
  2. rag_score:            top BM25+semantic retrieval score from guidelines search
     (higher = more relevant guideline evidence found)
  3. model_logprob_proxy:  model's self-reported confidence (from logprobs if
     available, or mapped from structured output confidence field)
  4. entity_match_rate:    fraction of proposed entities that were validated
     against ICD-10/NPI registries (0 if NLP not used)

Calibration:
  Raw logistic combination → isotonic regression calibration (Platt scaling).
  Calibration parameters are learned offline from the eval golden dataset (Day 6).
  Before calibration data is available, a linear blend is used.

Brier score tracking:
  After human approval/rejection, update_outcome() is called to accumulate
  Brier score (lower is better; 0.25 = random; perfect = 0.0).

PHI NOTE: Confidence scores operate on proposal metadata only — no PHI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProposalConfidence(BaseModel):
    """Calibrated confidence output for one proposal."""

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Calibrated probability this proposal is correct (0-1).",
    )
    uncertainty: float = Field(
        ge=0.0, le=1.0,
        description="Epistemic uncertainty (1 - confidence). High values mean the "
                    "model is unsure and the reviewer should scrutinise carefully.",
    )
    calibration_method: Literal["linear_blend", "isotonic", "platt"] = "linear_blend"
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)

    @property
    def tier(self) -> str:
        if self.confidence >= 0.85:
            return "high"
        elif self.confidence >= 0.60:
            return "medium"
        else:
            return "low"


@dataclass
class ConfidenceScorer:
    """Compute calibrated confidence scores for clinical proposals.

    Usage::

        scorer = ConfidenceScorer()
        conf = scorer.score(
            validation_pass_rate=1.0,
            rag_score=0.8,
            model_logprob_proxy=0.9,
            entity_match_rate=0.7,
        )
        print(conf.confidence, conf.tier)
    """

    # Feature weights (learned from eval data; these are priors)
    w_validation: float = 0.40
    w_rag: float = 0.30
    w_model: float = 0.20
    w_entity: float = 0.10

    # Calibration: isotonic regression lookup table (filled after eval Day 6)
    # Format: list of (raw_score, calibrated_score) sorted by raw_score
    calibration_table: list[tuple[float, float]] = field(default_factory=list)

    # Brier score tracking
    _outcomes: list[tuple[float, float]] = field(default_factory=list)  # (pred, actual)

    def score(
        self,
        validation_pass_rate: float = 1.0,
        rag_score: float = 0.5,
        model_logprob_proxy: float = 0.5,
        entity_match_rate: float = 0.0,
        has_validation_warnings: bool = False,
        value_above_flag: bool = False,
    ) -> ProposalConfidence:
        """Compute calibrated confidence for one proposal.

        Args:
            validation_pass_rate:    0=failed validation, 1=passed all checks
            rag_score:               Top guideline retrieval score (0-1)
            model_logprob_proxy:     Model's self-reported confidence (0-1)
            entity_match_rate:       Fraction of entities validated (0-1)
            has_validation_warnings: True if deterministic gate flagged the value
            value_above_flag:        True if value is above clinical flag threshold

        Returns:
            ProposalConfidence with calibrated confidence and uncertainty.
        """
        # Clamp inputs
        vpr = max(0.0, min(1.0, validation_pass_rate))
        rag = max(0.0, min(1.0, rag_score))
        mlp = max(0.0, min(1.0, model_logprob_proxy))
        emr = max(0.0, min(1.0, entity_match_rate))

        # Weighted linear combination
        raw = (
            self.w_validation * vpr
            + self.w_rag * rag
            + self.w_model * mlp
            + self.w_entity * emr
        )

        # Apply penalty modifiers
        flags: list[str] = []
        if has_validation_warnings:
            raw *= 0.85
            flags.append("validation_warning: value near clinical threshold")
        if value_above_flag:
            raw *= 0.75
            flags.append("value_above_flag: exceeds clinical flag threshold")
        if validation_pass_rate < 1.0:
            raw *= 0.5
            flags.append("validation_partial_fail: some checks did not pass")

        raw = max(0.0, min(1.0, raw))

        # Calibrate
        calibrated, method = self._calibrate(raw)

        return ProposalConfidence(
            confidence=round(calibrated, 4),
            uncertainty=round(1.0 - calibrated, 4),
            calibration_method=method,
            feature_contributions={
                "validation": round(self.w_validation * vpr, 4),
                "rag": round(self.w_rag * rag, 4),
                "model": round(self.w_model * mlp, 4),
                "entity": round(self.w_entity * emr, 4),
            },
            flags=flags,
        )

    def _calibrate(
        self, raw: float
    ) -> tuple[float, Literal["linear_blend", "isotonic", "platt"]]:
        """Calibrate a raw score using the isotonic lookup table if available."""
        if not self.calibration_table:
            return raw, "linear_blend"

        # Linear interpolation over the isotonic calibration table
        table = sorted(self.calibration_table, key=lambda x: x[0])
        if raw <= table[0][0]:
            return table[0][1], "isotonic"
        if raw >= table[-1][0]:
            return table[-1][1], "isotonic"

        for i in range(len(table) - 1):
            x0, y0 = table[i]
            x1, y1 = table[i + 1]
            if x0 <= raw <= x1:
                t = (raw - x0) / (x1 - x0) if x1 != x0 else 0.0
                return round(y0 + t * (y1 - y0), 4), "isotonic"

        return raw, "isotonic"

    def update_outcome(self, predicted: float, actual_correct: bool) -> None:
        """Record an outcome for Brier score tracking.

        Call this after a human approves (actual_correct=True) or rejects
        (actual_correct=False) a proposal, passing the confidence score that
        was generated at proposal time.
        """
        self._outcomes.append((predicted, 1.0 if actual_correct else 0.0))

    def brier_score(self) -> float | None:
        """Compute Brier score over all tracked outcomes.

        Returns None if no outcomes have been recorded.
        Brier score: mean((predicted - actual)^2). Lower is better.
        Perfect = 0.0, Random = 0.25.
        """
        if not self._outcomes:
            return None
        return round(
            sum((p - a) ** 2 for p, a in self._outcomes) / len(self._outcomes), 4
        )

    def load_calibration_table(
        self, table: list[tuple[float, float]]
    ) -> None:
        """Load an isotonic calibration lookup table.

        Format: list of (raw_score, calibrated_score) pairs sorted ascending.
        Generated by the eval harness (Day 6) from the golden dataset.
        """
        self.calibration_table = sorted(table, key=lambda x: x[0])

    def calibration_report(self) -> dict[str, Any]:
        """Return a summary of calibration quality."""
        brier = self.brier_score()
        n = len(self._outcomes)
        if n == 0:
            return {"brier_score": None, "n_outcomes": 0, "method": "linear_blend"}
        mean_pred = sum(p for p, _ in self._outcomes) / n
        mean_actual = sum(a for _, a in self._outcomes) / n
        return {
            "brier_score": brier,
            "n_outcomes": n,
            "mean_predicted_confidence": round(mean_pred, 4),
            "mean_actual_accuracy": round(mean_actual, 4),
            "calibration_gap": round(abs(mean_pred - mean_actual), 4),
            "method": "isotonic" if self.calibration_table else "linear_blend",
        }


# Module-level singleton
_scorer: ConfidenceScorer | None = None


def get_scorer() -> ConfidenceScorer:
    """Return the module-level ConfidenceScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer
