"""Entity-linking evaluation metrics for CUI extraction quality.

Grades predictions from ontology/cui_mapper.py against gold UMLS CUI
annotations from the MedQuAD corpus, using BioEL evaluation methodology.

Metrics
-------
top_k_accuracy(k=1)  fraction of items where the gold CUI appears in the
                     top-k ranked predictions (higher is better, max 1.0)
mrr                  mean reciprocal rank of the gold CUI across all items
                     (higher is better, max 1.0)

JD alignment: "Verify study outputs / QA-QC" and "structured test scenario
design for clinical AI outputs" (cover letter).

PHI NOTE: Operates on disease names and UMLS concept identifiers only.
No patient data is accessed. Zero PHI touchpoints.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger("evidence_pipeline.evals.entity_linking")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Graded prediction for a single MedQuAD item."""

    item_id: str
    focus: str                    # condition/disease name from the QA pair
    gold_cui: str                 # ground-truth UMLS CUI from MedQuAD
    predictions: list[str]        # ranked list of predicted CUIs (best first)
    question_type: str = "other"
    is_rare_disease: bool = False

    @property
    def rank(self) -> int | None:
        """1-indexed position of gold CUI in predictions; None if not found."""
        try:
            return self.predictions.index(self.gold_cui) + 1
        except ValueError:
            return None

    @property
    def is_hit(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        r = self.rank
        return 1.0 / r if r is not None else 0.0

    def hit_at(self, k: int) -> bool:
        r = self.rank
        return r is not None and r <= k

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "focus": self.focus,
            "gold_cui": self.gold_cui,
            "predictions": self.predictions,
            "rank": self.rank,
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "is_hit": self.is_hit,
            "question_type": self.question_type,
            "is_rare_disease": self.is_rare_disease,
        }


@dataclass
class EvalReport:
    """Aggregate evaluation report across all graded items."""

    suite: str
    dataset: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    results: list[EvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- aggregate metrics ------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.results)

    def top_k_accuracy(self, k: int = 1) -> float:
        """Fraction of items where gold CUI is in the top-k predictions."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit_at(k)) / self.n

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank across all items."""
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / self.n

    @property
    def coverage(self) -> float:
        """Fraction of items for which at least one prediction was returned."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.predictions) / self.n

    # ---- stratified metrics -----------------------------------------------

    def top_k_by_group(self, group: str, k: int = 1) -> dict[str, float]:
        """Top-k accuracy stratified by question_type or is_rare_disease."""
        groups: dict[str, list[EvalResult]] = {}
        for r in self.results:
            key = str(getattr(r, group, "unknown"))
            groups.setdefault(key, []).append(r)
        return {
            g: sum(1 for r in items if r.hit_at(k)) / len(items)
            for g, items in groups.items()
        }

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "dataset": self.dataset,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
            "metrics": {
                "n": self.n,
                "top_1_accuracy": round(self.top_k_accuracy(1), 4),
                "top_3_accuracy": round(self.top_k_accuracy(3), 4),
                "top_5_accuracy": round(self.top_k_accuracy(5), 4),
                "mrr": round(self.mrr, 4),
                "coverage": round(self.coverage, 4),
            },
            "stratified": {
                "by_question_type": self.top_k_by_group("question_type"),
                "by_rare_disease": self.top_k_by_group("is_rare_disease"),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        _logger.info("Eval report saved: %s", path)

    def print_summary(self) -> None:
        print(f"\nEval suite: {self.suite}  |  dataset: {self.dataset}")
        print(f"  items   : {self.n}")
        print(f"  top-1   : {self.top_k_accuracy(1):.1%}")
        print(f"  top-3   : {self.top_k_accuracy(3):.1%}")
        print(f"  top-5   : {self.top_k_accuracy(5):.1%}")
        print(f"  MRR     : {self.mrr:.4f}")
        print(f"  coverage: {self.coverage:.1%}")


# ---------------------------------------------------------------------------
# Metric functions (stateless helpers)
# ---------------------------------------------------------------------------


def top_k_accuracy(results: list[EvalResult], k: int = 1) -> float:
    """Top-k accuracy over a list of EvalResult objects."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit_at(k)) / len(results)


def mrr(results: list[EvalResult]) -> float:
    """Mean reciprocal rank over a list of EvalResult objects."""
    if not results:
        return 0.0
    return sum(r.reciprocal_rank for r in results) / len(results)
