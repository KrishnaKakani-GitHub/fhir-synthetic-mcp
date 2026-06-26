"""Tests for evidence_pipeline/evals/ -- entity-linking metrics and runner."""
from __future__ import annotations

from evidence_pipeline.evals.entity_linking import (
    EvalReport, EvalResult, mrr, top_k_accuracy,
)
from evidence_pipeline.evals.runner import _predict, run_smoke


# ---------------------------------------------------------------------------
# EvalResult unit tests
# ---------------------------------------------------------------------------

def _make_result(gold: str, preds: list[str], **kwargs) -> EvalResult:
    return EvalResult(item_id="t001", focus="test", gold_cui=gold,
                      predictions=preds, **kwargs)


def test_rank_exact_hit() -> None:
    r = _make_result("C0028344", ["C0028344", "C0011860"])
    assert r.rank == 1

def test_rank_second_position() -> None:
    r = _make_result("C0011860", ["C0028344", "C0011860"])
    assert r.rank == 2

def test_rank_miss_is_none() -> None:
    r = _make_result("C9999999", ["C0028344", "C0011860"])
    assert r.rank is None and not r.is_hit

def test_hit_at_k() -> None:
    r = _make_result("C0011860", ["C0028344", "C0011860", "C0020538"])
    assert not r.hit_at(1) and r.hit_at(2) and r.hit_at(3)

def test_reciprocal_rank_hit() -> None:
    r = _make_result("C0028344", ["C0028344"])
    assert r.reciprocal_rank == 1.0

def test_reciprocal_rank_miss() -> None:
    r = _make_result("C9999999", ["C0028344"])
    assert r.reciprocal_rank == 0.0


# ---------------------------------------------------------------------------
# Metric function tests
# ---------------------------------------------------------------------------

def test_top_k_accuracy_all_hits() -> None:
    results = [_make_result("C0028344", ["C0028344"]),
               _make_result("C0011860", ["C0011860"])]
    assert top_k_accuracy(results, k=1) == 1.0

def test_top_k_accuracy_partial() -> None:
    results = [_make_result("C0028344", ["C0028344"]),
               _make_result("C9999999", ["C0028344"])]
    assert top_k_accuracy(results, k=1) == 0.5

def test_mrr_all_rank_one() -> None:
    results = [_make_result("C0028344", ["C0028344"]),
               _make_result("C0011860", ["C0028344", "C0011860"])]
    # RR: 1.0 + 0.5 = 1.5 / 2 = 0.75
    assert abs(mrr(results) - 0.75) < 1e-9

def test_mrr_empty() -> None:
    assert mrr([]) == 0.0


# ---------------------------------------------------------------------------
# EvalReport tests
# ---------------------------------------------------------------------------

def test_eval_report_metrics() -> None:
    r1 = _make_result("C0028344", ["C0028344"])
    r2 = _make_result("C0011860", ["C0028344", "C0011860"])
    report = EvalReport(suite="test", dataset="test", results=[r1, r2])
    assert report.top_k_accuracy(1) == 0.5
    assert report.top_k_accuracy(2) == 1.0
    assert abs(report.mrr - 0.75) < 1e-9

def test_eval_report_to_dict_keys() -> None:
    report = EvalReport(suite="s", dataset="d", results=[])
    d = report.to_dict()
    assert all(k in d for k in ["suite", "dataset", "metrics", "results"])
    assert all(k in d["metrics"] for k in ["top_1_accuracy", "top_3_accuracy", "mrr"])


# ---------------------------------------------------------------------------
# Smoke runner tests
# ---------------------------------------------------------------------------

def test_smoke_runner_returns_report() -> None:
    report = run_smoke()
    assert report.suite == "smoke"
    assert report.n > 0

def test_smoke_top1_is_perfect() -> None:
    """Every alias in the crosswalk should resolve to rank 1."""
    report = run_smoke()
    assert report.top_k_accuracy(1) == 1.0, (
        f"Expected 100% top-1 on smoke suite, got {report.top_k_accuracy(1):.1%}\n"
        + "\n".join(f"  MISS: {r.focus!r} gold={r.gold_cui}"
                   for r in report.results if not r.is_hit)
    )

def test_smoke_mrr_is_one() -> None:
    report = run_smoke()
    assert report.mrr == 1.0

def test_predict_pnh_is_first() -> None:
    preds = _predict("PNH")
    assert preds and preds[0] == "C0028344"

def test_predict_unknown_returns_candidates() -> None:
    """Unknown focus returns crosswalk candidates (not empty)."""
    preds = _predict("xyzzy disease")
    assert len(preds) > 0  # fallback returns full crosswalk
