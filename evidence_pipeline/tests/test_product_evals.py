"""Tests for evidence_pipeline/evals/product.py -- Layer 3 AMIE-style product evals."""
from __future__ import annotations

from evidence_pipeline.evals.product import (
    CI_THRESHOLD,
    GOLDEN_DATASET,
    AxisScores,
    GoldenCase,
    ProductReport,
    run_product_eval,
)


# --- golden dataset --------------------------------------------------------

def test_golden_dataset_not_empty() -> None:
    assert len(GOLDEN_DATASET) >= 5

def test_golden_dataset_has_required_fields() -> None:
    for case in GOLDEN_DATASET:
        assert case.case_id and case.question and case.focus
        assert case.expected_icd10 and case.expected_cui

def test_golden_dataset_covers_rare_diseases() -> None:
    assert sum(1 for c in GOLDEN_DATASET if c.is_rare_disease) >= 2


# --- AMIE axis scores ------------------------------------------------------

def test_axis_scores_composite_mean() -> None:
    axes = AxisScores(
        ontology_accuracy=1.0,
        evidence_sourcing=1.0,
        metatag_completeness=1.0,
        grounding_score=1.0,
        safety_gate=1.0,
    )
    assert axes.composite == 1.0

def test_axis_scores_partial_composite() -> None:
    axes = AxisScores(
        ontology_accuracy=1.0,
        evidence_sourcing=1.0,
        metatag_completeness=0.0,
        grounding_score=1.0,
        safety_gate=1.0,
    )
    assert abs(axes.composite - 0.8) < 1e-9


# --- product eval run ------------------------------------------------------

def test_product_eval_runs() -> None:
    report = run_product_eval()
    assert report.n == len(GOLDEN_DATASET)

def test_composite_score_meets_ci_threshold() -> None:
    report = run_product_eval()
    assert report.mean_composite >= CI_THRESHOLD, (
        f"Mean composite {report.mean_composite:.3f} below threshold {CI_THRESHOLD}\n"
        + "\n".join(
            f"  [{r.case_id}] composite={r.axes.composite:.3f} axes={r.axes.to_dict()}"
            for r in report.results if not r.passed
        )
    )

def test_icd10_accuracy_is_perfect() -> None:
    report = run_product_eval()
    assert report.icd10_accuracy == 1.0

def test_cui_accuracy_is_perfect() -> None:
    report = run_product_eval()
    assert report.cui_accuracy == 1.0

def test_all_cases_pass_ci_threshold() -> None:
    report = run_product_eval()
    assert report.overall_pass_rate == 1.0

def test_axis_means_all_above_threshold() -> None:
    report = run_product_eval()
    am = report.axis_means()
    failures = {ax: v for ax, v in am.items() if v < CI_THRESHOLD}
    assert not failures, f"Axes below threshold: {failures}"

def test_summary_keys() -> None:
    report = run_product_eval()
    s = report.summary()
    assert all(k in s for k in ["mean_composite", "icd10_accuracy",
                                 "cui_accuracy", "axis_means", "overall_pass_rate"])
    assert all(ax in s["axis_means"] for ax in [
        "ontology_accuracy", "evidence_sourcing",
        "metatag_completeness", "grounding_score", "safety_gate",
    ])

def test_single_case_pnh_all_axes_pass() -> None:
    pnh = next(c for c in GOLDEN_DATASET if c.case_id == "prod_001")
    report = run_product_eval([pnh])
    r = report.results[0]
    assert r.icd10_correct
    assert r.cui_correct
    assert r.axes.composite >= CI_THRESHOLD
    assert r.axes.grounding_score == 1.0
    assert r.axes.safety_gate == 1.0
