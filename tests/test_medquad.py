"""Tests for the MedQuAD loader (datasets layer).

Covers:
  - CSV parsing with gold annotations
  - XML parsing (MedQuAD document shape)
  - filters: only_gold_cui, only_answered, limit
  - MedQuADItem properties: has_gold_cui, is_answered, is_rare_disease
  - graceful behaviour on a missing data dir (no raise)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fhir_mcp.medquad import MedQuADItem, MedQuADLoader, get_loader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CSV = """qid,source,question_type,question,answer,focus,focus_category,cui,semantic_type,synonyms
0000001-1,GARD,information,What is (are) Adult-onset Still's disease ?,A rare inflammatory condition.,Adult-onset Still's disease,Disease,C0085253,Disease or Syndrome,AOSD|Still disease adult
0000002-1,MedlinePlus,treatment,What are the treatments for hypertension ?,,Hypertension,Disease,C0020538,Disease or Syndrome,High blood pressure
0000003-1,NIDDK,causes,What causes diabetes ?,Multiple factors contribute.,Diabetes,Disease,,Disease or Syndrome,
"""

_XML = """<Document id="0000010" source="GARD" url="http://example">
  <Focus>Fabry disease</Focus>
  <FocusAnnotations>
    <UMLS>
      <CUIs><CUI>C0002986</CUI></CUIs>
      <SemanticTypes><SemanticType>Disease or Syndrome</SemanticType></SemanticTypes>
      <Synonyms>
        <Synonym>Anderson-Fabry disease</Synonym>
        <Synonym>Alpha-galactosidase A deficiency</Synonym>
      </Synonyms>
    </UMLS>
  </FocusAnnotations>
  <QAPairs>
    <QAPair pid="1">
      <Question qid="0000010-1" qtype="information">What is (are) Fabry disease ?</Question>
      <Answer>Fabry disease is a rare genetic disorder.</Answer>
    </QAPair>
  </QAPairs>
</Document>
"""


@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    d = tmp_path / "medquad_csv"
    d.mkdir()
    (d / "sample.csv").write_text(_CSV, encoding="utf-8")
    return d


@pytest.fixture()
def xml_dir(tmp_path: Path) -> Path:
    d = tmp_path / "medquad_xml"
    (d / "GARD").mkdir(parents=True)
    (d / "GARD" / "0000010.xml").write_text(_XML, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_loads_all_items(csv_dir: Path) -> None:
    items = MedQuADLoader(csv_dir).load()
    assert len(items) == 3
    first = items[0]
    assert first.qid == "0000001-1"
    assert first.focus == "Adult-onset Still's disease"
    assert first.cui == "C0085253"
    assert first.synonyms == ["AOSD", "Still disease adult"]


def test_csv_only_gold_cui_filters_missing_cui(csv_dir: Path) -> None:
    items = MedQuADLoader(csv_dir).load(only_gold_cui=True)
    # row 3 (diabetes) has an empty CUI and must be dropped
    assert {i.focus for i in items} == {
        "Adult-onset Still's disease",
        "Hypertension",
    }


def test_csv_only_answered_filters_empty_answers(csv_dir: Path) -> None:
    items = MedQuADLoader(csv_dir).load(only_answered=True)
    # row 2 (hypertension) has an empty answer and must be dropped
    foci = {i.focus for i in items}
    assert "Hypertension" not in foci
    assert "Adult-onset Still's disease" in foci


def test_limit_caps_results(csv_dir: Path) -> None:
    items = MedQuADLoader(csv_dir).load(limit=2)
    assert len(items) == 2


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------


def test_xml_parses_focus_and_cui(xml_dir: Path) -> None:
    items = MedQuADLoader(xml_dir).load()
    assert len(items) == 1
    item = items[0]
    assert item.qid == "0000010-1"
    assert item.focus == "Fabry disease"
    assert item.cui == "C0002986"
    assert item.question_type == "information"
    assert "Anderson-Fabry disease" in item.synonyms
    assert item.is_answered


# ---------------------------------------------------------------------------
# Item properties
# ---------------------------------------------------------------------------


def test_properties() -> None:
    item = MedQuADItem(
        qid="x-1", source="GARD", question="q?", answer="a", cui="C0001"
    )
    assert item.has_gold_cui
    assert item.is_answered
    assert item.is_rare_disease

    bare = MedQuADItem(qid="y-1", source="MedlinePlus", question="q?")
    assert not bare.has_gold_cui
    assert not bare.is_answered
    assert not bare.is_rare_disease


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_missing_dir_returns_empty_without_raising(tmp_path: Path) -> None:
    items = MedQuADLoader(tmp_path / "does_not_exist").load()
    assert items == []


def test_get_loader_is_singleton() -> None:
    assert get_loader() is get_loader()
