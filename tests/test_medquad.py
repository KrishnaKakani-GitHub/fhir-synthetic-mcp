"""Tests for src/fhir_mcp/medquad.py.

All tests run without the full MedQuAD corpus — fixtures are minimal
in-memory CSV and XML strings that exercise both parser paths.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fhir_mcp.medquad import (
    LoadResult,
    MedQuADItem,
    MedQuADLoader,
    get_loader,
)


# ---------------------------------------------------------------------------
# Fixtures — written to tmp_path so tests are hermetic
# ---------------------------------------------------------------------------

CSV_CONTENT = textwrap.dedent("""\
    qtype,Question,Answer,source,Focus,CUI,SemanticType,url
    information,What is Type 2 Diabetes?,"Type 2 diabetes is a chronic condition.",niddk,Type 2 Diabetes,C0011860,Disease or Syndrome,https://niddk.nih.gov/diabetes
    treatment,How is PNH treated?,"Treatment includes eculizumab.",GARD,Paroxysmal nocturnal hemoglobinuria,C0028344,Disease or Syndrome,https://rarediseases.info.nih.gov/diseases
    symptoms,What are the symptoms of diabetes?,,niddk,Type 2 Diabetes,C0011860,Disease or Syndrome,
    information,What causes anemia?,,medlineplus,,,,
""")

XML_CONTENT = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <Dataset>
      <Source>GARD</Source>
      <URL>https://rarediseases.info.nih.gov/diseases/7537</URL>
      <Focus>Gaucher disease</Focus>
      <FocusAnnotations>
        <UMLS>
          <CUIs><CUI>C0017205</CUI></CUIs>
          <SemanticTypeList>
            <SemanticType>Disease or Syndrome</SemanticType>
          </SemanticTypeList>
        </UMLS>
      </FocusAnnotations>
      <QAPairs>
        <QAPair pid="1">
          <Question qtype="information">What is Gaucher disease?</Question>
          <Answer>Gaucher disease is a rare genetic disorder.</Answer>
        </QAPair>
        <QAPair pid="2">
          <Question qtype="treatment">How is Gaucher disease treated?</Question>
          <Answer>Treatment includes enzyme replacement therapy.</Answer>
        </QAPair>
        <QAPair pid="3">
          <Question qtype="frequency">How common is Gaucher disease?</Question>
          <Answer></Answer>
        </QAPair>
      </QAPairs>
    </Dataset>
""")


@pytest.fixture()
def csv_corpus(tmp_path: Path) -> Path:
    """Write CSV fixture and return the corpus dir."""
    corpus_dir = tmp_path / "medquad"
    corpus_dir.mkdir()
    (corpus_dir / "test_source.csv").write_text(CSV_CONTENT, encoding="utf-8")
    return corpus_dir


@pytest.fixture()
def xml_corpus(tmp_path: Path) -> Path:
    """Write XML fixture and return the corpus dir."""
    corpus_dir = tmp_path / "medquad"
    corpus_dir.mkdir()
    (corpus_dir / "GARD_gaucher.xml").write_text(XML_CONTENT, encoding="utf-8")
    return corpus_dir


# ---------------------------------------------------------------------------
# CSV loader tests
# ---------------------------------------------------------------------------


def test_csv_loads_all_questions(csv_corpus: Path) -> None:
    """All four rows in the CSV fixture are loaded (even unanswered ones)."""
    result = MedQuADLoader(corpus_dir=csv_corpus).load()
    assert result.total == 4
    assert bool(result)  # LoadResult.__bool__


def test_csv_answered_count(csv_corpus: Path) -> None:
    """Only rows with non-empty answers are counted as answered."""
    result = MedQuADLoader(corpus_dir=csv_corpus).load()
    # Rows 0 and 1 have answers; rows 2 and 3 are empty
    assert result.answered == 2


def test_csv_rare_disease_flag(csv_corpus: Path) -> None:
    """GARD-sourced rows are identified as rare-disease items."""
    result = MedQuADLoader(corpus_dir=csv_corpus).load()
    rare = [i for i in result.items if i.is_rare_disease]
    assert len(rare) == 1
    assert rare[0].focus == "Paroxysmal nocturnal hemoglobinuria"


def test_csv_gold_cui_flag(csv_corpus: Path) -> None:
    """Items with a non-empty CUI column report has_gold_cui=True."""
    result = MedQuADLoader(corpus_dir=csv_corpus).load()
    with_cui = [i for i in result.items if i.has_gold_cui]
    without_cui = [i for i in result.items if not i.has_gold_cui]
    assert len(with_cui) == 3   # rows 0,1,2 have CUI C0011860/C0028344/C0011860
    assert len(without_cui) == 1  # row 3 (anemia) has no CUI


# ---------------------------------------------------------------------------
# XML loader tests
# ---------------------------------------------------------------------------


def test_xml_loads_qa_pairs(xml_corpus: Path) -> None:
    """All three QAPairs in the XML fixture are loaded."""
    result = MedQuADLoader(corpus_dir=xml_corpus).load()
    assert result.total == 3


def test_xml_metadata_propagated(xml_corpus: Path) -> None:
    """Source-level metadata (Focus, CUI, SemanticType) is applied to each item."""
    result = MedQuADLoader(corpus_dir=xml_corpus).load()
    for item in result.items:
        assert item.focus == "Gaucher disease"
        assert item.cui == "C0017205"
        assert item.semantic_type == "Disease or Syndrome"
        assert item.source == "GARD"
        assert item.is_rare_disease


def test_xml_unanswered_item(xml_corpus: Path) -> None:
    """A QAPair with an empty Answer element is loaded but marked unanswered."""
    result = MedQuADLoader(corpus_dir=xml_corpus).load()
    answered = [i for i in result.items if i.is_answered]
    unanswered = [i for i in result.items if not i.is_answered]
    assert len(answered) == 2
    assert len(unanswered) == 1


# ---------------------------------------------------------------------------
# Filter and missing-corpus tests
# ---------------------------------------------------------------------------


def test_filter_by_question_type(csv_corpus: Path) -> None:
    """filter(question_type=...) returns only matching items."""
    loader = MedQuADLoader(corpus_dir=csv_corpus)
    info_items = loader.filter(question_type="information")
    assert all(i.question_type == "information" for i in info_items)
    assert len(info_items) == 2  # rows 0 and 3


def test_missing_corpus_returns_empty(tmp_path: Path) -> None:
    """A non-existent corpus directory returns an empty LoadResult, never raises."""
    loader = MedQuADLoader(corpus_dir=tmp_path / "does_not_exist")
    result = loader.load()
    assert result.total == 0
    assert not bool(result)  # LoadResult.__bool__ returns False


def test_get_loader_singleton(csv_corpus: Path) -> None:
    """get_loader() returns a MedQuADLoader; repeated calls return the same instance."""
    loader_a = get_loader(corpus_dir=csv_corpus)
    loader_b = get_loader()  # no override — returns cached instance
    assert loader_a is loader_b
    assert isinstance(loader_a, MedQuADLoader)
