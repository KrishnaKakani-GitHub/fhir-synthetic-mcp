"""MedQuAD dataset loader -- NIH clinical QA corpus.

Loads the Medical Question Answering Dataset (MedQuAD) into typed Pydantic v2
models. MedQuAD contains 47,457 QA pairs drawn from 12 NIH websites including
cancer.gov, niddk.nih.gov, GARD (rare diseases), and MedlinePlus.

Dataset sources:
  XML  (original):  https://github.com/abachaa/MedQuAD          (CC BY 4.0)
  CSV  (converted): https://github.com/avery-lockwood/MedQuAD-CSVs

This loader supports both distribution formats.

Role in the evidence pipeline:
  MedQuAD items carry gold UMLS CUI annotations that the ontology/cui_mapper
  uses as ground truth. The pipeline is:
    MedQuADItem.cui -> cui_mapper.lookup() -> ICD-10/RxNorm/LOINC codes
    -> deterministic validation -> metatagged output

Citation (required by CC BY 4.0):
  Abacha, A.B. & Demner-Fushman, D. (2019). A Question-Entailment Approach
  to Question Answering. BMC Bioinformatics 20, 511.
  https://doi.org/10.1186/s12859-019-3119-8

PHI NOTE: MedQuAD contains de-identified public NIH content.
No patient data is present. Zero PHI touchpoints.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

_logger = logging.getLogger("evidence_pipeline.datasets.medquad")

_DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "medquad"
_CORPUS_DIR = Path(os.environ.get("EVIDENCE_PIPELINE_MEDQUAD_DIR", str(_DEFAULT_CORPUS_DIR)))

_RARE_DISEASE_SOURCES: frozenset[str] = frozenset({"GARD", "genetic_and_rare_diseases", "rarediseases"})


class MedQuADItem(BaseModel):
    """A single QA pair from the MedQuAD corpus."""

    item_id: str = Field(description="Unique identifier within the corpus")
    question: str = Field(description="The clinical question text")
    answer: str | None = Field(default=None, description="NIH-authored answer; None when removed for copyright")
    question_type: str = Field(default="other", description="Question category (information, treatment, diagnosis, ...)")
    focus: str | None = Field(default=None, description="Medical entity the question is about")
    cui: str | None = Field(default=None, description="UMLS Concept Unique Identifier for the focus entity")
    semantic_type: str | None = Field(default=None, description="UMLS semantic type (e.g. 'Disease or Syndrome')")
    source: str | None = Field(default=None, description="NIH source site (e.g. 'GARD', 'cancer.gov')")
    source_url: str | None = Field(default=None, description="URL of the source NIH page")

    @property
    def is_answered(self) -> bool:
        """True when the item has a non-empty NIH answer (~16K of 47K pairs)."""
        return bool(self.answer and self.answer.strip())

    @property
    def has_gold_cui(self) -> bool:
        """True when a UMLS CUI is available for entity-linking evaluation."""
        return bool(self.cui and self.cui.strip())

    @property
    def is_rare_disease(self) -> bool:
        """True when the question originates from a rare-disease source (GARD)."""
        if not self.source:
            return False
        src = self.source.lower()
        return any(tag.lower() in src for tag in _RARE_DISEASE_SOURCES)


@dataclass
class LoadResult:
    """Outcome of a corpus load operation."""
    items: list[MedQuADItem] = field(default_factory=list)
    total: int = 0
    answered: int = 0
    with_cui: int = 0
    rare_disease: int = 0
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.total > 0

    def __repr__(self) -> str:
        return (f"LoadResult(total={self.total}, answered={self.answered}, "
                f"with_cui={self.with_cui}, rare_disease={self.rare_disease})")


class MedQuADLoader:
    """Loads MedQuAD QA pairs from a local corpus directory.

    Supports CSV (avery-lockwood/MedQuAD-CSVs) and XML (abachaa/MedQuAD).

    Usage::

        loader = MedQuADLoader()
        result = loader.load()
        eval_cases = loader.eval_items()  # answered + has_gold_cui
        rare = loader.filter(is_rare_disease=True)
    """

    def __init__(self, corpus_dir: Path | None = None) -> None:
        self._dir = corpus_dir or _CORPUS_DIR
        self._cache: list[MedQuADItem] | None = None

    def load(self, *, force_reload: bool = False) -> LoadResult:
        """Load all QA pairs. Returns empty LoadResult if corpus not found -- never raises."""
        if self._cache is not None and not force_reload:
            return self._make_result(self._cache)
        if not self._dir.exists():
            _logger.warning("MedQuAD corpus not found: %s -- set EVIDENCE_PIPELINE_MEDQUAD_DIR", self._dir)
            return LoadResult()
        items: list[MedQuADItem] = []
        errors: list[str] = []
        csv_files = sorted(self._dir.rglob("*.csv"))
        xml_files = sorted(self._dir.rglob("*.xml"))
        if csv_files:
            for f in csv_files:
                loaded, errs = self._load_csv(f)
                items.extend(loaded); errors.extend(errs)
        elif xml_files:
            for f in xml_files:
                loaded, errs = self._load_xml(f)
                items.extend(loaded); errors.extend(errs)
        else:
            _logger.warning("No .csv or .xml files in %s", self._dir)
        self._cache = items
        result = self._make_result(items)
        result.errors = errors
        _logger.info("MedQuAD loaded: %s", result)
        return result

    def eval_items(self) -> list[MedQuADItem]:
        """Return items suitable for entity-linking eval: answered + has_gold_cui."""
        return [i for i in self.load().items if i.is_answered and i.has_gold_cui]

    def filter(self, *, question_type: str | None = None, is_rare_disease: bool | None = None,
               is_answered: bool | None = None, has_gold_cui: bool | None = None,
               source: str | None = None) -> list[MedQuADItem]:
        """Return a filtered subset. All filters AND-combined; None skips a filter."""
        def _keep(item: MedQuADItem) -> bool:
            if question_type is not None and item.question_type != question_type:
                return False
            if is_rare_disease is not None and item.is_rare_disease != is_rare_disease:
                return False
            if is_answered is not None and item.is_answered != is_answered:
                return False
            if has_gold_cui is not None and item.has_gold_cui != has_gold_cui:
                return False
            if source is not None:
                if item.source is None or source.lower() not in item.source.lower():
                    return False
            return True
        return [i for i in self.load().items if _keep(i)]

    def iter_items(self) -> Iterator[MedQuADItem]:
        yield from self.load().items

    def _load_csv(self, path: Path) -> tuple[list[MedQuADItem], list[str]]:
        items: list[MedQuADItem] = []
        errors: list[str] = []
        stem = path.stem
        try:
            with path.open(encoding="utf-8", newline="") as fh:
                for idx, row in enumerate(csv.DictReader(fh)):
                    try:
                        item = MedQuADItem(
                            item_id=f"{stem}_{idx}",
                            question=_s(row.get("Question") or row.get("question", "")),
                            answer=_s(row.get("Answer") or row.get("answer")) or None,
                            question_type=_s(row.get("qtype") or row.get("question_type", "other")) or "other",
                            focus=_s(row.get("Focus") or row.get("focus")) or None,
                            cui=_s(row.get("CUI") or row.get("cui")) or None,
                            semantic_type=_s(row.get("SemanticType") or row.get("semantic_type")) or None,
                            source=_s(row.get("source") or row.get("Source")) or stem,
                            source_url=_s(row.get("url") or row.get("URL")) or None,
                        )
                        if item.question:
                            items.append(item)
                    except Exception as exc:
                        errors.append(f"{path.name}:{idx}: {exc}")
        except Exception as exc:
            errors.append(f"Failed to open {path.name}: {exc}")
        return items, errors

    def _load_xml(self, path: Path) -> tuple[list[MedQuADItem], list[str]]:
        items: list[MedQuADItem] = []
        errors: list[str] = []
        stem = path.stem
        try:
            root = ET.parse(path).getroot()  # noqa: S314
        except ET.ParseError as exc:
            return items, [f"XML parse error in {path.name}: {exc}"]
        source = root.findtext("Source") or root.get("source") or stem
        source_url = root.findtext("URL") or root.get("url")
        focus = root.findtext("Focus") or root.findtext(".//Focus")
        cui = root.findtext(".//CUI") or root.findtext(".//cui")
        semantic_type = root.findtext(".//SemanticType") or root.findtext(".//semantictype")
        for pair in root.findall(".//QAPair"):
            pid = pair.get("pid", "0")
            q_elem = pair.find("Question")
            if q_elem is None:
                continue
            question = _s(q_elem.text)
            if not question:
                continue
            a_elem = pair.find("Answer")
            try:
                items.append(MedQuADItem(
                    item_id=f"{stem}_{pid}",
                    question=question,
                    answer=_s(a_elem.text) if a_elem is not None else None,
                    question_type=q_elem.get("qtype", "other") or "other",
                    focus=_s(focus) or None,
                    cui=_s(cui) or None,
                    semantic_type=_s(semantic_type) or None,
                    source=_s(source) or stem,
                    source_url=_s(source_url) or None,
                ))
            except Exception as exc:
                errors.append(f"{path.name}:pid={pid}: {exc}")
        return items, errors

    @staticmethod
    def _make_result(items: list[MedQuADItem]) -> LoadResult:
        return LoadResult(
            items=items, total=len(items),
            answered=sum(1 for i in items if i.is_answered),
            with_cui=sum(1 for i in items if i.has_gold_cui),
            rare_disease=sum(1 for i in items if i.is_rare_disease),
        )


def _s(value: str | None) -> str:
    return (value or "").strip()


_loader: MedQuADLoader | None = None


def get_loader(corpus_dir: Path | None = None) -> MedQuADLoader:
    """Return the module-level MedQuADLoader instance."""
    global _loader
    if _loader is None or corpus_dir is not None:
        _loader = MedQuADLoader(corpus_dir=corpus_dir)
    return _loader
