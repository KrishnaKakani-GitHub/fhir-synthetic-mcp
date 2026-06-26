"""MedQuAD dataset loader: real NIH clinical questions with ontology annotations.

MedQuAD (Medical Question Answering Dataset) is a corpus of 47,457 clinician
and consumer questions curated from 12 NIH websites (cancer.gov, GARD,
MedlinePlus, niddk.nih.gov, ...). Each question carries gold annotations that
this platform uses as ground truth for the ontology-mapping pipeline:

  - question_type   : one of 37 types (Treatment, Diagnosis, Side Effects, ...)
  - focus           : the disease/drug the question is about
  - focus_category  : Disease | Drug | Other
  - cui             : UMLS Concept Unique Identifier for the focus
  - semantic_type   : UMLS semantic type (TUI label)

Why this matters for the platform:
  The CUI is the universal hub that links a free-text clinical question to
  ICD-10-CM / RxNorm / LOINC / SNOMED codes. A question -> focus -> CUI ->
  ontology-code chain is exactly the "convert clinical questions into
  structured phenotypes / ontological representations" capability, and the
  gold CUI lets us *grade* the LLM extraction deterministically.

Source:  https://github.com/abachaa/MedQuAD  (CC BY 4.0)
CSV port: https://github.com/avery-lockwood/MedQuAD-CSVs

This loader supports two on-disk formats:
  1. The original XML tree (one .xml per question, grouped in topic folders).
  2. A flat CSV with columns: question, answer, focus, question_type,
     synonyms, cui, semantic_type, source.

PHI NOTE: MedQuAD is public, de-identified educational content. It contains
no PHI. This module has no PHI touchpoints.
"""
from __future__ import annotations

import csv
import logging
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

_logger = logging.getLogger("fhir_mcp.medquad")

# Default on-disk location; override with FHIR_MCP_MEDQUAD_DIR.
_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "medquad"
)
_DATA_DIR = Path(os.environ.get("FHIR_MCP_MEDQUAD_DIR", str(_DEFAULT_DATA_DIR)))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class MedQuADItem(BaseModel):
    """A single MedQuAD question with its gold ontology annotations.

    The gold fields (focus, cui, semantic_type) are the reference labels the
    eval harness grades the LLM extraction against. `answer` is the NIH-authored
    gold answer used for QA/QC of generated evidence summaries.
    """

    qid: str = Field(description="Stable question id, e.g. '0000001-1'")
    source: str = Field(default="", description="Origin NIH site, e.g. 'GARD'")
    question_type: str = Field(default="", description="One of 37 MedQuAD types")
    question: str
    answer: str = Field(default="", description="NIH gold answer (may be empty)")
    focus: str = Field(default="", description="Disease/drug the question is about")
    focus_category: Optional[str] = Field(
        default=None, description="Disease | Drug | Other"
    )
    cui: Optional[str] = Field(
        default=None, description="UMLS CUI for the focus (gold label)"
    )
    semantic_type: Optional[str] = Field(
        default=None, description="UMLS semantic type label"
    )
    synonyms: list[str] = Field(default_factory=list)

    @property
    def has_gold_cui(self) -> bool:
        """True if this item can be used as an entity-linking eval case."""
        return bool(self.cui and self.cui.strip())

    @property
    def is_answered(self) -> bool:
        """True if the NIH gold answer is present (many were removed for
        MedlinePlus copyright reasons, leaving ~16k answered pairs)."""
        return bool(self.answer and self.answer.strip())

    @property
    def is_rare_disease(self) -> bool:
        """Heuristic: GARD is the NIH Genetic and Rare Diseases program.

        Lets a caller build a rare-disease subset to demonstrate the JD's
        'from common to rare' coverage requirement.
        """
        return self.source.upper() == "GARD"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class MedQuADLoader:
    """Loads MedQuAD items from XML or CSV on disk.

    Usage::

        loader = MedQuADLoader()
        items = loader.load()                 # all items
        gradable = loader.load(only_gold_cui=True)
        rare = [i for i in loader.load() if i.is_rare_disease]
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR

    # -- public API --------------------------------------------------------

    def load(
        self,
        *,
        only_gold_cui: bool = False,
        only_answered: bool = False,
        limit: int | None = None,
    ) -> list[MedQuADItem]:
        """Load MedQuAD items, applying optional filters.

        Args:
            only_gold_cui: keep only items with a gold UMLS CUI (eval cases).
            only_answered: keep only items with an NIH gold answer.
            limit:         cap the number of returned items (after filtering).

        Returns:
            List of MedQuADItem. Empty if the data dir is missing — the loader
            never raises on a missing corpus, so the rest of the platform can
            run without the dataset checked in.
        """
        items: list[MedQuADItem] = []
        for item in self._iter_items():
            if only_gold_cui and not item.has_gold_cui:
                continue
            if only_answered and not item.is_answered:
                continue
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
        _logger.info(
            "Loaded %d MedQuAD items from %s (gold_cui=%s answered=%s)",
            len(items), self._data_dir, only_gold_cui, only_answered,
        )
        return items

    # -- format dispatch ---------------------------------------------------

    def _iter_items(self) -> Iterator[MedQuADItem]:
        if not self._data_dir.exists():
            _logger.warning(
                "MedQuAD data dir %s does not exist; returning no items. "
                "Download from https://github.com/abachaa/MedQuAD",
                self._data_dir,
            )
            return
        csvs = sorted(self._data_dir.rglob("*.csv"))
        if csvs:
            for path in csvs:
                yield from self._iter_csv(path)
            return
        xmls = sorted(self._data_dir.rglob("*.xml"))
        for path in xmls:
            item = self._parse_xml(path)
            if item is not None:
                yield item

    # -- CSV parsing -------------------------------------------------------

    @staticmethod
    def _iter_csv(path: Path) -> Iterator[MedQuADItem]:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                norm = {k.strip().lower(): (v or "").strip()
                        for k, v in row.items() if k}
                syn_raw = norm.get("synonyms", "")
                synonyms = [s.strip() for s in syn_raw.split("|") if s.strip()]
                yield MedQuADItem(
                    qid=norm.get("qid") or norm.get("id") or f"{path.stem}-{i}",
                    source=norm.get("source", ""),
                    question_type=norm.get("question_type", ""),
                    question=norm.get("question", ""),
                    answer=norm.get("answer", ""),
                    focus=norm.get("focus", ""),
                    focus_category=norm.get("focus_category") or None,
                    cui=norm.get("cui") or None,
                    semantic_type=norm.get("semantic_type") or None,
                    synonyms=synonyms,
                )

    # -- XML parsing -------------------------------------------------------

    @staticmethod
    def _parse_xml(path: Path) -> MedQuADItem | None:
        """Parse one MedQuAD XML document into a single MedQuADItem.

        MedQuAD XML packs multiple QA pairs per document under <QAPairs>. We
        collapse to the first answered pair so each file yields one canonical
        item; callers needing every pair can extend this. The document-level
        <Focus> / <FocusAnnotations> carry the gold CUI + semantic type.
        """
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            _logger.warning("Skipping malformed MedQuAD XML %s: %s", path, e)
            return None

        source = root.attrib.get("source", "")
        doc_id = root.attrib.get("id", path.stem)
        focus = (root.findtext("Focus") or "").strip()

        cui: str | None = None
        semantic_type: str | None = None
        synonyms: list[str] = []
        fa = root.find("FocusAnnotations")
        if fa is not None:
            cui = (fa.findtext(".//CUI") or "").strip() or None
            semantic_type = (
                fa.findtext(".//SemanticType") or fa.findtext(".//SemanticTypeName") or ""
            ).strip() or None
            synonyms = [
                s.text.strip()
                for s in fa.findall(".//Synonym")
                if s.text and s.text.strip()
            ]

        qa_first = root.find(".//QAPair")
        question = ""
        answer = ""
        question_type = ""
        qid = doc_id
        if qa_first is not None:
            q_el = qa_first.find("Question")
            question = (q_el.text or "").strip() if q_el is not None else ""
            if q_el is not None:
                question_type = q_el.attrib.get("qtype", "")
                qid = q_el.attrib.get("qid", doc_id)
            answer = (qa_first.findtext("Answer") or "").strip()

        if not question:
            return None

        return MedQuADItem(
            qid=qid,
            source=source,
            question_type=question_type,
            question=question,
            answer=answer,
            focus=focus,
            cui=cui,
            semantic_type=semantic_type,
            synonyms=synonyms,
        )


# Module-level singleton (matches get_nlp() / get_rules() pattern)
_loader: MedQuADLoader | None = None


def get_loader() -> MedQuADLoader:
    """Return the module-level MedQuAD loader instance."""
    global _loader
    if _loader is None:
        _loader = MedQuADLoader()
    return _loader
