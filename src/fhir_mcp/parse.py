"""Nemotron Parse integration — raw document → structured text.

Nemotron Parse (NVIDIA NIM) overcomes traditional OCR limitations:
  - Multi-column layout support (prior auth letters, EOB statements)
  - Table extraction with spatial grounding (formulary tables, benefit grids)
  - Reading-order reconstruction (critical for insurance documents)
  - Markdown formatting with structure preserved

This module is the first stage of the end-to-end pipeline:

  Raw PDF/Doc
    → Nemotron Parse (this module)  → structured markdown
    → ClinicalNLP.extract_entities()  → ICD-10/LOINC/NPI entities
    → LOINC deterministic validator    → accept / reject / warn
    → propose_observation              → staged write
    → human approve_write              → committed observation
    → SHA-256 audit chain              → tamper-evident record

API: NVIDIA NIM (OpenAI-compatible)
  https://integrate.api.nvidia.com/v1/chat/completions
  Model: nvidia/nemotron-parse
  Auth:  NVIDIA_API_KEY environment variable

NIM constraints:
  - Exactly one user message per request (no system message)
  - Image-only content (no text content items)
  - PDFs must be converted to PNG images first

PDFs are rendered page-by-page via pypdfium2; each page is a separate call.

For PHI compliance, set NEMOTRON_PARSE_BASE_URL to a self-hosted NIM
endpoint so document content stays on-premises.

PHI NOTE: Cloud NIM endpoint is for synthetic/de-identified documents only.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_logger = logging.getLogger("fhir_mcp.parse")

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
_BASE_URL = os.environ.get(
    "NEMOTRON_PARSE_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
_MODEL = "nvidia/nemotron-parse"
_TIMEOUT = 60
_MAX_PAGES = 10


class ParseResult:
    """Result from Nemotron Parse."""

    def __init__(
        self,
        structured_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = _MODEL,
        parse_method: str = "nemotron_parse",
        page_count: int = 0,
    ) -> None:
        self.structured_text = structured_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.parse_method = parse_method
        self.page_count = page_count
        self.output_word_count = len(structured_text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_text": self.structured_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "parse_method": self.parse_method,
            "page_count": self.page_count,
            "output_word_count": self.output_word_count,
        }


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def _is_pdf(source: str | Path | bytes) -> bool:
    if isinstance(source, bytes):
        return source[:4] == b"%PDF"
    return str(source).lower().endswith(".pdf")


def _download_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fhir-mcp/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _pdf_to_page_images(pdf_bytes: bytes) -> list[str]:
    """Render PDF pages to base64 PNG strings via pypdfium2.

    Install: pip install pypdfium2 Pillow
    """
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pypdfium2 required for PDF parsing: pip install pypdfium2 Pillow"
        ) from e

    pdf = pdfium.PdfDocument(pdf_bytes)
    page_images: list[str] = []
    n_pages = min(len(pdf), _MAX_PAGES)

    for i in range(n_pages):
        page = pdf[i]
        bitmap = page.render(scale=2)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        page_images.append(base64.b64encode(buf.getvalue()).decode())

    _logger.info("Rendered %d/%d PDF pages to PNG", n_pages, len(pdf))
    return page_images


# ---------------------------------------------------------------------------
# NIM API call — image-only, single user message, no system message
# ---------------------------------------------------------------------------


def _call_nim_api(image_b64: str) -> tuple[str, int, int]:
    """Send one base64 PNG to Nemotron Parse NIM.

    NIM constraints:
      - Single user message (no system message allowed)
      - Image-only content (no text items allowed)

    Returns (structured_text, input_tokens, output_tokens).
    structured_text is always a str (empty string if NIM returns None).
    """
    payload = json.dumps({
        "model": _MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 4096,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_NVIDIA_API_KEY}",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Nemotron Parse API error {e.code}: {body}") from e

    # Guard: content may be None if NIM returns an empty response
    text: str = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content") or ""
    )
    usage = data.get("usage", {})
    _logger.debug("NIM raw response keys: %s", list(data.keys()))
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_document(
    source: str | Path | bytes,
    document_type: str = "clinical",
) -> ParseResult:
    """Parse a clinical document using Nemotron Parse.

    Args:
        source: Local file path, public image URL, or raw bytes (PDF or image).
        document_type: Metadata hint. One of:
          'clinical', 'prior_auth', 'eob', 'treatment_plan', 'lab_report'

    Returns:
        ParseResult with structured_text (markdown).

    PDFs are rendered page-by-page; each page is a separate NIM call.
    Results are concatenated with page markers.

    PHI NOTE: Content sent to NVIDIA NIM cloud unless NEMOTRON_PARSE_BASE_URL
    is set to a self-hosted endpoint. Use self-hosted for PHI documents.
    """
    if not _NVIDIA_API_KEY:
        _logger.warning(
            "NVIDIA_API_KEY not set — falling back to raw text. "
            "Set NVIDIA_API_KEY to enable Nemotron Parse."
        )
        return _fallback_parse(source)

    # Resolve source to base64 PNG list (one per page)
    page_b64s: list[str] = []

    if isinstance(source, str) and source.startswith("http"):
        if _is_pdf(source):
            page_b64s = _pdf_to_page_images(_download_url(source))
        else:
            page_b64s = [base64.b64encode(_download_url(source)).decode()]

    elif isinstance(source, (str, Path)) and not str(source).startswith("http"):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        doc_bytes = path.read_bytes()
        if _is_pdf(source) or _is_pdf(doc_bytes):
            page_b64s = _pdf_to_page_images(doc_bytes)
        else:
            page_b64s = [base64.b64encode(doc_bytes).decode()]

    elif isinstance(source, bytes):
        if _is_pdf(source):
            page_b64s = _pdf_to_page_images(source)
        else:
            page_b64s = [base64.b64encode(source).decode()]

    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    if not page_b64s:
        return _fallback_parse(source)

    # One NIM call per page
    page_texts: list[str] = []
    total_input = 0
    total_output = 0

    for i, b64 in enumerate(page_b64s):
        try:
            text, inp, out = _call_nim_api(b64)
            page_texts.append(f"<!-- page {i + 1} -->\n{text}")
            total_input += inp
            total_output += out
            _logger.info("Page %d/%d: %d words", i + 1, len(page_b64s), len(text.split()))
        except (RuntimeError, urllib.error.URLError, TimeoutError) as e:
            _logger.warning("Page %d failed: %s — skipping", i + 1, e)
            page_texts.append(f"<!-- page {i + 1}: parse failed -->")

    structured_text = "\n\n".join(page_texts)
    _logger.info(
        "Nemotron Parse complete: pages=%d words=%d input_tokens=%d",
        len(page_b64s), len(structured_text.split()), total_input,
    )

    return ParseResult(
        structured_text=structured_text,
        input_tokens=total_input,
        output_tokens=total_output,
        model=_MODEL,
        parse_method="nemotron_parse_nim",
        page_count=len(page_b64s),
    )


def _fallback_parse(source: str | Path | bytes) -> ParseResult:
    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8", errors="replace")
        except Exception:
            text = "[Binary content — set NVIDIA_API_KEY for structured parsing]"
    elif isinstance(source, (str, Path)) and not str(source).startswith("http"):
        try:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = f"[Could not read {source} — set NVIDIA_API_KEY for structured parsing]"
    else:
        text = str(source)
    return ParseResult(structured_text=text, parse_method="raw_text_fallback")
