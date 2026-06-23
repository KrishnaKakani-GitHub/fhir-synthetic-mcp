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

NIM constraint: exactly one user message per request, no system message.
PDFs are rendered page-by-page via pypdfium2; each page is a separate call.

For PHI compliance, deploy NIM self-hosted on-premises so no document
content leaves your infrastructure. The cloud NIM endpoint is suitable
for synthetic or de-identified documents only.

PHI NOTE: If NVIDIA_API_KEY is set and documents contain PHI, use the
self-hosted NIM endpoint (NEMOTRON_PARSE_BASE_URL env var).
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

_USER_PROMPT = (
    "You are a clinical document parser. Extract and structure all content "
    "from this document page. Preserve table structure as markdown tables. "
    "Maintain reading order. Extract: patient demographics, diagnosis codes, "
    "procedure codes, medication lists, authorization decisions, dates, and "
    "provider information. Output clean, structured markdown."
)


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
# NIM API call — exactly one user message, no system message
# ---------------------------------------------------------------------------


def _call_nim_api(image_b64: str, document_type: str) -> tuple[str, int, int]:
    """Send one base64 PNG to Nemotron Parse NIM.

    NIM constraint: messages array must contain exactly one user message.
    No system message is permitted.

    Returns (structured_text, input_tokens, output_tokens).
    """
    user_text = f"{_USER_PROMPT} Document type: {document_type}."

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
                    {
                        "type": "text",
                        "text": user_text,
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

    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    usage = data.get("usage", {})
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
        document_type: 'clinical', 'prior_auth', 'eob', 'treatment_plan', 'lab_report'

    Returns:
        ParseResult with structured_text (markdown).

    PDFs are rendered page-by-page via pypdfium2; each page is a separate
    NIM call (NIM accepts exactly one user message per request).
    Results are concatenated with page markers.

    PHI NOTE: content sent to NVIDIA NIM cloud. Use NEMOTRON_PARSE_BASE_URL
    to point at a self-hosted NIM endpoint for PHI documents.
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
            text, inp, out = _call_nim_api(b64, document_type)
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
