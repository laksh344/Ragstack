"""Multi-modal vision extraction.

Uses GPT-4o Vision to extract text from images, charts, diagrams,
and complex tables embedded in PDFs that traditional parsers miss.

This is the key differentiator from 95% of RAG projects — most only
handle plain text. This module handles the visual content.

Reference: RAGFlow's deepdoc/vision/ uses custom OCR + layout models.
We use GPT-4o Vision for simpler, higher-quality extraction.
"""

import base64
from pathlib import Path

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from backend.config import settings
from backend.models.document import ParsedDocument, ParsedPage

logger = structlog.get_logger()

VISION_PROMPT = """Analyze this document page image and extract ALL content precisely:

1. **Text**: Extract all readable text, preserving paragraph structure.
2. **Tables**: Convert every table to markdown format with headers and alignment.
3. **Charts/Graphs**: Describe the chart type, axes, data points, and key trends.
4. **Diagrams**: Describe the diagram structure, components, and relationships.
5. **Formulas/Equations**: Transcribe using LaTeX notation.

Output the extracted content in a structured format. Be thorough — every piece
of information on this page matters for knowledge retrieval."""


async def enrich_with_vision(
    document: ParsedDocument,
    file_path: str | Path,
) -> ParsedDocument:
    """Process pages with tables/images through GPT-4o Vision.

    Only processes pages where the parser detected tables or images,
    keeping costs manageable by skipping text-only pages.

    Returns the document with enriched page content.
    """
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        logger.info("vision.skip", reason="not_pdf", file=path.name)
        return document

    pages_to_process = [
        p for p in document.pages if p.has_tables or p.has_images
    ]

    if not pages_to_process:
        logger.info("vision.skip", reason="no_visual_content", file=path.name)
        return document

    logger.info(
        "vision.start",
        file=path.name,
        pages_to_process=len(pages_to_process),
    )

    try:
        import fitz  # pymupdf

        pdf_doc = fitz.open(str(path))
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            max_tokens=4096,
            temperature=0,
        )

        vision_count = 0
        for page in pages_to_process:
            try:
                # Render page to image
                pdf_page = pdf_doc[page.page_number - 1]
                pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x for quality
                img_bytes = pix.tobytes("png")
                b64_image = base64.b64encode(img_bytes).decode("utf-8")

                # Send to GPT-4o Vision
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ]
                )

                response = await llm.ainvoke([message])
                vision_text = response.content

                # Merge vision output with existing parser output
                if vision_text:
                    page.content = (
                        page.content
                        + "\n\n--- Vision Extraction ---\n"
                        + vision_text
                    )
                    vision_count += 1

                logger.info(
                    "vision.page_complete",
                    page=page.page_number,
                    chars_added=len(vision_text),
                )

            except Exception as e:
                logger.warning(
                    "vision.page_error",
                    page=page.page_number,
                    error=str(e),
                )
                continue

        pdf_doc.close()
        logger.info("vision.complete", file=path.name, pages_enriched=vision_count)

    except Exception as e:
        logger.error("vision.error", file=path.name, error=str(e))

    return document


def get_page_image_b64(file_path: str | Path, page_number: int) -> str | None:
    """Extract a single page as base64 PNG image. Utility for the frontend chunk viewer."""
    try:
        import fitz

        path = Path(file_path)
        if path.suffix.lower() != ".pdf":
            return None

        doc = fitz.open(str(path))
        if page_number < 1 or page_number > len(doc):
            doc.close()
            return None

        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")

    except Exception:
        return None
