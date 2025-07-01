from pathlib import Path
from typing import List, Tuple

import PyPDF2
from loguru import logger


async def extract_pdf_content(filepath: str) -> List[Tuple[str, int]]:
    try:
        with open(filepath, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            pages_content = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages_content.append((text.strip(), page_num))
                except Exception as e:
                    logger.warning(f"Page {page_num} extraction failed: {e}")
            return pages_content
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF not found: {filepath}")
    except Exception as e:
        raise Exception(f"PDF read error: {e}")


def get_pdf_files(docs_path: str) -> List[str]:
    """Get all PDF files from a directory"""
    path = Path(docs_path)

    if path.is_dir():
        pdf_files = list(path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {docs_path}")
        return [str(pdf) for pdf in sorted(pdf_files)]

    raise ValueError(f"Invalid path: {docs_path}")
