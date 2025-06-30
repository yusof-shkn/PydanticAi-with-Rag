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
