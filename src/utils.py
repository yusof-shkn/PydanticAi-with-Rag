import hashlib
from datetime import datetime
from typing import List

import numpy as np
from loguru import logger
from ollama import AsyncClient
from pydantic_ai import ModelRetry

from src.models import DocsSection, DocStore
from src.settings import settings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            for i in range(min(100, chunk_size // 4)):
                if end - i > start and text[end - i] in ".!?":
                    end = end - i + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def get_file_hash(filepath: str) -> str:
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return str(datetime.now().timestamp())


async def retrieve_enhanced(
    store: DocStore, client: AsyncClient, search_query: str, max_results: int = 0
) -> list[tuple[DocsSection, float]]:
    """Enhanced retrieval with better error handling and flexible parameters"""
    logger.debug(f"Retrieving context for queryಸr query: {search_query}")

    try:
        response = await client.embeddings(
            model=settings.EMBEDDING_MODEL,
            prompt=search_query,
        )
        query_embedding = np.array(response["embedding"])

        similarities = []
        for i, emb in enumerate(store.embeddings):
            if (
                sim := cosine_similarity(query_embedding, emb)
            ) >= settings.SIMILARITY_THRESHOLD:
                similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # Use provided max_results if > 0, otherwise default to settings
        limit = max_results if max_results > 0 else settings.TOP_K_RESULTS
        results = [(store.sections[i], score) for i, score in similarities[:limit]]

        logger.debug(f"Retrieved {len(results)} relevant sections")
        return results

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise ModelRetry(f"Failed to retrieve context: {str(e)}")
