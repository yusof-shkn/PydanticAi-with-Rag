import hashlib
from datetime import datetime
from typing import List, Tuple

import numpy as np
from loguru import logger

from src.config.settings import settings
from src.models import DocsSection, DocStore


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
    store: DocStore, client, search_query: str, max_results=settings.TOP_K_RESULTS
) -> List[Tuple[DocsSection, float]]:
    """Enhanced retrieval with detailed logging"""
    try:
        logger.info(f"Starting enhanced retrieval for: '{search_query}'")

        # Generate query embedding
        response = await client.embeddings(
            model=settings.EMBEDDING_MODEL,
            prompt=search_query,
        )
        query_embedding = np.array(response["embedding"])
        logger.debug("Generated query embedding")

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(store.embeddings):
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            if sim >= settings.SIMILARITY_THRESHOLD:
                similarities.append((i, sim))

        logger.info(f"Found {len(similarities)} sections above threshold")

        # Sort and select top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:max_results]

        logger.info(f"Top {len(top_results)} results selected")

        # Prepare results with detailed logging
        results = []
        for i, score in top_results:
            section = store.sections[i]
            logger.debug(
                f"Result {len(results) + 1}: {section.title} | Score: {score:.3f} | Source: {section.source_document}"
            )
            results.append((section, score))

        return results
    except Exception as e:
        logger.error(f"Enhanced retrieval failed: {e}")
        raise
