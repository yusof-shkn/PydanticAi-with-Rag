import asyncio
from datetime import datetime

import logfire
import numpy as np
from ollama import AsyncClient

from src.models import DocsSection, DocStore
from src.pdf_processing import extract_pdf_content
from src.settings import settings
from src.utils import chunk_text, cosine_similarity, get_file_hash


async def generate_embedding_with_retry(
    client: AsyncClient, section: DocsSection
) -> np.ndarray:
    last_exception = None
    for attempt in range(settings.MAX_RETRIES):
        try:
            with logfire.span("creating embedding", section_id=section.section_id):
                response = await client.embeddings(
                    model=settings.EMBEDDING_MODEL,
                    prompt=section.embedding_content(),
                )
                return np.array(response["embedding"])
        except Exception as e:
            last_exception = e
            if attempt < settings.MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"Embedding failed for {section.section_id}") from last_exception


async def build_search_store(doc_file: str) -> DocStore:
    doc_hash = get_file_hash(doc_file)
    if cached_store := DocStore.load_from_cache(doc_hash):
        return cached_store

    pages_content = await extract_pdf_content(doc_file)
    sections = []
    for page_text, page_num in pages_content:
        chunks = chunk_text(page_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            title = f"Page {page_num}" + (f" - Part {i + 1}" if len(chunks) > 1 else "")
            url = f"local://page_{page_num}_chunk_{i}"
            section = DocsSection(
                url=url, title=title, content=chunk, page_num=page_num
            )
            sections.append(section)

    ollama = AsyncClient()
    embeddings = []
    sem = asyncio.Semaphore(5)

    async def bounded_embedding(section: DocsSection):
        async with sem:
            return await generate_embedding_with_retry(ollama, section)

    embeddings = await asyncio.gather(
        *[bounded_embedding(section) for section in sections]
    )

    store = DocStore(
        sections=sections,
        embeddings=embeddings,
        created_at=datetime.now(),
        doc_hash=doc_hash,
    )
    store.save_to_cache()
    return store


async def retrieve(
    store: DocStore, client: AsyncClient, search_query: str
) -> list[tuple[DocsSection, float]]:
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
    return [
        (store.sections[i], score)
        for i, score in similarities[: settings.TOP_K_RESULTS]
    ]
