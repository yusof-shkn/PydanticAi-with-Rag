import asyncio
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from loguru import logger
from ollama import AsyncClient
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.logger import loggerConfig
from src.models import DocsSection, DocStore
from src.pdf_processing import extract_pdf_content
from src.settings import settings
from src.utils import chunk_text, cosine_similarity, get_file_hash

# Set up environment variables
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
ollama_model = OpenAIModel(
    model_name="llama3.2", provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)

loggerConfig()


async def generate_embedding_with_retry(
    client: AsyncClient, section: DocsSection
) -> np.ndarray:
    last_exception = None
    for attempt in range(settings.MAX_RETRIES):
        try:
            logger.debug(f"Creating embedding for section: {section.section_id}")
            response = await client.embeddings(
                model=settings.EMBEDDING_MODEL,
                prompt=section.embedding_content(),
            )
            return np.array(response["embedding"])
        except Exception as e:
            last_exception = e
            logger.warning(
                f"Embedding attempt {attempt + 1} failed for {section.section_id}: {e}"
            )
            if attempt < settings.MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"Embedding failed for {section.section_id}") from last_exception


async def build_search_store(doc_file: str) -> DocStore:
    logger.debug(f"Building search store for: {doc_file}")
    doc_hash = get_file_hash(doc_file)
    if cached_store := DocStore.load_from_cache(doc_hash):
        logger.debug(f"Using cached store (hash: {doc_hash})")
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

    logger.debug(f"Generating embeddings for {len(sections)} sections...")
    embeddings = await asyncio.gather(
        *[bounded_embedding(section) for section in sections]
    )

    logger.debug(f"Built new store with {len(sections)} sections")
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
    logger.debug(f"Retrieving context for query: {search_query}")
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
    results = [
        (store.sections[i], score)
        for i, score in similarities[: settings.TOP_K_RESULTS]
    ]
    logger.debug(f"Retrieved {len(results)} relevant sections")
    return results


@dataclass
class Deps:
    store: DocStore
    ollama: AsyncClient


agent = Agent(
    ollama_model,
    deps_type=Deps,
    system_prompt="You are a helpful assistant answering questions about Logfire documentation. When a user asks a question, use the 'retrieve_context' tool to get relevant debugrmation, then use that debugrmation to answer the question.",
)


@agent.tool
async def retrieve_context(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve relevant context for the query."""
    deps = context.deps
    retrieved = await retrieve(deps.store, deps.ollama, search_query)
    if not retrieved:
        logger.debug("No relevant debugrmation found for query")
        return "No relevant debugrmation found in the documentation."

    context_str = "\n\n".join(section.display_content() for section, _ in retrieved)
    logger.debug(f"Retrieved context length: {len(context_str)} characters")
    return context_str


async def run_chatbot(store: DocStore):
    logger.debug("🤖 Enhanced RAG Chatbot with PydanticAI")
    logger.debug(f"📚 Loaded {len(store.sections)} document sections")
    logger.debug(f"🕒 Processed: {store.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💬 Ask questions (type 'exit', 'quit', or 'help')")

    ollama = AsyncClient()
    deps = Deps(store=store, ollama=ollama)

    while True:
        try:
            query = input("\n🔍 You: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Goodbye!")
                logger.debug("User exited the chatbot")
                break
            if query.lower() == "help":
                print("Commands: help, debug, exit")
                continue
            if query.lower() == "debug":
                print(f"Sections: {len(store.sections)}")
                continue

            logger.debug(f"User query: {query}")
            print("🤔 Thinking...")
            result = await agent.run(query, deps=deps)
            print(f"\n🤖 Bot: {result.output}")
            logger.debug(f"Bot response: {result.output}")

        except KeyboardInterrupt:
            print("\n👋 Interrupted")
            logger.warning("Chatbot interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(
                "\n❌ Error: I encountered an issue processing your request. Our engineering team has been notified."
            )
