import asyncio
import os
import re  # Add this import at the top
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from loguru import logger
from ollama import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from src.commands import ChatbotCommands, SearchQuery
from src.config.settings import settings
from src.embedding import generate_embedding_with_retry
from src.logger import loggerConfig, setup_logger
from src.models import DocsSection, DocStore
from src.pdf_processing import extract_pdf_content
from src.utils import chunk_text, get_file_hash, retrieve_enhanced

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

model_settings = ModelSettings(
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS,
    timeout=settings.TIMEOUT,
)

ollama_model = OpenAIModel(
    model_name=settings.GENERATION_MODEL,
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
)

loggerConfig()


class LoadingAnimation:
    """Enhanced loading animation class with better error handling"""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.animation_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.running = False
        self.task = None

    def start(self):
        """Start the loading animation"""
        self.running = True
        self.task = asyncio.create_task(self._animate())

    async def stop(self):
        """Stop the loading animation"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        # Clear the line
        print("\r" + " " * (len(self.message) + 5), end="\r")

    async def _animate(self):
        """Animation loop with better exception handling"""
        i = 0
        try:
            while self.running:
                char = self.animation_chars[i % len(self.animation_chars)]
                print(f"\r{char} {self.message}...", end="", flush=True)
                await asyncio.sleep(0.1)
                i += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Animation error: {e}")


async def build_multi_pdf_store(pdf_files: List[str]) -> DocStore:
    """Build combined store from multiple PDF files"""
    stores = []
    for pdf_file in pdf_files:
        store = await build_search_store(pdf_file)
        stores.append(store)

    # Combine all sections and embeddings
    all_sections = []
    all_embeddings = []
    for store in stores:
        all_sections.extend(store.sections)
        all_embeddings.extend(store.embeddings)

    return DocStore(
        sections=all_sections,
        embeddings=all_embeddings,
        created_at=datetime.now(),
        doc_hash="multi_pdf_store",  # Special hash for multi-doc
    )


async def build_search_store(doc_file: str) -> DocStore:
    """Enhanced search store building with better error handling"""
    logger.info(f"Building search store for: {doc_file}")
    try:
        doc_hash = get_file_hash(doc_file)
        if cached_store := DocStore.load_from_cache(doc_hash):
            logger.info(f"Using cached store (hash: {doc_hash})")
            return cached_store

        logger.info("Extracting PDF content...")
        pages_content = await extract_pdf_content(doc_file)
        sections = []

        for page_text, page_num in pages_content:
            chunks = chunk_text(page_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:
                    continue
                title = f"Page {page_num}" + (
                    f" - Part {i + 1}" if len(chunks) > 1 else ""
                )
                url = f"local://page_{page_num}_chunk_{i}"
                section = DocsSection(
                    url=url,
                    title=title,
                    content=chunk,
                    page_num=page_num,
                    source_document=Path(doc_file).name,
                )
                sections.append(section)

        ollama = AsyncClient()
        sem = asyncio.Semaphore(5)

        async def bounded_embedding(section: DocsSection):
            async with sem:
                return await generate_embedding_with_retry(ollama, section)

        logger.info(f"Generating embeddings for {len(sections)} sections...")
        embeddings = await asyncio.gather(
            *[bounded_embedding(section) for section in sections]
        )

        logger.info(f"Built new store with {len(sections)} sections")
        store = DocStore(
            sections=sections,
            embeddings=embeddings,
            created_at=datetime.now(),
            doc_hash=doc_hash,
            source_document=Path(doc_file).name,
        )
        store.save_to_cache()
        return store
    except Exception as e:
        logger.error(f"Error building search store: {e}")
        raise


@dataclass
class EnhancedDeps:
    """Enhanced dependencies with caching"""

    store: DocStore
    ollama: AsyncClient
    session_id: str = "default"
    context_cache: Dict[str, str] = field(default_factory=dict)
    sources_cache: Dict[str, set] = field(default_factory=dict)
    response_cache: Dict[str, tuple] = field(default_factory=dict)


agent = Agent(
    ollama_model,
    deps_type=EnhancedDeps,
    model_settings=model_settings,
    system_prompt=f"""You are a helpful assistant specialized in answering questions about Megafon company documentation.

IMPORTANT GUIDELINES:
- ALWAYS respond in {settings.LANGUAGE} only, regardless of the user's language
- Use the 'retrieve_context' tool to get relevant information for user questions
- Provide clear, accurate, and helpful responses based on the retrieved context
- If no relevant information is found, be honest about it and suggest alternative approaches
- Format your responses clearly with proper structure when appropriate
- Be conversational but professional and concise
- If asked about something not in the documentation, clearly state that the information is not available in the provided documentation
- Focus on being helpful while staying within the bounds of the available documentation
- Always ensure your response is in {settings.LANGUAGE}, even if the question is asked in another language

Remember: All responses must be in {settings.LANGUAGE} only.""",
    retries=settings.MAX_RETRIES,
)


# In chatbot.py


@agent.tool
async def retrieve_context(context: RunContext[EnhancedDeps], search_query: str) -> str:
    """Retrieve relevant context for the query from the Megafon documentation with caching"""
    deps = context.deps
    query_key = search_query.lower()
    logger.info(f"Starting context retrieval for: '{search_query}'")

    # Check if we have both context and sources cached
    if query_key in deps.context_cache and query_key in deps.sources_cache:
        logger.info(f"Using cached context for query: {search_query}")
        logger.debug(f"Cached sources: {deps.sources_cache[query_key]}")
        return deps.context_cache[query_key]

    try:
        # NEW: Detect multi-topic queries
        if " and " in search_query.lower():
            logger.info("Detected multi-topic query")
            topics = re.split(r"\s+and\s+", search_query, flags=re.IGNORECASE)
            topics = [t.strip() for t in topics if t.strip()]

            all_retrieved = []
            for topic in topics:
                logger.info(f"Retrieving context for subtopic: '{topic}'")
                topic_retrieved = await retrieve_enhanced(
                    deps.store, deps.ollama, topic, max_results=3
                )
                all_retrieved.extend(topic_retrieved)

            # Remove duplicates
            seen_sections = set()
            retrieved = []
            for section, score in all_retrieved:
                if section.section_id not in seen_sections:
                    seen_sections.add(section.section_id)
                    retrieved.append((section, score))
        else:
            logger.info(f"Retrieving enhanced context for: '{search_query}'")
            retrieved = await retrieve_enhanced(deps.store, deps.ollama, search_query)

        if not retrieved:
            logger.info("No relevant information found for query")
            context_str = "No relevant information found in the Megafon documentation for this query."
            sources_used = set()
        else:
            context_pieces = []
            sources_used = set()
            logger.info(f"Found {len(retrieved)} relevant sections")

            for section, score in retrieved:
                logger.debug(
                    f"Section source: {section.source_document} | Score: {score:.3f}"
                )
                context_pieces.append(f"[Source: {section.title}]\n{section.content}")

                if section.source_document:
                    logger.debug(f"Adding source: {section.source_document}")
                    sources_used.add(section.source_document)
                    # Mark section as used
                    section.was_used_in_context = True
                else:
                    logger.warning(f"Section has no source_document: {section.title}")

            context_str = "\n\n---\n\n".join(context_pieces)
            logger.info(f"Retrieved context length: {len(context_str)} characters")
            logger.info(f"Sources used: {sources_used}")

        # Cache both context and sources
        deps.context_cache[query_key] = context_str
        deps.sources_cache[query_key] = sources_used
        return context_str
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        raise ModelRetry(f"Failed to retrieve context: {str(e)}")


@agent.tool
async def advanced_search(context: RunContext[EnhancedDeps], query: SearchQuery) -> str:
    """Perform an advanced search with custom parameters"""
    try:
        deps = context.deps
        original_threshold = settings.SIMILARITY_THRESHOLD
        settings.SIMILARITY_THRESHOLD = query.similarity_threshold
        retrieved = await retrieve_enhanced(
            deps.store, deps.ollama, query.query, max_results=query.max_results
        )
        settings.SIMILARITY_THRESHOLD = original_threshold

        if not retrieved:
            return (
                f"No results found for '{query.query}' with the specified parameters."
            )

        results = []
        for i, (section, score) in enumerate(retrieved, 1):
            results.append(
                f"Result {i} (Score: {score:.3f}):\n{section.display_content()}"
            )

        return f"Advanced search results for '{query.query}':\n\n" + "\n\n---\n\n".join(
            results
        )
    except Exception as e:
        logger.error(f"Advanced search error: {e}")
        raise ModelRetry(f"Advanced search failed: {str(e)}")


async def run_enhanced_chatbot(store: DocStore):
    """Enhanced chatbot with caching for faster responses"""
    print("\n" + "=" * 70)
    print("🚀 Enhanced RAG Chatbot with Latest PydanticAI Features")
    print("=" * 70)
    print(f"📚 Loaded: {len(store.sections)} document sections")
    print(f"🕒 Processed: {store.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("📖 Document: Megafon Company Documentation")
    print(f"🤖 Model: {settings.GENERATION_MODEL}")
    print(f"⚙️ Temperature: {settings.TEMPERATURE}, Max Tokens: {settings.MAX_TOKENS}")
    print(f"🌍 Language: {settings.LANGUAGE} responses only")
    print("\n💡 Type 'help' to see all available commands")
    print("💬 Ask questions about the documentation or use commands")
    print("🔍 Try 'search <query>' for advanced searching")
    print("-" * 70)

    ollama = AsyncClient()
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    deps = EnhancedDeps(store=store, ollama=ollama, session_id=session_id)
    commands = ChatbotCommands(store)

    while True:
        try:
            query = input("\n🔍 You: ").strip()
            if not query:
                continue

            # Reset sources for new query
            used_sources = set()

            if query.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Thanks for using the Enhanced RAG Chatbot! Goodbye!")
                logger.info("User exited the chatbot")
                break

            if commands.is_command(query):
                loading = LoadingAnimation("Processing command")
                loading.start()
                try:
                    result = await commands.execute_command(query, deps)
                    await loading.stop()
                    print(f"\n{result}")
                except Exception as e:
                    await loading.stop()
                    print(f"\n❌ Command error: {str(e)}")
                    logger.error(f"Command error: {e}")
                continue

            # Handle regular queries with caching
            logger.debug(f"User query: {query}")
            query_lower = query.lower()
            if query_lower in deps.response_cache:
                response, sources = deps.response_cache[query_lower]
                print(f"\n🤖 Assistant: {response}")
                if sources:
                    print("\n📚 Source PDFs:")
                    for src in sources:
                        print(f"  - {src}")
                else:
                    print("\nℹ️ No specific PDF sources identified")
                print("\n⏱️ Response time: 0.00s (cached)")
                continue

            loading = LoadingAnimation("Generating response")
            loading.start()
            start_time = time.time()

            try:
                result = await agent.run(
                    query, deps=deps, model_settings=model_settings
                )
                await loading.stop()
                response_time = time.time() - start_time
                response = result.output
                logger.info(f"Response generated: {response[:100]}...")

                # Get sources from source_cache - handle missing entries
                used_sources = deps.sources_cache.get(query_lower, set())
                logger.info(f"Sources from cache: {used_sources}")

                if not used_sources:
                    # Find sections that were actually used
                    used_sources = {
                        section.source_document
                        for section in store.sections
                        if getattr(section, "was_used_in_context", False)
                        and section.source_document
                    }
                    # Clear the markers for next query
                    for section in store.sections:
                        if hasattr(section, "was_used_in_context"):
                            del section.was_used_in_context
                print(f"\n🤖 Assistant: {response}")

                # Display source PDFs
                if used_sources:
                    logger.info(f"Displaying sources: {used_sources}")
                    print("\n📚 Source PDFs:")
                    for src in used_sources:
                        print(f"  - {src}")
                else:
                    # Try to get document name from store if possible
                    doc_name = getattr(store, "source_document", None)
                    logger.info(f"Store source_document: {doc_name}")

                    if doc_name:
                        print(f"\n📚 Source PDF: {doc_name}")
                    else:
                        # For multi-doc stores, check if sections have sources
                        section_sources = {
                            s.source_document
                            for s in store.sections
                            if s.source_document
                        }
                        logger.info(f"Section sources: {section_sources}")

                        if section_sources:
                            print("\n📚 Source PDFs:")
                            for src in section_sources:
                                print(f"  - {src}")
                        else:
                            logger.warning("No sources found in store or sections")
                            print("\nℹ️ No specific PDF sources identified")

                print(f"\n⏱️ Response time: {response_time:.2f}s")

                # Cache response with sources
                deps.response_cache[query_lower] = (response, used_sources)
                commands.add_to_history(query, response)
            except Exception as e:
                await loading.stop()
                logger.error(f"Error processing query: {e}")
                print(
                    "\n❌ I encountered an issue processing your request. "
                    "Please try rephrasing your question or use 'debug' command for more info."
                )
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Use 'exit' to quit properly.")
            logger.warning("Chatbot interrupted by user")
            continue
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Type 'help' for available commands or browserify'exit' to quit.")


async def run_chatbot(store: DocStore):
    """Backward compatibility wrapper"""
    await run_enhanced_chatbot(store)


if __name__ == "__main__":
    import asyncio

    setup_logger()

    async def main():
        pass

    asyncio.run(main())
