import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from loguru import logger
from ollama import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from src.commands import ChatbotCommands, SearchQuery
from src.embedding import generate_embedding_with_retry
from src.logger import loggerConfig
from src.models import DocsSection, DocStore
from src.pdf_processing import extract_pdf_content
from src.settings import settings
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
                    url=url, title=title, content=chunk, page_num=page_num
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
    response_cache: Dict[str, str] = field(default_factory=dict)


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


@agent.tool
async def retrieve_context(context: RunContext[EnhancedDeps], search_query: str) -> str:
    """Retrieve relevant context for the query from the Megafon documentation with caching"""
    deps = context.deps
    query_key = search_query.lower()
    if query_key in deps.context_cache:
        logger.debug(f"Using cached context for query: {search_query}")
        return deps.context_cache[query_key]

    try:
        retrieved = await retrieve_enhanced(deps.store, deps.ollama, search_query)
        if not retrieved:
            logger.debug("No relevant information found for query")
            context_str = "No relevant information found in the Megafon documentation for this query."
        else:
            context_pieces = []
            for section, score in retrieved:
                context_pieces.append(f"[Source: {section.title}]\n{section.content}")
            context_str = "\n\n---\n\n".join(context_pieces)
            logger.debug(
                f"Retrieved context length: {len(context_str)} characters from {len(retrieved)} sections"
            )

        # Cache the context
        deps.context_cache[query_key] = context_str
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
                response = deps.response_cache[query_lower]
                print(f"\n🤖 Assistant: {response}")
                print("\n⏱️ Response time: 0.00s (cached)")
                commands.add_to_history(query, response)
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

                print(f"\n🤖 Assistant: {response}")
                print(f"\n⏱️ Response time: {response_time:.2f}s")

                # Cache the response
                deps.response_cache[query_lower] = response
                commands.add_to_history(query, response)
                logger.info(f"Query processed successfully in {response_time:.2f}s")
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

    async def main():
        pass

    asyncio.run(main())
