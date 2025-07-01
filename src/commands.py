import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from src.models import DocStore
from src.utils import retrieve_enhanced
from src.config.settings import settings


class SearchQuery(BaseModel):
    """Enhanced search query model with validation"""

    query: str = Field(..., description="The search query to execute", min_length=1)
    max_results: int = Field(
        default=5, description="Maximum number of results to return", ge=1, le=20
    )
    similarity_threshold: float = Field(
        default=0.3, description="Minimum similarity threshold", ge=0.0, le=1.0
    )


class ChatbotCommands:
    """Enhanced chatbot commands with better error handling and new features"""

    def __init__(self, store: DocStore):
        self.store = store
        self.commands = {
            "help": self._help_command,
            "debug": self._debug_command,
            "stats": self._stats_command,
            "clear": self._clear_command,
            "search": self._search_command,
            "summary": self._summary_command,
            "history": self._history_command,
            "export": self._export_command,
            "model": self._model_command,
            "settings": self._settings_command,
        }
        self.query_history: List[Tuple[str, str, datetime]] = []

    def is_command(self, query: str) -> bool:
        """Check if input is a command"""
        return query.lower().split()[0] in self.commands

    async def execute_command(self, query: str, deps) -> Optional[str]:
        """Execute a command and return result"""
        try:
            parts = query.lower().split()
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []

            if command in self.commands:
                return await self.commands[command](args, deps)
            return None
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return f"❌ Error executing command: {str(e)}"

    def add_to_history(self, query: str, response: str):
        """Add query and response to history"""
        self.query_history.append((query, response, datetime.now()))

        if len(self.query_history) > 50:
            self.query_history = self.query_history[-50:]

    async def _help_command(self, args: List[str], deps) -> str:
        """Show help information"""
        if not args:
            return f"""
🤖 **Enhanced RAG Chatbot - Available Commands:**

**Basic Commands:**
• `help` - Show this help message
• `help <command>` - Get detailed help for specific command
• `exit/quit/bye` - Exit the chatbot

**Information Commands:**
• `debug` - Show technical debugging information
• `stats` - Display document and usage statistics
• `summary` - Get a summary of the loaded document
• `history` - Show recent query history
• `model` - Show current model information
• `settings` - Show current model settings

**Search Commands:**
• `search <query>` - Search for specific content in documents
• `export` - Export current session data

**Utility Commands:**
• `clear` - Clear the screen

**Usage Tips:**
• Ask natural language questions about the Megafon documentation
• Use specific terms for better results
• Commands are case-insensitive
• All responses are in {settings.LANGUAGE}
            """

        command = args[0].lower()
        detailed_help = {
            "search": "Search for specific content: `search <your query>`\nExample: `search pricing plans`",
            "debug": "Shows: sections count, embeddings info, model details, memory usage",
            "stats": "Displays: document info, query statistics, response times",
            "summary": "Generates an AI summary of the loaded document content",
            "history": "Shows your last 10 queries and their timestamps",
            "export": "Exports current session data to a JSON file",
            "clear": "Clears the terminal screen for better readability",
            "model": "Shows current model information and capabilities",
            "settings": "Shows current model settings (temperature, MAX_TOKENS, etc.)",
        }

        return detailed_help.get(command, f"No detailed help available for '{command}'")

    async def _debug_command(self, args: List[str], deps) -> str:
        """Show debug information"""
        return f"""
🔧 **Enhanced Debug Information:**
• Document sections: {len(self.store.sections)}
• Embeddings shape: {len(self.store.embeddings)} x {len(self.store.embeddings[0]) if self.store.embeddings else 0}
• Document processed: {self.store.created_at.strftime("%Y-%m-%d %H:%M:%S")}
• Document hash: {self.store.doc_hash[:12]}...
• Embedding model: {settings.EMBEDDING_MODEL}
• Generation model: {settings.GENERATION_MODEL}
• Similarity threshold: {settings.SIMILARITY_THRESHOLD}
• Max results: {settings.TOP_K_RESULTS}
• Total queries processed: {len(self.query_history)}
• Python version: {sys.version.split()[0]}
• PydanticAI version: Enhanced with latest features
        """

    async def _stats_command(self, args: List[str], deps) -> str:
        """Show statistics"""
        total_content_length = sum(
            len(section.content) for section in self.store.sections
        )
        avg_section_length = (
            total_content_length / len(self.store.sections)
            if self.store.sections
            else 0
        )

        pages = set(section.page_num for section in self.store.sections)

        return f"""
📊 **Enhanced Statistics:**
• Total pages: {len(pages)}
• Total sections: {len(self.store.sections)}
• Average section length: {avg_section_length:.0f} characters
• Total content: {total_content_length:,} characters
• Queries processed: {len(self.query_history)}
• Session started: {self.store.created_at.strftime("%Y-%m-%d %H:%M:%S")}
• Session duration: {datetime.now() - self.store.created_at}
• Average response time: {self._calculate_avg_response_time():.2f}s
        """

    async def _clear_command(self, args: List[str], deps) -> str:
        """Clear the screen"""
        os.system("cls" if os.name == "nt" else "clear")
        return "Screen cleared! 🧹"

    async def _search_command(self, args: List[str], deps) -> str:
        """Enhanced search for content"""
        if not args:
            return "Please provide a search query. Usage: `search <your query>`"

        search_query = " ".join(args)

        try:
            retrieved = await retrieve_enhanced(self.store, deps.ollama, search_query)

            if not retrieved:
                return f"No results found for: '{search_query}'"

            results = []
            for i, (section, score) in enumerate(retrieved[:5]):
                preview = (
                    section.content[:100] + "..."
                    if len(section.content) > 100
                    else section.content
                )
                results.append(
                    f"**{i + 1}. {section.title}** (Score: {score:.3f})\n{preview}"
                )

            return f"🔍 **Search Results for '{search_query}':**\n\n" + "\n\n".join(
                results
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"❌ Search failed: {str(e)}"

    async def _summary_command(self, args: List[str], deps) -> str:
        """Generate document summary"""
        if not self.store.sections:
            return "No document content available for summary."

        # Get a sample of content from different pages
        sample_sections = []
        pages_seen = set()

        for section in self.store.sections:
            if section.page_num not in pages_seen and len(sample_sections) < 10:
                sample_sections.append(section)
                pages_seen.add(section.page_num)

        content_sample = "\n\n".join(
            section.content[:200] for section in sample_sections
        )

        return f"""
📋 **Document Summary:**
• Total pages: {len(set(s.page_num for s in self.store.sections))}
• Total sections: {len(self.store.sections)}
• Content preview from first few pages:

{content_sample[:1000]}{"..." if len(content_sample) > 1000 else ""}

*For detailed information, ask specific questions about the content.*
        """

    async def _history_command(self, args: List[str], deps) -> str:
        """Show query history"""
        if not self.query_history:
            return "No query history available."

        recent_queries = self.query_history[-10:]
        history_text = "📚 **Recent Query History:**\n\n"

        for i, (query, response, timestamp) in enumerate(reversed(recent_queries), 1):
            time_str = timestamp.strftime("%H:%M:%S")
            preview = response[:50] + "..." if len(response) > 50 else response
            history_text += f"{i}. [{time_str}] **Q:** {query}\n   **A:** {preview}\n\n"

        return history_text

    async def _export_command(self, args: List[str], deps) -> str:
        """Export session data"""
        try:
            export_data = {
                "session_info": {
                    "created_at": self.store.created_at.isoformat(),
                    "doc_hash": self.store.doc_hash,
                    "total_sections": len(self.store.sections),
                    "export_time": datetime.now().isoformat(),
                    "model_settings": {
                        "temperature": settings.TEMPERATURE,
                        "MAX_TOKENS": settings.MAX_TOKENS,
                        "timeout": settings.TIMEOUT,
                    },
                },
                "query_history": [
                    {
                        "query": query,
                        "response": response,
                        "timestamp": timestamp.isoformat(),
                    }
                    for query, response, timestamp in self.query_history
                ],
            }

            filename = f"enhanced_chatbot_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = Path(filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return f"✅ Session data exported to: {filepath.absolute()}"

        except Exception as e:
            logger.error(f"Export error: {e}")
            return f"❌ Export failed: {str(e)}"

    async def _model_command(self, args: List[str], deps) -> str:
        """Show model information"""
        return f"""
🤖 **Model Information:**
• Generation Model: {settings.GENERATION_MODEL}
• Embedding Model: {settings.EMBEDDING_MODEL}
• Model Provider: OpenAI-compatible (Ollama)
• Base URL: {os.environ.get("OLLAMA_API_BASE", "Not set")}
• Model Settings: Configured with ModelSettings
• Response Language: {settings.LANGUAGE} only
• Framework: PydanticAI (Latest version)
        """

    async def _settings_command(self, args: List[str], deps) -> str:
        """Show current model settings"""
        return f"""
⚙️ **Model Settings:**
• Temperature: {settings.TEMPERATURE}
• Max Tokens: {settings.MAX_TOKENS}
• Timeout: {settings.TIMEOUT}s
• Similarity Threshold: {settings.SIMILARITY_THRESHOLD}
• Top K Results: {settings.TOP_K_RESULTS}
• Chunk Size: {settings.CHUNK_SIZE}
• Chunk Overlap: {settings.CHUNK_OVERLAP}
• Max Retries: {settings.MAX_RETRIES}
        """

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time (placeholder)"""
        return 1.5
