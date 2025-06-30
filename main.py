# main.py
import argparse
import asyncio

from src.chatbot import build_search_store, run_chatbot
from src.settings import settings


async def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument("command", choices=["chat", "train"], help="Run mode")
    args = parser.parse_args()

    print("🚀 Starting RAG Chatbot...")

    store = await build_search_store(str(settings.DOCS_FILE))

    if not hasattr(store, "sections") or not store.sections:
        raise ValueError("❌ Invalid store: missing sections")

    print(f"✅ Store ready with {len(store.sections)} sections")

    if args.command == "chat":
        await run_chatbot(store)
        
    elif args.command == "train":
        print("📚 Training complete - Store is ready for chat")


if __name__ == "__main__":
    asyncio.run(main())
