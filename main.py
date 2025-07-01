import asyncio
from pathlib import Path

from src.chatbot import build_multi_pdf_store, build_search_store, run_chatbot
from src.config.settings import settings
from src.pdf_processing import get_pdf_files


async def run_bot():
    print("🚀 Starting Enhanced Multi-PDF RAG Chatbot...")

    try:
        pdf_files = get_pdf_files(settings.DOCS_PATH)

        print(f"📚 Found {len(pdf_files)} PDF files:")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"   {i}. {Path(pdf_file).name}")

        if len(pdf_files) == 1:
            store = await build_search_store(pdf_files[0])
        else:
            store = await build_multi_pdf_store(pdf_files)

        if not hasattr(store, "sections") or not store.sections:
            raise ValueError("❌ Invalid store: missing sections")

        print(f"✅ Store ready with {len(store.sections)} sections")

        doc_counts = {}
        for section in store.sections:
            source = getattr(section, "source_document", "Unknown")
            doc_counts[source] = doc_counts.get(source, 0) + 1

        print("\n📊 Document Contribution:")
        total_sections = len(store.sections)
        for doc, count in doc_counts.items():
            percentage = (count / total_sections) * 100
            print(f"   📄 {doc}: {count} sections ({percentage:.1f}%)")

        await run_chatbot(store)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_bot())
