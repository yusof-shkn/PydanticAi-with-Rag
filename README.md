# Enhanced RAG Chatbot with PydanticAI

![RAG Chatbot Demo](https://via.placeholder.com/800x400?text=RAG+Chatbot+Demo)

A sophisticated Retrieval-Augmented Generation chatbot using PydanticAI for structured interactions and Ollama for local LLM processing. Processes PDF documents to create a searchable knowledge base and answers questions using context retrieval.

## Features

- **PDF Processing**: Extract and chunk text from PDF documents
- **Embedding Generation**: Create vector embeddings using Ollama's NLP models
- **Semantic Search**: Find relevant context using cosine similarity
- **RAG Architecture**: Combine retrieval with generative AI for accurate answers
- **Caching System**: Store processed documents for faster reloads
- **Logging**: Comprehensive logging with Loguru for debugging

## Tech Stack

- **Python 3.11+**
- **Ollama** (local LLM server)
- **PydanticAI** (structured AI interactions)
- **Loguru** (logging)
- **uv** (Python package manager)
- **PyPDF2** (PDF processing)
- **NumPy** (vector operations)

## Prerequisites

1. Install [Ollama](https://ollama.com/download):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Download required models:
   ```bash
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pydanticAi_rag_chatbot.git
   cd pydanticAi_rag_chatbot
   ```

2. Install Python dependencies using `uv`:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

## Configuration

Edit `src/settings.py` to configure:
- Document path (`DOCS_FILE`)
- Chunk size and overlap
- Similarity threshold
- Model parameters

## Usage

### 1. Train Mode (Process Documents)

Process your PDF and build the search store:
```bash
uv run python main.py train
```

### 2. Chat Mode (Start Chatbot)

Interact with your documents:
```bash
uv run python main.py chat
```

### 3. Chatbot Commands
- `help`: Show available commands
- `info`: Display document statistics
- `exit`/`quit`: End session

## Project Structure

```
├── data/                   # PDF documents for processing
├── cache/                  # Cached document stores
├── src/
│   ├── chatbot.py          # Main chatbot logic
│   ├── models.py           # Data models (DocStore, DocsSection)
│   ├── pdf_processing.py   # PDF text extraction
│   ├── settings.py         # Configuration settings
│   └── utils.py            # Helper functions
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
└── bot.log                 # Generated log file
```

## Customization

### Change Document
1. Place new PDF in `data/` directory
2. Update `DOCS_FILE` in `src/settings.py`

### Modify Models
Edit `src/settings.py`:
```python
EMBEDDING_MODEL: str = "nomic-embed-text"  # Embedding model
GENERATION_MODEL: str = "llama3:8b"        # Generation model
```

### Adjust Chunking
```python
# src/settings.py
CHUNK_SIZE: int = 1000     # Characters per chunk
CHUNK_OVERLAP: int = 200   # Overlap between chunks
```

## Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   ollama serve
   ```

2. **Model not found**:
   ```bash
   ollama pull llama3:8b
   ```

3. **PDF extraction issues**:
   - Ensure PDF is text-readable (not scanned)
   - Try different PDF in `data/` directory

### View Logs
```bash
tail -f bot.log
```

## License

MIT License - see [LICENSE](LICENSE) for details
