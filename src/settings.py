import os
from pathlib import Path


class Settings:
    APP_ENV = "development"
    BASE_DIR: Path = Path(__file__).parent.parent
    DOCS_FILE: Path = BASE_DIR / "data/what_is_facebook.pdf"
    CACHE_DIR: Path = BASE_DIR / "cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    LOG_DIR: Path = BASE_DIR / "logs/bot.log"

    # Model configuration
    EMBEDDING_MODEL: str = "nomic-embed-text"
    # settings.py
    GENERATION_MODEL: str = "litellm/ollama/llama3"
    # Processing parameters
    MAX_RETRIES: int = 3
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.1


settings = Settings()
