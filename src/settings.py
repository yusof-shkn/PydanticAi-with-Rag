import os
from pathlib import Path


class Settings:
    APP_ENV = "development"
    BASE_DIR: Path = Path(__file__).parent.parent
    DOCS_FILE: Path = BASE_DIR / "data/MegaFon.pdf"
    CACHE_DIR: Path = BASE_DIR / "cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    LOG_DIR: Path = BASE_DIR / "logs/bot.log"

    # Model configuration
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LANGUAGE: str = "English"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2000
    TIMEOUT: int = 60
    # settings.py
    GENERATION_MODEL: str = "llama3.2"
    # Processing parameters
    MAX_RETRIES: int = 3
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.1


settings = Settings()
