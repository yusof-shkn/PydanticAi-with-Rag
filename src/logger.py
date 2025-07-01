# logger.py - Fixed to use LOG_FILE instead of LOG_DIR
import sys

from loguru import logger

from src.config.settings import settings


def loggerConfig():
    """Configure logger with file and console output"""
    logger.remove()

    logger.add(
        settings.LOG_FILE,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG" if settings.APP_ENV == "development" else "INFO",
    )

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
    )

    logger.info("Logger configured successfully")
    logger.info(f"Log file: {settings.LOG_FILE}")
    logger.info(f"Environment: {settings.APP_ENV}")


def setup_logger():
    """Setup logger - wrapper function"""
    try:
        loggerConfig()
    except Exception as e:
        print(f"Failed to configure logger: {e}")
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.error(f"Logger configuration failed, using console only: {e}")
