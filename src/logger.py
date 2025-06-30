import sys

from loguru import logger

from src.settings import settings


def loggerConfig():
    logger.remove()
    logger.add(
        settings.LOG_DIR,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG" if settings.APP_ENV == "development" else "INFO",
    )
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
        level="DEBUG" if settings.APP_ENV == "development" else "INFO",
    )
