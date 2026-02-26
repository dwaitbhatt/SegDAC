from segdac_dev.logging.loggers.logger import Logger
from pathlib import Path


def log_model_weights(
    console_logger,
    best_model_file_path: Path,
    final_model_file_path: Path,
    logger: Logger,
    best_model_name: str,
    final_model_name: str
):
    if best_model_file_path is None or final_model_file_path is None:
        console_logger.info("No trained model found, skipping logging!")
        return

    console_logger.info("Logging best model...")
    logger.log_model(best_model_name, best_model_file_path)
    console_logger.info("Best model logged!")

    console_logger.info("Logging final model...")
    logger.log_model(final_model_name, final_model_file_path)
    console_logger.info("Final model logged!")
