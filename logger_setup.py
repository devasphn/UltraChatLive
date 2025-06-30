import logging
import config

def setup_logging():
    """Configures logging for the application based on settings in config.py."""
    log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(level=log_level, format=config.LOG_FORMAT)

    # Suppress overly verbose logs from specific libraries if not in DEBUG mode
    if log_level > logging.DEBUG:
        logging.getLogger('aioice.ice').setLevel(logging.WARNING)
        # Example: logging.getLogger('another_verbose_library').setLevel(logging.WARNING)

    logging.info(f"Logging initialized with level {log_level_str}.")

if __name__ == '__main__':
    # Example of how to use it (for testing this module directly)
    # In the main application, you'd just call setup_logging()
    setup_logging()
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
