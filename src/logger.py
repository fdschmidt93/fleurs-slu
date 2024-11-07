import logging
import sys


def setup_logger(name: str, log_file: str = "app.log", level=logging.DEBUG):
    """Function to set up a logger that logs to both console and a file.

    Args:
        name (str): Name of the logger (usually the module name).
        log_file (str): The file to log to. Defaults to 'app.log'.
        level (int): Logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Configured logger.
    """

    # Create a custom logger
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if logger already exists
    if not logger.hasHandlers():
        # Set the level of logging
        logger.setLevel(level)

        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)  # To log to console
        file_handler = logging.FileHandler(log_file)  # To log to a file

        # Set levels for handlers (optional, can be customized)
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to the handlers
        console_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(console_format)
        file_handler.setFormatter(file_format)

        # Add the handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
