"""
My logging setup.

I want to have consistent logging across the whole app.
"""

import logging
import logging.handlers
import os
from datetime import datetime


# A flag to make sure I only configure logging once.
_logging_configured = False


def setup_logging(log_level=logging.INFO):
    # This sets up the logger for the whole application.
    global _logging_configured
    
    if _logging_configured:
        return logging.getLogger()
    
    # Make sure the 'logs' folder exists.
    os.makedirs("logs", exist_ok=True)
    
    # My standard log format.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Log to a file, and rotate it so it doesn't get huge.
    log_filename = f"logs/chunscraper_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_filename, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Also log to the console.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Get the root logger and set it up.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Get rid of any old handlers.
    root_logger.handlers.clear()
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Quiet down some of the noisy libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    _logging_configured = True
    
    return root_logger


def get_logger(name):
    # A simple way to get a logger instance for any module.
    # It will automatically set up logging if it hasn't been done yet.
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name)