"""
Logging configuration module for Intelligent Systems Models and Methods
Provides centralized logging setup for better output management and analysis
"""

import logging
import sys
from datetime import datetime
import os

def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    console_output=True,
    format_string=None
):
    """
    Set up logging configuration for the project
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, creates timestamped log file
        console_output: Whether to output to console (default: True)
        format_string: Custom format string for log messages
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"intelligent_systems_{timestamp}.log")
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        if console_output:
            logger.info(f"Logging to file: {log_file}")
    
    return logger

def get_logger(name=None):
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__ or module name)
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)