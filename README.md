# Intelligent Systems Models and Methods
This repository contains the source code and necessary dependencies to test the model examples presented in the "Models and Methods of Intelligent Systems" module, part of the Master's in Data Analysis and Intelligent Systems. The module covers the following areas:

*   **Rule-Based Systems**
*   **Machine Learning and Neural Networks**
*   **Deep Learning**
*   **Evolutionary Algorithms and Swarm Intelligence**

Each section includes model implementations and practical examples to facilitate understanding of the concepts.

## Recent Updates

**Translation and Logging Enhancement (Latest)**
- All code has been translated from Spanish to English
- Print statements have been replaced with proper logging for better output management and analysis
- Added centralized logging configuration for consistent output formatting
- All 30 Jupyter notebooks and Python files have been updated
- Logging output is saved to timestamped files in the `logs/` directory

## Project Structure

- `seccion_1/`: Rule-based systems and fuzzy logic examples
- `seccion_2/`: Machine learning and neural network examples  
- `seccion_3/`: Deep learning and preprocessing techniques
- `seccion_4/`: Evolutionary algorithms and optimization examples
- `logging_config.py`: Centralized logging configuration
- `logs/`: Directory for log files (auto-created)

## Logging System

The project now uses a centralized logging system that:
- Creates timestamped log files for each execution
- Provides different log levels (INFO, WARNING, ERROR)
- Outputs to both console and file
- Enables better analysis and debugging of model outputs

## Usage

To use the enhanced logging system in your code:

```python
from logging_config import setup_logging, get_logger

# Set up logging
logger = setup_logging(log_level=logging.INFO)
module_logger = get_logger(__name__)

# Use logging instead of print
module_logger.info("Your message here")
module_logger.warning("Warning message")
module_logger.error("Error message")
```
