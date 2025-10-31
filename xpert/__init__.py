"""
X-Pert: Single Cell Perturbation Analysis

A comprehensive framework for analyzing gene perturbations in single-cell data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# package-level logger used by submodules via `from .. import logger`
import logging as _logging

logger = _logging.getLogger("xpert")
if not logger.handlers:
    _logging.basicConfig(level=_logging.INFO)
    logger.setLevel(_logging.INFO)

# Import main classes and functions from actual files
from .data import Byte_Pert_Data

__all__ = [
    "Byte_Pert_Data",
    "__version__",
    "__author__",
    "__email__",
    "logger",
]