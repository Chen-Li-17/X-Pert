"""
X-Pert: Single Cell Perturbation Analysis

A comprehensive framework for analyzing gene perturbations in single-cell data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes and functions
from .analysis import PerturbationAnalyzer
from .models import PerturbationModel
from .data import DataLoader, DataPreprocessor
from .visualization import PerturbationPlotter

# Import utilities
from .utils import (
    validate_data,
    compute_perturbation_scores,
    identify_differential_genes,
)

__all__ = [
    # Main classes
    "PerturbationAnalyzer",
    "PerturbationModel", 
    "DataLoader",
    "DataPreprocessor",
    "PerturbationPlotter",
    
    # Utility functions
    "validate_data",
    "compute_perturbation_scores",
    "identify_differential_genes",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
]
