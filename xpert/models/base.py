"""
Base perturbation model class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from anndata import AnnData


class PerturbationModel(ABC):
    """
    Abstract base class for perturbation models.
    
    This class defines the interface that all perturbation models must implement.
    """
    
    def __init__(self, **kwargs):
        """Initialize the perturbation model."""
        self.model_params = kwargs
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, 
            adata: AnnData, 
            perturbation_genes: list,
            control_genes: Optional[list] = None,
            **kwargs) -> 'PerturbationModel':
        """
        Fit the perturbation model to the data.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data with gene expression
        perturbation_genes : list
            List of genes that were perturbed
        control_genes : list, optional
            List of control genes for comparison
        **kwargs
            Additional parameters for model fitting
            
        Returns
        -------
        self : PerturbationModel
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, 
                adata: AnnData,
                perturbation_genes: list,
                **kwargs) -> np.ndarray:
        """
        Predict perturbation effects.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data to make predictions on
        perturbation_genes : list
            List of genes that were perturbed
        **kwargs
            Additional parameters for prediction
            
        Returns
        -------
        predictions : np.ndarray
            Predicted perturbation effects
        """
        pass
    
    @abstractmethod
    def score(self, 
              adata: AnnData,
              perturbation_genes: list,
              **kwargs) -> Dict[str, float]:
        """
        Compute model performance scores.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data
        perturbation_genes : list
            List of genes that were perturbed
        **kwargs
            Additional parameters for scoring
            
        Returns
        -------
        scores : Dict[str, float]
            Dictionary of performance metrics
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'PerturbationModel':
        """Set model parameters."""
        self.model_params.update(params)
        return self
    
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        # Implementation would depend on the specific model type
        raise NotImplementedError("Model saving not implemented")
    
    def load(self, filepath: str) -> 'PerturbationModel':
        """Load the model from disk."""
        # Implementation would depend on the specific model type
        raise NotImplementedError("Model loading not implemented")
