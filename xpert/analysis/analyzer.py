"""
Main perturbation analyzer class.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

from ..models import PerturbationModel
from ..utils import validate_data, compute_perturbation_scores


class PerturbationAnalyzer:
    """
    Main class for analyzing single-cell perturbation data.
    
    This class provides a high-level interface for performing perturbation
    analysis on single-cell RNA-seq data.
    """
    
    def __init__(self, 
                 adata: AnnData,
                 model: Optional[PerturbationModel] = None,
                 **kwargs):
        """
        Initialize the perturbation analyzer.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data with gene expression
        model : PerturbationModel, optional
            Perturbation model to use for analysis
        **kwargs
            Additional parameters for the analyzer
        """
        self.adata = adata.copy()
        self.model = model
        self.analyzer_params = kwargs
        self.results = {}
        
        # Validate input data
        validate_data(self.adata)
        
    def identify_perturbed_genes(self, 
                                method: str = "differential",
                                **kwargs) -> List[str]:
        """
        Identify genes that show significant perturbation effects.
        
        Parameters
        ----------
        method : str
            Method to use for identifying perturbed genes
        **kwargs
            Additional parameters for the identification method
            
        Returns
        -------
        perturbed_genes : List[str]
            List of genes showing significant perturbation effects
        """
        if method == "differential":
            from .differential import DifferentialAnalysis
            diff_analyzer = DifferentialAnalysis(self.adata)
            results = diff_analyzer.identify_differential_genes(**kwargs)
            return results['significant_genes'].tolist()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_perturbation_effects(self,
                                   perturbation_genes: List[str],
                                   control_genes: Optional[List[str]] = None,
                                   **kwargs) -> Dict[str, Any]:
        """
        Analyze the effects of gene perturbations.
        
        Parameters
        ----------
        perturbation_genes : List[str]
            List of genes that were perturbed
        control_genes : List[str], optional
            List of control genes for comparison
        **kwargs
            Additional parameters for analysis
            
        Returns
        -------
        results : Dict[str, Any]
            Dictionary containing analysis results
        """
        if self.model is None:
            raise ValueError("No model specified for analysis")
        
        # Fit the model if not already fitted
        if not self.model.is_fitted:
            self.model.fit(self.adata, perturbation_genes, control_genes, **kwargs)
        
        # Make predictions
        predictions = self.model.predict(self.adata, perturbation_genes, **kwargs)
        
        # Compute scores
        scores = self.model.score(self.adata, perturbation_genes, **kwargs)
        
        # Store results
        self.results = {
            'predictions': predictions,
            'scores': scores,
            'perturbation_genes': perturbation_genes,
            'control_genes': control_genes,
        }
        
        return self.results
    
    def plot_perturbation_heatmap(self, 
                                 genes: Optional[List[str]] = None,
                                 **kwargs) -> None:
        """
        Plot a heatmap of perturbation effects.
        
        Parameters
        ----------
        genes : List[str], optional
            Genes to include in the heatmap
        **kwargs
            Additional parameters for plotting
        """
        from ..visualization import PerturbationHeatmap
        
        if 'predictions' not in self.results:
            raise ValueError("No analysis results available. Run analyze_perturbation_effects first.")
        
        plotter = PerturbationHeatmap(self.results['predictions'])
        plotter.plot(genes=genes, **kwargs)
    
    def get_perturbation_scores(self, 
                               perturbation_genes: List[str],
                               **kwargs) -> np.ndarray:
        """
        Compute perturbation scores for specified genes.
        
        Parameters
        ----------
        perturbation_genes : List[str]
            List of genes to compute scores for
        **kwargs
            Additional parameters for scoring
            
        Returns
        -------
        scores : np.ndarray
            Perturbation scores for each gene
        """
        return compute_perturbation_scores(
            self.adata, 
            perturbation_genes, 
            **kwargs
        )
    
    def save_results(self, filepath: str) -> None:
        """
        Save analysis results to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the results
        """
        from ..utils import save_results
        save_results(self.results, filepath)
    
    def load_results(self, filepath: str) -> None:
        """
        Load analysis results from disk.
        
        Parameters
        ----------
        filepath : str
            Path to load the results from
        """
        from ..utils import load_results
        self.results = load_results(filepath)
