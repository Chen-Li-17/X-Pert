"""
Data validation utilities.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from anndata import AnnData
import warnings


def validate_data(adata: AnnData, 
                  min_cells: int = 10,
                  min_genes: int = 100,
                  check_duplicates: bool = True) -> None:
    """
    Validate single-cell data for perturbation analysis.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell data to validate
    min_cells : int
        Minimum number of cells required
    min_genes : int
        Minimum number of genes required
    check_duplicates : bool
        Whether to check for duplicate gene names
        
    Raises
    ------
    ValueError
        If data validation fails
    """
    # Check basic data structure
    if not isinstance(adata, AnnData):
        raise ValueError("Data must be an AnnData object")
    
    # Check dimensions
    if adata.n_obs < min_cells:
        raise ValueError(f"Data must have at least {min_cells} cells, got {adata.n_obs}")
    
    if adata.n_vars < min_genes:
        raise ValueError(f"Data must have at least {min_genes} genes, got {adata.n_vars}")
    
    # Check for missing values
    if np.isnan(adata.X.data).any():
        warnings.warn("Data contains NaN values", UserWarning)
    
    # Check for duplicate gene names
    if check_duplicates and adata.var_names.duplicated().any():
        warnings.warn("Data contains duplicate gene names", UserWarning)
    
    # Check for empty cells/genes
    if (adata.X.sum(axis=1) == 0).any():
        warnings.warn("Data contains cells with zero expression", UserWarning)
    
    if (adata.X.sum(axis=0) == 0).any():
        warnings.warn("Data contains genes with zero expression", UserWarning)


def validate_perturbation_genes(adata: AnnData, 
                               perturbation_genes: List[str]) -> List[str]:
    """
    Validate and filter perturbation genes.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell data
    perturbation_genes : List[str]
        List of genes to validate
        
    Returns
    -------
    valid_genes : List[str]
        List of valid perturbation genes present in the data
    """
    valid_genes = []
    missing_genes = []
    
    for gene in perturbation_genes:
        if gene in adata.var_names:
            valid_genes.append(gene)
        else:
            missing_genes.append(gene)
    
    if missing_genes:
        warnings.warn(f"Missing genes in data: {missing_genes}", UserWarning)
    
    if not valid_genes:
        raise ValueError("No valid perturbation genes found in the data")
    
    return valid_genes


def validate_control_genes(adata: AnnData,
                          control_genes: Optional[List[str]] = None,
                          perturbation_genes: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Validate control genes.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell data
    control_genes : List[str], optional
        List of control genes
    perturbation_genes : List[str], optional
        List of perturbation genes to exclude from controls
        
    Returns
    -------
    valid_controls : List[str], optional
        List of valid control genes
    """
    if control_genes is None:
        return None
    
    valid_controls = []
    missing_controls = []
    
    for gene in control_genes:
        if gene in adata.var_names:
            # Exclude perturbation genes from controls
            if perturbation_genes is None or gene not in perturbation_genes:
                valid_controls.append(gene)
        else:
            missing_controls.append(gene)
    
    if missing_controls:
        warnings.warn(f"Missing control genes in data: {missing_controls}", UserWarning)
    
    return valid_controls if valid_controls else None
