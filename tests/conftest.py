"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    n_cells = 100
    n_genes = 50
    
    # Generate random expression data
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Create gene names
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    
    # Create cell names
    cell_names = [f"Cell_{i:03d}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    
    # Add some metadata
    adata.obs['cell_type'] = np.random.choice(['A', 'B', 'C'], n_cells)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], n_cells)
    
    return adata


@pytest.fixture
def perturbation_genes():
    """Sample perturbation genes for testing."""
    return ['Gene_001', 'Gene_005', 'Gene_010', 'Gene_015']


@pytest.fixture
def control_genes():
    """Sample control genes for testing."""
    return ['Gene_020', 'Gene_025', 'Gene_030', 'Gene_035']


@pytest.fixture
def sample_perturbation_data():
    """Create sample data with perturbation effects."""
    n_cells = 200
    n_genes = 100
    
    # Generate base expression
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Add perturbation effects to some cells
    perturbed_cells = np.random.choice(n_cells, size=50, replace=False)
    perturbation_genes = [10, 20, 30, 40]  # Gene indices
    
    for cell_idx in perturbed_cells:
        for gene_idx in perturbation_genes:
            X[cell_idx, gene_idx] *= 2  # 2-fold increase
    
    # Create AnnData object
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:03d}" for i in range(n_cells)]
    
    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    
    # Add perturbation metadata
    adata.obs['perturbed'] = False
    adata.obs.loc[adata.obs_names[perturbed_cells], 'perturbed'] = True
    
    return adata
