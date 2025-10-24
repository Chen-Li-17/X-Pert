"""
Unit tests for validation utilities.
"""

import pytest
import numpy as np
from anndata import AnnData

from xpert.utils.validation import validate_data, validate_perturbation_genes, validate_control_genes


class TestValidateData:
    """Test data validation functions."""
    
    def test_valid_data(self, sample_adata):
        """Test validation of valid data."""
        # Should not raise any exceptions
        validate_data(sample_adata)
    
    def test_insufficient_cells(self):
        """Test validation with insufficient cells."""
        X = np.random.rand(5, 50)  # Only 5 cells
        adata = AnnData(X=X)
        adata.var_names = [f"Gene_{i}" for i in range(50)]
        adata.obs_names = [f"Cell_{i}" for i in range(5)]
        
        with pytest.raises(ValueError, match="Data must have at least 10 cells"):
            validate_data(adata, min_cells=10)
    
    def test_insufficient_genes(self):
        """Test validation with insufficient genes."""
        X = np.random.rand(100, 5)  # Only 5 genes
        adata = AnnData(X=X)
        adata.var_names = [f"Gene_{i}" for i in range(5)]
        adata.obs_names = [f"Cell_{i}" for i in range(100)]
        
        with pytest.raises(ValueError, match="Data must have at least 100 genes"):
            validate_data(adata, min_genes=100)
    
    def test_wrong_type(self):
        """Test validation with wrong data type."""
        with pytest.raises(ValueError, match="Data must be an AnnData object"):
            validate_data("not_an_ann_data")


class TestValidatePerturbationGenes:
    """Test perturbation gene validation."""
    
    def test_valid_genes(self, sample_adata, perturbation_genes):
        """Test validation of valid perturbation genes."""
        valid_genes = validate_perturbation_genes(sample_adata, perturbation_genes)
        assert len(valid_genes) == len(perturbation_genes)
        assert all(gene in sample_adata.var_names for gene in valid_genes)
    
    def test_missing_genes(self, sample_adata):
        """Test validation with missing genes."""
        missing_genes = ['NonExistentGene1', 'NonExistentGene2']
        valid_genes = ['Gene_001', 'Gene_005']  # Some valid genes
        all_genes = missing_genes + valid_genes
        
        with pytest.warns(UserWarning, match="Missing genes in data"):
            result = validate_perturbation_genes(sample_adata, all_genes)
        
        assert len(result) == len(valid_genes)
        assert all(gene in valid_genes for gene in result)
    
    def test_no_valid_genes(self, sample_adata):
        """Test validation with no valid genes."""
        missing_genes = ['NonExistentGene1', 'NonExistentGene2']
        
        with pytest.raises(ValueError, match="No valid perturbation genes found"):
            validate_perturbation_genes(sample_adata, missing_genes)


class TestValidateControlGenes:
    """Test control gene validation."""
    
    def test_valid_controls(self, sample_adata, control_genes):
        """Test validation of valid control genes."""
        valid_controls = validate_control_genes(sample_adata, control_genes)
        assert valid_controls is not None
        assert len(valid_controls) == len(control_genes)
        assert all(gene in sample_adata.var_names for gene in valid_controls)
    
    def test_none_controls(self, sample_adata):
        """Test validation with None control genes."""
        result = validate_control_genes(sample_adata, None)
        assert result is None
    
    def test_exclude_perturbation_genes(self, sample_adata, perturbation_genes):
        """Test exclusion of perturbation genes from controls."""
        # Mix perturbation and control genes
        mixed_genes = perturbation_genes + ['Gene_020', 'Gene_025']
        
        valid_controls = validate_control_genes(
            sample_adata, 
            mixed_genes, 
            perturbation_genes
        )
        
        assert valid_controls is not None
        assert len(valid_controls) == 2  # Only the control genes
        assert all(gene not in perturbation_genes for gene in valid_controls)
