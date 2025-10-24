"""
Integration tests for the full perturbation analysis pipeline.
"""

import pytest
import numpy as np
from anndata import AnnData

from xpert.analysis import PerturbationAnalyzer
from xpert.models.base import PerturbationModel


class TestFullPipeline:
    """Test the complete perturbation analysis pipeline."""
    
    def test_end_to_end_analysis(self, sample_perturbation_data):
        """Test complete end-to-end analysis pipeline."""
        # Create a simple mock model for testing
        class SimplePerturbationModel(PerturbationModel):
            def fit(self, adata, perturbation_genes, control_genes=None, **kwargs):
                self.perturbation_genes = perturbation_genes
                self.control_genes = control_genes
                self.is_fitted = True
                return self
            
            def predict(self, adata, perturbation_genes, **kwargs):
                # Simple prediction based on expression levels
                predictions = np.zeros((adata.n_obs, len(perturbation_genes)))
                for i, gene in enumerate(perturbation_genes):
                    if gene in adata.var_names:
                        gene_idx = adata.var_names.get_loc(gene)
                        predictions[:, i] = adata.X[:, gene_idx].toarray().flatten()
                return predictions
            
            def score(self, adata, perturbation_genes, **kwargs):
                return {
                    'correlation': 0.75,
                    'mse': 0.25,
                    'r2': 0.60
                }
        
        # Set up analysis
        model = SimplePerturbationModel()
        analyzer = PerturbationAnalyzer(sample_perturbation_data, model=model)
        
        # Define perturbation genes
        perturbation_genes = ['Gene_010', 'Gene_020', 'Gene_030', 'Gene_040']
        control_genes = ['Gene_050', 'Gene_060', 'Gene_070', 'Gene_080']
        
        # Run analysis
        results = analyzer.analyze_perturbation_effects(
            perturbation_genes=perturbation_genes,
            control_genes=control_genes
        )
        
        # Verify results structure
        assert 'predictions' in results
        assert 'scores' in results
        assert 'perturbation_genes' in results
        assert 'control_genes' in results
        
        # Verify prediction shape
        assert results['predictions'].shape == (sample_perturbation_data.n_obs, len(perturbation_genes))
        
        # Verify scores
        assert 'correlation' in results['scores']
        assert 'mse' in results['scores']
        assert 'r2' in results['scores']
        
        # Verify model is fitted
        assert model.is_fitted
        
        # Test perturbation score computation
        scores = analyzer.get_perturbation_scores(perturbation_genes)
        assert len(scores) == len(perturbation_genes)
        assert all(isinstance(score, (int, float)) for score in scores)
    
    def test_data_validation_in_pipeline(self):
        """Test that data validation works in the pipeline."""
        # Create invalid data (too few cells)
        X = np.random.rand(5, 50)
        adata = AnnData(X=X)
        adata.var_names = [f"Gene_{i}" for i in range(50)]
        adata.obs_names = [f"Cell_{i}" for i in range(5)]
        
        with pytest.raises(ValueError):
            PerturbationAnalyzer(adata)
    
    def test_perturbation_gene_validation(self, sample_perturbation_data):
        """Test perturbation gene validation in pipeline."""
        model = MockPerturbationModel()
        analyzer = PerturbationAnalyzer(sample_perturbation_data, model=model)
        
        # Test with invalid genes
        invalid_genes = ['NonExistentGene1', 'NonExistentGene2']
        
        with pytest.raises(ValueError):
            analyzer.analyze_perturbation_effects(invalid_genes)


class MockPerturbationModel(PerturbationModel):
    """Mock model for integration testing."""
    
    def fit(self, adata, perturbation_genes, control_genes=None, **kwargs):
        self.is_fitted = True
        return self
    
    def predict(self, adata, perturbation_genes, **kwargs):
        return np.random.rand(adata.n_obs, len(perturbation_genes))
    
    def score(self, adata, perturbation_genes, **kwargs):
        return {'accuracy': 0.85}
