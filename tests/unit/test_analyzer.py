"""
Unit tests for perturbation analyzer.
"""

import pytest
import numpy as np
from anndata import AnnData

from xpert.analysis import PerturbationAnalyzer
from xpert.models.base import PerturbationModel


class MockPerturbationModel(PerturbationModel):
    """Mock perturbation model for testing."""
    
    def fit(self, adata, perturbation_genes, control_genes=None, **kwargs):
        """Mock fit method."""
        self.is_fitted = True
        return self
    
    def predict(self, adata, perturbation_genes, **kwargs):
        """Mock predict method."""
        return np.random.rand(adata.n_obs, len(perturbation_genes))
    
    def score(self, adata, perturbation_genes, **kwargs):
        """Mock score method."""
        return {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75}


class TestPerturbationAnalyzer:
    """Test perturbation analyzer functionality."""
    
    def test_initialization(self, sample_adata):
        """Test analyzer initialization."""
        analyzer = PerturbationAnalyzer(sample_adata)
        assert analyzer.adata is not None
        assert analyzer.model is None
        assert analyzer.results == {}
    
    def test_initialization_with_model(self, sample_adata):
        """Test analyzer initialization with model."""
        model = MockPerturbationModel()
        analyzer = PerturbationAnalyzer(sample_adata, model=model)
        assert analyzer.model is not None
        assert analyzer.model == model
    
    def test_analyze_perturbation_effects_no_model(self, sample_adata, perturbation_genes):
        """Test analysis without model raises error."""
        analyzer = PerturbationAnalyzer(sample_adata)
        
        with pytest.raises(ValueError, match="No model specified"):
            analyzer.analyze_perturbation_effects(perturbation_genes)
    
    def test_analyze_perturbation_effects_with_model(self, sample_adata, perturbation_genes):
        """Test analysis with model."""
        model = MockPerturbationModel()
        analyzer = PerturbationAnalyzer(sample_adata, model=model)
        
        results = analyzer.analyze_perturbation_effects(perturbation_genes)
        
        assert 'predictions' in results
        assert 'scores' in results
        assert 'perturbation_genes' in results
        assert results['perturbation_genes'] == perturbation_genes
        assert results['predictions'].shape == (sample_adata.n_obs, len(perturbation_genes))
    
    def test_analyze_perturbation_effects_with_controls(self, sample_adata, perturbation_genes, control_genes):
        """Test analysis with control genes."""
        model = MockPerturbationModel()
        analyzer = PerturbationAnalyzer(sample_adata, model=model)
        
        results = analyzer.analyze_perturbation_effects(
            perturbation_genes, 
            control_genes=control_genes
        )
        
        assert results['control_genes'] == control_genes
    
    def test_plot_perturbation_heatmap_no_results(self, sample_adata):
        """Test plotting without results raises error."""
        analyzer = PerturbationAnalyzer(sample_adata)
        
        with pytest.raises(ValueError, match="No analysis results available"):
            analyzer.plot_perturbation_heatmap()
    
    def test_get_perturbation_scores(self, sample_adata, perturbation_genes):
        """Test getting perturbation scores."""
        analyzer = PerturbationAnalyzer(sample_adata)
        
        scores = analyzer.get_perturbation_scores(perturbation_genes)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(perturbation_genes)
