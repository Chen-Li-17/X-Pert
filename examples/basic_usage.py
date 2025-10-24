"""
X-Pert Basic Usage Example

This example demonstrates how to use X-Pert for single-cell perturbation analysis.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import xpert as xp


def create_sample_data(n_cells=1000, n_genes=2000):
    """Create sample single-cell data"""
    print("Creating sample data...")
    
    # Generate random expression data
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Create gene and cell names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    
    # Add cell metadata
    adata.obs['cell_type'] = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], n_cells)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2', 'batch3'], n_cells)
    
    # Add gene metadata
    adata.var['gene_type'] = np.random.choice(['protein_coding', 'lncRNA', 'pseudogene'], n_genes)
    
    return adata


def add_perturbation_effects(adata, perturbation_genes, effect_size=2.0):
    """Add perturbation effects to the data"""
    print(f"Adding perturbation effects to {len(perturbation_genes)} genes...")
    
    # Randomly select some cells for perturbation
    n_perturbed_cells = adata.n_obs // 4
    perturbed_cells = np.random.choice(adata.n_obs, size=n_perturbed_cells, replace=False)
    
    # Add effects to perturbed genes
    for gene in perturbation_genes:
        if gene in adata.var_names:
            gene_idx = adata.var_names.get_loc(gene)
            adata.X[perturbed_cells, gene_idx] *= effect_size
    
    # Add perturbation metadata
    adata.obs['perturbed'] = False
    adata.obs.loc[adata.obs_names[perturbed_cells], 'perturbed'] = True
    
    return adata


def main():
    """Main function"""
    print("X-Pert Basic Usage Example")
    print("=" * 50)
    
    # 1. Create sample data
    adata = create_sample_data()
    print(f"Data shape: {adata.shape}")
    print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
    
    # 2. Data preprocessing
    print("\nData preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    print(f"Data shape after preprocessing: {adata.shape}")
    
    # 3. Define perturbation genes
    perturbation_genes = ['Gene_0010', 'Gene_0020', 'Gene_0030', 'Gene_0040', 'Gene_0050']
    control_genes = ['Gene_1000', 'Gene_1100', 'Gene_1200', 'Gene_1300', 'Gene_1400']
    
    # 4. Add perturbation effects
    adata = add_perturbation_effects(adata, perturbation_genes)
    
    # 5. Create perturbation analyzer
    print("\nCreating perturbation analyzer...")
    analyzer = xp.PerturbationAnalyzer(adata)
    
    # 6. Identify perturbed genes
    print("Identifying perturbed genes...")
    identified_genes = analyzer.identify_perturbed_genes()
    print(f"Identified {len(identified_genes)} perturbed genes")
    print(f"Top 10: {identified_genes[:10]}")
    
    # 7. Analyze perturbation effects
    print("\nAnalyzing perturbation effects...")
    results = analyzer.analyze_perturbation_effects(
        perturbation_genes=perturbation_genes,
        control_genes=control_genes
    )
    
    print("Analysis results:")
    print(f"Prediction shape: {results['predictions'].shape}")
    print(f"Scores: {results['scores']}")
    
    # 8. Compute perturbation scores
    print("\nComputing perturbation scores...")
    scores = analyzer.get_perturbation_scores(perturbation_genes)
    print(f"Perturbation scores: {scores}")
    
    # 9. Visualization (if available)
    try:
        print("\nCreating visualization...")
        analyzer.plot_perturbation_heatmap(genes=perturbation_genes)
        print("Heatmap created")
    except Exception as e:
        print(f"Visualization creation failed: {e}")
    
    # 10. Save results
    print("\nSaving results...")
    analyzer.save_results("perturbation_results.h5ad")
    print("Results saved to perturbation_results.h5ad")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()
