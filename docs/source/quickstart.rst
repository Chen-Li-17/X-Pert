Quick Start
===========

This guide will help you quickly get started with X-Pert for single-cell perturbation analysis.

Basic Usage
-----------

Loading Data
~~~~~~~~~~~~

First, load your single-cell data:

.. code-block:: python

   import xpert as xp
   import scanpy as sc
   
   # Load data
   adata = sc.read_h5ad("your_data.h5ad")
   
   # View basic data information
   print(adata)
   print(adata.obs.head())
   print(adata.var.head())

Creating Analyzer
~~~~~~~~~~~~~~~~~

Create a perturbation analyzer:

.. code-block:: python

   # Create analyzer
   analyzer = xp.PerturbationAnalyzer(adata)
   
   # Or use specific model
   from xpert.models import LinearPerturbationModel
   model = LinearPerturbationModel()
   analyzer = xp.PerturbationAnalyzer(adata, model=model)

Identifying Perturbed Genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Identify perturbed genes
   perturbed_genes = analyzer.identify_perturbed_genes()
   print(f"Found {len(perturbed_genes)} perturbed genes")
   print(perturbed_genes[:10])  # Show first 10

Analyzing Perturbation Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define perturbation genes
   perturbation_genes = ['GENE1', 'GENE2', 'GENE3']
   control_genes = ['CTRL1', 'CTRL2', 'CTRL3']
   
   # Analyze perturbation effects
   results = analyzer.analyze_perturbation_effects(
       perturbation_genes=perturbation_genes,
       control_genes=control_genes
   )
   
   print("Analysis results:")
   print(f"Prediction shape: {results['predictions'].shape}")
   print(f"Scores: {results['scores']}")

Visualizing Results
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create perturbation heatmap
   analyzer.plot_perturbation_heatmap(genes=perturbation_genes)
   
   # Save results
   analyzer.save_results("perturbation_results.h5ad")

Complete Example
----------------

Here's a complete analysis example:

.. code-block:: python

   import xpert as xp
   import scanpy as sc
   import matplotlib.pyplot as plt
   
   # 1. Load data
   print("Loading data...")
   adata = sc.read_h5ad("data.h5ad")
   
   # 2. Data preprocessing
   print("Preprocessing data...")
   sc.pp.filter_cells(adata, min_genes=200)
   sc.pp.filter_genes(adata, min_cells=3)
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   
   # 3. Create analyzer
   print("Creating analyzer...")
   analyzer = xp.PerturbationAnalyzer(adata)
   
   # 4. Identify perturbed genes
   print("Identifying perturbed genes...")
   perturbed_genes = analyzer.identify_perturbed_genes()
   
   # 5. Analyze perturbation effects
   print("Analyzing perturbation effects...")
   results = analyzer.analyze_perturbation_effects(
       perturbation_genes=perturbed_genes[:10]  # Use first 10 genes
   )
   
   # 6. Visualization
   print("Creating visualization...")
   analyzer.plot_perturbation_heatmap()
   plt.title("Perturbation Effects Heatmap")
   plt.show()
   
   # 7. Save results
   print("Saving results...")
   analyzer.save_results("results.h5ad")
   
   print("Analysis complete!")

Command Line Usage
------------------

X-Pert also provides a command-line interface:

.. code-block:: bash

   # Analyze perturbation effects
   xpert analyze data.h5ad --perturbation-genes GENE1,GENE2,GENE3 --output results.h5ad
   
   # Create visualization
   xpert plot data.h5ad --genes GENE1,GENE2 --output plot.png --plot-type heatmap

Advanced Features
-----------------

Custom Models
~~~~~~~~~~~~~

.. code-block:: python

   from xpert.models import PerturbationModel
   
   class CustomModel(PerturbationModel):
       def fit(self, adata, perturbation_genes, control_genes=None, **kwargs):
           # Implement your model
           self.is_fitted = True
           return self
       
       def predict(self, adata, perturbation_genes, **kwargs):
           # Implement prediction logic
           return predictions
       
       def score(self, adata, perturbation_genes, **kwargs):
           # Implement scoring logic
           return scores
   
   # Use custom model
   custom_model = CustomModel()
   analyzer = xp.PerturbationAnalyzer(adata, model=custom_model)

Batch Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze multiple gene sets
   gene_sets = [
       ['GENE1', 'GENE2'],
       ['GENE3', 'GENE4'],
       ['GENE5', 'GENE6']
   ]
   
   all_results = {}
   for i, genes in enumerate(gene_sets):
       print(f"Analyzing gene set {i+1}/{len(gene_sets)}")
       results = analyzer.analyze_perturbation_effects(genes)
       all_results[f"set_{i}"] = results

Next Steps
----------

Now that you understand the basic usage of X-Pert, you can:

1. Check out :doc:`tutorial` for more detailed tutorials
2. Browse :doc:`examples` for more examples
3. Read :doc:`api/analysis` for API details
4. See :doc:`contributing` to participate in project development

Need help? Check :doc:`troubleshooting` or submit an issue.