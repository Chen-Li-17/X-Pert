X-Pert Documentation
====================

X-Pert is a Python package specifically designed for single-cell perturbation analysis, providing powerful tools to analyze the effects of gene perturbations on single-cell transcriptomes.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/analysis
   api/models
   api/data
   api/visualization
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing
   development
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Other

   license
   authors

Features
--------

* 🧬 **Single-cell Perturbation Modeling**: Support for multiple perturbation types and modeling methods
* 📊 **Data Preprocessing**: Complete single-cell data preprocessing pipeline
* 🔍 **Perturbation Effect Analysis**: Identify and analyze the effects of gene perturbations on cell states
* 📈 **Visualization Tools**: Rich visualization capabilities for result presentation
* ⚡ **High-Performance Computing**: Optimized algorithms for large-scale single-cell data
* 🔧 **Modular Design**: Easy to extend and customize modular architecture

Quick Start
-----------

.. code-block:: python

   import xpert as xp
   import scanpy as sc

   # Load single-cell data
   adata = sc.read_h5ad("your_data.h5ad")

   # Create perturbation analyzer
   perturbator = xp.PerturbationAnalyzer(adata)

   # Identify perturbed genes
   perturbed_genes = perturbator.identify_perturbed_genes()

   # Analyze perturbation effects
   effects = perturbator.analyze_perturbation_effects()

   # Visualize results
   perturbator.plot_perturbation_heatmap()

Installation
------------

.. code-block:: bash

   pip install x-pert

Or install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/X-Pert.git
   cd X-Pert
   pip install -e .

Citation
--------

If you use X-Pert in your research, please cite our paper:

.. code-block:: bibtex

   @article{xpert2024,
     title={X-Pert: A Comprehensive Framework for Single Cell Perturbation Analysis},
     author={Your Name and Collaborators},
     journal={Nature Methods},
     year={2024}
   }

License
-------

This project is licensed under the MIT License - see :doc:`license` for details.

Contact
-------

* Project Homepage: https://github.com/yourusername/X-Pert
* Issue Tracker: https://github.com/yourusername/X-Pert/issues
* Email: your.email@example.com

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`