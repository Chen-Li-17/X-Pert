Quick Start
===========

This guide summarizes the end-to-end workflow for reproducing the experiments described in the X-Pert manuscript. Each stage corresponds to a curated Jupyter notebook that you can launch after installing the dependencies.

Workflow Overview
-----------------

1. **Prepare the Environment**

   #. Install scGPT and X-Pert as described in :doc:`installation`.
   #. Download the required perturbation datasets (links provided inside each notebook).

2. **Construct Perturbation Datasets**

   * :doc:`Genetic perturbation preprocessing <notebooks/create_genetic_data>`
   * :doc:`Chemical perturbation preprocessing <notebooks/create_chemical_data>`
   * :doc:`Joint (co-train) perturbation preprocessing <notebooks/create_cotrain_data>`

3. **Train Perturbation Models**

   * :doc:`Train X-Pert on genetic perturbations <notebooks/train_genetic_model>`
   * :doc:`Train X-Pert on chemical perturbations <notebooks/train_chemical_model>`

4. **Explore the Perturbverse**

   * :doc:`Visualize latent perturbation embeddings <notebooks/visualize_perturbverse>`

Minimal Python Check
--------------------

After completing the environment setup, verify that both scGPT and X-Pert import correctly:

.. code-block:: python

   import scgpt
   import xpert

   print(f"scGPT version: {scgpt.__version__}")
   print(f"X-Pert version: {xpert.__version__}")

High-level Python APIs for perturbation modeling are actively evolving. For complete, reproducible pipelines—including data preparation, training, and visualization—use the tutorial notebooks linked above.

Next Steps
----------

* Switch to the tutorial notebooks listed above for full data-to-results pipelines.
* Follow ongoing updates and discussions on GitHub: https://github.com/Chen-Li-17/X-Pert
* Reach out to the maintainers at chen-li21@mails.tsinghua.edu.cn if you run into issues.