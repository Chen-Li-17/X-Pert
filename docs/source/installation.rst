Installation Guide
==================

X-Pert builds on the single-cell foundation model **scGPT**. Install and configure scGPT first, then set up X-Pert in the same environment.

.. important::

   X-Pert requires `scGPT <https://github.com/bowang-lab/scGPT>`_ for pretrained gene embeddings and tokenizers. Install ``scgpt`` before installing X-Pert.

Prerequisites
-------------

* Python 3.8 or later (tested with CPython 3.8–3.11)
* CUDA-capable GPU recommended for large-scale experiments (optional)
* ``scgpt`` package installed in the target environment

1. Install scGPT
----------------

Create a fresh environment (conda or virtualenv) and install scGPT:

.. code-block:: bash

   # optional: create a dedicated environment
   conda create -n xpert python=3.10
   conda activate xpert

   # install scGPT (required dependency)
   pip install scgpt

For advanced setups (e.g., FlashAttention, custom CUDA builds), follow the scGPT instructions at https://github.com/bowang-lab/scGPT.

2. Install X-Pert
-----------------

Clone the repository and install X-Pert in editable mode:

.. code-block:: bash

   git clone https://github.com/Chen-Li-17/X-Pert.git
   cd X-Pert
   pip install -e .

To recreate the documented environment, use the provided ``environment.yml``:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate xpert

3. Optional: Documentation and Development Extras
-------------------------------------------------

Install optional dependencies for documentation or development tasks:

.. code-block:: bash

   # Documentation build
   pip install -e ".[docs]"

   # Development tooling (linters, tests, pre-commit)
   pip install -e ".[dev]"
   pre-commit install

4. Verify the Installation
--------------------------

Run a quick import check to ensure scGPT and X-Pert are both available:

.. code-block:: python

   import scgpt
   import xpert

   print(f"scGPT version: {scgpt.__version__}")
   print(f"X-Pert version: {xpert.__version__}")

If both imports succeed and versions print correctly, your environment is ready for the tutorial notebooks.

Getting Help
------------

* Open an issue on GitHub: https://github.com/Chen-Li-17/X-Pert/issues
* Contact the maintainers: chen-li21@mails.tsinghua.edu.cn