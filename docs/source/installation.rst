Installation Guide
==================

X-Pert supports multiple installation methods. We recommend using conda or pip for installation.

System Requirements
--------------------

* Python 3.8 or higher
* At least 4GB RAM (8GB or more recommended)
* At least 1GB available disk space

Install with pip
----------------

Recommended installation method using pip:

.. code-block:: bash

   pip install x-pert

If you need the development version, install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/yourusername/X-Pert.git

Install with conda
------------------

Recommended method using conda to create an isolated environment:

.. code-block:: bash

   # Create new environment
   conda env create -f environment.yml
   
   # Activate environment
   conda activate xpert

Or install manually:

.. code-block:: bash

   conda create -n xpert python=3.9
   conda activate xpert
   conda install -c conda-forge scanpy pandas numpy scipy scikit-learn matplotlib seaborn
   pip install x-pert

Install from Source
--------------------

If you want to install from source or participate in development:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/X-Pert.git
   cd X-Pert
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install

Verify Installation
-------------------

After installation, you can verify the installation was successful:

.. code-block:: python

   import xpert as xp
   print(xp.__version__)

If the version number is successfully output, the installation was successful.

Optional Dependencies
---------------------

X-Pert supports various optional dependencies for specific features:

Deep Learning Models
~~~~~~~~~~~~~~~~~~~~

If you need to use deep learning models, install the following dependencies:

.. code-block:: bash

   # PyTorch
   pip install torch torchvision torchaudio
   
   # TensorFlow
   pip install tensorflow
   
   # JAX
   pip install jax jaxlib

GPU Support
~~~~~~~~~~~

For GPU acceleration, install the corresponding CUDA version:

.. code-block:: bash

   # PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # TensorFlow with GPU
   pip install tensorflow[and-cuda]

Documentation Building
~~~~~~~~~~~~~~~~~~~~~~

If you want to build documentation:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'xpert'**

Make sure you have correctly installed X-Pert and the Python environment is correct.

**ModuleNotFoundError: No module named 'scanpy'**

X-Pert depends on scanpy, make sure it's installed:

.. code-block:: bash

   pip install scanpy

**Out of memory errors**

For large datasets, it's recommended to increase available memory or use data chunking.

**CUDA-related errors**

If using GPU, make sure the CUDA version is compatible with PyTorch/TensorFlow versions.

Getting Help
------------

If you encounter installation issues, please:

1. Check the :doc:`troubleshooting` page
2. Submit an issue on GitHub: https://github.com/yourusername/X-Pert/issues
3. Send email to: your.email@example.com