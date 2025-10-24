# X-Pert: Single Cell Perturbation Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/yourusername/X-Pert/workflows/Tests/badge.svg)](https://github.com/yourusername/X-Pert/actions)

X-Pert is a Python package specifically designed for single-cell perturbation analysis, providing powerful tools to analyze the effects of gene perturbations on single-cell transcriptomes.

## Features

- 🧬 **Single-cell Perturbation Modeling**: Support for multiple perturbation types and modeling methods
- 📊 **Data Preprocessing**: Complete single-cell data preprocessing pipeline
- 🔍 **Perturbation Effect Analysis**: Identify and analyze the effects of gene perturbations on cell states
- 📈 **Visualization Tools**: Rich visualization capabilities for result presentation
- ⚡ **High-Performance Computing**: Optimized algorithms for large-scale single-cell data
- 🔧 **Modular Design**: Easy to extend and customize modular architecture

## Installation

### Install with pip

```bash
pip install x-pert
```

### Install from source

```bash
git clone https://github.com/yourusername/X-Pert.git
cd X-Pert
pip install -e .
```

### Install with conda

```bash
conda env create -f environment.yml
conda activate xpert
```

## Quick Start

```python
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
```

## Main Modules

- **`xpert.models`**: Perturbation modeling algorithms
- **`xpert.data`**: Data preprocessing and loading tools
- **`xpert.analysis`**: Perturbation effect analysis
- **`xpert.visualization`**: Visualization tools
- **`xpert.utils`**: Utility functions

## Documentation

Detailed API documentation and tutorials are available at: [https://x-pert.readthedocs.io](https://x-pert.readthedocs.io)

## Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to participate in project development.

## Citation

If you use X-Pert in your research, please cite our paper:

```bibtex
@article{xpert2024,
  title={X-Pert: A Comprehensive Framework for Single Cell Perturbation Analysis},
  author={Your Name and Collaborators},
  journal={Nature Methods},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Project Homepage: https://github.com/yourusername/X-Pert
- Issue Tracker: https://github.com/yourusername/X-Pert/issues
- Email: your.email@example.com

## Acknowledgments

Thanks to all contributors and the open source community for their support!
