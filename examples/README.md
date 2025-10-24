# X-Pert Examples

This directory contains usage examples and tutorials for the X-Pert package.

## Example List

### Basic Examples

- **`basic_usage.py`**: Basic usage example demonstrating X-Pert's core functionality
- **`advanced_analysis.py`**: Advanced analysis example with multiple analysis methods
- **`custom_models.py`**: Custom model example

### Tutorials

- **`tutorial_1_data_preprocessing.ipynb`**: Data preprocessing tutorial
- **`tutorial_2_perturbation_analysis.ipynb`**: Perturbation analysis tutorial
- **`tutorial_3_visualization.ipynb`**: Visualization tutorial

### Application Cases

- **`crispr_screen_analysis.py`**: CRISPR screen data analysis
- **`drug_perturbation_analysis.py`**: Drug perturbation analysis
- **`time_series_analysis.py`**: Time series perturbation analysis

## Running Examples

### Environment Setup

```bash
# Install X-Pert
pip install x-pert

# Or install from source
git clone https://github.com/yourusername/X-Pert.git
cd X-Pert
pip install -e .
```

### Running Python Scripts

```bash
# Run basic example
python examples/basic_usage.py

# Run advanced analysis example
python examples/advanced_analysis.py
```

### Running Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then open the corresponding `.ipynb` files.

## Data Requirements

Most examples use simulated data, but you can also use your own data:

- **Format**: H5AD format (recommended) or CSV format
- **Gene Expression**: Rows are cells, columns are genes
- **Metadata**: Annotation information for cells and genes

## Customizing Examples

You can modify examples according to your needs:

1. Change data paths
2. Adjust parameters
3. Add new analysis steps
4. Modify visualization settings

## Contributing Examples

If you have interesting use cases, please submit a Pull Request!

## Getting Help

If you encounter problems:

1. Check documentation: https://x-pert.readthedocs.io
2. Submit an Issue: https://github.com/yourusername/X-Pert/issues
3. Send email: your.email@example.com
