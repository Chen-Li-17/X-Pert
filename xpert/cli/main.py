"""
Main CLI entry point for X-Pert.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import scanpy as sc
from anndata import AnnData

from ..analysis import PerturbationAnalyzer
from ..models import LinearPerturbationModel


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="X-Pert: Single Cell Perturbation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xpert analyze data.h5ad --perturbation-genes GENE1,GENE2 --output results.h5ad
  xpert plot data.h5ad --genes GENE1,GENE2 --output plot.png
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze perturbation effects')
    analyze_parser.add_argument('input', help='Input data file (h5ad format)')
    analyze_parser.add_argument('--perturbation-genes', required=True,
                               help='Comma-separated list of perturbation genes')
    analyze_parser.add_argument('--control-genes',
                               help='Comma-separated list of control genes')
    analyze_parser.add_argument('--output', required=True,
                               help='Output file for results')
    analyze_parser.add_argument('--model', default='linear',
                               choices=['linear', 'neural', 'ensemble'],
                               help='Model type to use')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Create visualizations')
    plot_parser.add_argument('input', help='Input data file (h5ad format)')
    plot_parser.add_argument('--genes',
                            help='Comma-separated list of genes to plot')
    plot_parser.add_argument('--output', required=True,
                            help='Output file for plot')
    plot_parser.add_argument('--plot-type', default='heatmap',
                            choices=['heatmap', 'volcano', 'trajectory'],
                            help='Type of plot to create')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'plot':
        run_plotting(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_analysis(args):
    """Run perturbation analysis."""
    # Load data
    print(f"Loading data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    # Parse gene lists
    perturbation_genes = args.perturbation_genes.split(',')
    control_genes = args.control_genes.split(',') if args.control_genes else None
    
    # Create model
    if args.model == 'linear':
        model = LinearPerturbationModel()
    else:
        raise ValueError(f"Model type '{args.model}' not implemented yet")
    
    # Create analyzer
    analyzer = PerturbationAnalyzer(adata, model=model)
    
    # Run analysis
    print("Running perturbation analysis...")
    results = analyzer.analyze_perturbation_effects(
        perturbation_genes=perturbation_genes,
        control_genes=control_genes
    )
    
    # Save results
    print(f"Saving results to {args.output}...")
    analyzer.save_results(args.output)
    
    print("Analysis complete!")


def run_plotting(args):
    """Create visualizations."""
    # Load data
    print(f"Loading data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    # Parse gene list
    genes = args.genes.split(',') if args.genes else None
    
    # Create analyzer (without model for plotting)
    analyzer = PerturbationAnalyzer(adata)
    
    # Create plot
    print(f"Creating {args.plot_type} plot...")
    if args.plot_type == 'heatmap':
        analyzer.plot_perturbation_heatmap(genes=genes)
    else:
        raise ValueError(f"Plot type '{args.plot_type}' not implemented yet")
    
    print(f"Plot saved to {args.output}")


if __name__ == '__main__':
    main()
