#!/usr/bin/env python3
"""Setup script for X-Pert package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="x-pert",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive framework for single cell perturbation analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/X-Pert",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/X-Pert/issues",
        "Source": "https://github.com/yourusername/X-Pert",
        "Documentation": "https://x-pert.readthedocs.io",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "xpert=xpert.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
