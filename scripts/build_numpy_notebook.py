#!/usr/bin/env python3
"""Builds and executes the NumPy Playground notebook from clean Python source."""
import os
import sys

# Ensure scripts dir is on sys.path
sys.path.insert(0, os.path.dirname(__file__))
from build_notebooks import script_to_notebook

if __name__ == "__main__":
    script_to_notebook(
        "01_numpy_playground/numpy_playground.py",
        "01_numpy_playground/numpy_playground.ipynb"
    )
