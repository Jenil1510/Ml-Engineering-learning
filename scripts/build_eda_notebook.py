#!/usr/bin/env python3
"""Builds and executes the EDA notebook from clean Python source."""
import os
import sys

# Ensure scripts dir is on sys.path
sys.path.insert(0, os.path.dirname(__file__))
from build_notebooks import script_to_notebook

if __name__ == "__main__":
    script_to_notebook(
        "02_exploratory_data_analysis/eda_bank_marketing.py",
        "02_exploratory_data_analysis/eda_bank_marketing.ipynb"
    )
