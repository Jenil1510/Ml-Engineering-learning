#!/usr/bin/env python3
"""
Simple, elegant builder script that converts clean percent-format Python scripts
into executed Jupyter Notebooks (.ipynb) with all figures and outputs rendered.
"""

import os
import sys

# Auto-switch to virtual environment if run with system python
try:
    import nbformat as nbf
    from nbclient import NotebookClient
except ModuleNotFoundError:
    venv_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "python3"))
    if os.path.exists(venv_py) and sys.executable != venv_py:
        print(f"[*] Activating virtual environment: {venv_py}")
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        print("Error: Missing packages. Run: pip install -r requirements.txt")
        sys.exit(1)


def script_to_notebook(py_filepath: str, ipynb_filepath: str):
    """Parses a percent-script (.py) into an executed notebook (.ipynb)."""
    print(f"\nBuilding notebook: {ipynb_filepath}")
    with open(py_filepath, "r", encoding="utf-8") as f:
        content = f.read()

    cells = []
    # Split cells by the standard '# %%' delimiter
    for chunk in content.split("# %%"):
        chunk = chunk.strip()
        if not chunk:
            continue

        if chunk.startswith("[markdown]"):
            md = chunk[len("[markdown]"):].strip()
            # Strip triple quotes if present
            if md.startswith('"""') and md.endswith('"""'):
                md = md[3:-3].strip()
            elif md.startswith("'''") and md.endswith("'''"):
                md = md[3:-3].strip()
            cells.append(nbf.v4.new_markdown_cell(md))
        else:
            cells.append(nbf.v4.new_code_cell(chunk))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    }

    # Execute cells to capture all outputs and plots
    print(f"  Executing {len(cells)} cells...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    os.makedirs(os.path.dirname(ipynb_filepath), exist_ok=True)
    with open(ipynb_filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  Successfully saved: {ipynb_filepath}")


if __name__ == "__main__":
    targets = [
        ("01_numpy_playground/numpy_playground.py", "01_numpy_playground/numpy_playground.ipynb"),
        ("02_exploratory_data_analysis/eda_bank_marketing.py", "02_exploratory_data_analysis/eda_bank_marketing.ipynb"),
    ]

    for py_path, ipynb_path in targets:
        script_to_notebook(py_path, ipynb_path)

    print("\nAll notebooks built and validated successfully!")
