"""
Assembles all cell_*.py files into a valid Jupyter Notebook (.ipynb).
Run this script ONCE:
    python assemble_notebook.py
"""
import json, os, glob

CELL_DIR    = 'notebook_cells'
OUTPUT_FILE = 'i221931-NLP-Assignment2.ipynb'

cell_files = sorted(glob.glob(os.path.join(CELL_DIR, 'cell_*.py')))

cells = []
for fpath in cell_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        src = f.read()
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    })

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook assembled: {OUTPUT_FILE}  ({len(cells)} cells)")
