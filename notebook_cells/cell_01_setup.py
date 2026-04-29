# ============================================================
# CS-4063 NLP Assignment 3 — Transformers + RAG
# Student: Muhammad Suleman | Roll: i221931
# Save this file as: i221931-NLP-Assignment3.ipynb
#
# EXECUTION INSTRUCTIONS:
#   1. pip install torch matplotlib scikit-learn tqdm ipywidgets
#   2. Place all 5 .json files in the same directory as this notebook
#   3. Run cells top-to-bottom; GPU recommended but CPU works
# ============================================================

import os, json, re, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Directories
os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print("Directories 'results/' and 'models/' ready.")
