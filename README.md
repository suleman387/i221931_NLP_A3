# NLP Assignment 3: Transformers + RAG on Amazon Reviews

**Student:** Muhammad Suleman  
**Roll No.:** i221931  
**Course:** CS-4063 Natural Language Processing  
**Institution:** FAST-NUCES, Islamabad  

## Project Overview
This repository contains a complete, from-scratch implementation of a three-stage Natural Language Processing pipeline:
1. **Part A (Encoder):** A multi-task, encoder-only Transformer that classifies sentiment (Positive/Neutral/Negative) and Product Category (Beauty/Cellphones/Electronics).
2. **Part B (Retrieval):** A vector search module using Cosine Similarity to find the top-k most relevant reviews from the training set based on encoder embeddings.
3. **Part C (Decoder/RAG):** A causal, decoder-only Transformer that generates natural language explanations (1-2 sentences) justifying the sentiment using the retrieved context (Retrieval-Augmented Generation).

**Strict Constraints Followed:** No pretrained models (BERT, GPT) were used. The `nn.Transformer`, `nn.MultiheadAttention`, and `nn.TransformerEncoder` abstractions were strictly avoided. All attention mechanisms and Transformer blocks were built manually using PyTorch primitives.

## Repository Structure
- `i221931_NLP_Assignment2.ipynb`: The main Jupyter Notebook containing the full implementation, training loops, evaluation, and an interactive UI.
- `Technical_Report.pdf`: A comprehensive 5-page report detailing architectural justifications, hyperparameter tuning logs, RAG ablation studies, and qualitative analysis.
- `notebook_cells/`: The modular Python scripts used during the development phase to construct the final notebook.
- `models/`: Directory where the trained `encoder.pt` and `decoder.pt` weights are automatically saved during execution.
- `results/`: Directory where loss curves, retrieval distribution plots, and embeddings are saved.

## How to Run
1. **Dependencies:** Ensure you have PyTorch, matplotlib, scikit-learn, and ipywidgets installed (`pip install torch matplotlib scikit-learn ipywidgets`).
2. **Dataset Setup:** 
   - Download the Amazon Product Reviews (5-core) dataset.
   - Place `Beauty_5.json`, `Cell_Phones_and_Accessories_5.json`, and `Electronics_5.json` in the same directory as the notebook (or update the paths in the Google Colab configuration block at the top of the notebook).
3. **Execution:** Open `i221931_NLP_Assignment2.ipynb` in Jupyter Notebook, VS Code, or Google Colab, and run all cells sequentially. The interactive RAG UI will appear at the bottom of the notebook.

## Evaluation Highlights
- **Encoder Sentiment Accuracy:** ~80%
- **Encoder Category Accuracy:** ~85%
- **RAG Ablation:** Generating explanations *with* retrieved context yielded significant reductions in perplexity compared to the non-retrieval baseline, confirming the successful integration of the RAG mechanism.
