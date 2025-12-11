# Basic RAG for Movie Recommendation and Information Retrieval 

This repository is based on my learning with Information Retrieval (IR), embeddings, chunking strategies, and a RAG-based movie recommending agent. It is organized around Jupyter notebooks that demonstrate practical workflows for Retrieval-Augmented Generation (RAG)-style pipelines and classical IR tasks.

## Repository Structure

- `Information_Retrieval/`
  - `chunking.ipynb`: Experiments with document/text chunking strategies (fixed-size, token-aware, overlap) discussing advantages and disadvantages of each strategy.
  - `bert_embedding.ipynb`: Generates BERT-based vector embeddings for text and explores similarity search (cosine, dot product), dimensionality, and performance considerations.
  - `a_movie_reccomender.ipynb`: A RAG-based movie recommending agent that uses embeddings, vector search, and an LLM to recommend movies based on natural-language scenarios.
- `data/`
  - `movies_metadata.csv`: Public movie metadata used by the recommender notebook (e.g., title, overview, genres, etc.).
- `.gitignore`: Standard ignore rules for Python and notebooks.
- `pyvenv.cfg`: Python virtual environment configuration.

## Getting Started

### Prerequisites
- Python 3.12+ used and recommended
- pip and virtualenv (or conda)
- JupyterLab or Jupyter Notebook

### Setup (using venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install common dependencies
pip install jupyter numpy pandas scikit-learn matplotlib seaborn
# Embeddings/IR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers faiss-cpu
```

If you have a CUDA-capable GPU and want acceleration, install the appropriate PyTorch wheel for your CUDA version. The notebooks can also run CPU-only with smaller SentenceTransformer models.

### Launch Notebooks

```bash
# In the virtual environment
jupyter lab
# or
jupyter notebook
```
Open the notebooks under `Information_Retrieval/` and run the cells top-to-bottom.

## Notebooks Overview

1. Chunking (`chunking.ipynb`)
   - Implements multiple chunkers: fixed-length, overlap, token-aware (BPE/word-level approximation)
   - Compares chunk size and overlap effects on recall/precision proxies for retrieval
   - Produces visualizations and basic metrics to guide RAG context window design

2. BERT Embedding (`bert_embedding.ipynb`)
   - Loads a pre-trained BERT/SentenceTransformer model
   - Encodes text into vector embeddings and normalizes vectors
   - Demonstrates similarity search (cosine) with FAISS
   - Benchmarks latency vs. batch size, embedding dimension, and model choice

3. RAG Movie Recommender (`a_movie_reccomender.ipynb`)
   - Loads `data/movies_metadata.csv`
   - Indexes movies with sentence embeddings in a vector store (FAISS)
   - Retrieves relevant movie contexts for a user’s natural-language query
   - Uses an LLM to generate recommendations grounded in the retrieved context

## Data

- `data/movies_metadata.csv` should be present by default. If missing, supply a CSV with at least the following columns: `title`, `overview`, `genres` (string or list-like). Update the notebook path if you use a different file/location.
