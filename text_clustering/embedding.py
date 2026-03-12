"""
embedding.py — Document embedding generation using sentence-transformers.

Provides a thin wrapper around sentence-transformers so the rest of the
pipeline does not depend on the embedding library directly.

Functions
---------
compute_embeddings(texts, model_name=None, batch_size=64)
    Return a numpy array of shape (n_texts, dim) containing the embeddings.

load_embedding_model(model_name=None)
    Return a pre-loaded SentenceTransformer instance (useful when the caller
    needs to generate embeddings in multiple steps).

Usage
-----
    from text_clustering.embedding import compute_embeddings

    embeddings = compute_embeddings(["Hello world", "Another text"])
    # embeddings.shape == (2, 384) for all-MiniLM-L6-v2
"""

from __future__ import annotations

import logging

import numpy as np

from text_clustering.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def load_embedding_model(model_name: str | None = None):
    """Load and return a SentenceTransformer model.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to the value of the
        ``EMBEDDING_MODEL`` environment variable (see config.py).

    Returns
    -------
    SentenceTransformer
        The loaded model, ready for ``model.encode()``.
    """
    from sentence_transformers import SentenceTransformer

    name = model_name or EMBEDDING_MODEL
    logger.info("Loading embedding model: %s", name)
    model = SentenceTransformer(name)
    logger.info("Embedding dimension: %d", model.get_sentence_embedding_dimension())
    return model


def compute_embeddings(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute embeddings for a list of texts.

    Parameters
    ----------
    texts : list[str]
        The raw text strings to embed.
    model_name : str, optional
        HuggingFace model identifier.  Defaults to ``EMBEDDING_MODEL``.
    batch_size : int
        Encoding batch size passed to sentence-transformers.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(texts), embedding_dim)``.
    """
    model = load_embedding_model(model_name)
    logger.info("Computing embeddings for %d texts (batch_size=%d)…", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    logger.info("Embeddings computed — shape: %s", embeddings.shape)
    return embeddings
