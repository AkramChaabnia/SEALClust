"""
config.py — Environment-variable configuration shared by all pipeline steps.

All LLM-related settings are read once at import time from the environment
(populated by python-dotenv / .env).  Any pipeline module that needs these
values imports them from here instead of re-reading os.environ itself.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── LLM settings ──────────────────────────────────────────────────────────
MODEL: str = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
FORCE_JSON_MODE: bool = os.getenv("LLM_FORCE_JSON_MODE", "false").lower() == "true"
REQUEST_DELAY: float = float(os.getenv("LLM_REQUEST_DELAY", "2"))

# ── K-Medoids pre-clustering settings ─────────────────────────────────────
KMEDOIDS_ENABLED: bool = os.getenv("KMEDOIDS_ENABLED", "false").lower() == "true"
KMEDOIDS_K: int = int(os.getenv("KMEDOIDS_K", "100"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── GMM pre-clustering settings ───────────────────────────────────────────
GMM_K: int = int(os.getenv("GMM_K", "100"))             # 0 = auto-select via BIC
GMM_COVARIANCE_TYPE: str = os.getenv("GMM_COVARIANCE_TYPE", "tied")  # full|tied|diag|spherical

# ── SEAL-Clust settings ──────────────────────────────────────────────────
# t-SNE dimensionality reduction
TSNE_N_COMPONENTS: int = int(os.getenv("TSNE_N_COMPONENTS", "2"))
TSNE_PERPLEXITY: float = float(os.getenv("TSNE_PERPLEXITY", "30"))
TSNE_N_ITER: int = int(os.getenv("TSNE_N_ITER", "1000"))
TSNE_METRIC: str = os.getenv("TSNE_METRIC", "cosine")

# Clustering & auto-k
SEALCLUST_K: int = int(os.getenv("SEALCLUST_K", "0"))   # 0 = auto (Elbow method)
SEALCLUST_ELBOW_K_MIN: int = int(os.getenv("SEALCLUST_ELBOW_K_MIN", "5"))
SEALCLUST_ELBOW_K_MAX: int = int(os.getenv("SEALCLUST_ELBOW_K_MAX", "200"))
SEALCLUST_ELBOW_STEP: int = int(os.getenv("SEALCLUST_ELBOW_STEP", "5"))
