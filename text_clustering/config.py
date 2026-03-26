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

# ── Responses API (gpt-5.x / o-series) ───────────────────────────────
# Auto-detect: if the model starts with "gpt-5" or "o1" or "o3" or "o4-mini",
# use the Responses API instead of Chat Completions.  Override explicitly
# with USE_RESPONSES_API=true/false.
_auto_responses = MODEL.startswith(("gpt-5", "o1", "o3", "o4"))
USE_RESPONSES_API: bool = (
    os.getenv("USE_RESPONSES_API", "").lower() in ("true", "1", "yes")
    if os.getenv("USE_RESPONSES_API")
    else _auto_responses
)
REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "medium")  # low | medium | high

# ── K-Medoids pre-clustering settings ─────────────────────────────────────
KMEDOIDS_ENABLED: bool = os.getenv("KMEDOIDS_ENABLED", "false").lower() == "true"
KMEDOIDS_K: int = int(os.getenv("KMEDOIDS_K", "100"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── GMM pre-clustering settings ───────────────────────────────────────────
GMM_K: int = int(os.getenv("GMM_K", "100"))             # 0 = auto-select via BIC
GMM_COVARIANCE_TYPE: str = os.getenv("GMM_COVARIANCE_TYPE", "tied")  # full|tied|diag|spherical

# ── SEAL-Clust settings ──────────────────────────────────────────────────
# Dimensionality reduction
SEALCLUST_REDUCTION: str = os.getenv("SEALCLUST_REDUCTION", "pca")  # pca | tsne
SEALCLUST_PCA_DIMS: int = int(os.getenv("SEALCLUST_PCA_DIMS", "50"))

# t-SNE (legacy / optional)
TSNE_N_COMPONENTS: int = int(os.getenv("TSNE_N_COMPONENTS", "2"))
TSNE_PERPLEXITY: float = float(os.getenv("TSNE_PERPLEXITY", "30"))
TSNE_N_ITER: int = int(os.getenv("TSNE_N_ITER", "1000"))
TSNE_METRIC: str = os.getenv("TSNE_METRIC", "cosine")

# Overclustering (K₀)
SEALCLUST_K0: int = int(os.getenv("SEALCLUST_K0", "300"))  # initial overcluster count

# BIC-based optimal K* estimation
SEALCLUST_BIC_K_MIN: int = int(os.getenv("SEALCLUST_BIC_K_MIN", "5"))
SEALCLUST_BIC_K_MAX: int = int(os.getenv("SEALCLUST_BIC_K_MAX", "50"))
SEALCLUST_K_METHOD: str = os.getenv(
    "SEALCLUST_K_METHOD", "silhouette",
)  # silhouette | bic | ensemble

# Legacy Elbow settings (kept for backward compatibility)
SEALCLUST_K: int = int(os.getenv("SEALCLUST_K", "0"))   # 0 = auto (BIC), >0 = manual K*
SEALCLUST_ELBOW_K_MIN: int = int(os.getenv("SEALCLUST_ELBOW_K_MIN", "5"))
SEALCLUST_ELBOW_K_MAX: int = int(os.getenv("SEALCLUST_ELBOW_K_MAX", "200"))
SEALCLUST_ELBOW_STEP: int = int(os.getenv("SEALCLUST_ELBOW_STEP", "5"))

# Label generation chunk size for representative batches
SEALCLUST_LABEL_CHUNK_SIZE: int = int(os.getenv("SEALCLUST_LABEL_CHUNK_SIZE", "30"))

# ── SEAL-Clust v3 settings ───────────────────────────────────────────
SEALCLUST_V3_CLUSTER_METHOD: str = os.getenv(
    "SEALCLUST_V3_CLUSTER_METHOD", "kmedoids",
)  # kmedoids | gmm | kmeans
SEALCLUST_V3_CLASSIFY_BATCH: int = int(os.getenv("SEALCLUST_V3_CLASSIFY_BATCH", "20"))

# ── Hybrid pipeline settings ─────────────────────────────────────────
HYBRID_LLM_BATCH_SIZE: int = int(os.getenv("HYBRID_LLM_BATCH_SIZE", "30"))
HYBRID_P: float = float(os.getenv("HYBRID_P", "0.1"))  # overclustering proportion
HYBRID_K_MIN: int = int(os.getenv("HYBRID_K_MIN", "2"))
HYBRID_K_MAX: int = int(os.getenv("HYBRID_K_MAX", "50"))

# ── Graph Community Clustering settings ──────────────────────────────
GRAPHCLUST_KNN: int = int(os.getenv("GRAPHCLUST_KNN", "15"))
GRAPHCLUST_MIN_SIMILARITY: float = float(os.getenv("GRAPHCLUST_MIN_SIMILARITY", "0.3"))
GRAPHCLUST_RESOLUTION: float = float(os.getenv("GRAPHCLUST_RESOLUTION", "1.0"))
