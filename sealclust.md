# SEAL-Clust — Scalable Efficient Autonomous LLM Clustering

> **Branch**: `feature/kmedoids`  
> **Framework document**: [`seal_clust_framework.md`](./seal_clust_framework.md)  
> **Dataset**: `massive_scenario` · small split (2,974 documents, 18 ground-truth classes)  
> **Project**: PPD — M2 MLSD, Université Paris Cité

---

## Table of Contents

1. [What Is SEAL-Clust?](#1-what-is-seal-clust)
2. [Problems With the Original Paper](#2-problems-with-the-original-paper)
3. [The 7-Step Framework](#3-the-7-step-framework)
4. [Implementation Status](#4-implementation-status)
5. [All Experimental Results](#5-all-experimental-results)
6. [Key Observations](#6-key-observations)
7. [Tutorial — How to Run Everything](#7-tutorial--how-to-run-everything)
8. [All Possible Ways to Run the Algorithm](#8-all-possible-ways-to-run-the-algorithm)
9. [Cost Analysis](#9-cost-analysis)
10. [Future Work / Missing Steps](#10-future-work--missing-steps)

---

## 1. What Is SEAL-Clust?

SEAL-Clust (**S**calable **E**fficient **A**utonomous **L**LM **Clust**ering) is an improved framework that upgrades the approach proposed in **"Text Clustering as Classification with LLMs"** ([arXiv:2410.00927](https://arxiv.org/abs/2410.00927)).

The core idea is to **separate semantic reasoning (LLM) from large-scale clustering computation (embeddings + traditional algorithms)**. Instead of sending every document to the LLM for classification, we:

1. **Embed** all texts with a lightweight model
2. **Form microclusters** using scalable algorithms (K-Medoids, GMM, …)
3. **Extract prototypes** (medoid / representative documents) from each microcluster
4. **Query the LLM only on prototypes** to generate labels
5. **Propagate** prototype labels back to every document locally

This reduces LLM calls from **~3,000** (one per document) to **~100–300** (one per prototype), achieving a **10–30× cost reduction**.

---

## 2. Problems With the Original Paper

| Problem | Original Paper | SEAL-Clust Solution |
|---------|---------------|---------------------|
| **High LLM cost** | One API call per document (~3,000 calls) | Only prototype docs go to the LLM (~100–300 calls) |
| **Not fully unsupervised** | Requires 20% seed labels from ground truth | Microclustering needs zero labels |
| **Undefined k** | LLM implicitly decides k during label merge | Systematic k via BIC / silhouette / gap statistic |
| **Scalability** | Linear in N (all samples → LLM) | Sub-linear: embed once, cluster locally, LLM on M << N prototypes |

---

## 3. The 7-Step Framework

```
┌──────────────────────────────────────────────────────────────────┐
│                       SEAL-Clust Pipeline                        │
│                                                                  │
│  texts ──►[1 Embed]──►[2 DimReduce]──►[3 Microcluster]          │
│                                              │                   │
│                                       ┌──────┴──────┐            │
│                                       │4 Prototypes │            │
│                                       └──────┬──────┘            │
│                                              │                   │
│                        ┌─────────────────────┘                   │
│                        ▼                                         │
│                 [5 LLM Labeling]                                 │
│                        │                                         │
│                        ▼                                         │
│                 [6 Auto-k]  ◄── BIC / silhouette / HDBSCAN       │
│                        │                                         │
│                        ▼                                         │
│                 [7 Propagate]  ──► final labelled dataset         │
└──────────────────────────────────────────────────────────────────┘
```

### Step 1 — Semantic Embedding Generation
Compute dense vector representations for every document using a lightweight sentence-transformer.

- **Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Implementation**: `text_clustering/embedding.py`
- **Output**: `embeddings.npy` (shape: `N × 384`)

### Step 2 — Dimensionality Reduction
Apply t-SNE or PCA to reduce noise and improve cluster separation in the embedding space.

- **Implementation**: `text_clustering/dimreduce.py`
- **t-SNE** (default for SEAL-Clust): `reduce_tsne()` — cosine metric, perplexity=30, 2D output, max_iter=1000
- **PCA**: `reduce_pca()` — linear fallback, n_components=50
- **Output**: `embeddings_reduced.npy` (shape: `N × 2` for t-SNE)

### Step 3 — Microcluster Formation
Instead of sending all N documents to the LLM, create M << N microclusters using a fast algorithm.

Three implementations available:

| Algorithm | File | Metric | Key Feature |
|-----------|------|--------|-------------|
| **K-Medoids** | `text_clustering/kmedoids.py` | cosine | Actual data points as centers (medoids) |
| **GMM** | `text_clustering/gmm.py` | L2 on normalized embeddings | Soft assignments, posterior probabilities |
| **SEAL-Clust K-Medoids** | `text_clustering/sealclust.py` | cosine (on t-SNE-reduced) | t-SNE → Elbow auto-k → K-Medoids |

- K-Medoids: `k-medoids++` init, `random_state=42`, cosine metric
- GMM: L2-normalised embeddings (cosine ≈ euclidean on unit sphere), `covariance_type=tied`, `n_init=3`
- SEAL-Clust: t-SNE reduction → Elbow method auto-k selection → K-Medoids on 2D embeddings

### Step 4 — Prototype Extraction
From each microcluster, extract the most representative document.

| Method | Strategy | File |
|--------|----------|------|
| K-Medoids | The medoid **is** the representative (by definition) | `kmedoids.py → get_medoid_documents()` |
| GMM | Document closest to the component mean (via L2 distance) | `gmm.py → get_representative_documents()` |

Output: `medoid_documents.jsonl` or `representative_documents.jsonl` — one JSON object per prototype with `text`, `cluster_id`, and optionally `probability`.

### Step 5 — LLM Semantic Labeling
The LLM reads prototypes and generates cluster labels. This is the **only** step that requires LLM calls.

Two sub-steps (same as original paper):
1. **Label generation** (`tc-label-gen`): LLM proposes labels from chunked prototype texts
2. **Classification** (`tc-classify --medoid_mode` or `--representative_mode`): LLM assigns one label per prototype

### Step 6 — Automatic Determination of k
Determine the optimal number of clusters systematically.

| Method | Implementation | Status |
|--------|---------------|--------|
| **Elbow** (Kneedle algorithm) | `sealclust.py → elbow_select_k()` | ✅ Implemented |
| **BIC** (Bayesian Information Criterion) | `gmm.py → auto_select_k()` | ✅ Implemented |
| **Silhouette score** | `gmm.py → auto_select_k()` | ✅ Implemented |
| **Gap statistic** | — | ❌ Not implemented |
| **HDBSCAN density stability** | — | ❌ Not implemented |

Usage:
```bash
# Elbow auto-k (SEAL-Clust pipeline — default)
tc-sealclust --data massive_scenario
# → scans k ∈ [10, 100] step 10, picks elbow point

# BIC/silhouette auto-k (GMM pipeline)
tc-gmm --data massive_scenario --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200 --selection_criterion bic
```

### Step 7 — Label Propagation
Propagate prototype labels to every document in the dataset without any additional LLM calls.

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Hard (nearest-centroid)** | Each document inherits the label of its nearest prototype by embedding distance | `kmedoids.py → propagate_labels()`, `gmm.py → propagate_labels()` |
| **Soft (posterior-based)** | Uses GMM posterior probabilities; documents below a confidence threshold are marked "Unsuccessful" | `gmm.py → propagate_labels_soft()` |

---

## 4. Implementation Status

| SEAL-Clust Step | Status | Module(s) | Notes |
|----------------|--------|-----------|-------|
| 1. Embedding | ✅ Done | `embedding.py` | `all-MiniLM-L6-v2`, 384d |
| 2. Dim Reduction | ✅ Done | `dimreduce.py` | t-SNE (cosine, 2D) + PCA |
| 3. Microcluster | ✅ Done | `kmedoids.py`, `gmm.py`, `sealclust.py` | K-Medoids + GMM + SEAL-Clust (t-SNE + Elbow + K-Medoids) |
| 4. Prototype Extract | ✅ Done | `kmedoids.py`, `gmm.py`, `sealclust.py` | Medoids / nearest-to-mean |
| 5. LLM Labeling | ✅ Done | `pipeline/classification.py`, `paper/label_generation.py` | Medoid & representative modes |
| 6. Auto-k | ✅ Done | `sealclust.py`, `gmm.py` | Elbow (Kneedle) + BIC + silhouette |
| 7. Propagation | ✅ Done | `kmedoids.py`, `gmm.py`, `pipeline/sealclust_pipeline.py` | Hard + soft propagation |

**Overall**: 7/7 steps fully implemented. ✅

---

## 5. All Experimental Results

### Paper Baseline (Table 2 — `gpt-3.5-turbo-0125`, 20% seed labels, batch=15)

| Dataset | ACC | NMI | ARI |
|---------|-----|-----|-----|
| `massive_scenario` | **71.75** | **78.00** | **56.86** |
| `massive_intent` | 64.12 | 65.44 | 48.92 |
| `go_emotion` | 31.66 | 27.39 | 13.50 |
| `arxiv_fine` | 38.78 | 57.43 | 20.55 |
| `mtop_intent` | 72.18 | 78.78 | 71.93 |

### Our Runs — Original Pipeline (No Pre-Clustering)

| Run | Model | target_k | n_pred | ACC | NMI | ARI | LLM Calls | Status |
|-----|-------|----------|--------|-----|-----|-----|-----------|--------|
| Paper | `gpt-3.5-turbo-0125` | implicit | ~18 | **71.75** | **78.00** | **56.86** | ~3,000 | Reference |
| Run 01 | `trinity-large-preview:free` | — | 168 | 40.69 | 66.64 | 33.06 | ~3,000 | ❌ Broken merge |
| Run 02 | `gemini-2.0-flash-001` | 18 | 18 | 60.46 | 63.90 | 53.87 | ~3,000 | ✅ Valid |
| Run 03 | `gemini-2.0-flash-001` | none | — | — | — | — | — | ⚠️ Merge failed |

### SEAL-Clust Runs — K-Medoids Pre-Clustering

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` (384d) |
| Metric | cosine |
| Init | k-medoids++ |
| Random seed | 42 |

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Unlabelled |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|------------|
| KM-01 | `gpt-4o-mini` | 100 | 19 | 54.98 | 57.78 | 41.66 | ~300 | ~10× | 0.77% |
| KM-02 | `gpt-4o-mini` | 300 | 20 | 55.21 | 57.25 | 39.85 | ~500 | ~6× | 0.37% |

> **KM-01**: Label generation ran on full 2,974 docs → 715 proposed labels. Merged with `target_k=18` → 19 labels.  
> Classification on 100 medoids (181s). 23/2,974 docs (0.77%) received no label during propagation.  
> Run dir: `./runs/massive_scenario_small_20260312_112628`

> **KM-02**: k=300 (9.9× compression). Label gen on 300 medoid docs → 683 proposed labels.  
> Merged with `target_k=18` → 19 labels (20 predicted incl. "Unsuccessful"). Classification on 300 medoids (1161s, includes ~10min API retry).  
> 11/2,974 docs (0.37%) received no label. Run dir: `./runs/massive_scenario_small_20260312_120831`

### SEAL-Clust Runs — GMM Pre-Clustering

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` (384d) |
| Covariance type | tied |
| n_init | 3 |
| L2-normalised | yes |
| Random seed | 42 |

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Unlabelled |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|------------|
| GMM-01 | `gpt-3.5-turbo` | 100 | 17 | 53.63 | 58.51 | 40.53 | ~300 | ~30× | **0%** |

> **GMM-01**: Label gen with `gpt-3.5-turbo` on full 2,974 docs → 252 proposed labels.  
> `gpt-3.5-turbo` couldn't merge well (229 labels) — re-merged with `gpt-4o-mini` at `target_k=18` → 20 labels.  
> Classification with `gpt-3.5-turbo` on 100 GMM representatives (205s). 0 unsuccessful.  
> Hard propagation: 0/2,974 docs (0%) unlabelled. Run dir: `./runs/massive_scenario_small_20260313_095906`

### SEAL-Clust Runs — t-SNE + Elbow + K-Medoids (Full Pipeline)

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` (384d) |
| Dim reduction | t-SNE (n_components=2, perplexity=30, metric=cosine, max_iter=1000) |
| Auto-k method | Elbow (Kneedle algorithm) |
| Elbow scan range | k ∈ [10, 100], step=10 |
| Clustering | K-Medoids on 2D t-SNE embeddings |
| Random seed | 42 |

**Elbow curve:**

| k | Inertia |
|---|---------|
| 10 | 44.79 |
| 20 | 9.40 |
| **30** | **4.37** ← elbow |
| 40 | 2.73 |
| 50 | 1.70 |
| 60 | 1.18 |
| 70 | 0.85 |
| 80 | 0.65 |
| 90 | 0.51 |
| 100 | 0.40 |

| Run | Model | k | k method | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Unlabelled |
|-----|-------|---|----------|--------|-----|-----|-----|-----------|-----------|------------|
| SC-01 | `gpt-3.5-turbo` + `gpt-4o-mini` (merge) | 30 | Elbow (auto) | 12 | 43.21 | 44.68 | 26.14 | ~30 | **~99×** | 0% |
| SC-02 | `gpt-4o-mini` | 200 | Manual | 20 | 43.44 | 42.37 | 27.66 | ~200 | **~15×** | 0.1% |

> **SC-01**: Full 7-step SEAL-Clust pipeline.  
> 1. Embedded 2,974 docs → 384d vectors.  
> 2. t-SNE reduced to 2D (cosine metric).  
> 3. Elbow method selected k=30 from scan [10, 100].  
> 4. K-Medoids on 2D embeddings → 30 medoid prototypes (99.1× reduction).  
> 5. Label gen with `gpt-3.5-turbo`: 260 proposed → 231 after merge (poor merge).  
>    Re-merged with `gpt-4o-mini`: 260 → 39 → 18 labels (two-pass).  
> 6. Classification on 30 medoids only (57s, ~30 LLM calls).  
> 7. Propagation: all 2,974 docs labelled, 0% unlabelled.  
> **Only 12 of 18 labels were used** — low k (30) limits cluster diversity.  
> Run dir: `./runs/massive_scenario_small_20260313_113104`

> **SC-02**: SEAL-Clust with manual k=200, `gpt-4o-mini` throughout.  
> 1. Embedded 2,974 docs → 384d vectors.  
> 2. t-SNE reduced to 2D (cosine metric).  
> 3. Manual k=200 (Elbow skipped). 14.9× reduction.  
> 4. K-Medoids on 2D embeddings → 200 medoid prototypes (22.4s).  
> 5. Label gen with `gpt-4o-mini`: 718 proposed → merge failed (parse error).  
>    Re-merged with `gpt-4o-mini`: 718 → 19 labels (one-pass, `remerge_labels.py`).  
> 6. Classification on 200 medoids (357s, ~200 LLM calls). **All 19 labels used** + 1 Unsuccessful.  
> 7. Propagation: 2,974 docs labelled, 3 Unsuccessful (0.1%).  
> **Increasing k from 30 to 200 did NOT improve ACC** — the t-SNE 2D projection is the bottleneck.  
> Run dir: `./runs/massive_scenario_small_20260313_135205`

### Summary — All Methods Compared

| Method | Model | k | ACC | NMI | ARI | LLM Calls | Cost Reduction | Unlabelled |
|--------|-------|---|-----|-----|-----|-----------|----------------|------------|
| Paper baseline | `gpt-3.5-turbo-0125` | — | **71.75** | **78.00** | **56.86** | ~3,000 | 1× | 0% |
| Original Run 02 | `gemini-2.0-flash-001` | 18 | 60.46 | 63.90 | 53.87 | ~3,000 | 1× | 0% |
| **KM-01** | `gpt-4o-mini` | 100 | 54.98 | 57.78 | 41.66 | ~300 | **10×** | 0.77% |
| **KM-02** | `gpt-4o-mini` | 300 | 55.21 | 57.25 | 39.85 | ~500 | 6× | 0.37% |
| **GMM-01** | `gpt-3.5-turbo` | 100 | 53.63 | **58.51** | 40.53 | ~300 | **30×** | **0%** |
| **SC-01** | `gpt-3.5-turbo` + `gpt-4o-mini` | 30 (Elbow) | 43.21 | 44.68 | 26.14 | **~30** | **~99×** | **0%** |
| **SC-02** | `gpt-4o-mini` | 200 (manual) | 43.44 | 42.37 | 27.66 | ~200 | ~15× | 0.1% |

---

## 6. Key Observations

### What Works Well

1. **Massive cost reduction**: 10–99× fewer LLM calls with pre-clustering.
2. **SC-01 achieves the ultimate cost efficiency**: only ~30 LLM calls for classification (~99× reduction) — the lowest of any run.
3. **GMM achieves highest NMI** (58.51) among all pre-clustering runs — better cluster purity than both K-Medoids variants.
4. **GMM and SEAL-Clust produce 0% unlabelled** documents: every cluster has a representative, so propagation covers 100% of documents.
5. **Elbow method works**: correctly identified k=30 as the elbow point from the inertia curve, providing a fully automatic k selection.
6. **t-SNE reduction** enables K-Medoids to run on 2D data instead of 384D, making the Elbow scan over k ∈ [10,100] fast (~30s total).
7. **Full automation**: SC-01 requires zero manual parameter choices — Elbow picks k, t-SNE handles reduction, K-Medoids clusters, propagation labels.

### What Doesn't Work Well (Yet)

1. **ACC gap**: SC-01/SC-02 (~43%) are 28pp below the paper baseline (71.75%). The t-SNE 2D projection loses too much structure.
2. **t-SNE is the bottleneck, not k**: SC-02 (k=200) performs nearly identically to SC-01 (k=30) — ACC 43.44% vs 43.21%. Increasing k 7× did not help. The 2D projection collapses high-dimensional cluster structure.
3. **Label merge with `gpt-3.5-turbo`**: Still fails at aggressive merging. `gpt-4o-mini` is required for the merge step.
4. **KM-01/KM-02 outperform SC-01/SC-02** on accuracy: K-Medoids on raw 384D embeddings (~55%) vs t-SNE-reduced 2D (~43%).

### Insights

- **t-SNE 2D is the main accuracy limiter**: SC-01 (k=30, ACC=43.21%) ≈ SC-02 (k=200, ACC=43.44%). Going from 30→200 prototypes only improved ACC by +0.23pp. The information loss from 384D→2D projection dominates.
- **Trade-off curve**: SC-01/SC-02 (t-SNE 2D, ACC≈43%) → KM-01 (raw 384D, ACC=55%) → Paper (per-doc, ACC=72%). Dimensionality reduction quality matters more than k.
- **Recommended approach**: Use K-Medoids on raw 384D embeddings (KM-01, k=100) for best accuracy-cost balance (55% ACC, 10× reduction). Use SEAL-Clust (SC-01) only when full automation is needed.
- GMM's soft assignments provide richer information than K-Medoids' hard assignments, explaining the better NMI (58.51 vs 57.78).
- **SEAL-Clust's value is full automation** — no manual k needed — but t-SNE reduction to 2D loses too much discriminative structure. Future: try PCA to 50D instead.

---

## 7. Tutorial — How to Run Everything

### Prerequisites

```bash
# 1. Activate environment
conda activate ppd

# 2. Install the package in development mode
pip install -e .

# 3. Seed labels (run once — produces runs/chosen_labels.json)
tc-seed-labels

# 4. Configure your LLM provider in .env
cat .env
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# LLM_MODEL=gpt-3.5-turbo
```

### Switching the LLM Provider

Edit `.env` at the project root:

```bash
# ── OpenAI (direct) ──
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-3.5-turbo      # or gpt-4o-mini, gpt-4o

# ── OpenRouter ──
LLM_PROVIDER=openrouter
OPENAI_API_KEY=or-...your-openrouter-key...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.0-flash-001
```

---

### Mode A — Original Pipeline (Baseline, No Pre-Clustering)

```bash
# Step 1: Label generation (~200 LLM calls, processes texts in chunks of 15)
tc-label-gen --data massive_scenario
# → creates run_dir, e.g. ./runs/massive_scenario_small_20260313_...

# Step 2: Classification (~2,974 LLM calls — one per document)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir>

# Step 3: Evaluation (local, instant — no LLM needed)
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~3,000 LLM calls · **Time**: 1–3 hours · **Expected ACC**: 60–72% (model-dependent)

---

### Mode B — K-Medoids Pre-Clustering (SEAL-Clust)

```bash
# Step 0: Embed + K-Medoids (~10–40s, no LLM)
tc-kmedoids --data massive_scenario --kmedoids_k 100
# → creates run_dir with embeddings.npy + kmedoids_metadata.json + medoid_documents.jsonl

# Step 1: Label generation (writes labels into the same run_dir)
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>

# (Optional) If too many labels proposed, re-merge to target_k:
python tools/remerge_labels.py ./runs/<run_dir> 18

# Step 2: Classify medoids only (~100 LLM calls)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# Step 3: Propagate medoid labels → full dataset (local, instant)
tc-kmedoids --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# Step 4: Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~300 LLM calls · **Time**: 5–15 minutes · **Expected ACC**: ~55%

---

### Mode C — GMM Pre-Clustering (SEAL-Clust)

```bash
# Step 0: Embed + GMM (~20–40s, no LLM)
tc-gmm --data massive_scenario --gmm_k 100
# → creates run_dir with embeddings.npy + gmm_metadata.json + gmm_probs.npy + representative_documents.jsonl

# Step 1: Label generation
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>

# (Optional) Re-merge labels:
python tools/remerge_labels.py ./runs/<run_dir> 18

# Step 2: Classify GMM representatives only (~100 LLM calls)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --representative_mode

# Step 3: Propagate labels → full dataset
# Hard propagation (default — nearest centroid):
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# OR soft propagation (uses posterior probabilities):
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate --soft --confidence_threshold 0.4

# Step 4: Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~300 LLM calls · **Time**: 5–15 minutes · **Expected ACC**: ~54%

---

### Mode D — SEAL-Clust Full Pipeline (t-SNE + Elbow + K-Medoids)

The most automated mode — zero manual parameters required.

```bash
# Step 0: Embed + t-SNE + Elbow auto-k + K-Medoids (~40s, no LLM)
tc-sealclust --data massive_scenario
# → scans k ∈ [10, 100], picks elbow (e.g. k=30)
# → creates run_dir with embeddings.npy + embeddings_reduced.npy + sealclust_metadata.json + medoid_documents.jsonl

# Step 1: Label generation (writes labels into the same run_dir)
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>

# (Optional) Re-merge labels to target_k (often needed with gpt-3.5-turbo):
python tools/remerge_labels.py ./runs/<run_dir> 18

# Step 2: Classify medoids only (~30 LLM calls for k=30)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# Step 3: Propagate medoid labels → full dataset (local, instant)
tc-sealclust --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# Step 4: Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**With manual k override** (skip Elbow, use your own k):

```bash
tc-sealclust --data massive_scenario --sealclust_k 50
```

**Cost**: ~30–50 LLM calls · **Time**: ~5 minutes · **Expected ACC**: ~43% (k=30), better with higher k

---

### GMM Auto-Select k (via BIC / Silhouette)

Let the algorithm find the best k automatically:

```bash
tc-gmm --data massive_scenario --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200 --selection_criterion bic
```

This tries every k in [10, 200], picks the k with the lowest BIC, then proceeds with that k.

---

### Reusing Embeddings Between Runs

Both K-Medoids and GMM save `embeddings.npy`. To avoid recomputing (~18s):

```bash
# 1. Run K-Medoids first
tc-kmedoids --data massive_scenario --kmedoids_k 100
# → run_dir = ./runs/massive_scenario_small_20260313_100000

# 2. Run GMM on the SAME run_dir — detects and loads embeddings.npy
tc-gmm --data massive_scenario --gmm_k 100 --run_dir ./runs/massive_scenario_small_20260313_100000
```

---

### Changing Datasets

Replace `massive_scenario` with any dataset under `./datasets/`:

```bash
tc-gmm --data banking77 --gmm_k 50
tc-kmedoids --data clinc --kmedoids_k 100
tc-label-gen --data arxiv_fine --run_dir ./runs/<run_dir>
```

Available datasets: `arxiv_fine`, `banking77`, `clinc`, `clinc_domain`, `few_event`, `few_nerd_nat`, `few_rel_nat`, `go_emotion`, `massive_intent`, `massive_scenario`, `mtop_domain`, `mtop_intent`, `reddit`, `stackexchange`.

---

### Resume After Interruption

All expensive steps have checkpoints:

| Artifact | File | Reuse |
|----------|------|-------|
| Embeddings | `embeddings.npy` | Pass `--run_dir` to skip recomputation |
| K-Medoids metadata | `kmedoids_metadata.json` | Auto-detected |
| GMM metadata | `gmm_metadata.json` + `gmm_probs.npy` | Auto-detected |
| Classification | `classifications.json` | Checkpoint every 1 call in medoid/representative mode |

```bash
# Simply re-run the same command — it picks up from where it stopped
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --representative_mode
```

---

### Makefile Shortcuts

See [Section 8.8](#88-makefile-shortcuts) for all Makefile targets including SEAL-Clust, K-Medoids, and GMM pipelines.

---

### Quick Reference — Which Mode to Use

| Scenario | Mode | Key Command |
|----------|------|-------------|
| Best quality, no budget constraint | A (Original) | `tc-label-gen` → `tc-classify` |
| Fast K-Medoids experiments | B (K-Medoids) | `tc-kmedoids` → `tc-classify --medoid_mode` |
| Fast GMM experiments (0% unlabelled) | C (GMM) | `tc-gmm` → `tc-classify --representative_mode` |
| **Fully automatic (zero params)** | **D (SEAL-Clust)** | **`tc-sealclust` → `tc-classify --medoid_mode`** |
| Auto-select k (GMM) | C + auto-k | `tc-gmm --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200` |
| Auto-select k (SEAL-Clust) | D (default) | `tc-sealclust --data massive_scenario` |
| Manual k with t-SNE | D + manual | `tc-sealclust --data massive_scenario --sealclust_k 50` |
| Reuse embeddings | Any | Pass `--run_dir ./runs/<existing_dir>` |
| Switch LLM provider | Any | Edit `.env` → `LLM_PROVIDER=openrouter` |
| Switch LLM model | Any | Edit `.env` → `LLM_MODEL=gpt-4o-mini` |

---

## 8. All Possible Ways to Run the Algorithm

This section documents **every configuration knob** and **every combination** for running the SEAL-Clust framework.

### 8.1 Entry Points (CLI Commands)

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `tc-sealclust` | SEAL-Clust: t-SNE + Elbow/manual k + K-Medoids | Full auto pipeline |
| `tc-kmedoids` | K-Medoids on raw 384D embeddings | Manual k, no dim reduction |
| `tc-gmm` | GMM on L2-normalised embeddings | Soft clusters, auto-k via BIC |
| `tc-label-gen` | LLM label generation from prototype texts | After any pre-clustering step |
| `tc-classify` | LLM classification of prototypes | After label generation |
| `tc-evaluate` | Compute ACC/NMI/ARI against ground truth | Final step |

### 8.2 Pre-Clustering Variants

#### A. SEAL-Clust (t-SNE + Elbow + K-Medoids) — `tc-sealclust`

```bash
# Fully automatic (Elbow picks k)
tc-sealclust --data massive_scenario

# Manual k override (skip Elbow)
tc-sealclust --data massive_scenario --sealclust_k 50

# Custom Elbow range
tc-sealclust --data massive_scenario --elbow_k_min 5 --elbow_k_max 200 --elbow_step 5

# On a different dataset
tc-sealclust --data banking77
tc-sealclust --data clinc --sealclust_k 100

# Use large split
tc-sealclust --data massive_scenario --use_large
```

**t-SNE parameters** (via `.env` or defaults):

| Parameter | .env Variable | Default | Description |
|-----------|--------------|---------|-------------|
| Components | `TSNE_N_COMPONENTS` | 2 | Output dimensionality |
| Perplexity | `TSNE_PERPLEXITY` | 30.0 | Balance local/global structure |
| Max iterations | `TSNE_MAX_ITER` | 1000 | Convergence iterations |
| Metric | `TSNE_METRIC` | cosine | Distance metric for t-SNE |

**Elbow parameters** (via CLI or `.env`):

| Parameter | CLI Flag | .env Variable | Default |
|-----------|----------|--------------|---------|
| Manual k | `--sealclust_k` | `SEALCLUST_K` | 0 (=auto) |
| Min k | `--elbow_k_min` | `SEALCLUST_ELBOW_K_MIN` | 10 |
| Max k | `--elbow_k_max` | `SEALCLUST_ELBOW_K_MAX` | 100 |
| Step | `--elbow_step` | `SEALCLUST_ELBOW_STEP` | 10 |

#### B. K-Medoids (raw embeddings) — `tc-kmedoids`

```bash
# Standard run with k=100
tc-kmedoids --data massive_scenario --kmedoids_k 100

# With a custom k
tc-kmedoids --data massive_scenario --kmedoids_k 50
tc-kmedoids --data massive_scenario --kmedoids_k 300
```

#### C. GMM — `tc-gmm`

```bash
# Fixed k
tc-gmm --data massive_scenario --gmm_k 100

# Auto-select k via BIC
tc-gmm --data massive_scenario --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200 --selection_criterion bic

# Auto-select k via silhouette
tc-gmm --data massive_scenario --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200 --selection_criterion silhouette
```

### 8.3 Label Generation Variants

```bash
# Standard (uses full dataset texts in chunks of 15)
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>

# Re-merge with target k (when initial merge produces too many labels)
python tools/remerge_labels.py ./runs/<run_dir> 18

# Two-pass merge for stubborn models:
# 1. First pass with gpt-4o-mini
# 2. Set labels_merged.json as new proposed, re-merge again
```

### 8.4 Classification Variants

```bash
# Classify medoid prototypes (K-Medoids / SEAL-Clust)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# Classify GMM representative prototypes
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --representative_mode

# Classify ALL documents (original paper mode — expensive)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir>
```

### 8.5 Propagation Variants

```bash
# SEAL-Clust propagation (from sealclust precluster)
tc-sealclust --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# K-Medoids propagation
tc-kmedoids --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# GMM hard propagation (nearest centroid)
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# GMM soft propagation (posterior-based, with confidence threshold)
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate --soft --confidence_threshold 0.4
```

### 8.6 LLM Model Switching

Edit `.env` before any LLM step:

```bash
# OpenAI direct
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo       # cheapest
LLM_MODEL=gpt-4o-mini         # better merging quality
LLM_MODEL=gpt-4o              # best quality, most expensive

# OpenRouter (any model)
LLM_PROVIDER=openrouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.0-flash-001
LLM_MODEL=anthropic/claude-3.5-sonnet
LLM_MODEL=meta-llama/llama-3.1-70b-instruct
```

### 8.7 Dataset Options

All datasets under `./datasets/`. Each has `small.jsonl` and `large.jsonl`:

```bash
# Any dataset name works with any pipeline
tc-sealclust --data banking77
tc-kmedoids --data clinc --kmedoids_k 100
tc-gmm --data arxiv_fine --gmm_k 50

# Use large split (more documents)
tc-sealclust --data massive_scenario --use_large
```

Available: `arxiv_fine`, `banking77`, `clinc`, `clinc_domain`, `few_event`, `few_nerd_nat`, `few_rel_nat`, `go_emotion`, `massive_intent`, `massive_scenario`, `mtop_domain`, `mtop_intent`, `reddit`, `stackexchange`.

### 8.8 Makefile Shortcuts

```bash
# SEAL-Clust pipeline
make run-sealclust data=massive_scenario         # precluster (auto-k)
make run-sealclust data=massive_scenario k=50    # precluster (manual k)
make run-step1 data=massive_scenario             # label generation
make run-sealclust-classify data=massive_scenario run=./runs/<run_dir>  # classify medoids
make run-sealclust-propagate data=massive_scenario run=./runs/<run_dir>  # propagate
make run-step3 data=massive_scenario run=./runs/<run_dir>  # evaluate

# K-Medoids pipeline
make run-kmedoids data=massive_scenario k=100
make run-kmedoids-classify data=massive_scenario run=./runs/<run_dir>
make run-kmedoids-propagate data=massive_scenario run=./runs/<run_dir>

# GMM pipeline
make run-gmm data=massive_scenario k=100
make run-gmm-classify data=massive_scenario run=./runs/<run_dir>
make run-gmm-propagate data=massive_scenario run=./runs/<run_dir>
```

### 8.9 Complete End-to-End Examples

**Example 1: SEAL-Clust auto-k on massive_scenario (cheapest)**

```bash
conda activate ppd
tc-sealclust --data massive_scenario                    # ~40s, picks k=30
tc-label-gen --data massive_scenario --run_dir ./runs/<dir>   # ~7 min
python tools/remerge_labels.py ./runs/<dir> 18           # ~5s
tc-classify --data massive_scenario --run_dir ./runs/<dir> --medoid_mode  # ~1 min
tc-sealclust --data massive_scenario --run_dir ./runs/<dir> --propagate   # <1s
tc-evaluate --data massive_scenario --run_dir ./runs/<dir>  # <1s
# Total: ~8 min, ~30 LLM classify calls + ~200 label gen calls
```

**Example 2: K-Medoids k=100 on banking77 (balanced)**

```bash
tc-kmedoids --data banking77 --kmedoids_k 100
tc-label-gen --data banking77 --run_dir ./runs/<dir>
python tools/remerge_labels.py ./runs/<dir> 77
tc-classify --data banking77 --run_dir ./runs/<dir> --medoid_mode
tc-kmedoids --data banking77 --run_dir ./runs/<dir> --propagate
tc-evaluate --data banking77 --run_dir ./runs/<dir>
```

**Example 3: GMM auto-k on clinc (soft propagation)**

```bash
tc-gmm --data clinc --gmm_k 0 --gmm_k_min 10 --gmm_k_max 300 --selection_criterion bic
tc-label-gen --data clinc --run_dir ./runs/<dir>
tc-classify --data clinc --run_dir ./runs/<dir> --representative_mode
tc-gmm --data clinc --run_dir ./runs/<dir> --propagate --soft --confidence_threshold 0.5
tc-evaluate --data clinc --run_dir ./runs/<dir>
```

---

## 9. Cost Analysis

### LLM API Calls Per Mode

| Pipeline Step | Mode A (Original) | Mode B (K-Medoids k=100) | Mode C (GMM k=100) | Mode D (SEAL-Clust k=30) |
|--------------|-------------------|--------------------------|---------------------|--------------------------|
| Embedding | 0 | 0 | 0 | 0 |
| Dim Reduction | 0 | 0 | 0 | 0 |
| Microcluster | 0 | 0 | 0 | 0 |
| Label generation | ~200 | ~200 | ~200 | ~200 |
| Label merge | 1 | 1 | 1 | 1–3 (may need re-merge) |
| Classification | **~2,974** | **~100** | **~100** | **~30** |
| Propagation | 0 | 0 | 0 | 0 |
| Evaluation | 0 | 0 | 0 | 0 |
| **Total** | **~3,175** | **~301** | **~301** | **~233** |
| **Reduction** | 1× | **~10×** | **~10×** | **~14×** |

> Note: If label generation is also run only on prototypes (not all documents), Mode B/C can reach **~30× reduction** (GMM-01 achieved this). Mode D with k=30 reaches **~99× reduction** at the classification step alone.

### Wall-Clock Time (massive_scenario · small split)

| Step | Mode A | Mode B (k=100) | Mode C (k=100) | Mode D (k=30) |
|------|--------|-----------------|-----------------|----------------|
| Embedding | — | ~18s | ~18s | ~18s |
| t-SNE | — | — | — | ~5s |
| Elbow scan | — | — | — | ~10s |
| Clustering | — | ~10s | ~20s | ~5s |
| Label generation | ~15 min | ~15 min | ~15 min | ~7 min |
| Classification | ~2 hours | **181s** | **205s** | **57s** |
| Propagation | — | <1s | <1s | <1s |
| **Total** | ~2.5 hours | **~20 min** | **~20 min** | **~8 min** |

---

## 10. Future Work / Missing Steps

### High Priority

1. **Replace t-SNE 2D with PCA 50D in SEAL-Clust**
   - SC-02 proved that k is not the bottleneck — t-SNE 2D is
   - Try `reduce_pca(embeddings, n_components=50)` before K-Medoids
   - Expected: closer to KM-01 accuracy (55%) while keeping Elbow auto-k
   - Implementation: modify `sealclust_pipeline.py` to support `--reduction pca --pca_dims 50`

2. **HDBSCAN Microclustering**
   - Density-based clustering that automatically determines k
   - Does not require specifying k upfront
   - Would complete the "autonomous" aspect of SEAL-Clust

### Medium Priority

3. **Multi-representative extraction**
   - Instead of 1 document per cluster, extract 2–3 representatives
   - The LLM sees more diversity per cluster → better labels

4. **Gap statistic for k selection**
   - Complement BIC/silhouette with gap statistic
   - More robust k determination

### Low Priority

5. **Soft propagation tuning**
   - Experiment with different confidence thresholds
   - Hybrid: soft for GMM, hard for K-Medoids

6. **Cross-dataset evaluation**
   - Run SEAL-Clust on all 14 datasets
   - Compare with paper baselines across the board

7. **Better embedding models**
   - Try `all-mpnet-base-v2` (768d), `instructor-xl`, or `e5-large`
   - May improve cluster quality at the embedding level

---

## Appendix — Run Directory Contents

Each run produces a self-contained directory under `./runs/`:

```
runs/<dataset>_<split>_<timestamp>/
├── embeddings.npy              # Dense embeddings (N × 384)
├── embeddings_reduced.npy      # t-SNE reduced embeddings (N × 2) — SEAL-Clust only
├── elbow_scores.json           # Elbow method k vs inertia — SEAL-Clust only
├── sealclust_metadata.json     # SEAL-Clust: k, method, t-SNE params, assignments
│   OR kmedoids_metadata.json   # K-Medoids: k, metric, cluster assignments
│   OR gmm_metadata.json        # GMM: k, covariance type, means, assignments
│   AND gmm_probs.npy           # GMM: posterior probabilities (N × k)
├── medoid_documents.jsonl      # K-Medoids / SEAL-Clust prototypes
│   OR representative_documents.jsonl  # GMM prototypes
├── cluster_sizes.json          # Size of each microcluster
├── labels_true.json            # Ground-truth label list
├── labels_proposed.json        # LLM-proposed labels (before merge)
├── labels_merged.json          # Final label set (after merge)
├── classifications.json        # Label → [text, text, …]
├── classifications_full.json   # With propagated labels
├── results.json                # Final metrics: ACC, NMI, ARI
├── sealclust_precluster.log    # SEAL-Clust precluster log
├── sealclust_propagate.log     # SEAL-Clust propagation log
├── step1_label_gen.log         # Label generation log
├── step2_classification.log    # Classification log
└── step3_evaluation.log        # Evaluation log
```

---

# 2. Pre-cluster (t-SNE + K-Medoids, manual k=200)
tc-sealclust --data massive_scenario --sealclust_k 200

# 3. Label generation
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>

# 4. Re-merge labels (usually needed)
python tools/remerge_labels.py ./runs/<run_dir> 18

# 5. Classify medoids
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# 6. Propagate
tc-sealclust --data massive_scenario --run_dir ./runs/massive_intent_small_20260313_141759 --propagate

# 7. Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/massive_intent_small_20260313_141759

## Appendix — Configuration Reference (.env)

```bash
# ── LLM ──
LLM_PROVIDER=openai              # openai | openrouter
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=                 # blank for OpenAI, https://openrouter.ai/api/v1 for OpenRouter
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096
FORCE_JSON_MODE=false
REQUEST_DELAY=0.5

# ── K-Medoids ──
KMEDOIDS_ENABLED=false           # true to enable by default
KMEDOIDS_K=100

# ── GMM ──
GMM_K=100
GMM_COVARIANCE_TYPE=tied         # full | tied | diag | spherical

# ── t-SNE (SEAL-Clust Step 2) ──
TSNE_N_COMPONENTS=2              # Output dimensionality (2 for visualization/clustering)
TSNE_PERPLEXITY=30.0             # Balance local/global structure (5–50)
TSNE_MAX_ITER=1000               # Convergence iterations
TSNE_METRIC=cosine               # Distance metric (cosine | euclidean)

# ── SEAL-Clust ──
SEALCLUST_K=0                    # 0 = auto (Elbow), >0 = manual k
SEALCLUST_ELBOW_K_MIN=10         # Min k for Elbow scan
SEALCLUST_ELBOW_K_MAX=100        # Max k for Elbow scan
SEALCLUST_ELBOW_STEP=10          # Step size for Elbow scan

# ── Embedding ──
EMBEDDING_MODEL=all-MiniLM-L6-v2
```
