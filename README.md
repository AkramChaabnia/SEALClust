# Text Clustering as Classification with LLMs

> PPD reproduction — M2 MLSD, Université Paris Cité
> Based on: [Text Clustering as Classification with LLMs](https://arxiv.org/abs/2410.00927) (Chen Huang, Guoxiu He, 2024)
> Original code: [ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)

This repository reproduces the paper's baseline and introduces **SEAL-Clust** (**S**calable **E**fficient **A**utonomous **L**LM **Clust**ering) — a 9-stage pipeline that reduces LLM cost by **10×** while maintaining competitive accuracy through overclustering + representative-based label discovery. It also provides a **Hybrid Pipeline** that combines LLM-based label generation with embedding-based K optimisation and GMM overclustering, **Graph Community Clustering** — a fundamentally different approach that builds a k-NN embedding graph, discovers clusters via Louvain community detection, and uses the LLM only for post-hoc labelling, plus **LLM-free baselines** (KMeans / GMM) for benchmarking.

For the full research log — experimental results, code fixes, model investigation, and key findings — see [FINDINGS.md](./FINDINGS.md).

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Setup](#2-setup)
3. [Architecture](#3-architecture)
4. [Available Datasets](#4-available-datasets)
5. [Usage — All Pipeline Modes](#5-usage--all-pipeline-modes)
6. [CLI Reference](#6-cli-reference)
7. [Resuming an Interrupted Run](#7-resuming-an-interrupted-run)
8. [Token Usage Tracking](#8-token-usage-tracking)
9. [Label Reuse (Caching)](#9-label-reuse-caching)
10. [Run Directory Structure](#10-run-directory-structure)
11. [Evaluation & Metrics](#11-evaluation--metrics)
12. [Configuration Reference](#12-configuration-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Repository Structure](#14-repository-structure)
15. [Tutorial — Running SEAL-Clust v4 Step by Step](#15-tutorial--running-seal-clust-v4-step-by-step)
16. [Development](#16-development)
17. [Citation](#17-citation)

---

## 1. Quick Start

```bash
# ── Option 1: Fully automatic (K* estimated via silhouette-elbow) ──
tc-sealclust --data massive_scenario --k0 300 --full

# ── Option 2: You know the target K (e.g., 18 clusters for massive_scenario) ──
tc-sealclust --data massive_scenario --k0 300 --k_star 18 --full

# ── Option 3: Using Make ──
make run-sealclust-full data=massive_scenario
make run-sealclust-full data=massive_scenario kstar=18
```

One command runs the entire pipeline end-to-end: embedding → clustering → LLM label discovery → classification → propagation → evaluation.

---

## 2. Setup

### Requirements

- Python 3.12+
- [conda](https://docs.conda.io/) (recommended), `venv`, or [uv](https://docs.astral.sh/uv/) (fastest)
- An API key for [OpenAI](https://platform.openai.com/) or [OpenRouter](https://openrouter.ai)

### Installation

```bash
# Clone and setup
git clone https://github.com/AkramChaabnia/SEALClust.git
cd SEALClust

# Option A: Using conda (recommended for data science workflows)
conda create -n env_name python=3.12
conda activate env_name
pip install -e ".[dev]"

# Option B: Using venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Option C: Using uv (fastest, modern Python package manager)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

See [CONTRIBUTING.md](./CONTRIBUTING.md#environment-setup) for detailed setup instructions for each method.

### Dependency Groups

Choose which features you need to minimize installation size:

```bash
# Baseline pipeline only (6 core packages — smallest footprint)
pip install -e .

# Baseline + embeddings (for hybrid/SEAL-Clust pipelines)
pip install -e ".[embeddings]"

# Baseline + visualization (for plotting & analysis)
pip install -e ".[viz]"

# Development tools (linting, testing, type-checking)
pip install -e ".[dev]"

# Everything (all features + dev tools)
pip install -e ".[all]"

# Mix and match as needed
pip install -e ".[embeddings,viz,dev]"
```

| Group | Includes | Use Case |
|-------|----------|----------|
| **(default)** | python-dotenv, tqdm, openai, numpy, scipy, scikit-learn | Baseline: `tc-seed-labels`, `tc-label-gen`, `tc-classify`, `tc-evaluate` |
| **embeddings** | sentence-transformers, umap-learn | Hybrid/SEAL-Clust: `tc-hybrid`, `tc-sealclust`, `tc-sealclust-v3/v4` |
| **viz** | matplotlib | Visualization: `tc-visualize` |
| **http** | httpx | Alternative HTTP client (rarely needed) |
| **dev** | ruff, commitizen, pre-commit, mypy, pytest | Development & contributions |
| **all** | All of the above | Full feature set |

### API Key Configuration

Create a `.env` file in the project root:

```bash
# Required
LLM_PROVIDER=openai            # openai | openrouter
LLM_MODEL=gpt-4o-mini          # or gpt-4o, gpt-3.5-turbo
OPENAI_API_KEY=sk-...          # Your OpenAI API key

# For OpenRouter
LLM_PROVIDER=openrouter
OPENAI_API_KEY=or-...your-openrouter-key...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.0-flash-001
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=4096
LLM_FORCE_JSON_MODE=false
LLM_PROVIDER=openrouter
LLM_REQUEST_DELAY=2
```

### Verify Installation

```bash
tc-sealclust --help
tc-preflight   # checks LLM connectivity
```

---

## 3. Architecture

### The 9-Stage SEAL-Clust v2 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Document Embedding    all-MiniLM-L6-v2 → 384D        │
│  Stage 2: Dimensionality Reduction    PCA 384D → 50D           │
│  Stage 3: Overclustering    K-Medoids K₀=N → N clusters    │
│  Stage 4: Representative Selection    Extract K₀ medoid docs   │
│  Stage 5: Label Discovery    LLM reads reps → candidate labels  │
│  Stage 6: K* Estimation    Silhouette-elbow / manual → K*       │
│  Stage 7: Label Consolidation    LLM merges → exactly K* labels │
│  Stage 8: Classification    LLM classifies K₀ reps → K* labels │
│  Stage 9: Label Propagation    Each doc inherits medoid's label  │
└─────────────────────────────────────────────────────────────────┘
```

| Symbol | Meaning | Typical Value |
|--------|---------|:-------------:|
| **K₀** | Overclustering size (micro-clusters) | 300 |
| **K\*** | Final number of categories | 5–50 (auto) or manual |

K₀ is always much larger than K\*. The pipeline over-clusters for good representative
coverage, then uses the LLM to discover the right number of meaningful categories.

### Architecture — SEAL-Clust v4 (Mode S)

#### What's New in v4

| Feature | v3 | v4 |
|---------|----|----|
| **Label style** | One-word only | 1–3 word descriptive labels |
| **Dataset context** | None | Domain descriptions injected into all prompts |
| **Consolidation** | Simple merge | Structured semantic grouping with preserve rules |
| **Classification bias** | Fixed label order | Label list **shuffled per batch** |
| **Token tracking** | None | Built-in input/output token counters |
| **Prompt design** | Generic | Dataset-aware with scenario/service framing |
| **CLI** | `tc-sealclust-v3` | `tc-sealclust-v4` |

#### Dataset-Aware Prompts

v4 automatically injects **domain context** into every LLM prompt based on the dataset name. This helps the LLM produce correctly-scoped labels without leaking any ground-truth information.

For example, when processing `massive_scenario`, the LLM sees:

> *DATASET CONTEXT: Short commands spoken to a virtual assistant (like Alexa or Siri). Each command asks the assistant to perform an action or retrieve information using one of the assistant's built-in features or connected services... Categories correspond to the assistant SCENARIO — the high-level service or feature the user is interacting with.*

All 14 benchmark datasets have pre-written descriptions in `text_clustering/data.py`. Unknown datasets get a generic fallback.

#### Label-Order Shuffling

In Stage 8 (classification), the label list is **randomly shuffled for each batch** to prevent the LLM from consistently favouring labels that appear early in the list (positional bias).

---

### Stage Cache Table

| Stage | Cache File | If Exists → |
|:-----:|-----------|-------------|
| 1 | `embeddings.npy` | Skip embedding |
| 2 | `embeddings_reduced.npy` | Skip PCA/t-SNE |
| 3 | `sealclust_metadata.json` | Skip overclustering |
| 5 | `labels_proposed.json` | Skip label discovery (has intra-stage checkpoint) |
| 6 | `k_estimation.json` | Skip K\* estimation |
| 7 | `labels_merged.json` | Skip label consolidation |
| 8 | `classifications.json` | Has intra-stage checkpoint (`checkpoint_classify.json`) |
| 9 | `classifications_full.json` | Skip propagation |

---


## 4. Available Datasets

All datasets are under `./datasets/<name>/` with `small.jsonl` and `large.jsonl` splits.

| Dataset | Documents (small) | Ground Truth K | Description |
|---------|:-----------------:|:--------------:|-------------|
| `massive_scenario` | 2,974 | **18** | Virtual assistant scenarios |
| `massive_intent` | 2,974 | **59** | Virtual assistant intents (fine-grained) |
| `clinc_domain` | 4,500 | **10** | CLINC intent domains (broad) |
| `clinc` | 4,500 | **150** | CLINC intents (fine-grained) |
| `mtop_domain` | 4,386 | **11** | MTOP task-oriented domains |
| `mtop_intent` | 4,386 | **102** | MTOP intents (fine-grained) |
| `banking77` | 3,080 | **77** | Banking customer queries |
| `go_emotion` | 5,940 | **27** | Emotion classification |
| `few_event` | 4,742 | **34** | Event type detection |
| `few_nerd_nat` | 3,789 | **58** | Named entity types |
| `few_rel_nat` | 4,480 | **64** | Relation types |
| `arxiv_fine` | 3,674 | **93** | ArXiv paper categories |
| `reddit` | 3,217 | **50** | Reddit post topics |
| `stackexchange` | 4,156 | **121** | StackExchange question topics |

**Download**: [Google Drive (ClusterLLM)](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view) — unzip into `./datasets/`

---

## 5. Usage — All Pipeline Modes

### Mode A — Original Pipeline (No Pre-Clustering)

The paper's method. Most expensive but highest quality with strong models.

```bash
# Step 0: Seed labels (run once)
tc-seed-labels

# Step 1: Label generation (~200 LLM calls)
tc-label-gen --data massive_scenario
# ⚠️ COPY THE PRINTED RUN DIR
# or if you have the selected number k
tc-label-gen --data massive_scenario --target_k 18

# Step 2: Classification (~2,974 LLM calls — one per document)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir>

# Step 2 (optimised): Batched classification (~298 LLM calls with batch_size=10)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --batch_size 10

# Step 3: Evaluation
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~3,000 LLM calls (unbatched) or ~500 (batched, `--batch_size 10`) · **Time**: 1–3h (unbatched) or ~20min (batched)

---

### Mode B — K-Medoids Pre-Clustering

K-Medoids on raw 384D embeddings. Medoids are actual documents from the dataset.

```bash
# Step 1: Embed + K-Medoids (~10–40s, no LLM)
tc-kmedoids --data massive_scenario --kmedoids_k 100
# ⚠️ COPY THE PRINTED RUN DIR

# Step 2: Label generation
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>
# (Optional) Re-merge: tc-remerge-labels ./runs/<run_dir> 18

# Step 3: Classify medoids only (~100 LLM calls)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# Step 4: Propagate → full dataset
tc-kmedoids --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# Step 5: Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~300 LLM calls · **Time**: 5–15 min

---

### Mode C — GMM Pre-Clustering

GMM on L2-normalised 384D embeddings with soft (probabilistic) assignments.

```bash
# Step 1: Embed + GMM (~20–40s, no LLM)
tc-gmm --data massive_scenario --gmm_k 100
# (Alternative) Auto-select k: tc-gmm --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200 --selection_criterion bic
# ⚠️ COPY THE PRINTED RUN DIR

# Step 2: Label generation
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>

# Step 3: Classify representatives (~100 LLM calls)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --representative_mode

# Step 4: Propagate
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate
# OR soft propagation:
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate --soft --confidence_threshold 0.4

# Step 5: Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~300 LLM calls · **Time**: 5–15 min

---
### Mode D — SEAL-Clust v2 Step-by-Step

Run Stages 1–7, inspect intermediate results, then complete the pipeline.

```bash
# Stages 1–7
conda run -n ppd tc-sealclust --data massive_scenario --k0 300 --k_star 18
# ⚠️ COPY THE PRINTED RUN DIR

# Inspect
cat ./runs/<run_dir>/labels_proposed.json | python3 -m json.tool
cat ./runs/<run_dir>/labels_merged.json | python3 -m json.tool

# Stage 8: Classify representatives (~300 LLM calls)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# Stage 9: Propagate labels → full dataset
tc-sealclust --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```
> **💡 Tip**: For best accuracy, set `--k_star` to the ground-truth number of classes.
> All automated K\* methods tend to under-estimate.

#### K\* Estimation Methods

| Method | Flag | Description |
|--------|------|-------------|
| **Silhouette-elbow** | `--k_method silhouette` | Elbow detection on silhouette curve (default) |
| **Calinski-Harabasz** | `--k_method calinski` | Peaks at optimal K via variance ratio |
| **BIC** | `--k_method bic` | GMM + Bayesian Information Criterion |
| **Ensemble** | `--k_method ensemble` | Median of all three (most robust) |

---

### Mode E — SEAL-Clust v2 Full Pipeline

Single `--full` command runs all 9 stages + evaluation.

```bash
# Auto K* (algorithm estimates the number of clusters)
tc-sealclust --data massive_scenario --k0 300 --full

# Manual K* (you know the ground-truth K)
tc-sealclust --data massive_scenario --k0 300 --k_star 18 --full

# Try different K* estimation methods
tc-sealclust --data massive_scenario --k0 300 --k_method ensemble --full
```

**Cost**: ~310 LLM calls · **Time**: ~10 min ·

---

### Mode F — Hybrid Pipeline (LLM + Embedding Optimisation)

Combines LLM semantic label generation with embedding-based K optimisation and GMM overclustering for robust clustering without manual K specification.

**The 8 Steps:**

```
┌────────────────────────────────────────────────────────────────────────┐
│  Step 1: LLM Label Generation    Batch texts → one-word labels (K₀)  │
│  Step 2: Embedding               all-MiniLM-L6-v2 → 384D             │
│  Step 3: LLM Label Reduction     Merge synonymous labels (K₀ → K₁)   │
│  Step 4: K Optimisation          KMeans + silhouette sweep → best K   │
│  Step 5: LLM Label Alignment     Force exactly K labels (K₁ → K)     │
│  Step 6: GMM Overclustering      p×N micro-clusters + medoid extract  │
│  Step 7: LLM Medoid Labelling    Classify each medoid → one of K     │
│  Step 8: Label Propagation       GMM membership → full-dataset labels │
└────────────────────────────────────────────────────────────────────────┘
```

#### Full Pipeline (one command)

```bash
# Auto K (silhouette-optimised)
tc-hybrid --data massive_scenario --full

# Manual target K
tc-hybrid --data massive_scenario --full --target_k 18

# Custom parameters
tc-hybrid --data massive_scenario --full --p 0.15 --k_min 3 --k_max 40
```

#### Step-by-Step

```bash
# Steps 1–5 (label discovery + K optimisation)
tc-hybrid --data massive_scenario
# ⚠️ COPY THE PRINTED RUN DIR

# Inspect intermediate results
cat ./runs/<run_dir>/hybrid_labels_k1.json | python3 -m json.tool   # reduced labels
cat ./runs/<run_dir>/hybrid_k_optimisation.json | python3 -m json.tool  # best K
cat ./runs/<run_dir>/labels_merged.json | python3 -m json.tool       # final labels

# Individual steps
tc-hybrid --data massive_scenario --step 6 --continue_from ./runs/<run_dir>
tc-hybrid --data massive_scenario --step 7 --continue_from ./runs/<run_dir>
tc-hybrid --data massive_scenario --step 8 --continue_from ./runs/<run_dir>
```

#### Cache Files

| Step | Cache File | If Exists → |
|:----:|-----------|-------------|
| 1 | `hybrid_labels_k0.json`, `hybrid_per_doc_labels.json` | Skip LLM label gen |
| 2 | `embeddings.npy` | Skip embedding |
| 3 | `hybrid_labels_k1.json` | Skip label reduction |
| 4 | `hybrid_k_optimisation.json` | Skip K optimisation |
| 5 | `labels_merged.json` | Skip label alignment |
| 6 | `hybrid_gmm_metadata.json` | Skip GMM overclustering |
| 7 | `hybrid_medoid_labels.json` | Skip medoid labelling |
| 8 | `classifications.json`, `classifications_full.json` | Skip propagation |

**Cost**: ~80–200 LLM calls (depending on dataset size) · **Time**: 5–15 min

---

### Mode G — Baselines (No LLM)

Pure embedding-based clustering for benchmarking. No LLM calls required.

#### KMeans Baseline

```bash
# Fixed K
tc-baseline --data massive_scenario --method kmeans --k 18

# Auto-K via silhouette sweep
tc-baseline --data massive_scenario --method kmeans --auto_k --k_min 5 --k_max 30

# With PCA pre-reduction
tc-baseline --data massive_scenario --method kmeans --k 18 --pca_dims 50
```

#### GMM Baseline

```bash
# Fixed K
tc-baseline --data massive_scenario --method gmm --k 18

# Auto-K via BIC minimisation
tc-baseline --data massive_scenario --method gmm --auto_k --k_min 5 --k_max 30

# Custom covariance type
tc-baseline --data massive_scenario --method gmm --k 18 --covariance_type diag
```

**Cost**: 0 LLM calls · **Time**: 30s–2min · **Expected ACC**: 30–45% (depends on dataset)

---

### Mode H — Graph Community Clustering (Louvain + LLM Labelling)

A **fundamentally different approach** to text clustering.  Instead of using the LLM as a classifier (shared by Modes A–F) or geometric partitioning (KMeans/GMM in Mode G), Mode H discovers clusters via **graph community detection** on a k-NN embedding similarity graph.

**Architecture** — 3-Step Pipeline:
1. **Build k-NN Graph** — Each document connects to its k most similar documents (cosine similarity in embedding space, above a minimum threshold)
2. **Community Detection** — Custom Louvain modularity optimisation discovers natural clusters via graph topology (binary search over resolution γ to match target_k)
3. **LLM Post-hoc Labelling** — The LLM names each discovered community from representative samples (~K LLM calls)

**Key differences from all other modes**:
- No labels exist during clustering — they are extracted **post-hoc**
- Clustering emerges from **graph community structure** (modularity Q), not geometric partitioning (centroids/Gaussians)
- Louvain can discover **non-convex, non-spherical** clusters that KMeans/GMM cannot
- The LLM is used **only for naming** communities, not for clustering decisions

```bash
# Full pipeline (all 3 steps + evaluation)
tc-graphclust --data massive_scenario --target_k 18 --full

# Auto-detect K (no target_k — Louvain decides)
tc-graphclust --data massive_scenario --full

# Steps 1–2 only (clustering without labels)
tc-graphclust --data massive_scenario --target_k 18

# Step 3 + evaluation only (requires existing run)
tc-graphclust --data massive_scenario --run_dir ./runs/<dir> --label_only

# Using Make
make run-graphclust-full data=massive_scenario target_k=18
make run-graphclust-full data=massive_scenario knn=20 resolution=1.5
```

**Cost**: ~K LLM calls (post-hoc labelling only, e.g. ~18 for massive_scenario) · **Time**: 2–5min

---
### Mode S — SEAL-Clust v4 ⭐ RECOMMENDED

The latest and most capable pipeline. Dataset-aware prompts, label shuffling, and token tracking.

```bash
# ── Full pipeline (one command) ──
tc-sealclust-v4 --data massive_scenario --k0 300 --full
tc-sealclust-v4 --data massive_scenario --k0 300 --k_star 18 --full

# ── Step-by-step ──
tc-sealclust-v4 --data massive_scenario --k0 300 --k_star 18      # Stages 1–7
tc-sealclust-v4 --data massive_scenario --run_dir ./runs/<dir> --classify   # Stage 8
tc-sealclust-v4 --data massive_scenario --run_dir ./runs/<dir> --propagate  # Stage 9
tc-evaluate --data massive_scenario --run_dir ./runs/<dir>                  # Evaluate

# ── Using Make ──
make run-sealclust-v4-full data=massive_scenario kstar=18
make run-sealclust-v4 data=massive_scenario kstar=18
make run-sealclust-v4-classify data=massive_scenario run=./runs/<dir>
make run-sealclust-v4-propagate data=massive_scenario run=./runs/<dir>
```

**Cost**: ~116 API calls, ~88K tokens · **Time**: ~5 min

---

### Modes Y/Z — SEAL-Clust v3 (Multi-Method + One-Word Labels)

SEAL-Clust v3 is an improved version of the SEAL-Clust framework with:

- **Multiple clustering backends**: K-Medoids (default), GMM, or KMeans
- **No default dimensionality reduction**: Clusters on raw 384D embeddings by default; optionally `--reduction pca` or `--reduction tsne`
- **Label discovery from ALL documents**: Sends all documents (not just representatives) to the LLM in chunks for better label coverage
- **Configurable label source**: `--label_source all` (default, all documents) or `--label_source representatives` (only K₀ rep docs — faster, fewer LLM calls)
- **One-word label constraint**: Labels are single general category words (e.g. "travel", "finance")
- **Iterative chunked consolidation**: Handles 500+ candidate labels reliably
- **Retry-based label discovery**: If fewer labels than K* are discovered, retries with reshuffled chunks
- **Batched representative classification**: Classify multiple docs per LLM call (20× default)
- **Flexible representative selection**: Medoids for K-Medoids; closest-to-centroid for GMM/KMeans

**v3 Architecture (9 Stages):**

```
Stage 1: Embed documents (sentence-transformers)
Stage 2: Dimensionality reduction (optional — none by default)
Stage 3: Overclustering (K₀ clusters via kmedoids/gmm/kmeans)
Stage 4: Select representative per cluster
Stage 5: LLM label discovery (one-word labels, chunked — all docs or reps only)
Stage 6: K* estimation (manual or automatic)
Stage 7: LLM label consolidation (iterative merge → K* labels)
Stage 8: LLM representative classification (batched)
Stage 9: Label propagation (rep → all documents)
```

**Mode Z — Full Pipeline** (all 9 stages + evaluation):

```bash
# Full end-to-end with default K-Medoids
tc-sealclust-v3 --data massive_scenario --k0 300 --full

# With manual K*
tc-sealclust-v3 --data massive_scenario --k0 300 --k_star 18 --full

# Using GMM clustering
tc-sealclust-v3 --data massive_scenario --k0 300 --cluster_method gmm --full

# Using KMeans
tc-sealclust-v3 --data massive_scenario --k0 300 --cluster_method kmeans --full

# With PCA dimensionality reduction
tc-sealclust-v3 --data massive_scenario --k0 300 --reduction pca --full

# Label discovery from representatives only (faster)
tc-sealclust-v3 --data massive_scenario --k0 300 --label_source representatives --full

# Using Make
make run-sealclust-v3-full data=massive_scenario
make run-sealclust-v3-full data=massive_scenario kstar=18 cluster_method=gmm
make run-sealclust-v3-full data=massive_scenario reduction=pca
make run-sealclust-v3-full data=massive_scenario label_source=representatives
```

**Mode Y — Step-by-Step** (run stages separately):

```bash
# Stages 1–7: embed + cluster + discover labels + consolidate
tc-sealclust-v3 --data massive_scenario --k0 300 --k_star 18

# Stage 8: classify representatives
tc-sealclust-v3 --data massive_scenario --run_dir ./runs/<dir> --classify

# Stage 9: propagate labels
tc-sealclust-v3 --data massive_scenario --run_dir ./runs/<dir> --propagate

# Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<dir>

# Using Make
make run-sealclust-v3 data=massive_scenario kstar=18
make run-sealclust-v3-classify data=massive_scenario run=./runs/<dir>
make run-sealclust-v3-propagate data=massive_scenario run=./runs/<dir>
```

**v2 vs v3 Comparison:**

| Feature | v2 (Mode D/E) | v3 (Mode Y/Z) |
|---------|---------------|----------------|
| Clustering | K-Medoids only | K-Medoids / GMM / KMeans |
| Labels | 2–5 word phrases | One word (general) |
| Label source | Representatives only | All docs (default) or representatives (`--label_source`) |
| Consolidation | Single-pass merge | Iterative chunked merge |
| Classification | One doc per LLM call | Batched (20 per call) |
| Stage 8 | Shared `tc-classify` | Built-in batched classification |
| CLI | `tc-sealclust` | `tc-sealclust-v3` |

**Cost**: ~K₀/30 + K₀/20 LLM calls (label discovery + classification) · **Time**: 5–15min

---

### Mode Quick Reference

| Scenario | Mode | Command |
|----------|------|---------|
| **One command, full automation** ⭐ | **S** | `tc-sealclust-v4 --data X --k0 300 --full` |
| **One command, known K\*** ⭐ | **S** | `tc-sealclust-v4 --data X --k0 300 --k_star N --full`|
| **One command, full automation**| E | `tc-sealclust --data X --k0 300 --full` |
| **One command, known K\***| E | `tc-sealclust --data X --k0 300 --k_star N --full` |
| **v3: multi-method + one-word labels** | Z | `tc-sealclust-v3 --data X --k0 300 --full` |
| **v3: GMM clustering backend** | Z | `tc-sealclust-v3 --data X --cluster_method gmm --full` |
| **v3: with PCA reduction** | Z | `tc-sealclust-v3 --data X --k0 300 --reduction pca --full` |
| **Hybrid: LLM + embedding K-opt** | F | `tc-hybrid --data X --full` |
| **Graph community clustering** | H | `tc-graphclust --data X --target_k N --full` |
| **Baseline: no LLM benchmark** | G | `tc-baseline --data X --method kmeans --k N` |
| **Debug / inspect stages** | D | `tc-sealclust` → `tc-classify` → `--propagate` |
| **K-Medoids on raw embeddings** | B | `tc-kmedoids` → `tc-label-gen` → `tc-classify --medoid_mode` |
| **GMM soft clusters** | C | `tc-gmm` → `tc-label-gen` → `tc-classify --representative_mode` |
| **Paper baseline (most expensive)** | A | `tc-seed-labels` → `tc-label-gen` → `tc-classify` → `tc-evaluate` |
| **Paper baseline (batched, 10× faster)** | A | `tc-classify --batch_size 10 --run_dir <dir>` |
| **Reuse cached embeddings** | Any | Pass `--run_dir ./runs/<existing_dir>` |

---

### Makefile Shortcuts

```bash
# ── SEAL-Clust v4 (Mode S — recommended) ──
make run-sealclust-v4-full data=massive_scenario
make run-sealclust-v4-full data=massive_scenario kstar=18
make run-sealclust-v4-full data=massive_scenario cluster_method=gmm k0=200
make run-sealclust-v4 data=massive_scenario kstar=18
make run-sealclust-v4-classify data=massive_scenario run=./runs/<run_dir>
make run-sealclust-v4-propagate data=massive_scenario run=./runs/<run_dir>

# ── SEALClust v2 (Mode E ) ──
make run-sealclust-full data=massive_scenario
make run-sealclust-full data=massive_scenario kstar=18
make run-sealclust-full data=massive_scenario k0=200 kmethod=ensemble

# ── SEALClust Stages 1–7 (Mode D) ──
make run-sealclust data=massive_scenario kstar=18

# ── SEALClust Stage 8 / Stage 9 separately ──
make run-sealclust-classify data=massive_scenario run=./runs/<run_dir>
make run-sealclust-propagate data=massive_scenario run=./runs/<run_dir>

# ── Original pipeline (Mode A) ──
make run-step0
make run-step1 data=massive_scenario
make run-step2 data=massive_scenario run=./runs/<run_dir>
make run-step2 data=massive_scenario run=./runs/<run_dir> classify_batch=10  # 10× faster
make run-step3 data=massive_scenario run=./runs/<run_dir>

# ── K-Medoids (Mode B) ──
make run-kmedoids data=massive_scenario k=100
make run-kmedoids-classify data=massive_scenario run=./runs/<run_dir>
make run-kmedoids-propagate data=massive_scenario run=./runs/<run_dir>

# ── GMM (Mode C) ──
make run-gmm data=massive_scenario k=100
make run-gmm-classify data=massive_scenario run=./runs/<run_dir>
make run-gmm-propagate data=massive_scenario run=./runs/<run_dir>

# ── Hybrid Pipeline (Mode F) ──
make run-hybrid-full data=massive_scenario
make run-hybrid-full data=massive_scenario hybrid_p=0.15 hybrid_k_max=40
make run-hybrid data=massive_scenario step=4     # single step

# ── Baselines (Mode G) ──
make run-baseline-kmeans data=massive_scenario k=18
make run-baseline-gmm data=massive_scenario k=18
make run-baseline-kmeans data=massive_scenario auto_k=1 k_min=5 k_max=30

# ── Graph Community Clustering (Mode H) ──
make run-graphclust-full data=massive_scenario target_k=18
make run-graphclust-full data=massive_scenario knn=20 resolution=1.5
make run-graphclust data=massive_scenario

# ── SEAL-Clust v3 (Mode Z) ──
make run-sealclust-v3-full data=massive_scenario
make run-sealclust-v3-full data=massive_scenario kstar=18 cluster_method=gmm
make run-sealclust-v3 data=massive_scenario kstar=18
make run-sealclust-v3-classify data=massive_scenario run=./runs/<run_dir>
make run-sealclust-v3-propagate data=massive_scenario run=./runs/<run_dir>
```

| Variable | Default | Description |
|----------|---------|-------------|
| `data` | *(required)* | Dataset name |
| `k` | `100` | K-Medoids / GMM / Baseline k |
| `k0` | `300` | SEALClust overclustering K₀ |
| `kstar` | `0` | SEALClust manual K\* (`0` = auto) |
| `kmethod` | `silhouette` | K\* estimation method |
| `run` | *(for separate stages)* | Run directory path |
| `classify_batch` | `1` | Classification batch size (`10` = 10× fewer LLM calls) |
| `hybrid_p` | `0.1` | Hybrid overclustering fraction |
| `hybrid_k_min` | `2` | Hybrid K sweep minimum |
| `hybrid_k_max` | `50` | Hybrid K sweep maximum |
| `hybrid_batch` | `30` | Hybrid LLM batch size |
| `auto_k` | — | Enable auto-K for baselines (`1`) |
| `pca` | — | PCA dims for baselines |
| `graph_knn` | `15` | Graph clustering: k-NN neighbours |
| `min_sim` | `0.3` | Graph clustering: min cosine similarity |
| `resolution` | `1.0` | Graph clustering: Louvain resolution |
| `target_k` | — | Graph clustering / Hybrid: target K |
| `cluster_method` | `kmedoids` | SEAL-Clust v3: clustering backend (`kmedoids` / `gmm` / `kmeans`) |
| `v3_classify_batch` | `20` | SEAL-Clust v3: representatives per classification call |
| `reduction` | *(empty → none)* | SEAL-Clust v3: dim reduction (`none` / `pca` / `tsne`) |
| `label_source` | `all` | SEAL-Clust v3: label discovery source (`all` / `representatives`) |

---

## 6. CLI Reference

### `tc-sealclust` — Main Pipeline Command

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--full` | flag | — | Run all 9 stages + evaluation |
| `--propagate` | flag | — | Run Stage 9 only (requires `--run_dir`) |
| `--k0 N` | int | `300` | Overclustering size K₀ |
| `--k_star N` | int | `0` | Manual K\* (`0` = auto-estimate) |
| `--k_method M` | str | `silhouette` | `silhouette` / `calinski` / `bic` / `ensemble` |
| `--bic_k_min N` | int | `5` | Min K for estimation search |
| `--bic_k_max N` | int | `50` | Max K for estimation search |
| `--reduction M` | str | `pca` | `pca` (recommended) / `tsne` |
| `--pca_dims N` | int | `50` | PCA output dimensions |
| `--run_dir PATH` | str | — | Reuse existing run directory |
| `--use_large` | flag | — | Use `large.jsonl` split |
| `--embedding_model M` | str | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `--label_chunk_size N` | int | `30` | Docs per LLM label-discovery call |
| `--batch_size N` | int | `64` | Embedding batch size |
| `--seed N` | int | `42` | Random seed |

### `tc-sealclust-v3` — SEAL-Clust v3 Pipeline (Modes Y/Z)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--full` | flag | — | Run all 9 stages + evaluation (Mode Z) |
| `--classify` | flag | — | Run Stage 8 only (requires `--run_dir`) |
| `--propagate` | flag | — | Run Stage 9 only (requires `--run_dir`) |
| `--k0 N` | int | `300` | Overclustering size K₀ |
| `--k_star N` | int | `0` | Manual K\* (`0` = auto-estimate) |
| `--k_method M` | str | `silhouette` | `silhouette` / `calinski` / `bic` / `ensemble` |
| `--cluster_method M` | str | `kmedoids` | `kmedoids` / `gmm` / `kmeans` |
| `--classify_batch_size N` | int | `20` | Representatives per LLM classification call |
| `--reduction M` | str | `none` | `none` / `pca` / `tsne` — dimensionality reduction |
| `--pca_dims N` | int | `50` | PCA output dimensions (only when `--reduction pca`) |
| `--label_source M` | str | `all` | `all` = every document, `representatives` = K₀ reps only |
| `--run_dir PATH` | str | — | Existing run directory |
| `--use_large` | flag | — | Use `large.jsonl` split |
| `--embedding_model M` | str | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `--label_chunk_size N` | int | `30` | Docs per LLM label-discovery call |
| `--reuse_labels` | flag | — | Enable label cache reuse |
| `--seed N` | int | `42` | Random seed |

### `tc-sealclust-v4` — SEAL-Clust v4 Pipeline (Mode S) ⭐

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--full` | flag | — | Run all 9 stages + evaluation (Mode S) |
| `--classify` | flag | — | Run Stage 8 only (requires `--run_dir`) |
| `--propagate` | flag | — | Run Stage 9 only (requires `--run_dir`) |
| `--k0 N` | int | `300` | Overclustering size K₀ |
| `--k_star N` | int | `0` | Manual K\* (`0` = auto-estimate) |
| `--k_method M` | str | `silhouette` | `silhouette` / `calinski` / `bic` / `ensemble` |
| `--cluster_method M` | str | `kmedoids` | `kmedoids` / `gmm` / `kmeans` |
| `--reduction M` | str | `none` | `none` / `pca` / `tsne` |
| `--pca_dims N` | int | `50` | PCA output dimensions (only when `--reduction pca`) |
| `--classify_batch_size N` | int | `20` | Representatives per LLM classification call |
| `--label_source M` | str | `all` | `all` = every document, `representatives` = K₀ reps only |
| `--label_chunk_size N` | int | `30` | Docs per LLM label-discovery call |
| `--run_dir PATH` | str | — | Existing run directory |
| `--use_large` | flag | — | Use `large.jsonl` split |
| `--embedding_model M` | str | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `--reuse_labels` | flag | — | Enable label cache reuse |
| `--seed N` | int | `42` | Random seed |

### `tc-kmedoids` — K-Medoids Pre-Clustering (Mode B)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--kmedoids_k N` | int | `100` | Number of clusters |
| `--run_dir PATH` | str | — | Existing run dir (for `--propagate` or reuse embeddings) |
| `--propagate` | flag | — | Propagate medoid labels → full dataset |

### `tc-gmm` — GMM Pre-Clustering (Mode C)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--gmm_k N` | int | `100` | Number of components (`0` = auto-select) |
| `--gmm_k_min N` | int | `10` | Min k for auto-selection |
| `--gmm_k_max N` | int | `200` | Max k for auto-selection |
| `--selection_criterion M` | str | `bic` | `bic` / `silhouette` |
| `--covariance_type M` | str | `tied` | `full` / `tied` / `diag` / `spherical` |
| `--propagate` | flag | — | Propagate labels → full dataset |
| `--soft` | flag | — | Soft (probability-weighted) propagation |
| `--confidence_threshold F` | float | `0.4` | Min posterior for soft propagation |

### `tc-label-gen` — LLM Label Generation (Modes A–D)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `arxiv_fine` | Dataset name |
| `--run_dir PATH` | str | — | Existing run dir (from pre-clustering) |
| `--chunk_size N` | int | `15` | Texts per LLM call |
| `--target_k N` | int | — | Target number of labels for merge |

### `tc-classify` — LLM Classification (All Modes)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `arxiv_fine` | Dataset name |
| `--run_dir PATH` | str | **required** | Run dir with `labels_merged.json` |
| `--batch_size N` | int | `1` | Sentences per LLM call (`10`–`20` recommended for 10× speedup) |
| `--medoid_mode` | flag | — | Classify only medoid docs (Modes B, D, E) |
| `--representative_mode` | flag | — | Classify only GMM representative docs (Mode C) |

### `tc-evaluate` — Evaluation (All Modes)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `arxiv_fine` | Dataset name |
| `--run_dir PATH` | str | **required** | Run dir with `classifications.json` or `classifications_full.json` |

### `tc-hybrid` — Hybrid Pipeline (Mode F)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--full` | flag | — | Run all 8 steps + evaluation |
| `--step N` | int | — | Run only step N (1–8) |
| `--continue_from PATH` | str | — | Resume from existing run dir |
| `--target_k N` | int | — | Override automatic K (skip step 4) |
| `--p F` | float | `0.1` | GMM overclustering fraction |
| `--k_min N` | int | `2` | Min K for silhouette sweep |
| `--k_max N` | int | `50` | Max K for silhouette sweep |
| `--llm_batch_size N` | int | `30` | Documents per LLM label-gen call |
| `--covariance_type M` | str | `full` | GMM covariance: `full`/`tied`/`diag`/`spherical` |
| `--use_large` | flag | — | Use `large.jsonl` split |

### `tc-baseline` — Baselines (Mode G)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--method M` | str | **required** | `kmeans` or `gmm` |
| `--k N` | int | — | Number of clusters (required unless `--auto_k`) |
| `--auto_k` | flag | — | Automatic K selection (silhouette / BIC) |
| `--k_min N` | int | `2` | Min K for auto-selection sweep |
| `--k_max N` | int | `50` | Max K for auto-selection sweep |
| `--pca_dims N` | int | — | Optional PCA pre-reduction |
| `--covariance_type M` | str | `full` | GMM covariance: `full`/`tied`/`diag`/`spherical` |
| `--use_large` | flag | — | Use `large.jsonl` split |

### `tc-graphclust` — Graph Clustering (Mode H)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data NAME` | str | `massive_scenario` | Dataset name |
| `--full` | flag | — | Run all 5 steps + evaluation |
| `--label_only` | flag | — | Run Step 5 + evaluation only (requires `--run_dir`) |
| `--step N` | int | — | Run only step N (1–5) |
| `--n_anchors N` | int | `15` | Number of anchor documents |
| `--anchor_method M` | str | `farthest_point` | `farthest_point` / `random` |
| `--judge_batch_size N` | int | `50` | Documents per LLM pairwise-judge call |
| `--similarity_threshold F` | float | `0.15` | Min embedding cosine sim for LLM judging |
| `--top_k_anchors N` | int | `0` | Only judge each doc against top-K anchors (`0` = all) |
| `--edge_threshold F` | float | `0.3` | Min affinity-vector cosine sim for graph edges |
| `--knn N` | int | `15` | k-nearest neighbours per node |
| `--resolution F` | float | `1.0` | Louvain resolution γ (higher → more clusters) |
| `--target_k N` | int | `0` | Target communities (`0` = auto via resolution) |
| `--samples_per_community N` | int | `8` | Docs sampled per community for labelling |
| `--run_dir PATH` | str | — | Reuse existing run directory |
| `--use_large` | flag | — | Use `large.jsonl` split |

### Other Commands

| Command | Purpose |
|---------|---------|
| `tc-seed-labels` | Generate seed labels for Mode A (run once) |
| `tc-preflight` | Verify LLM connectivity and configuration |
| `tc-probe-models` | Test model compatibility before full runs |
| `tc-remerge-labels` | Re-merge labels to a target K (iterative) |
| `tc-analyze` | Generate dataset statistics JSON reports |
| `tc-visualize` | Generate t-SNE visualisation of clustering results |

---

## 7. Resuming an Interrupted Run

Every pipeline supports **two layers** of fault tolerance:

1. **Stage-level caching** — Each stage writes its output to a file (e.g. `embeddings.npy`, `labels_proposed.json`). If the file already exists when the pipeline is re-run with the same `--run_dir`, the stage is skipped entirely. This is automatic and requires no special flags.

2. **Intra-stage checkpointing** — Long-running LLM loops (label generation, classification, medoid labelling) save progress to a `checkpoint_*.json` file at regular intervals. If interrupted, the same command resumes from the last checkpoint. Checkpoint files are automatically deleted once the stage completes successfully.

### 7.1 Quick Resume (All Pipelines)

Pass the same `--run_dir` to continue from where it stopped:

```bash
# SEAL-Clust: stages 1-7 cached, continues from the first incomplete stage
tc-sealclust --data massive_scenario --k0 300 --k_star 18 \
    --run_dir ./runs/massive_scenario_small_20260314_150000 --full

# Hybrid: same pattern
tc-hybrid --data massive_scenario --target_k 18 \
    --run_dir ./runs/<run_dir> --full

# Graph clustering: same pattern
tc-graphclust --data massive_scenario --target_k 18 \
    --run_dir ./runs/<run_dir> --full

# Original 3-step: re-run Step 2 (it resumes from checkpoint)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir>
```

### 7.2 Checkpoint Files Reference

| Pipeline | Stage | Checkpoint File | What Is Saved | Save Interval |
|----------|-------|-----------------|---------------|:-------------:|
| **Original** (Step 1) | Label Generation | `checkpoint_labelgen.json` | Processed chunk count + all labels discovered so far | Every ~5% of chunks |
| **Original** (Step 2) | Classification | `checkpoint_classify.json` | Processed sample count + classification dict | Every 200 samples (unbatched) or every ~10 batches (batched) |
| **Hybrid** (Step 1) | LLM Label Gen | `checkpoint_hybrid_step1.json` | Processed batch count + per-document labels | Every ~5% of batches |
| **Hybrid** (Step 7) | Medoid Labelling | `checkpoint_hybrid_step7.json` | Processed count + medoid label assignments | Every ~5% of medoids |
| **SEAL-Clust** (Stage 5) | Label Discovery | `checkpoint_sealclust_labels.json` | Processed chunk count + candidate labels | Every ~10% of chunks |
| **Graph Clust** (Step 3) | Community Labelling | `checkpoint_graphclust_step3.json` | Labelled community IDs + names | Every ~10% of communities |

> **Note**: Checkpoint files are ephemeral — they exist only during an active run. Once the stage finishes, the checkpoint is deleted automatically.

### 7.3 How Checkpoints Work

When a pipeline stage is interrupted (Ctrl+C, timeout, crash):

```
runs/massive_scenario_small_20260314_150000/
├── checkpoint_classify.json          ← checkpoint exists → stage was interrupted
├── classifications.json              ← partial results (also written at each checkpoint)
├── labels_merged.json                ← completed by previous stage
└── ...
```

Simply re-run the **exact same command**. The pipeline will:
1. Skip all stages whose output files already exist (stage-level cache)
2. Within the interrupted stage, load the checkpoint and skip already-processed items
3. Continue from where it left off
4. Delete the checkpoint file when the stage finishes

```bash
# Example: classification was interrupted at sample 1500/2974
# Just re-run — it resumes from sample 1500
tc-classify --data massive_scenario --run_dir ./runs/<run_dir>
```

### 7.4 Special Cases

**SEAL-Clust `--full` and Stage 8**: The `--full` flag chains all stages. If Stage 8 (classification) was interrupted, re-running with `--full` will skip Stages 1–7 (cached) and resume Stage 8 from its checkpoint. You can also run Stage 8 separately for finer control:

```bash
tc-classify --data massive_scenario \
    --run_dir ./runs/<run_dir> --medoid_mode
```

**Re-run with different K\***: Delete K\*-dependent files and re-run:

```bash
rm ./runs/<run_dir>/labels_merged.json ./runs/<run_dir>/classifications.json \
   ./runs/<run_dir>/classifications_full.json ./runs/<run_dir>/results.json
tc-sealclust --data massive_scenario --k0 300 --k_star 25 \
    --run_dir ./runs/<run_dir> --full
```

> **💡** Keep `labels_proposed.json` — candidate labels don't depend on K\*.

**Original Step 1 (label generation)**: Uses a fixed random seed (`42`) for shuffling, so checkpoint resume sees the same document order. If you need a different shuffle, delete the checkpoint before re-running.

### 7.5 Limitations

- **Baselines** (`tc-baseline`): KMeans and GMM baselines are fully deterministic and fast (no LLM calls), so checkpointing is not needed. If interrupted, simply re-run — embedding computation is cached.
- **Evaluation** (`tc-evaluate`): Pure computation on existing files, finishes in seconds. No checkpoint needed.
- **Cross-stage dependencies**: Deleting a stage's output and re-running will not automatically invalidate downstream stages. If you re-run Stage 5 (labels), you must also delete Stage 7+ outputs.

---

## 8. Token Usage Tracking

SEAL-Clust v4 automatically tracks all LLM API usage throughout a pipeline run:

- **Input tokens**: total prompt tokens sent to the API
- **Output tokens**: total completion tokens received
- **Total tokens**: combined (input + output)
- **API calls**: number of individual LLM requests

### Where Token Usage Appears

1. **Pipeline logs** — printed at the end of every `--full` run:
   ```
   ─── Token Usage ───
   API calls     : 116
   Input tokens  : 83944
   Output tokens : 4287
   Total tokens  : 88231
   ```

2. **`results.json`** — saved alongside accuracy metrics:
   ```json
   {
     "ACC": 0.5898,
     "NMI": 0.5702,
     "ARI": 0.4417,
     "token_usage": {
       "input_tokens": 83944,
       "output_tokens": 4287,
       "total_tokens": 88231,
       "api_calls": 116
     }
   }
   ```

### Estimating Cost

To estimate the cost of a run on `massive_scenario` , check your LLM provider's pricing:

| Provider / Model | Input (per 1M tokens) | Output (per 1M tokens) | ~Cost per Run |
|------------------|-----------------------|------------------------|:-------------:|
| Gemini 2.0 Flash (OpenRouter) | $0.10 | $0.40 | ~$0.01 |
| GPT-4o-mini | $0.15 | $0.60 | ~$0.02 |
| GPT-4o | $2.50 | $10.00 | ~$0.25 |
| GPT-5.4-mini | $0.75 | $4.50 | ~$0.10 – $0.12 |
| GPT-5.4 | $2.50 | $15.00 | ~$0.40 – $0.50 |

A typical v4 run on `massive_scenario` uses ~84K input + ~4K output tokens ≈ **88K total tokens**.

---

## 9. Label Reuse (Caching)

By default every pipeline run regenerates labels from scratch via LLM calls.
When you are iterating on the *same dataset* with the *same number of clusters*,
this is wasteful.  The **`--reuse_labels`** flag enables a shared label cache
that persists across runs.

### 9.1 How It Works

| Run # | Cache state | Behaviour |
|-------|-------------|-----------|
| 1st   | miss        | Generate labels normally via LLM, then **save** them to `runs/label_cache/` |
| 2nd+  | hit         | **Load** cached labels — skip all LLM label-generation calls |

The cache key is `{dataset}_{split}_k{n_labels}`, e.g. `massive_scenario_small_k18.json`.

### 9.2 Supported Pipelines

| Pipeline | Flag | Stages skipped on cache hit |
|----------|------|-----------------------------|
| Original (`tc-label-gen`) | `--reuse_labels` | Label generation + merge (entire Step 1) |
| SEAL-Clust (`tc-sealclust`) | `--reuse_labels` | Stage 5 (Label Discovery) + Stage 7 (Consolidation) |
| Hybrid (`tc-hybrid`) | `--reuse_labels` | Step 5 (LLM Label Alignment) |

Graph clustering (`tc-graphclust`) generates labels per-community post-hoc and does not use this feature.

### 9.3 CLI Examples

```bash
# Original pipeline — first run (generates + caches)
tc-label-gen --data massive_scenario --reuse_labels

# Original pipeline — second run (loads from cache, 0 LLM calls)
tc-label-gen --data massive_scenario --reuse_labels

# SEAL-Clust — with manual K* (best for cache reuse)
tc-sealclust --data massive_scenario --k_star 18 --reuse_labels --full

# SEAL-Clust — auto K* (cache checked after K* estimation in Stage 6)
tc-sealclust --data massive_scenario --reuse_labels --full

# Hybrid — with explicit target K
tc-hybrid --data massive_scenario --target_k 18 --reuse_labels --full

# Makefile shortcuts — add reuse_labels=1
make run-step1 data=massive_scenario reuse_labels=1
make run-sealclust-full data=massive_scenario kstar=18 reuse_labels=1
make run-hybrid-full data=massive_scenario target_k=18 reuse_labels=1
```

### 9.4 Cache Directory

```
runs/
└── label_cache/
    ├── massive_scenario_small_k18.json    # ["Alarm & Timer", "Audio & Music", ...]
    ├── massive_scenario_small_k19.json
    ├── clinc_small_k150.json
    └── ...
```

Each file is a plain JSON array of label strings.

Use `--label_cache_dir <path>` to override the default location (`runs/label_cache/`).

### 9.5 Tips

- **Specify K explicitly** for deterministic cache hits (`--target_k`, `--k_star`).
  When K is determined automatically (e.g. silhouette search), the cache key depends
  on the estimated K which may vary between runs.
- **Clear the cache** by deleting `runs/label_cache/` or individual files.
- Default behaviour (no `--reuse_labels`) is **unchanged** — labels are always regenerated.

---

## 10. Run Directory Structure

**SEAL-Clust Run**

```
runs/
└── massive_scenario_small_v4_20260325_094026/
    ├── embeddings.npy                  # Stage 1: Raw 384D embeddings (N × 384)
    ├── sealclust_v4_metadata.json      # Stage 3: Cluster assignments, rep indices
    ├── representative_documents.jsonl  # Stage 4: Representative documents
    ├── labels_proposed.json            # Stage 5: ~60 candidate labels
    ├── labels_merged.json              # Stage 7: Final K* label names
    ├── labels_true.json                # Ground-truth label list
    ├── classifications.json            # Stage 8: {label: [sentences...]} for reps
    ├── classifications_full.json       # Stage 9: {label: [sentences...]} for all docs
    ├── results.json                    # Evaluation: ACC, NMI, ARI + token usage
    ├── sealclust_v4_pipeline.log       # Stages 1–7 log
    ├── sealclust_v4_classify.log       # Stage 8 log
    ├── sealclust_v4_propagate.log      # Stage 9 log
    ├── step3_evaluation.log            # Evaluation log
    ├── visualization.log               # Visualisation log
    └── assets/                         # Generated plots
        ├── tsne_predicted.png
        ├── tsne_true.png
        ├── umap_predicted.png
        ├── umap_true.png
        ├── comparison_pca.png
        ├── distribution_predicted.png
        ├── distribution_true.png
        └── histogram_cluster_sizes.png
```

**Graph Clustering (Mode H)** produces a different set of files:

```
runs/
└── massive_scenario_small_graphclust_20260318_150000/
    ├── embeddings.npy                    # 384D embeddings (shared with other modes)
    ├── graphclust_anchors.json           # Step 1: Anchor indices + method
    ├── graphclust_affinity.npy           # Step 2: Binary affinity matrix (N × A)
    ├── graphclust_graph.npz              # Step 3: Sparse edge list (rows, cols, weights)
    ├── graphclust_communities.json       # Step 4: Community assignments + sizes
    ├── graphclust_community_names.json   # Step 5: {community_id: topic_label}
    ├── graphclust_metadata.json          # Pipeline parameters + stats
    ├── labels_true.json                  # Ground-truth label list
    ├── labels_merged.json                # Community labels (for evaluation compat)
    ├── classifications.json              # {label: [sentences...]}
    ├── classifications_full.json         # Same (no propagation needed — all docs assigned)
    ├── results.json                      # ACC, NMI, ARI
    └── graphclust_pipeline.log           # Full pipeline log
```

---

## 11. Evaluation & Metrics

Three standard clustering metrics computed via **Hungarian matching** (optimal 1-to-1 alignment):

| Metric | Range | What It Measures |
|--------|:-----:|-----------------|
| **ACC** | 0–1 | Fraction of correctly assigned documents |
| **NMI** | 0–1 | Mutual information between predicted and true labels (normalized) |
| **ARI** | −1–1 | Adjusted Rand Index (chance-corrected pairwise agreement) |

```bash
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
# → writes results.json
```

> K\* affects all metrics: too low → clusters too broad (ACC drops); too high → too fine (ARI drops). Best results when K\* matches ground truth K.

---

## 12. Configuration Reference

### Environment Variables (`.env`)

```bash
# ── LLM ──
LLM_PROVIDER=openai              # openai | openrouter
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=                 # blank for OpenAI, https://openrouter.ai/api/v1 for OpenRouter
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096
FORCE_JSON_MODE=false
REQUEST_DELAY=0.5

# ── SEAL-Clust (v2/v3/v4) ──
SEALCLUST_K0=300
SEALCLUST_K=0                    # 0 = auto, >0 = manual K*
SEALCLUST_K_METHOD=silhouette
SEALCLUST_REDUCTION=pca          # v2 default (v3/v4 default: none)
SEALCLUST_PCA_DIMS=50
SEALCLUST_BIC_K_MIN=5
SEALCLUST_BIC_K_MAX=50
SEALCLUST_LABEL_CHUNK_SIZE=30
SEALCLUST_V3_CLUSTER_METHOD=kmedoids
SEALCLUST_V3_CLASSIFY_BATCH=20


# ── K-Medoids ──
KMEDOIDS_K=100

# ── GMM ──
GMM_K=100
GMM_COVARIANCE_TYPE=tied

# ── Hybrid Pipeline ──
HYBRID_LLM_BATCH_SIZE=30         # Documents per LLM label-gen call
HYBRID_P=0.1                     # GMM overclustering fraction (p × N)
HYBRID_K_MIN=2                   # Min K for silhouette sweep
HYBRID_K_MAX=50                  # Max K for silhouette sweep

# ── Graph Clustering (LPSGC) ──
GRAPHCLUST_N_ANCHORS=15          # Number of anchor documents
GRAPHCLUST_EDGE_THRESHOLD=0.3    # Min cosine sim for graph edges
GRAPHCLUST_KNN=15                # k-nearest neighbours per node
GRAPHCLUST_RESOLUTION=1.0        # Louvain resolution γ
GRAPHCLUST_SIMILARITY_THRESHOLD=0.15  # Embedding pre-filter threshold

# ── Embedding ──
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Choosing K₀

| K₀ | When to Use |
|:--:|-------------|
| 100 | Faster Stage 8 (100 API calls) |
| **300** | **Default — best coverage for most datasets** |
| 500 | Datasets with many classes (clinc=150, stackexchange=121) |

**Rule of thumb**: K₀ ≥ 3× expected K\*.

### Choosing PCA Dimensions

| PCA Dims | Notes |
|:--------:|-------|
| 20 | Too aggressive, loses structure |
| **50** | **Default — good balance (~57% explained variance)** |
| 100 | More detail, slightly slower |

---

## 13. Troubleshooting

### "Command not found: tc-sealclust"

```bash
conda activate ppd          # or: source .venv/bin/activate
pip install -e ".[dev]"     # reinstall entry points
```

### "OPENAI_API_KEY not set"

Create `.env` in the project root with `LLM_PROVIDER`, `LLM_MODEL`, and `OPENAI_API_KEY`.

### Stage 8 is too slow

Stage 8 makes ~300 LLM calls (~1.5s each). Options:
- Reduce K₀: `--k0 100`
- Use a faster model: `LLM_MODEL=gpt-3.5-turbo`

### K\* estimation gives wrong number

All automated methods estimate based on embedding geometry. If clusters overlap, K\* will be under-estimated. **Use `--k_star N` for best accuracy.**

### "Unsuccessful" label in results

Some documents couldn't be classified. Try increasing K\*, or re-run Stage 7 for different label names.

---

## 14. Repository Structure

```
SEALClust/
├── text_clustering/                  # Python package
│   ├── __init__.py
│   ├── client.py                     # OpenRouter/OpenAI client factory
│   ├── config.py                     # Centralised env-var config
│   ├── llm.py                        # LLM helpers: chat, retry, token tracking
│   ├── data.py                       # Dataset loading + domain descriptions
│   ├── prompts.py                    # Prompt construction (all versions)
│   ├── embedding.py                  # Sentence-transformers embedding
│   ├── dimreduce.py                  # PCA / t-SNE reduction
│   ├── kmedoids.py                   # K-Medoids clustering + propagation
│   ├── _kmedoids_impl.py            # Custom K-Medoids (PAM alternate)
│   ├── gmm.py                        # GMM clustering + propagation
│   ├── sealclust.py                  # SEAL-Clust v2 core algorithms
│   ├── sealclust_v3.py              # SEAL-Clust v3 core algorithms
│   ├── sealclust_v4.py              # SEAL-Clust v4 core algorithms ← NEW
│   ├── hybrid.py                     # Hybrid pipeline: 8-step LLM + embedding
│   ├── graphclust.py                # Graph community clustering (Louvain)
│   ├── baselines.py                  # KMeans / GMM baselines (no LLM)
│   ├── visualization.py             # t-SNE / UMAP cluster visualisation
│   ├── logging_config.py            # Logging setup
│   ├── label_cache.py               # Shared label cache for --reuse_labels
│   └── pipeline/                     # CLI entry points for all modes
│       ├── sealclust_v4_pipeline.py # v4 Mode S pipeline ← NEW
│       ├── sealclust_v3_pipeline.py # v3 Modes Y/Z pipeline
│       ├── sealclust_pipeline.py    # v2 Modes D/E pipeline
│       ├── hybrid_pipeline.py       # Mode F pipeline
│       ├── graphclust_pipeline.py   # Mode H pipeline
│       ├── baseline_pipeline.py     # Mode G pipeline
│       ├── label_generation.py      # Mode A Step 1
│       ├── classification.py        # Modes A–E Step 2 / Stage 8
│       ├── evaluation.py            # Evaluation (all modes)
│       ├── kmedoids_preprocessing.py# Mode B
│       ├── gmm_preprocessing.py     # Mode C
│       └── seed_labels.py           # Mode A Step 0
│   └── tools/                        # Developer utilities
│       ├── preflight.py             # Pre-run check (tc-preflight)
│       ├── probe_models.py          # Model compatibility probe (tc-probe-models)
│       ├── remerge_labels.py        # Re-merge labels tool (tc-remerge-labels)
│       └── analyze_datasets.py     # Dataset profiling (tc-analyze)
├── reference_impl/                   # Original paper shims (backward-compat)
├── datasets/                         # 14 datasets (not in git)
├── runs/                             # All outputs (not in git)
├── logs/                             # Background run logs (not in git)
├── Makefile                          # Convenience targets
├── pyproject.toml                    # Metadata + dependencies + entry points
├── requirements.txt                  # Pinned fallback for pip
├── FINDINGS.md                       # Research log
├── RESEARCH_ASSESSMENT.md           # Research assessment
├── CONTRIBUTING.md                  # Contribution guidelines
└── CHANGELOG.md                     # Version history
```
---

## 15. Tutorial — Running SEAL-Clust v4 Step by Step

This tutorial walks through a complete run on `massive_scenario` (2,974 virtual assistant commands, 18 ground-truth classes).

### Prerequisites

```bash
conda activate ppd                        # Activate environment
tc-preflight                              # Verify LLM connectivity
ls datasets/massive_scenario/small.jsonl  # Verify dataset exists
```

### Option A: One-Command Full Pipeline

```bash
# Full pipeline — everything automated
tc-sealclust-v4 --data massive_scenario --k0 300 --k_star 18 --full

# Or via Make:
make run-sealclust-v4-full data=massive_scenario kstar=18
```

This will:
1. Embed all 2,974 documents (all-MiniLM-L6-v2, ~10s with GPU)
2. Overcluster into 300 micro-clusters via K-Medoids (~30s)
3. Select 300 representative documents
4. Discover candidate labels by sending all docs to the LLM in 30-doc chunks (~100 LLM calls)
5. Consolidate ~60 candidates into exactly 18 final labels (~1 LLM call)
6. Classify 300 representatives into the 18 labels (~15 LLM calls with batch_size=20)
7. Propagate labels to all 2,974 documents
8. Evaluate (ACC, NMI, ARI) and generate visualisations

Total: **~116 API calls, ~88K tokens, ~5 minutes**

### Option B: Step-by-Step (Inspect Intermediate Results)

#### Step 1: Run Stages 1–7

```bash
tc-sealclust-v4 --data massive_scenario --k0 300 --k_star 18
# Note the printed run directory, e.g.:
#   Run dir: ./runs/massive_scenario_small_v4_20260325_094026
```

#### Step 2: Inspect discovered labels

```bash
# Candidate labels from discovery (Stage 5) — typically 50–80 candidates
cat ./runs/<run_dir>/labels_proposed.json | python3 -m json.tool

# Final consolidated labels (Stage 7) — exactly K* labels
cat ./runs/<run_dir>/labels_merged.json | python3 -m json.tool
```

Example output:
```json
["alarm", "smart_home", "music", "weather", "food", "time", "joke",
 "date", "news", "volume", "stock", "notification", "brightness",
 "temperature", "reminder", "shopping", "calendar", "health"]
```

#### Step 3: Classify representatives

```bash
tc-sealclust-v4 --data massive_scenario --run_dir ./runs/<run_dir> --classify

# Or via Make:
make run-sealclust-v4-classify data=massive_scenario run=./runs/<run_dir>
```

#### Step 4: Propagate labels to all documents

```bash
tc-sealclust-v4 --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# Or via Make:
make run-sealclust-v4-propagate data=massive_scenario run=./runs/<run_dir>
```

#### Step 5: Evaluate

```bash
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
# → results.json with ACC, NMI, ARI
```

### Option C: Different Clustering Backends

```bash
# GMM instead of K-Medoids
tc-sealclust-v4 --data massive_scenario --k0 300 --cluster_method gmm --full

# KMeans
tc-sealclust-v4 --data massive_scenario --k0 300 --cluster_method kmeans --full

# With PCA pre-reduction
tc-sealclust-v4 --data massive_scenario --k0 300 --reduction pca --pca_dims 50 --full
```

### Option D: Auto K\* (No Ground-Truth Knowledge)

```bash
# Let the pipeline estimate K* automatically
tc-sealclust-v4 --data massive_scenario --k0 300 --full

# Try different estimation methods
tc-sealclust-v4 --data massive_scenario --k0 300 --k_method ensemble --full
tc-sealclust-v4 --data massive_scenario --k0 300 --k_method calinski --full
```

> **💡 Tip**: For best accuracy, set `--k_star` to the ground-truth number of classes. All automated K\* methods tend to under-estimate.

### Re-Running with Different K\*

Delete the K\*-dependent files and re-run:

```bash
rm ./runs/<run_dir>/labels_merged.json \
   ./runs/<run_dir>/classifications.json \
   ./runs/<run_dir>/classifications_full.json \
   ./runs/<run_dir>/results.json

# Re-run with a different K*
tc-sealclust-v4 --data massive_scenario --run_dir ./runs/<run_dir> --k_star 20 --full
```

> **💡** Keep `labels_proposed.json` — candidate labels don't depend on K\*.

---


## 16. Development

Branching: `main` ← `develop` ← `feature/<desc>` / `fix/<desc>` / `docs/<desc>`
Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

```bash
make setup            # venv + install + git hooks
make lint             # ruff check
cz commit             # conventional commit prompt
make branch name=my-feature type=feature
make release          # bump version, merge develop→main, push tags
```

---

## 17. Citation

```bibtex
@inproceedings{huang2024text,
  title={Text Clustering as Classification with LLMs},
  author={Huang, Chen and He, Guoxiu},
  year={2024},
  url={https://arxiv.org/abs/2410.00927}
}
```
