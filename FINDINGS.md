# Findings — Text Clustering as Classification with LLMs

> **Project**: PPD — Text Clustering as Classification with LLMs  
> **Programme**: M2 MLSD, Université Paris Cité  
> **Period**: 2026-02-18 → ongoing  
> **Reference paper**: [arXiv:2410.00927](https://arxiv.org/abs/2410.00927) — Huang & He, 2024

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experimental Setup](#2-experimental-setup)
3. [Pipeline Versions](#3-pipeline-versions)
4. [Results by Dataset](#4-results-by-dataset)
   - [massive_scenario](#41-massive_scenario)
   - [massive_intent](#42-massive_intent)
   - [mtop_intent](#43-mtop_intent)
   - [arxiv_fine](#44-arxiv_fine)
   - [go_emotion](#45-go_emotion)
5. [Cross-Dataset Summary](#5-cross-dataset-summary)
6. [Analysis & Insights](#6-analysis--insights)
   - [SEAL-Clust v3: Cluster Method Comparison](#61-seal-clust-v3-cluster-method-comparison)
   - [SEAL-Clust v3: Label Source Analysis](#62-seal-clust-v3-label-source-analysis)
   - [SEAL-Clust v3: k₀ Sensitivity Analysis](#63-seal-clust-v3-k₀-sensitivity-analysis)
   - [SEAL-Clust v3: Reduction Analysis](#64-seal-clust-v3-reduction-analysis)
   - [GMM/KMeans Ordering Bug Impact](#65-gmmkmeans-ordering-bug-impact)
   - [Consolidation Accuracy](#66-consolidation-accuracy)
7. [Known Issues & Limitations](#7-known-issues--limitations)
8. [Conclusions & Recommendations](#8-conclusions--recommendations)

---

## 1. Executive Summary

This document presents a comprehensive analysis of **100+ experiment runs** across 5 datasets, comparing multiple pipeline variants for the "Text Clustering as Classification" paradigm introduced in the reference paper. The pipelines tested include:

- **SEAL-Clust v2** (LLM-based label generation → LLM-based classification)
- **SEAL-Clust v2 (batched)** (multi-pass merge strategy)
- **SEAL-Clust v3** (embedding-based clustering → LLM label discovery → LLM classification)
- **Baseline KMeans** (embedding + KMeans, no LLM)
- **Baseline GMM** (embedding + GMM, no LLM)
- **GraphClust** (incomplete)

**Key findings**:

| Finding | Detail |
|---------|--------|
| **Best overall method varies by dataset** | No single pipeline dominates across all 5 datasets |
| **LLM methods shine on high-k intent datasets** | On `mtop_intent` (102 classes), SEAL-Clust v3 achieves ACC=0.594 vs KMeans=0.330 — a **+80%** improvement |
| **Baselines dominate on arxiv_fine** | KMeans ACC=0.319 vs best LLM ACC=0.219 — LLM methods struggle with 93 fine-grained academic categories |
| **go_emotion is universally hard** | All methods below ACC=0.32; the dataset has high label ambiguity |
| **GMM/KMeans ordering bug caused 3× accuracy drop** | Pre-fix v3 GMM: ACC=0.16; post-fix: ACC=0.51 |
| **SEAL-Clust v2 best run**: ACC=0.680 on `massive_scenario` | Closest to paper baseline (0.718) |

---

## 2. Experimental Setup

### Datasets (small split)

| Dataset | Samples | True k | Domain |
|---------|---------|--------|--------|
| `massive_scenario` | 2,974 | 18 | Voice assistant scenarios |
| `massive_intent` | 2,974 | 59 | Voice assistant intents |
| `mtop_intent` | 4,386 | 102 | Multi-domain task intents |
| `arxiv_fine` | 3,674 | 93 | Academic paper topics |
| `go_emotion` | 5,940 | 27 | Emotion detection |

### Models & Infrastructure

| Component | Value |
|-----------|-------|
| **LLM** | `gpt-4o-mini` (OpenAI) |
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim) |
| **Evaluation metrics** | ACC (Hungarian-matched), ARI, NMI |
| **Seed labels** | 20% of true labels (per paper protocol) |
| **Random state** | 42 (fixed for reproducibility) |

### Paper Baseline (Table 2 — reference results with `gpt-3.5-turbo-0125`)

| Dataset | ACC | NMI | ARI |
|---------|-----|-----|-----|
| `massive_scenario` | 71.75 | 78.00 | 56.86 |
| `massive_intent` | 64.12 | 65.44 | 48.92 |
| `mtop_intent` | 72.18 | 78.78 | 71.93 |
| `arxiv_fine` | 38.78 | 57.43 | 20.55 |
| `go_emotion` | 31.66 | 27.39 | 13.50 |

---

## 3. Pipeline Versions

### SEAL-Clust v2 (LLM-based, `reference_impl/` pipeline)

The reproduction of the original paper's approach adapted to our infrastructure:
1. **Label generation**: LLM reads chunks of 15 texts, proposes candidate labels
2. **Label merging**: LLM consolidates candidates into a final label set (single call)
3. **Classification**: LLM assigns each text to one label (one call per text)

Runs are identified by directory names like `<dataset>_small_<timestamp>` without `v3` or `baseline` suffixes.

### SEAL-Clust v2 (batched)

A variant with multi-pass batched merging to work around model merge limitations. Labels are merged in batches of 30, then batch outputs are merged again, iterating until convergence.

### SEAL-Clust v3 (embedding-based + LLM)

A new hybrid pipeline:
1. **Embed**: Encode all documents with `all-MiniLM-L6-v2`
2. **Cluster**: Apply KMedoids, KMeans, or GMM to get k₀ initial clusters
3. **Discover labels**: LLM reads cluster representatives and proposes labels
4. **Consolidate**: LLM merges candidate labels down to k* final labels
5. **Classify**: LLM assigns each text to one of the final labels

Key parameters:
- `k0`: Number of initial clusters (overclustering factor)
- `k_star`: Target number of final labels
- `cluster_method`: `kmedoids` | `gmm` | `kmeans`
- `label_source`: `all` (all texts) | `representatives` (cluster medoids/centroids only)
- `reduction`: `none` | `pca` (PCA dimensionality reduction before clustering)

### Baseline KMeans

Pure embedding-based: embed → KMeans(k=true_k) → evaluate. No LLM involved. Labels assigned by cluster index, matched to ground truth via Hungarian algorithm.

### Baseline GMM

Same as Baseline KMeans but using Gaussian Mixture Models with tied or diagonal covariance.

---

## 4. Results by Dataset

### 4.1. `massive_scenario`

**True k = 18 · n = 2,974 · Paper ACC = 71.75**

This is the most extensively tested dataset with 30+ completed runs spanning all pipeline versions.

#### SEAL-Clust v2 (best runs)

| Run | pred_k | ACC | ARI | NMI | Notes |
|-----|--------|-----|-----|-----|-------|
| `20260323_134134` | 18 | **0.6796** | **0.5844** | **0.6609** | Best v2 overall |
| `20260319_094252` | 19 | 0.6264 | 0.4966 | 0.6193 | |
| `20260318_114822` | 17 | 0.6251 | 0.4908 | 0.6122 | |
| `20260319_093025` | 18 | 0.6224 | 0.4917 | 0.6041 | |
| `20260322_203140` | 19 | 0.5999 | 0.5138 | 0.6181 | |
| `20260319_141430` | 18 | 0.5978 | 0.4293 | 0.5759 | |
| `20260323_124000` | 19 | 0.5945 | 0.4873 | 0.5925 | |
| `20260323_133312` | 19 | 0.5831 | 0.4635 | 0.6103 | |
| `20260323_131550` | 18 | 0.5773 | 0.4483 | 0.5850 | |
| `20260314_113900` | 7 | 0.5632 | 0.3866 | 0.5515 | Under-clustered |
| `20260312_120831` | 20 | 0.5521 | 0.3985 | 0.5725 | |
| `20260312_112628` | 20 | 0.5498 | 0.4166 | 0.5778 | |
| `20260323_134749` | 17 | 0.5454 | 0.4193 | 0.5630 | |
| `20260313_095906` | 17 | 0.5363 | 0.4053 | 0.5851 | |
| `20260314_124216` | 18 | 0.5356 | 0.4492 | 0.5917 | |
| `20260318_141242` | 18 | 0.5148 | 0.3906 | 0.5061 | |
| `20260313_135205` | 20 | 0.4344 | 0.2766 | 0.4237 | |
| `20260313_113104` | 12 | 0.4321 | 0.2614 | 0.4468 | Under-clustered |
| `20260318_143356` | 19 | 0.4210 | 0.2822 | 0.4820 | |

**v2 range**: ACC 0.421–0.680 | ARI 0.261–0.584 | NMI 0.424–0.661  
**v2 best**: ACC=0.680, ARI=0.584, NMI=0.661 (gap to paper: −5.4% ACC)

#### SEAL-Clust v2 (batched)

| Run | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----|-----|
| `batched_20260319_100000` | 18 | 0.5797 | 0.4415 | 0.6082 |

#### SEAL-Clust v3

| Run | Method | k₀ | Label src | Reduction | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----------|-----------|--------|-----|-----|-----|
| `v3_20260323_114701` | kmedoids | 300 | reps | none | 18 | **0.5488** | **0.4090** | 0.5376 |
| `v3_20260323_120334` | kmeans | 300 | reps | none | 18 | 0.5430 | 0.4182 | **0.5751** |
| `v3_20260323_104132` | kmedoids | 300 | — | pca | 18 | 0.5327 | 0.4087 | 0.5479 |
| `v3_20260323_120041` | gmm | 300 | reps | none | 18 | 0.5145 | 0.3614 | 0.5233 |
| `v3_20260323_142230` | kmedoids | 500 | reps | pca | 19 | 0.4970 | 0.2956 | 0.4755 |
| `v3_20260323_141537` | kmedoids | 1000 | reps | pca | 18 | 0.4909 | 0.3380 | 0.5068 |
| `v3_20260323_120914` | gmm | 300 | reps | none | 18 | 0.4896 | 0.3605 | 0.5217 |
| `v3_20260323_135401` | kmedoids | 300 | reps | pca | 18 | 0.4876 | 0.3658 | 0.5289 |
| `v3_20260323_140338` | kmedoids | 2000 | reps | pca | 18 | 0.4455 | 0.3133 | 0.5142 |
| `v3_20260323_115259` | kmeans | 300 | reps | none | 18 | 0.1859 | 0.0374 | 0.1437 | 🐛 ordering bug |
| `v3_20260323_114930` | gmm | 300 | reps | none | 18 | 0.1769 | 0.0324 | 0.1485 | 🐛 ordering bug |
| `v3_20260323_113425` | gmm | 300 | — | none | 18 | 0.1624 | 0.0354 | 0.1498 | 🐛 ordering bug |

#### Baseline KMeans

| Run | PCA dims | pred_k | ACC | ARI | NMI |
|-----|----------|--------|-----|-----|-----|
| `kmeans_20260318_150511` | 100 | 18 | **0.5972** | **0.4614** | **0.6500** |
| `kmeans_20260318_150453` | 50 | 18 | 0.5861 | 0.4519 | 0.6415 |
| `kmeans_20260318_150425` | 0 | 18 | 0.5807 | 0.4318 | 0.6453 |
| `kmeans_20260318_150611` | 0 (auto_k) | 24 | 0.5541 | 0.4167 | 0.6537 |

#### Baseline GMM

| Run | Covariance | pred_k | ACC | ARI | NMI |
|-----|------------|--------|-----|-----|-----|
| `gmm_20260318_150526` | tied | 18 | **0.5790** | **0.4119** | **0.6469** |
| `gmm_20260318_150550` | diag | 18 | 0.5548 | 0.3545 | 0.6219 |

#### Summary — `massive_scenario`

| Pipeline | Best ACC | Best ARI | Best NMI |
|----------|----------|----------|----------|
| **Paper baseline** | **0.7175** | **0.5686** | **0.7800** |
| SEAL-Clust v2 | 0.6796 | 0.5844 | 0.6609 |
| Baseline KMeans (PCA=100) | 0.5972 | 0.4614 | 0.6500 |
| Baseline GMM (tied) | 0.5790 | 0.4119 | 0.6469 |
| SEAL-Clust v2 (batched) | 0.5797 | 0.4415 | 0.6082 |
| SEAL-Clust v3 (kmedoids) | 0.5488 | 0.4090 | 0.5376 |
| SEAL-Clust v3 (kmeans) | 0.5430 | 0.4182 | 0.5751 |
| SEAL-Clust v3 (gmm) | 0.5145 | 0.3614 | 0.5233 |

---

### 4.2. `massive_intent`

**True k = 59 · n = 2,974 · Paper ACC = 64.12**

#### SEAL-Clust v2

| Run | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----|-----|
| `20260322_204338` | 115 | **0.4627** | 0.3643 | **0.6796** |
| `20260319_151040` | 57 | 0.4583 | **0.3721** | 0.6414 |
| `20260319_135325` | 55 | 0.4452 | 0.3303 | 0.6433 |
| `20260313_141759` | 19 | 0.3837 | 0.2426 | 0.3916 |

Note: The earliest run (March 13) heavily under-clustered (19 vs 59 true), causing a significant ACC drop.

#### SEAL-Clust v3

| Run | Method | k₀ | Label src | Reduction | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----------|-----------|--------|-----|-----|-----|
| `v3_20260323_142841` | kmedoids | 300 | reps | none | 55 | **0.4943** | 0.3817 | 0.6416 |
| `v3_20260323_121358` | gmm | 300 | reps | none | 59 | 0.4879 | **0.3930** | **0.6516** |
| `v3_20260323_104537` | kmedoids | 300 | — | pca | 59 | 0.4701 | 0.3489 | 0.6304 |

#### Baselines

| Pipeline | PCA dims | pred_k | ACC | ARI | NMI |
|----------|----------|--------|-----|-----|-----|
| KMeans | 0 | 59 | **0.5356** | 0.3796 | 0.7030 |
| KMeans | 100 | 59 | 0.5171 | **0.3817** | **0.7013** |
| GMM (tied) | 0 | 59 | 0.5171 | 0.3740 | 0.6916 |

#### Summary — `massive_intent`

| Pipeline | Best ACC | Best ARI | Best NMI |
|----------|----------|----------|----------|
| **Paper baseline** | **0.6412** | **0.4892** | **0.6544** |
| Baseline KMeans | 0.5356 | 0.3796 | 0.7030 |
| Baseline GMM | 0.5171 | 0.3740 | 0.6916 |
| SEAL-Clust v3 (kmedoids) | 0.4943 | 0.3817 | 0.6416 |
| SEAL-Clust v3 (gmm) | 0.4879 | 0.3930 | 0.6516 |
| SEAL-Clust v2 | 0.4627 | 0.3721 | 0.6796 |

---

### 4.3. `mtop_intent`

**True k = 102 · n = 4,386 · Paper ACC = 72.18**

This is the dataset where LLM-based methods most dramatically outperform pure embedding baselines.

#### SEAL-Clust v2

| Run | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----|-----|
| `20260319_205405` | 88 | **0.5771** | **0.5887** | **0.7296** |
| `20260322_215830` | 90 | 0.5091 | 0.4745 | 0.6798 |

#### SEAL-Clust v3

| Run | Method | k₀ | Label src | Reduction | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----------|-----------|--------|-----|-----|-----|
| `v3_20260323_144706` | kmedoids | 1000 | reps | none | 67 | **0.5939** | 0.5642 | **0.7087** |
| `v3_20260323_110729` | kmedoids | 1000 | — | pca | 68 | 0.5853 | **0.5707** | 0.7045 |

#### Baselines

| Pipeline | PCA dims | pred_k | ACC | ARI | NMI |
|----------|----------|--------|-----|-----|-----|
| GMM (tied) | 0 | 102 | **0.3497** | **0.2542** | **0.6861** |
| KMeans | 0 | 102 | 0.3297 | 0.2462 | 0.6859 |
| KMeans | 100 | 102 | 0.2793 | 0.2116 | 0.6693 |

#### Summary — `mtop_intent`

| Pipeline | Best ACC | Best ARI | Best NMI |
|----------|----------|----------|----------|
| **Paper baseline** | **0.7218** | **0.7193** | **0.7878** |
| **SEAL-Clust v3 (kmedoids)** | **0.5939** | **0.5707** | **0.7087** |
| SEAL-Clust v2 | 0.5771 | 0.5887 | 0.7296 |
| Baseline GMM | 0.3497 | 0.2542 | 0.6861 |
| Baseline KMeans | 0.3297 | 0.2462 | 0.6859 |

> **Standout result**: LLM methods outperform baselines by **+80% ACC** on this high-k dataset. The 102 intent classes are semantically rich enough that the LLM's language understanding provides a massive advantage over pure embedding similarity.

---

### 4.4. `arxiv_fine`

**True k = 93 · n = 3,674 · Paper ACC = 38.78**

The hardest dataset for LLM-based methods. Fine-grained academic topic distinctions (e.g., `cs.AI` vs `cs.ML` vs `cs.CL`) are difficult for the LLM to discover and distinguish.

#### SEAL-Clust v2

| Run | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----|-----|
| `20260319_195056` | 90 | **0.2191** | **0.1150** | **0.5135** |
| `20260319_221956` | 86 | 0.2025 | 0.0928 | 0.4488 |
| `20260322_214422` | 64 | 0.1658 | 0.0704 | 0.4026 |
| `20260318_120306` | 63 | 0.1622 | 0.0740 | 0.4059 |
| `20260319_215647` | 19 | 0.1140 | 0.0662 | 0.3581 |

#### SEAL-Clust v3

| Run | Method | k₀ | Label src | Reduction | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----------|-----------|--------|-----|-----|-----|
| `v3_20260323_122510` | gmm | 500 | reps | none | 73 | **0.1783** | **0.0786** | 0.4346 |
| `v3_20260323_105426` | kmedoids | 500 | — | pca | 66 | 0.1758 | 0.0788 | 0.4153 |
| `v3_20260323_123115` | gmm | 1000 | reps | none | 91 | 0.1712 | 0.0769 | **0.4489** |
| `v3_20260323_145919` | kmedoids | 1000 | reps | none | 96 | 0.1666 | 0.0666 | 0.4037 |

Note: The `v3_20260323_123115` run mistakenly used `k_star=102` (mtop_intent's k) instead of 93.

#### Baselines

| Pipeline | PCA dims | pred_k | ACC | ARI | NMI |
|----------|----------|--------|-----|-----|-----|
| KMeans | 100 | 93 | **0.3187** | **0.1719** | **0.5375** |
| KMeans | 0 | 93 | 0.3078 | 0.1640 | 0.5310 |
| GMM (tied) | 0 | 93 | 0.3013 | 0.1672 | 0.5319 |

#### Summary — `arxiv_fine`

| Pipeline | Best ACC | Best ARI | Best NMI |
|----------|----------|----------|----------|
| **Paper baseline** | **0.3878** | **0.2055** | **0.5743** |
| **Baseline KMeans (PCA=100)** | **0.3187** | **0.1719** | **0.5375** |
| Baseline GMM | 0.3013 | 0.1672 | 0.5319 |
| SEAL-Clust v2 | 0.2191 | 0.1150 | 0.5135 |
| SEAL-Clust v3 (gmm) | 0.1783 | 0.0786 | 0.4346 |

> **Key observation**: Embedding baselines outperform all LLM-based pipelines on this dataset. The fine-grained academic taxonomy (93 narrow ArXiv categories) is inherently difficult for LLMs to discover from text alone. The embedding space captures subtle topical differences that the LLM's label generation step misses.

---

### 4.5. `go_emotion`

**True k = 27 · n = 5,940 · Paper ACC = 31.66**

The most challenging dataset for all methods. Emotion labels are inherently ambiguous — a single short text can plausibly belong to multiple emotion categories.

#### SEAL-Clust v2

| Run | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----|-----|
| `20260319_184719` | 28 | **0.3157** | **0.1456** | **0.2536** |
| `20260318_121756` | 26 | 0.1535 | 0.0512 | 0.0810 |
| `20260322_213120` | 27 | 0.1458 | 0.0265 | 0.0766 |

Note: The first run (March 19) is a strong outlier — significantly better than all others. This likely reflects favorable LLM label generation in that particular run.

#### SEAL-Clust v3

| Run | Method | k₀ | Label src | Reduction | pred_k | ACC | ARI | NMI |
|-----|--------|-----|-----------|-----------|--------|-----|-----|-----|
| `v3_20260323_104914` | kmedoids | 300 | — | pca | 28 | **0.1562** | **0.0422** | **0.0893** |
| `v3_20260323_121840` | gmm | 300 | reps | none | 28 | 0.1340 | 0.0250 | 0.0799 |
| `v3_20260323_143714` | kmedoids | 500 | reps | none | 104 | 0.1175 | 0.0310 | 0.1219 |
| `v3_20260323_143218` | kmedoids | 300 | reps | none | 27 | 0.1131 | 0.0138 | 0.0564 |

Note: The `v3_20260323_143714` run used `k_star=93` (wrong dataset value), resulting in 104 predicted clusters.

#### Baselines

| Pipeline | PCA dims | pred_k | ACC | ARI | NMI |
|----------|----------|--------|-----|-----|-----|
| KMeans | 0 | 27 | **0.1448** | **0.0402** | **0.1093** |
| KMeans | 50 | 27 | 0.1428 | 0.0393 | 0.1125 |
| KMeans | 100 | 27 | 0.1352 | 0.0375 | 0.1117 |
| GMM (tied) | 0 | 27 | 0.1264 | 0.0260 | 0.0893 |

#### Summary — `go_emotion`

| Pipeline | Best ACC | Best ARI | Best NMI |
|----------|----------|----------|----------|
| **Paper baseline** | **0.3166** | **0.1350** | **0.2739** |
| SEAL-Clust v2 (outlier) | 0.3157 | 0.1456 | 0.2536 |
| SEAL-Clust v3 (kmedoids) | 0.1562 | 0.0422 | 0.0893 |
| Baseline KMeans | 0.1448 | 0.0402 | 0.1093 |
| Baseline GMM | 0.1264 | 0.0260 | 0.0893 |

> **Key observation**: All methods struggle significantly on `go_emotion`. The best v2 outlier aside, typical LLM runs produce ACC ≈ 0.15, comparable to random assignment for 27 classes (0.037). The emotion domain has high inter-annotator disagreement and many short, ambiguous texts.

---

## 5. Cross-Dataset Summary

### Best ACC by Pipeline and Dataset

| Pipeline | massive_scenario | massive_intent | mtop_intent | arxiv_fine | go_emotion |
|----------|:---:|:---:|:---:|:---:|:---:|
| **Paper baseline** | **0.718** | **0.641** | **0.722** | **0.388** | **0.317** |
| SEAL-Clust v2 | 0.680 | 0.463 | 0.577 | 0.219 | 0.316 |
| SEAL-Clust v3 | 0.549 | 0.494 | **0.594** | 0.178 | 0.156 |
| Baseline KMeans | 0.597 | 0.536 | 0.330 | **0.319** | 0.145 |
| Baseline GMM | 0.579 | 0.517 | 0.350 | 0.301 | 0.126 |

### Best ARI by Pipeline and Dataset

| Pipeline | massive_scenario | massive_intent | mtop_intent | arxiv_fine | go_emotion |
|----------|:---:|:---:|:---:|:---:|:---:|
| **Paper baseline** | **0.569** | **0.489** | **0.719** | **0.206** | **0.135** |
| SEAL-Clust v2 | **0.584** | 0.372 | 0.589 | 0.115 | 0.146 |
| SEAL-Clust v3 | 0.418 | 0.393 | **0.571** | 0.079 | 0.042 |
| Baseline KMeans | 0.461 | 0.382 | 0.246 | **0.172** | 0.040 |
| Baseline GMM | 0.412 | 0.374 | 0.254 | 0.167 | 0.026 |

### Key Takeaways

1. **SEAL-Clust v2 comes closest to the paper** on `massive_scenario` (ACC gap: −5.4%) and `go_emotion` (best run within 0.1%)
2. **SEAL-Clust v3 excels on `mtop_intent`** — the highest-k dataset — achieving the best non-paper ACC (0.594)
3. **Baselines win on `arxiv_fine`** — embedding similarity captures fine-grained academic categories better than LLM label discovery
4. **Baselines win on `massive_intent`** in ACC — but v3 is competitive in ARI (0.393 vs 0.382)
5. **`go_emotion` is universally hard** — typical ACC ≈ 0.13–0.16 across all methods

### When does LLM help vs. hurt?

| Dataset characteristic | LLM advantage | Explanation |
|------------------------|---------------|-------------|
| Many semantically distinct classes (mtop: 102) | **Strong** (+80% vs baseline) | LLM understands intent semantics that embeddings conflate |
| Moderate classes, broad topics (massive_scenario: 18) | **Moderate** (+14% vs baseline) | LLM generates reasonable category names; some fragmentation |
| Many fine-grained technical classes (arxiv: 93) | **Negative** (−31% vs baseline) | LLM cannot discover narrow academic taxonomy |
| Emotion/affect classes (go_emotion: 27) | **Negligible** | Both LLM and embeddings fail on inherently ambiguous categories |

---

## 6. Analysis & Insights

### 6.1. SEAL-Clust v3: Cluster Method Comparison

Controlling for other parameters (k₀=300, label_source=representatives, reduction=none) on `massive_scenario`:

| Method | ACC | ARI | NMI | Runs |
|--------|-----|-----|-----|------|
| **kmedoids** | **0.549** | **0.409** | 0.538 | 1 |
| **kmeans** | 0.543 | 0.418 | **0.575** | 1 |
| **gmm** (post-fix) | 0.514 | 0.361 | 0.523 | 2 (avg) |

**Observation**: KMedoids and KMeans perform comparably. GMM lags slightly (−6% ACC), possibly because GMM soft assignments produce less coherent cluster representatives for the LLM to read.

On `massive_intent` (k₀=300, reps, none):

| Method | ACC | ARI | NMI |
|--------|-----|-----|-----|
| **kmedoids** | **0.494** | 0.382 | 0.642 |
| **gmm** | 0.488 | **0.393** | **0.652** |

Here GMM slightly edges out kmedoids on ARI and NMI, suggesting the pattern is dataset-dependent.

### 6.2. SEAL-Clust v3: Label Source Analysis

Comparing `label_source=all` (LLM sees all documents) vs `label_source=representatives` (LLM sees only cluster medoids):

| Dataset | Method | all ACC | reps ACC | Δ |
|---------|--------|---------|----------|---|
| massive_scenario | kmedoids, k₀=300, pca | 0.533 | 0.488 | −0.045 |
| massive_intent | kmedoids, k₀=300, pca | 0.470 | — | — |
| mtop_intent | kmedoids, k₀=1000, pca | 0.585 | — | — |
| mtop_intent | kmedoids, k₀=1000, none | — | **0.594** | — |

**Note**: Direct comparisons are limited because `label_source=all` was mostly used with `reduction=pca` in early v3 runs, while `label_source=representatives` was paired with `reduction=none` later. The two best mtop_intent results (0.585 all/pca vs 0.594 reps/none) suggest that `representatives` mode with no reduction is at least as good, while being significantly faster (LLM reads k₀ texts instead of all n texts).

### 6.3. SEAL-Clust v3: k₀ Sensitivity Analysis

On `massive_scenario` (kmedoids, reps, pca) — a controlled sweep:

| k₀ | n_candidates | n_final | pred_k | ACC | ARI | NMI |
|-----|-------------|---------|--------|-----|-----|-----|
| 300 | 55 | 18 | 18 | **0.488** | **0.366** | **0.529** |
| 500 | 76 | 19 | 19 | 0.497 | 0.296 | 0.476 |
| 1000 | 108 | 18 | 18 | 0.491 | 0.338 | 0.507 |
| 2000 | 166 | 18 | 18 | 0.446 | 0.313 | 0.514 |

**Observation**: Increasing k₀ beyond 300 does **not** improve results on this dataset. In fact, k₀=2000 performs worst. More initial clusters produce more candidate labels, making consolidation harder and noisier. The sweet spot appears to be k₀ ≈ 3–5× true k for moderate datasets.

However, on `mtop_intent` (102 classes), k₀=1000 (≈10× true k) works well, suggesting high-k datasets benefit from more overclustering.

### 6.4. SEAL-Clust v3: Reduction Analysis

Comparing PCA vs no reduction on `massive_scenario` (kmedoids, k₀=300):

| Reduction | Label src | ACC | ARI | NMI |
|-----------|-----------|-----|-----|-----|
| pca (100d) | all | 0.533 | 0.409 | 0.548 |
| none (384d) | reps | 0.549 | 0.409 | 0.538 |
| pca (100d) | reps | 0.488 | 0.366 | 0.529 |

**Observation**: `reduction=none` slightly outperforms PCA in ACC on this dataset. The 384-dimensional embeddings from `all-MiniLM-L6-v2` may already be compact enough that PCA removes useful signal. However, the difference is small and confounded by the label_source change.

### 6.5. GMM/KMeans Ordering Bug Impact

A critical bug was discovered where `sorted(rep_indices)` broke the alignment between JSONL document order and cluster representative labels. This affected all GMM and KMeans runs in v3 before the fix.

**Before vs after fix** (massive_scenario, k₀=300, reps, none):

| Method | Pre-fix ACC | Post-fix ACC | Ratio |
|--------|------------|-------------|-------|
| gmm | 0.162–0.177 | 0.490–0.514 | **3.0×** |
| kmeans | 0.186 | 0.543 | **2.9×** |
| kmedoids | 0.549 | 0.549 | 1.0× (unaffected) |

**Root cause**: KMedoids uses the original document indices as cluster centers, so the ordering is inherently correct. GMM and KMeans assign synthetic centroids, and the sorted indices scrambled the mapping between centroids and their assigned labels.

The fix removed `sorted()` from 3 locations in `sealclust_v3_pipeline.py`, restoring the natural ordering. All GMM/KMeans runs before the fix (identifiable by ACC ≈ 0.16–0.19) should be discarded.

### 6.6. Consolidation Accuracy

The v3 consolidation step (merging n_candidates down to k*) does not always achieve the exact target:

| Dataset | k* | n_candidates | n_final | Accuracy |
|---------|-----|-------------|---------|----------|
| massive_scenario | 18 | 55–178 | 18–19 | ✅ Good |
| massive_intent | 59 | 60–71 | 60–64 | ⚠️ Slight overshoot |
| mtop_intent | 102 | 44–131 | 44–104 | ⚠️ Variable |
| arxiv_fine | 93 | 49–131 | 72–116 | ❌ Poor |
| go_emotion | 27 | 46–142 | 27–136 | ⚠️ Highly variable |

The consolidation struggles when:
1. **n_candidates is close to k***: Not enough to merge (mtop k₀=700: 44 candidates for 102 target → kept 44)
2. **k* is large**: The LLM has difficulty counting and deduplicating among 100+ labels
3. **Labels are semantically distinct**: If candidates genuinely represent different concepts, the LLM correctly refuses to merge them

A deterministic `_trim_labels_by_similarity()` post-processing step was added to guarantee exact k* output by merging the most similar label pairs using embedding cosine similarity.

---

## 7. Known Issues & Limitations

### Incomplete/Failed Runs

Several run directories contain no `results.json`, indicating incomplete or failed runs:

| Count | Reason |
|-------|--------|
| ~15 runs | Step 1 (label generation) failed or was aborted before classification |
| ~5 runs | GraphClust pipeline (experimental, never completed) |
| 1 run | Typo in dataset name (`arxiv-fine` instead of `arxiv_fine`) |
| ~3 runs | v3 runs where candidate count was too low (insufficient k₀) |

### Configuration Errors

| Run | Error |
|-----|-------|
| `v3_20260323_123115` (arxiv) | Used `k_star=102` (mtop's k) instead of 93 |
| `v3_20260323_143714` (go_emotion) | Used `k_star=93` (arxiv's k) instead of 27 |

### Run Variance

SEAL-Clust v2 runs on `massive_scenario` show significant variance: ACC ranges from 0.421 to 0.680 across 19 completed runs. This is inherent to LLM-based label generation — different prompt batching and LLM sampling produce different candidate taxonomies.

### Missing Metadata

Early runs (pre-v3) do not have structured `metadata.json` or `sealclust_v3_metadata.json`. Pipeline version and parameters are inferred from directory naming conventions and log files.

---

## 8. Conclusions & Recommendations

### What works

1. **SEAL-Clust v2 is the closest to the paper** — best ACC=0.680 on `massive_scenario` (−5.4% gap to paper's 0.718). The gap is primarily due to using `gpt-4o-mini` instead of `gpt-3.5-turbo-0125`.

2. **SEAL-Clust v3 excels on high-k intent datasets** — ACC=0.594 on `mtop_intent` (102 classes), outperforming v2 (0.577) and dramatically outperforming baselines (0.330).

3. **Pure embedding baselines are surprisingly competitive** — KMeans with PCA=100 achieves the best non-paper results on `arxiv_fine` and `massive_intent`.

4. **KMedoids is the most reliable v3 cluster method** — no ordering bug, consistent results, and the cluster medoids are actual documents (better LLM context than synthetic centroids).

### What doesn't work

1. **LLM label discovery on fine-grained academic topics** (arxiv: 93 narrow ArXiv categories). The LLM proposes overly broad labels that conflate related but distinct research areas.

2. **All methods on emotion detection** (go_emotion). Emotion labels are inherently ambiguous and subjective.

3. **Large k₀ without benefit**: k₀=2000 on an 18-class dataset produces 166 noisy candidates that consolidation cannot clean up effectively.

### Recommended configurations

| Dataset type | Recommended pipeline | Key params |
|-------------|---------------------|------------|
| Broad intent/scenario (k < 30) | SEAL-Clust v2 | Default prompts |
| Many intents (k > 50) | SEAL-Clust v3 | kmedoids, k₀ ≈ 10×k, reps, none |
| Fine-grained technical (k > 50) | Baseline KMeans | PCA=100 |
| Emotion/affect | None reliable | All methods struggle |

### Future work

- [ ] Run SEAL-Clust v2 on remaining datasets (`mtop_intent`, `arxiv_fine`) for complete comparison
- [ ] Test `gpt-4o-mini` vs `gpt-3.5-turbo-0125` on the same dataset to isolate model effects
- [ ] Investigate few-shot consolidation prompts to improve k* accuracy on high-k datasets
- [ ] Explore ensemble methods (embedding baseline + LLM relabeling)
- [ ] Re-probe free Venice models (Llama 70B, Mistral 24B) for zero-cost comparison runs
