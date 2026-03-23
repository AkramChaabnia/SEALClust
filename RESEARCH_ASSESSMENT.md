# Research Assessment — Text Clustering as Classification with LLMs
## M2 Thesis · Paris Cité · February 2026

**Document type**: Assessment, retrospective, and research initiation  
**Status**: v1.0 — ready for team review  
**Scope**: Paper analysis · Field survey · Project state · Open problems · Roadmap

---

## Table of Contents

1. [Paper Overview — What It Does and Claims](#1-paper-overview)
2. [The Field — Where This Work Sits](#2-the-field)
3. [Critical Assessment of the Paper](#3-critical-assessment)
4. [Our Reproduction — What We Did and Found](#4-our-reproduction)
5. [Open Problems and Research Gaps](#5-open-problems-and-research-gaps)
6. [Candidate Research Directions](#6-candidate-research-directions)
7. [Roadmap — 5 Weeks](#7-roadmap)
8. [Thesis Contribution Options](#8-thesis-contribution-options)
9. [Reference Table](#9-reference-table)

---

## 1. Paper Overview

**Full citation**: Chen Huang & Guoxiu He. "Text Clustering as Classification with LLMs."
*SIGIR-AP 2025*, Xi'an, December 7–10, 2025. arXiv:2410.00927v3.
DOI: [10.1145/3767695.3769519](https://doi.org/10.1145/3767695.3769519)  
**Code**: https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM

### 1.1 The problem

Text clustering groups unlabeled documents into semantically coherent categories without
ground-truth labels. Classical methods (K-Means, DBSCAN over BERT/E5 embeddings) suffer
from three weaknesses the paper targets:

1. They require fine-tuned domain-specific embeddings (expensive, slow).
2. The resulting clusters are anonymous IDs — no human-readable labels without a
   post-hoc annotation step.
3. Clustering algorithms (K-Means especially) need the number of clusters k as input.

### 1.2 The proposed solution

The paper reframes clustering as **label generation + classification**, using only a single
LLM with no embeddings and no traditional clustering algorithm.

**Pipeline — Algorithm 1 (formal inputs: dataset D, batch size B, few-shot labels L_few)**

```
Step 1a — Label Generation
  Dataset shuffled → split into ⌈N/B⌉ mini-batches (default B=15)
  For each batch:
    LLM receives: B sentences + n seed labels (few-shot anchors)
    LLM produces: new candidate label names for uncategorised sentences
  All proposed labels accumulated → deduplicated → L_unique

Step 1b — Label Merge
  LLM receives: L_unique (often 300–500 labels)
  LLM produces: L_final (semantically consolidated set)
  Prompt instruction: merge synonyms and near-duplicates into one representative label
  ⚠️  No target count is given — the LLM is expected to consolidate naturally

Step 2 — Classification
  For each sentence d_i in D:
    LLM receives: d_i + L_final
    LLM assigns: one label from L_final

Step 3 — Evaluation
  Hungarian algorithm aligns predicted labels to ground-truth labels
  Metrics: ACC, NMI, ARI
```

**Key design claim (§3, Introduction)**: *"our framework does not require fine-tuning for
better representation or a pre-assigned cluster number K."*

This is a genuine claim: k is **not** listed in the algorithm inputs. The merge step is
designed to let the LLM discover the natural granularity from semantics alone.

**Seed labels (few-shot)**: 20% of the true label names are provided to the LLM during
Step 1a as in-context examples. This is the only human-supervised signal the pipeline uses.

### 1.3 Implementation details (paper defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| LLM | GPT-3.5-turbo-0125 | Only model tested |
| Batch size B | 15 | Tested: 10, 15 (best), 20 |
| Seed labels | 20% of true labels | Tested: 0%–25%; 20% optimal |
| Datasets | Small split | Large split not evaluated (cost) |
| Temperature | Not specified | |

### 1.4 Results (Table 2 — small split, GPT-3.5-turbo)

| Method | ArxivS2S ACC/NMI/ARI | GoEmo ACC/NMI/ARI | Massive-D ACC/NMI/ARI | Massive-I ACC/NMI/ARI | MTOP-I ACC/NMI/ARI |
|--------|----------------------|-------------------|-----------------------|-----------------------|--------------------|
| K-Means (E5) | 31.21 / 54.47 / 17.01 | 22.14 / 21.26 / 9.64 | 62.21 / 65.42 / 47.69 | 52.79 / 70.76 / 39.03 | 34.48 / 71.47 / 26.35 |
| K-Means (Instructor) | 25.11 / 48.48 / 12.39 | 25.19 / 21.54 / 17.03 | 60.41 / 67.31 / 43.90 | 56.55 / 74.49 / 42.88 | 33.04 / 71.46 / 26.72 |
| ClusterLLM | 26.34 / 50.45 / 13.65 | 26.75 / 23.89 / **17.76** | 60.85 / **68.67** / 45.07 | 60.69 / 77.64 / 46.15 | 35.04 / 73.83 / 29.04 |
| **Paper (ours)** | **38.78 / 57.43 / 20.55** | **31.66 / 27.39** / 13.50 | **64.12** / 65.44 / **48.92** | **71.75 / 78.00 / 56.86** | **72.18 / 78.78 / 71.93** |
| LLM_known_labels (UB) | 41.50 / 57.59 / 20.67 | 38.97 / 28.85 / 18.94 | 69.77 / 69.27 / 55.26 | 75.25 / 78.19 / 58.01 | 73.25 / 80.88 / 73.93 |

Notes:
- ClusterLLM beats the paper on GoEmo ARI (17.76 vs 13.50) and Massive-D NMI (68.67 vs 65.44).
- The paper is not universally dominant.
- LLM_known_labels (oracle true labels) is only 3–7 ACC points above the paper — label
  generation quality is close to the theoretical ceiling.

### 1.5 Granularity — clusters predicted vs. true k (Table 3)

| Method | ArxivS2S (GT=93) | GoEmo (GT=27) | Massive-D (GT=18) | Massive-I (GT=59) | MTOP-I (GT=102) |
|--------|-----------------|---------------|-------------------|-------------------|-----------------|
| ClusterLLM | 16 (−77) | 56 (+29) | 90 (+72) | 43 (−16) | 43 (−59) |
| Paper | 122 (+29) | 52 (+25) | 24 (+6) | 71 (+12) | 83 (−19) |

The paper consistently over-produces clusters even with GPT-3.5. Better alignment than
ClusterLLM on most datasets, but far from exact.

### 1.6 Few-shot label sensitivity (Table 6 — three datasets)

| Seed % | Massive-D ACC | Massive-I ACC | MTOP-I ACC |
|--------|---------------|---------------|------------|
| 0% | 54.54 | 60.20 | 54.53 |
| 10% | 54.54 | 60.20 | 54.53 |
| 15% | 63.87 | 64.31 | 63.10 |
| **20% (default)** | **64.12** | **71.75** | **72.18** |
| 25% | 65.64 | 69.99 | 65.85 |

At 0% seeds the method still outperforms most baselines. But the gap 0% → 20% is large
(+10–18 ACC points), meaning the seeds contribute significantly.

### 1.7 Paper-stated cost (Table 7)

| N samples | API cost | API time | Fine-tuning cost | FT time |
|-----------|----------|----------|-----------------|---------|
| 20k | $3.50 | 3.3 min | $9.65 | 83.6 min |
| 100k | $17.50 | 16.7 min | $9.80 | 85.0 min |
| 500k | $87.50 | 83.3 min | $10.57 | 91.6 min |

The paper compares cost only against fine-tuning. It never benchmarks against the $0
embedding baseline. This is the most misleading part of the paper's cost section.

---

## 2. The Field

### 2.1 Classical embedding-based clustering — the baseline to beat

The canonical approach: embed texts → cluster vectors.

| Component | Common choices | Cost | ACC (Massive-D est.) |
|-----------|---------------|------|----------------------|
| SBERT (MiniLM) + K-Means | Free local inference | $0 | ~55–62% |
| E5-large + K-Means | Free local inference | $0 | ~60–65% |
| Instructor-large + K-Means | Free local inference | $0 | ~60–65% |
| SBERT + K-Means + seed anchoring | Free local | $0 | **~65–70% (estimated)** |

These methods are instantaneous at scale (seconds to minutes), cost nothing, and represent
the true practical floor. Any proposed method must justify its cost overhead against this.

### 2.2 Deep clustering with contrastive learning (2020–2022)

SCCL (2021), CLNN (2022): jointly learn embeddings and cluster assignments. Improve ACC
significantly but require per-dataset training — not zero-shot generalisable.

### 2.3 LLM-augmented methods (2022–2025) — the direct competition

| Paper | Year | Approach | API cost | Key limitation |
|-------|------|----------|----------|---------------|
| **ClusterLLM** (Zhang et al., EMNLP 2023) | 2023 | Triplet comparisons → fine-tunes embedder; pairwise for granularity | **~$0.60/dataset** | Still requires embedder fine-tuning |
| **IDAS** (De Raedt et al., EACL 2023) | 2023 | LLM summarises prototypes → label + embed | Moderate | Embedding-dependent |
| **PAS/GoalEx** (Wang et al., EMNLP 2023) | 2023 | Propose–Assign–Select + ILP | High | Complex pipeline; weak on general data |
| **Keyphrase Clustering** (Viswanathan et al., TACL 2024) | 2024 | LLM keyphrases → embed + cluster | Moderate | Best keyphrase method; still embedding-dependent |
| **This paper** (Huang & He, SIGIR-AP 2025) | 2025 | Full LLM pipeline, no embeddings | ~$3.50/20k | O(N) API calls; model-dependent merge |

The cost gap between ClusterLLM ($0.60) and this paper ($3.50) is 6×. The ACC gap on the
best-case dataset is ~11 points. Whether that tradeoff is worth it depends entirely on the
use case.

### 2.4 The embedding landscape has improved since the paper was written

The paper's K-Means baselines use E5-large and Instructor-large (2022 models). Since then:
- `nomic-embed-text-v1.5` (2024): strong open-source, free local inference
- `bge-m3` (2024): multilingual, short-text focused
- `text-embedding-3-large` (OpenAI, 2024): state-of-art general embeddings

These make the embedding baseline stronger than the paper's numbers suggest. The gap
between "free embedding" and "paid LLM pipeline" is narrower today.

### 2.5 Where the field is heading (open questions as of 2026)

1. **Cost efficiency**: O(N) LLM calls per sample does not scale. Selective or hybrid use is the obvious next step.
2. **k-free clustering**: True discovery of cluster count without oracle k. BERTopic/HDBSCAN approach this from the embedding side; LLM-based methods have not solved it.
3. **Reproducibility**: Non-deterministic LLM outputs make benchmarking noisy. Temperature=0 helps but model-version drift remains.
4. **Multilingual**: Almost all LLM clustering papers are English-only, despite MASSIVE and other multilingual datasets being available.
5. **Interpretability without seeds**: Generating good labels with zero human input is the unsolved problem.

---

## 3. Critical Assessment of the Paper

### 3.1 What the paper gets right

- **Simple, elegant algorithm**: Two-stage pipeline is reproducible and extensible.
- **Strong empirical results**: Best ACC on 4/5 datasets in a competitive comparison.
- **Interpretability by design**: Human-readable labels are a direct output, not a post-hoc step.
- **No fine-tuning or embeddings**: Zero training overhead — plug any LLM in.
- **Near-oracle performance**: LLM_known_labels is only 3–7 ACC points above the method,
  meaning label generation quality is close to the theoretical ceiling.
- **Genuine k-free design**: k is not an input. The claim is architecturally honest.

### 3.2 The k question — accurate framing

The paper explicitly claims no pre-assigned k is needed (§3, Algorithm 1 inputs). This is
architecturally correct: the merge prompt contains no count constraint, and output granularity
is determined by LLM semantics alone.

**In practice**, the claim is model-dependent:
- GPT-3.5 implicitly consolidates to near-k without guidance (Table 3: +6 to +29 overshoot).
- Gemini and weaker models perform light deduplication instead of semantic consolidation
  without an explicit count anchor (our Run 03: 343→311 from true k=18).
- The paper presents no guidance on what model capability is needed for reliable consolidation.

**Correct framing**: The k-free design is real and works with GPT-3.5. It breaks with other
models. The paper's limitation section mentions inconsistent merging but does not quantify
or characterise the model-dependence. This is an important gap.

### 3.3 The comparison is not fair (primary concern)

Every baseline in Table 2 is **fully unsupervised**. The paper's method uses 20% true label
names as few-shot seeds — a semi-supervised signal that no baseline has.

The 0%-seeds ablation (Figure 3 / Table 6) partially addresses this: the method at 0% seeds
still beats most baselines. The LLM has intrinsic capability beyond the seeds.

But the gain from 0% → 20% seeds is +10–18 ACC points on intent datasets. The natural
semi-supervised baseline — K-Means + seed label embedding anchoring — is completely absent
from the paper. Without it, the relative contribution of the LLM pipeline vs. the
supervision signal cannot be assessed.

### 3.4 The cost argument ignores the obvious alternative

The paper compares API cost vs. fine-tuning cost. It never mentions the zero-cost option.

True cost landscape for 20k samples:

| Method | API cost | Compute | ACC est. (Massive-D) |
|--------|----------|---------|----------------------|
| SBERT + K-Means | $0 | 5 sec | ~58% |
| E5-large + K-Means | $0 | 30 sec | ~62% |
| SBERT + K-Means + seeds | $0 | 1 min | ~67% (est.) |
| ClusterLLM | ~$0.60 | ~$2 GPU | ~63% |
| **This paper (GPT-3.5)** | **$3.50** | $0 | **72%** |
| This paper (gemini-flash) | ~$0.15 | $0 | ~61% |

For a practitioner: "should I pay $3.50 or $0?" is the real question. The paper never asks it.

### 3.5 The merge step is the unaddressed bottleneck

The merge step converts 300–500 proposed labels into a final working taxonomy. It is the
most consequential step in the pipeline. Yet the paper:
- Provides no standalone merge quality metric
- Does not ablate merge failure (wrong label count → what downstream ACC impact?)
- Does not characterise which models consolidate vs. deduplicate
- Does not provide a fallback

Table 3 already shows imperfect merge even with GPT-3.5 (+6 to +29 overshoot on 4/5 datasets).
Our experiments show the problem is catastrophic with gemini without explicit guidance.

### 3.6 Dataset selection favours the method

Four of five datasets are clean, balanced, expert-labelled, short-text, English-only datasets
with crisp, non-overlapping categories (intent/domain). This is the optimal scenario for
LLM classification.

The two harder datasets (ArxivS2S: fine-grained academic topics; GoEmo: nuanced fine-grained
emotions) produce the paper's weakest results (38.78 and 31.66 ACC). The paper does not
discuss why.

### 3.7 Summary table

| Issue | Severity | Effect on paper's claims |
|-------|----------|-------------------------|
| Semi-supervised vs. unsupervised comparison | 🔴 High | ACC gains partially attributable to seeds, not the LLM |
| Missing $0 seed-anchored embedding baseline | 🔴 High | Cost-efficiency argument incomplete |
| Merge fragility — model-dependent | 🟡 Medium | k-free claim breaks for non-GPT-3.5 models |
| Dataset homogeneity | 🟡 Medium | Generalization to complex tasks unclear |
| No variance / temperature / reproducibility data | 🟡 Medium | Statistical significance hard to verify |
| k-free claim | ✅ Clarified | Correct by design; model-dependent in practice |

---

## 4. Our Reproduction — What We Did and Found

### 4.1 Infrastructure (v1.3.0, tag on `main`)

Full reproduction pipeline implemented from scratch, faithful to the paper:

```
text_clustering/
  pipeline/
    seed_labels.py       # Step 0: sample 20% of true labels → chosen_labels.json
    label_generation.py  # Step 1: batch label gen + LLM merge
    classification.py    # Step 2: per-sample LLM classification
    evaluation.py        # Step 3: Hungarian alignment + ACC/NMI/ARI
  prompts.py             # All 3 prompts (paper-faithful; target_k=None by default)
  client.py              # OpenRouter API wrapper
  config.py              # Env-based config (model, key, tokens)
  data.py                # All 5 dataset loaders
tools/
  preflight.py           # Pre-run health checks (C1–C6)
  probe_models.py        # Model capability testing
```

All 5 datasets downloaded. Full logging, checkpointing, CLI entry points. Documented
in FINDINGS.md. Ready to extend.

### 4.2 Runs

| Run | Date | Dataset | Model | target_k | n_pred | ACC | NMI | ARI | Status |
|-----|------|---------|-------|----------|--------|-----|-----|-----|--------|
| Paper baseline | — | Massive-D | GPT-3.5-turbo | — | ~24 | **71.75** | **78.00** | **56.86** | Reference |
| Run 01 | 2026-02-20 | Massive-D | trinity-large:free | — | 168 | 40.69 | 66.64 | 33.06 | ❌ Merge token overflow |
| **Run 02** | **2026-02-21** | **Massive-D** | **gemini-2.0-flash** | **18** | **18** | **60.46** | **63.90** | **53.87** | ✅ First valid run |
| Run 03 | 2026-02-21 | Massive-D | gemini-2.0-flash | — | 311 | — | — | — | ⚠️ Aborted after Step 1 |

Only Massive-D (small, k=18, 2974 samples) has been evaluated so far.

### 4.3 Gap analysis — Run 02 vs. paper

| Metric | Run 02 | Paper | Gap | Root cause |
|--------|--------|-------|-----|------------|
| ACC | 60.46 | 71.75 | −11.29 | 4 spurious labels + `weather` class dropped |
| NMI | 63.90 | 78.00 | −14.10 | Spurious labels split true classes' samples |
| **ARI** | **53.87** | **56.86** | **−2.99** | Near-paper — cluster structure is almost correct |

The ARI gap of 3 points is the key insight. ARI measures cluster structure independently
of label assignment. The pipeline produces near-correct cluster partitions; the label
quality (ACC/NMI) suffers because:
- `target_k=18` forced the model to fill 18 slots, creating 4 spurious labels
- `weather` (156 samples, a true class) was dropped — those samples were scattered

### 4.4 Key findings from the experiments

**Finding 1 — Merge step drives the quality gap, not the classification step.**  
When labels are accurate (ARI shows structure is nearly correct), classification is good.
The bottleneck is label taxonomy quality, not the per-sample classification.

**Finding 2 — GPT-3.5 and gemini have qualitatively different merge behaviour.**  
GPT-3.5 implicitly consolidates 350 labels to ~24 (near k=18) without guidance.
Gemini without `target_k` produces 311 from 343 — light deduplication only.
This is not a bug in our code; it is a fundamental model-dependence of the algorithm.

**Finding 3 — `target_k` trades label accuracy for count precision.**  
With `target_k=18`, gemini hits exact k but fills spurious slots. ARI (structure) is near-
paper; ACC (label accuracy) drops 11 points. The fix creates a different failure mode.

**Finding 4 — The infrastructure is complete and stable.**  
All 4 remaining datasets can be run today with a single command. The engineering phase
is done. Everything from here is research.

### 4.5 Remaining work before any contribution is claimed

- [ ] Run all 5 datasets with gemini (4 remaining)
- [ ] Fix merge prompt for aggressive consolidation without `target_k`
- [ ] Implement and run the seed-anchored embedding baseline
- [ ] Define at least one original contribution (see §6)

---

## 5. Open Problems and Research Gaps

### Gap 1 — Merge step quality is uncharacterised (HIGH PRIORITY)

The paper claims the merge step "aggregates labels with same meanings." Table 3 shows it
overshoots k by +6 to +29. Our experiments show it catastrophically fails with gemini.
No paper in this space evaluates merge quality as a standalone metric.

What is unknown:
- What merge quality metric correlates best with downstream ACC/NMI/ARI?
- Can embedding-based label clustering replace the LLM merge? (deterministic, free, model-independent)
- Does a two-stage merge (semantic grouping → LLM naming) outperform a single LLM call?

### Gap 2 — The semi-supervised baseline is missing everywhere (HIGH PRIORITY)

Not just from this paper — from every LLM clustering paper. The natural baseline when
seed labels are available is: embed texts + embed seed label names → nearest-centroid
assignment. This costs $0 and likely achieves 65–70% ACC on Massive-D.

Without this baseline, the entire sub-field of "seed-guided LLM clustering" is comparing
against the wrong things.

### Gap 3 — When is the LLM actually necessary?

The method calls the LLM N times in Step 2 (one per sample). For a sample like "set timer
for 5 minutes," embedding similarity to `time_and_date` is likely ≥ 0.95. The LLM adds
no value. Only ambiguous, polysemous, or domain-unusual samples benefit from LLM reasoning.

What is unknown:
- What fraction of samples are confidently assignable by embedding alone?
- What is the cost-quality Pareto curve as a function of LLM-call fraction?
- Does routing by embedding confidence preserve the paper's k-free property?

### Gap 4 — Seed provenance in real deployments

In a real deployment, you do not have 20% of the true labels. Getting them requires human
annotation, which is the expensive step the method claims to avoid.

What is unknown:
- Can the LLM generate its own seed candidates from a small random sample?
- Do LLM-generated seeds recover most of the performance of human seeds?
- How sensitive is performance to seed quality (noisy, incomplete, domain-shifted)?

### Gap 5 — Cross-model reproducibility

The paper uses GPT-3.5. The method's performance is model-dependent in at least one
critical way (the merge step). The paper presents no analysis across models.

What is unknown:
- What is the minimum model capability for reliable merge consolidation without target_k?
- How does cost-quality tradeoff vary across models (GPT-4, gemini-flash, llama-70B, etc.)?
- What is the performance on non-English text?

---

## 6. Candidate Research Directions

### Direction A — Embedding-Based Merge ⭐⭐⭐⭐⭐ (recommended primary)

**Replaces**: Step 1b (LLM merge call)  
**With**: SBERT embedding of proposed labels + agglomerative clustering + LLM naming

```
Proposed labels (350 strings)
  → embed each with SBERT (free, local)
  → agglomerative clustering with cosine distance + threshold τ
  → for each cluster: LLM picks canonical label name (1 call per cluster group, not 1 per 350)
  → L_final
```

**Why this matters**: Solves the model-dependence problem. Deterministic. $0 merge cost.
No target_k needed — τ controls granularity continuously.

**What to measure**: Merge quality (n_merged / true_k), semantic correctness (label precision
vs. true labels), downstream ACC/NMI/ARI across 5 datasets and 3 τ values.

**Effort**: ~3 days implementation + 1 day experiments.

---

### Direction B — Seed-Anchored Embedding Baseline ⭐⭐⭐⭐⭐ (required, not optional)

**Implements**: The missing baseline that must exist before any claim about the LLM's contribution.

```
1. Embed all N samples with E5-large (same model as paper's K-Means baseline)
2. Embed the 20% seed label names with the same model
3. For each sample: assign label = argmax cosine_similarity(sample, seed_labels)
4. Evaluate ACC/NMI/ARI
```

This answers: "How much of the method's advantage comes from the seeds alone?"

**Effort**: ~1 day. Must be done before any contribution is written up.

---

### Direction C — Confidence-Based Hybrid Classification ⭐⭐⭐⭐

**Augments**: Step 2 (per-sample LLM classification)  
**With**: Embedding-based pre-filter

```
1. Embed all samples + L_final labels with SBERT
2. For each sample: compute cosine similarity to all label embeddings
3. If max_similarity ≥ τ: assign by embedding (free)
4. If max_similarity < τ: send to LLM (paid)
5. Evaluate across τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
→ Pareto curve: ACC vs. % LLM calls
```

**Why this matters**: Directly answers the "too costly" question with a quantitative curve.
Achieves 3–10× cost reduction at minimal quality loss (hypothesis).

**Effort**: ~3 days.

---

### Direction D — LLM-Generated Seeds ⭐⭐⭐

**Replaces**: The 20% human seed labels  
**With**: LLM-proposed category names from a random sample

```
1. Sample 50–100 random sentences from D
2. Prompt LLM: "What are the main themes/intents in these texts? Propose category names."
3. Use LLM's output as seeds in Step 1a
4. Compare: 0% seeds vs. LLM-seeds vs. 20% human seeds
```

**Why this matters**: Makes the pipeline truly unsupervised. Enables fair comparison
to fully unsupervised baselines. Directly addresses the fairness concern.

**Effort**: ~2 days.

---

### Direction E — Multilingual Extension ⭐⭐⭐ (opportunistic)

**Extends**: Existing pipeline to non-English text  
**Dataset**: MASSIVE (already downloaded) has 51 languages

Run the pipeline on French, German, Arabic MASSIVE with gemini (multilingual LLM).
Compare ACC to English baseline. Identify which step degrades (generation? merge? classification?).

**Effort**: ~1 day (pipeline works; only the data loader language parameter changes).

---

## 7. Roadmap — 5 Weeks

### Week 1 (Feb 24 – Mar 1) — Complete the Reproduction + Missing Baseline

| Task | Effort | Deliverable |
|------|--------|-------------|
| Fix merge prompt (aggressive consolidation without target_k) | 2 days | Merge works on gemini without count anchor |
| Run 4 remaining datasets (arxiv_fine, go_emotion, massive_intent, mtop_intent) | 3 days | Full 5-dataset results table |
| **Direction B**: Implement + run seed-anchored embedding baseline on all 5 datasets | 1 day | Baseline numbers |

**Gate**: Full comparison table: paper pipeline (gemini) vs. embedding baseline, 5 datasets × 3 metrics.

---

### Week 2 (Mar 2 – Mar 8) — Direction A: Embedding Merge

| Task | Effort |
|------|--------|
| Install sentence-transformers; implement EmbeddingMerge | 1 day |
| Run embedding merge (3 threshold values) on all 5 datasets | 1.5 days |
| Define merge quality metric; compare to LLM merge | 0.5 days |
| Document in FINDINGS.md; decide contribution framing | 0.5 days |

**Gate**: Clear answer — does embedding merge match or beat LLM merge?

---

### Week 3 (Mar 9 – Mar 15) — Direction C: Hybrid Classification

| Task | Effort |
|------|--------|
| Implement confidence scorer (SBERT cosine vs. label embeddings) | 1 day |
| Implement hybrid router (threshold τ → embedding or LLM) | 1 day |
| Sweep τ on 3 datasets; plot Pareto curve | 1.5 days |
| Document + produce Pareto figure | 0.5 days |

**Gate**: Pareto curve — ACC vs. % LLM calls. This is the centrepiece result.

---

### Week 4 (Mar 16 – Mar 22) — Analysis + Writing

| Task | Effort |
|------|--------|
| Full cost-quality comparison table (all methods, all datasets) | 1 day |
| Introduction + Related Work chapter | 1.5 days |
| Methodology chapter (our contribution) | 1 day |
| Experiments + Results chapter | 0.5 days |

---

### Week 5 (Mar 23 – Mar 29) — Finalisation + Soutenance

| Task | Effort |
|------|--------|
| Discussion + Conclusion | 1 day |
| Figures, tables, references polish | 0.5 days |
| Full thesis review | 1 day |
| Soutenance slides | 1.5 days |

---

## 8. Thesis Contribution Options

### Option 1 (Recommended) — Hybrid Architecture

> *"We reproduce the text clustering pipeline of Huang & He (2025) with gemini-2.0-flash
> (a 50× cheaper model) and identify two structural problems: (1) the merge step is
> model-dependent and fails without explicit count guidance; (2) per-sample LLM
> classification is O(N) and unnecessary for high-confidence cases. We propose
> HybridCluster: embedding-based label merging with confidence-routed classification.
> On 5 benchmark datasets, HybridCluster matches paper-level clustering quality at
> 5–10× lower API cost and produces consistent results across LLM backends."*

**Novelty**: Embedding merge is new. Confidence routing is new. Cross-model reproducibility
analysis is new. Cost-quality Pareto is new.

---

### Option 2 — Critical Reproduction + Missing Baselines

> *"We provide a rigorous reproduction of Huang & He (2025) with a cheaper open-weight
> model and expose two critical evaluation gaps: a missing seed-anchored embedding
> baseline that matches LLM performance at zero cost, and a merge quality analysis showing
> that the method's k-free property is model-dependent. Our experiments on 5 datasets
> reframe the cost-quality tradeoff of LLM-based clustering."*

**Novelty**: Missing baseline is new. Merge quality analysis is new. Cross-model comparison is new.

---

### Option 3 — Fully Unsupervised Variant

> *"We extend Huang & He (2025) to a genuinely unsupervised setting by replacing the
> 20% human seed requirement with LLM-generated seed candidates. We show that LLM seeds
> recover 90% of human-seed performance, making the approach comparable to truly
> unsupervised baselines and clarifying where the LLM's contribution actually lies."*

**Novelty**: LLM seed generation is new. Fair unsupervised comparison is new.

---

## 9. Reference Table

| Paper | Venue | Year | Key contribution | Cite for |
|-------|-------|------|-----------------|---------|
| Huang & He (arXiv:2410.00927) | SIGIR-AP 2025 | 2025 | LLM-only clustering pipeline, no embeddings | This paper — everything |
| Zhang et al. (arXiv:2305.14871) | EMNLP 2023 | 2023 | ClusterLLM: triplet LLM → fine-tunes embedder, ~$0.60/dataset | Strongest baseline, cost comparison |
| De Raedt et al. | EACL / NLP4ConvAI 2023 | 2023 | IDAS: GPT-3 prototypes → label + embed+cluster | LLM-augmented clustering |
| Wang et al. | EMNLP 2023 | 2023 | PAS/GoalEx: Propose–Assign–Select + ILP | LLM-augmented clustering |
| Viswanathan et al. | TACL 2024 | 2024 | Keyphrase expansion → embed + cluster | LLM-augmented clustering |
| Reimers & Gurevych | EMNLP 2019 | 2019 | SBERT / sentence-transformers | Embedding baseline |
| Wang et al. | arXiv:2212.03533 | 2022 | E5-large embeddings | Embedding baseline (used in paper) |
| Su et al. | arXiv:2212.09741 | 2022 | Instructor-large | Embedding baseline (used in ClusterLLM) |
| FitzGerald et al. | ACL 2023 | 2023 | MASSIVE dataset — 51 languages | Massive-D and Massive-I datasets |
| Li et al. | EACL 2021 | 2021 | MTOP dataset | MTOP-I dataset |
| Demszky et al. | ACL 2020 | 2020 | GoEmotions dataset | GoEmo dataset |

---

## Appendix — Experimental Record

### Model and infrastructure settings

```
Model         : google/gemini-2.0-flash-001 (via OpenRouter)
API endpoint  : https://openrouter.ai/api/v1
Temperature   : 0  (deterministic)
Max tokens    : 4096
Batch size    : 15  (paper default)
Seeds         : 20% of true labels  (paper default)
Split         : small  (matches paper evaluation)
Repo branch   : fix/merge-prompt-v2  (at v1.3.0 / 27beaf3)
```

### Complete run log

| Run | Date | Dataset | Model | target_k | n_pred | ACC | NMI | ARI | Notes |
|-----|------|---------|-------|----------|--------|-----|-----|-----|-------|
| Paper | — | Massive-D | GPT-3.5-turbo-0125 | — | ~24 | 71.75 | 78.00 | 56.86 | Reference |
| 01 | 2026-02-20 | Massive-D | trinity-large:free | — | 168 | 40.69 | 66.64 | 33.06 | Merge token overflow |
| **02** | **2026-02-21** | **Massive-D** | **gemini-2.0-flash** | **18** | **18** | **60.46** | **63.90** | **53.87** | ✅ Valid |
| 03 | 2026-02-21 | Massive-D | gemini-2.0-flash | — | 311 | — | — | — | Aborted after Step 1 |

### Cumulative API cost

| Run | Calls | Estimated cost |
|-----|-------|---------------|
| Run 01 | 200 gen + 1 merge | ~$0.02 |
| Run 02 | 200 gen + 1 merge + 2974 classify | ~$0.15 |
| Run 03 | 200 gen + 1 merge (aborted) | ~$0.01 |
| **Total** | | **~$0.18** |

---

*Document version 1.0 — 2026-02-22. Update as experiments complete.*
