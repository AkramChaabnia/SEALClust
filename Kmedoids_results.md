# K-Medoids Pre-Clustering — Results

> **Branch**: `feature/kmedoids`
> **Goal**: Reduce LLM API calls by ~30× via K-Medoids document compression while preserving clustering quality.

---

## Paper Baseline (Table 2 — `gpt-3.5-turbo-0125`, 20% seed labels, batch=15)

| Dataset | ACC | NMI | ARI |
|---------|-----|-----|-----|
| `massive_scenario` | 71.75 | 78.00 | 56.86 |
| `massive_intent` | 64.12 | 65.44 | 48.92 |
| `go_emotion` | 31.66 | 27.39 | 13.50 |
| `arxiv_fine` | 38.78 | 57.43 | 20.55 |
| `mtop_intent` | 72.18 | 78.78 | 71.93 |

---

## Previous Runs (Original Pipeline — No K-Medoids)

### `massive_scenario` · small split

| Run | Model | Pipeline | target_k | n_pred | ACC | NMI | ARI | LLM Calls | Status |
|-----|-------|----------|----------|--------|-----|-----|-----|-----------|--------|
| Paper | `gpt-3.5-turbo-0125` | Original | implicit | ~18 | **71.75** | **78.00** | **56.86** | ~3,000 | Reference |
| Run 01 | `trinity-large-preview:free` | Original | — | 168 | 40.69 | 66.64 | 33.06 | ~3,000 | ❌ Broken merge |
| Run 02 | `gemini-2.0-flash-001` | Original | 18 | 18 | 60.46 | 63.90 | 53.87 | ~3,000 | ✅ Valid |
| Run 03 | `gemini-2.0-flash-001` | Original | none | — | — | — | — | — | ⚠️ Merge failed |

---

## K-Medoids Runs

### Configuration

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` |
| K-Medoids k | 100 / 300 |
| Metric | cosine |
| Init | k-medoids++ |
| Random seed | 42 |

### `massive_scenario` · small split (2,974 docs)

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|--------|
| KM-01 | `gpt-4o-mini` | 100 | 19 | 54.98 | 57.78 | 41.66 | ~300 | ~10× | ✅ Done |
| KM-02 | `gpt-4o-mini` | 300 | 20 | 55.21 | 57.25 | 39.85 | ~500 | ~6× | ✅ Done |

> **KM-01 notes**: Label generation ran on full 2974 docs (not medoids only), producing 715 proposed labels.
> Re-merged with `target_k=18` → 19 labels. Classification ran on 100 medoids only (181s).
> 23/2974 docs (0.77%) received no label during propagation.
> Run dir: `./runs/massive_scenario_small_20260312_112628`
>
> **KM-02 notes**: k=300 (9.9× reduction). Label generation on 300 medoid docs produced 683 proposed labels.
> Re-merged with `target_k=18` → 19 labels (20 predicted clusters incl. "Unsuccessful").
> Classification ran on 300 medoids (1161.3s, includes ~10min API retry pause).
> 11/2974 docs (0.37%) received no label during propagation.
> Run dir: `./runs/massive_scenario_small_20260312_120831`

### `massive_intent` · small split (2,974 docs → 100 medoids)

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|--------|
| KM-01 | `gpt-4o-mini` | 100 | — | — | — | — | ~100 | ~30× | ⏳ Pending |

### `go_emotion` · small split (5,940 docs → 100 medoids)

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|--------|
| KM-01 | `gpt-4o-mini` | 100 | — | — | — | — | ~100 | ~59× | ⏳ Pending |

### `arxiv_fine` · small split (3,674 docs → 100 medoids)

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|--------|
| KM-01 | `gpt-4o-mini` | 100 | — | — | — | — | ~100 | ~37× | ⏳ Pending |

### `mtop_intent` · small split (4,386 docs → 100 medoids)

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|--------|
| KM-01 | `gpt-4o-mini` | 100 | — | — | — | — | ~100 | ~44× | ⏳ Pending |

---

## Summary Comparison (once runs are complete)

| Dataset | Paper ACC | Paper NMI | Paper ARI | KM ACC | KM NMI | KM ARI | LLM Calls (Original → KM) |
|---------|-----------|-----------|-----------|--------|--------|--------|--------------------------|
| `massive_scenario` (k=100) | 71.75 | 78.00 | 56.86 | **54.98** | **57.78** | **41.66** | 3,000 → ~300 |
| `massive_scenario` (k=300) | 71.75 | 78.00 | 56.86 | **55.21** | **57.25** | **39.85** | 3,000 → ~500 |
| `massive_intent` | 64.12 | 65.44 | 48.92 | — | — | — | 3,000 → 100 |
| `go_emotion` | 31.66 | 27.39 | 13.50 | — | — | — | 5,940 → 100 |
| `arxiv_fine` | 38.78 | 57.43 | 20.55 | — | — | — | 3,674 → 100 |
| `mtop_intent` | 72.18 | 78.78 | 71.93 | — | — | — | 4,386 → 100 |

---

## How to Run

```bash
# 1. Activate environment
conda activate ppd

# 2. Pre-cluster (embedding + K-Medoids)
tc-kmedoids --data massive_scenario --kmedoids_k 100
# → prints run_dir, e.g. ./runs/massive_scenario_small_20260312_...

# 3. Seed labels (if not already done)
tc-seed-labels

# 4. Label generation (writes labels_merged.json into the same run_dir)
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>
# If merge produces too many labels, re-merge with target_k:
# python tools/remerge_labels.py ./runs/<run_dir> 18

# 5. Classify medoids only (saves checkpoint after EVERY sample)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# 6. Propagate labels to full dataset
tc-kmedoids --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# 7. Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

### Resume after interruption

All expensive steps have checkpoints:
- **Embeddings**: cached as `embeddings.npy` — pass `--run_dir` to reuse
- **K-Medoids metadata**: cached as `kmedoids_metadata.json`
- **Classification**: checkpoint after every API call in medoid mode

```bash
# Re-run the same command — it picks up from where it stopped
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode
```
