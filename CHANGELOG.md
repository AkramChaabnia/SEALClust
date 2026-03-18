## v1.5.0 (2026-03-18)

### Feat

- **visualization**: add post-run visualization module with confusion matrix, PCA/t-SNE/UMAP scatter plots, side-by-side cluster comparison, and cluster distribution charts
- **visualization**: auto-trigger visualizations after evaluation and full SEAL-Clust pipeline runs
- **visualization**: add `tc-visualize` CLI entry point for standalone use on existing runs
- **build**: add `matplotlib` and `umap-learn` as project dependencies

## v1.4.0 (2026-03-15)

### Feat

- **sealclust**: add SEAL-Clust 9-stage pipeline with embedding, PCA/t-SNE reduction, overclustering, K\* estimation, label discovery, consolidation, classification, and propagation
- **sealclust**: multi-method K\* estimation (BIC, silhouette-elbow, Calinski-Harabasz, ensemble with Kneedle elbow detection)
- **clustering**: add K-Medoids pre-clustering module with medoid extraction and label propagation
- **clustering**: add GMM clustering alternative to K-Medoids
- **prompts**: add seed-free label discovery and K\*-targeted consolidation prompts (`prompt_discover_labels`, `prompt_consolidate_labels`)
- **pipeline**: dual provider support, stage checkpoints, and results table output
- **tools**: add `remerge_labels` utility for re-running only the merge step with a target_k
- **docs**: consolidate research findings into unified FINDINGS.md with all experimental results (SC-01–SC-05, KM-01, KM-02, GMM-01)
- **docs**: update README with SEAL-Clust framework overview

### Fix

- **pipeline**: fix pipeline stage wiring and import issues
- **data**: fix dataset path `./dataset/` → `./datasets/` in seed_labels

## v1.3.0 (2026-02-21)

### Feat

- **model**: merge fix/model-gemini-flash into develop (gemini + target_k fix + Run 02/03 docs)
- **tools**: add tc-preflight pre-run check command
- **model**: switch primary model to google/gemini-2.0-flash-001

### Fix

- **prompts**: restore paper-faithful merge (no default target_k)
- merge run-02-prep into develop (token budget, target-k prompt, merge warning, per-step logs)
- **logging**: use separate log file per pipeline step
- **pipeline**: warn when merged label count exceeds 2x true class count
- **prompts**: add target_k to merge prompt to guide consolidation count
- **llm**: allow per-call max_tokens override; use 4096 for merge step

## v1.2.0 (2026-02-20)

### Feat

- **logging**: merge logging into develop
- **logging**: replace print calls with structured logging to run.log
- **runs**: merge run management into develop
- **runs**: add timestamped run dirs, checkpoint/resume, and results.json
- **package**: merge package restructure into develop
- **package**: restructure into text_clustering package with entry points

## v1.1.0 (2026-02-20)

### Feat

- **openrouter**: add probe_models CLI for model eligibility testing
- **openrouter**: add OpenRouter adapter and env template

### Fix

- **scripts**: rewrite given_label_classification to work with OpenRouter
- **scripts**: rewrite label_generation to work with OpenRouter
- **scripts**: fix dataset path resolution in select_part_labels

## v1.0.0 (2026-02-19)
