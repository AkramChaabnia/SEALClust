## v1.7.0 (2026-03-23)

### Feat

- **sealclustv3**: adding the mode x and mode y in the pipeline

### Fix

- fixing the issue of labels by slightlighly modifing the prompt
- fixing the calculation of accuracy with GMM and Kmeans for mode x and mode y
- Creating two modes of label generation either with representative or with all docs
- fix the problem of generating less labels than the ground truth
- **docs & prompts**: correct model-name references and prompts typo (use gpt-3.5-turbo-0125 / gemini-2.0-flash)
- restore FINDINGS.md to clean state
- modifying the --target_k for the original implementation #9
- modifying the --target_k for the original implementation

## v1.6.0 (2026-03-19)

### Feat

- **labels**: rewritten the code so we using generated labels multiple times without calling llm each time if needed
- **original**: adding an optimized version of running the original baseline with less llm calls
- **graphclust**: propse LLM pairwise similarity graph clustering #7
- **graphclust**: propse LLM pairwise similarity graph clustering
- **graphclust**: propse LLM pairwise similarity graph clustering
- adding baseline and hybrid approach v3 #6
- **sealcust-v3**: we propose a new version of the hybrid approach v3 of sealclust and test it
- **baseline**: creating baseline test with Kmeans and GMM

### Fix

- fixing the length of lines in files to pass lint
- **checkpoint**: about checkpoint system to match all the pipelines
- fixing kmedoids problem in sklearn extra #5
- removing sklearn extra
- **kmedoids**: modifying the pipeline to work with the new implementation
- **kmedoid**: adding kmedoid implementation to fix the problem of sklearn extra

## v1.5.1 (2026-03-18)

### Fix

- **makefile**: support both conda and venv environments #4
- **lint**: resolve all 32 ruff errors across codebase
- **makefile**: support both conda and venv environments

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
