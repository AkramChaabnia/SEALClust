.DEFAULT_GOAL := help
.PHONY: help setup setup-conda lint branch release run-step0 run-step1 run-step2 run-step3 run-kmedoids run-kmedoids-classify run-kmedoids-propagate run-gmm run-gmm-classify run-gmm-propagate run-sealclust run-sealclust-classify run-sealclust-propagate run-hybrid run-hybrid-full run-baseline-kmeans run-baseline-gmm run-graphclust run-graphclust-full run-sealclust-v3 run-sealclust-v3-full run-sealclust-v3-classify run-sealclust-v3-propagate analyze-datasets

# ── Environment auto-detection ──────────────────────────────────────────
# Priority: $(CONDA_PREFIX)/bin/ (if active) → .venv/bin/ → bare (system PATH)
ifdef CONDA_PREFIX
  BIN := $(CONDA_PREFIX)/bin/
else ifneq ($(wildcard .venv/bin/),)
  BIN := .venv/bin/
else
  BIN :=
endif

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  Environment: $(if $(BIN),$(BIN),system PATH)"
	@echo ""
	@echo "  setup                              venv + install + git hooks  (uv)"
	@echo "  setup-conda                        install into active conda env (pip)"
	@echo "  lint                               ruff check"
	@echo "  branch name=<n> type=<t>           create and push a branch off develop"
	@echo "  release                            bump version, merge develop->main, push tags"
	@echo ""
	@echo "  ── Original pipeline ──"
	@echo "  run-step0                              seed labels (run once)"
	@echo "  run-step1 data=<dataset>               label generation — prints the run directory"
	@echo "            reuse_labels=1                reuse cached labels (see Section 8 in README)"
	@echo "  run-step2 data=<d> run=<run_dir>       classification (background, resumes on restart)"
	@echo "            classify_batch=10             batched mode: 10× fewer LLM calls"
	@echo "  run-step3 data=<d> run=<run_dir>       evaluation → results.json"
	@echo ""
	@echo "  ── K-Medoids accelerated pipeline ──"
	@echo "  run-kmedoids data=<d> [k=100]      pre-cluster → medoid_documents.jsonl"
	@echo "  run-kmedoids-classify data=<d> run=<run_dir>"
	@echo "                                     classify medoids only (--medoid_mode)"
	@echo "  run-kmedoids-propagate data=<d> run=<run_dir>"
	@echo "                                     propagate medoid labels → full dataset"
	@echo ""
	@echo "  ── GMM accelerated pipeline ──"
	@echo "  run-gmm data=<d> [k=100]           pre-cluster → representative_documents.jsonl"
	@echo "  run-gmm-classify data=<d> run=<run_dir>"
	@echo "                                     classify representatives (--representative_mode)"
	@echo "  run-gmm-propagate data=<d> run=<run_dir>"
	@echo "                                     propagate labels → full dataset"
	@echo ""
	@echo "  ── SEAL-Clust full framework ──"
	@echo "  run-sealclust data=<d> [k0=300] [kstar=0] [kmethod=silhouette]"
	@echo "                                     Stages 1–7: embed + PCA + overcluster + labels + K* + consolidate"
	@echo "                                     kmethod: silhouette (default) | calinski | bic | ensemble"
	@echo "                                     k0=N → overclustering size, kstar=N → manual K*"
	@echo "                                     reuse_labels=1 → reuse cached labels"
	@echo "  run-sealclust-full data=<d> [k0=300] [kstar=0] [kmethod=silhouette]"
	@echo "                                     Stages 1–9 + evaluation in one command"
	@echo "  run-sealclust-classify data=<d> run=<run_dir>"
	@echo "                                     classify prototypes (--medoid_mode)"
	@echo "  run-sealclust-propagate data=<d> run=<run_dir>"
	@echo "                                     propagate labels → full dataset"
	@echo ""
	@echo "  Example (K-Medoids pipeline):"
	@echo "    make run-kmedoids data=massive_scenario k=100"
	@echo "    make run-step1 data=massive_scenario"
	@echo "    make run-kmedoids-classify data=massive_scenario run=./runs/<run_dir>"
	@echo "    make run-kmedoids-propagate data=massive_scenario run=./runs/<run_dir>"
	@echo "    make run-step3 data=massive_scenario run=./runs/<run_dir>"
	@echo ""
	@echo "  Example (GMM pipeline):"
	@echo "    make run-gmm data=massive_scenario k=100"
	@echo "    make run-step1 data=massive_scenario"
	@echo "    make run-gmm-classify data=massive_scenario run=./runs/<run_dir>"
	@echo "    make run-gmm-propagate data=massive_scenario run=./runs/<run_dir>"
	@echo "    make run-step3 data=massive_scenario run=./runs/<run_dir>"
	@echo ""
	@echo "  Example (SEAL-Clust pipeline, full end-to-end):"
	@echo "    make run-sealclust-full data=massive_scenario"
	@echo "    make run-sealclust-full data=massive_scenario k0=200 kmethod=ensemble"
	@echo "    make run-sealclust-full data=massive_scenario kstar=18"
	@echo ""
	@echo "  ── Hybrid Pipeline (LLM + Embedding) ──"
	@echo "  run-hybrid data=<d> [step=N]       Single step 1-8 or steps 1-5 (default)"
	@echo "                                     step=N → run only step N; cont=<dir> → resume"
	@echo "                                     reuse_labels=1 → reuse cached labels"
	@echo "  run-hybrid-full data=<d>            Full pipeline: steps 1-8 + evaluation"
	@echo "                                     hybrid_p=0.1 hybrid_k_min=2 hybrid_k_max=50"
	@echo "                                     target_k=N → override automatic K optimisation"
	@echo ""
	@echo "  ── Baselines (no LLM) ──"
	@echo "  run-baseline-kmeans data=<d> k=<K>  KMeans baseline (auto_k=1 for silhouette sweep)"
	@echo "  run-baseline-gmm data=<d> k=<K>    GMM baseline   (auto_k=1 for BIC sweep)"
	@echo "                                     pca=50 → optional PCA pre-reduction"
	@echo ""
	@echo "  Example (Hybrid pipeline, full end-to-end):"
	@echo "    make run-hybrid-full data=massive_scenario"
	@echo "    make run-hybrid-full data=massive_scenario hybrid_p=0.15 hybrid_k_max=40"
	@echo "    make run-hybrid data=massive_scenario step=4   # run only step 4"
	@echo ""
	@echo "  Example (Baselines):"
	@echo "    make run-baseline-kmeans data=massive_scenario k=18"
	@echo "    make run-baseline-gmm data=massive_scenario auto_k=1 k_min=5 k_max=30"
	@echo ""
	@echo "  ── Graph Community Clustering (Mode H) ──"
	@echo "  run-graphclust data=<d> [target_k=N]"
	@echo "                                     Steps 1–2: k-NN graph + Louvain communities"
	@echo "  run-graphclust-full data=<d> [target_k=N]"
	@echo "                                     Steps 1–3 + evaluation end-to-end"
	@echo "                                     knn=15 min_sim=0.3 resolution=1.0"
	@echo ""
	@echo "  Example (Graph Community Clustering):"
	@echo "    make run-graphclust-full data=massive_scenario target_k=18"
	@echo "    make run-graphclust-full data=massive_scenario knn=20 resolution=1.5"
	@echo "    make run-graphclust data=massive_scenario"
	@echo ""
	@echo "  ── SEAL-Clust v3 (Modes Y/Z) ──"
	@echo "  run-sealclust-v3 data=<d> [k0=300] [kstar=0] [kmethod=silhouette] [cluster_method=kmedoids]"
	@echo "                                     Stages 1–7: embed + overcluster + labels + K* + consolidate"
	@echo "                                     cluster_method: kmedoids (default) | gmm | kmeans"
	@echo "                                     v3_reduction: none (default) | pca | tsne"
	@echo "                                     label_source: all (default) | representatives"
	@echo "  run-sealclust-v3-full data=<d> [k0=300] [kstar=0] [kmethod=silhouette] [cluster_method=kmedoids]"
	@echo "                                     Stages 1–9 + evaluation in one command (Mode Z)"
	@echo "  run-sealclust-v3-classify data=<d> run=<run_dir>"
	@echo "                                     Stage 8: classify representatives via LLM"
	@echo "  run-sealclust-v3-propagate data=<d> run=<run_dir>"
	@echo "                                     Stage 9: propagate labels → full dataset"
	@echo ""
	@echo "  Example (SEAL-Clust v3, full end-to-end):"
	@echo "    make run-sealclust-v3-full data=massive_scenario"
	@echo "    make run-sealclust-v3-full data=massive_scenario kstar=18"
	@echo "    make run-sealclust-v3-full data=massive_scenario cluster_method=gmm k0=200"
	@echo "    make run-sealclust-v3-full data=massive_scenario v3_reduction=pca v3_pca_dims=50"
	@echo "    make run-sealclust-v3-full data=massive_scenario label_source=representatives"
	@echo "    make run-sealclust-v3 data=massive_scenario kstar=18"
	@echo ""
	@echo "  ── Dataset Analysis ──"
	@echo "  analyze-datasets                   Profile all datasets → assets/*.json"
	@echo "  analyze-datasets data=<d>          Profile a single dataset"
	@echo ""

setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"
	.venv/bin/pre-commit install

setup-conda:
	@[ -n "$(CONDA_PREFIX)" ] || (echo "error: no conda env is active — run 'conda activate <env>' first" && exit 1)
	pip install -e ".[dev]"
	$(BIN)pre-commit install

lint:
	$(BIN)ruff check .

# usage: make branch name=openrouter-retry type=fix
branch:
ifndef name
	$(error name is required, e.g. make branch name=my-feature type=feature)
endif
ifndef type
	$(error type is required: feature | fix | docs)
endif
	git checkout develop
	git pull origin develop
	git checkout -b $(type)/$(name)
	git push -u origin $(type)/$(name)

release:
	@[ "$$(git branch --show-current)" = "develop" ] || \
		(echo "error: must be on develop" && exit 1)
	$(BIN)cz bump
	$(eval NEW_TAG := $(shell git describe --tags --abbrev=0))
	git checkout main
	git merge --no-ff develop -m "release: merge develop into main for $(NEW_TAG)"
	git checkout develop
	git push origin main
	git push origin develop
	git push origin --tags

run-step0:
	$(BIN)tc-seed-labels

# usage: make run-step1 data=massive_scenario
# Prints the created run_dir — copy it for use in steps 2 and 3.
run-step1:
ifndef data
	$(error data is required, e.g. make run-step1 data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-label-gen --data $(data) $(if $(reuse_labels),--reuse_labels,) 2>&1 | tee logs/$(data)_label_gen.log

# usage: make run-step2 data=massive_scenario run=./runs/massive_scenario_small_20260220_143012
# Runs in the background; resumes automatically if a checkpoint.json exists in run_dir.
classify_batch ?= 1
run-step2:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260220_143012)
endif
	mkdir -p logs
	nohup $(BIN)tc-classify --data $(data) --run_dir $(run) --batch_size $(classify_batch) \
		>> logs/$(data)_classification.log 2>&1 &
	@echo "running in background — tail -f logs/$(data)_classification.log"
	@echo "to resume after interruption, re-run the same command"

# usage: make run-step3 data=massive_scenario run=./runs/massive_scenario_small_20260220_143012
run-step3:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260220_143012)
endif
	$(BIN)tc-evaluate --data $(data) --run_dir $(run)

# ── K-Medoids accelerated pipeline ──────────────────────────────────────

# usage: make run-kmedoids data=massive_scenario k=100
# Prints the created run_dir — copy it for use in subsequent steps.
k ?= 100
run-kmedoids:
ifndef data
	$(error data is required, e.g. make run-kmedoids data=massive_scenario k=100)
endif
	mkdir -p logs
	$(BIN)tc-kmedoids --data $(data) --kmedoids_k $(k) 2>&1 | tee logs/$(data)_kmedoids.log

# usage: make run-kmedoids-classify data=massive_scenario run=./runs/<run_dir>
# Classifies only the medoid documents (uses --medoid_mode).
run-kmedoids-classify:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260220_143012)
endif
	mkdir -p logs
	nohup $(BIN)tc-classify --data $(data) --run_dir $(run) --medoid_mode \
		>> logs/$(data)_kmedoids_classification.log 2>&1 &
	@echo "running in background — tail -f logs/$(data)_kmedoids_classification.log"

# usage: make run-kmedoids-propagate data=massive_scenario run=./runs/<run_dir>
# Propagates medoid labels to the full dataset.
run-kmedoids-propagate:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260220_143012)
endif
	$(BIN)tc-kmedoids --data $(data) --run_dir $(run) --propagate

# ── GMM accelerated pipeline ────────────────────────────────────────────

# usage: make run-gmm data=massive_scenario k=100
run-gmm:
ifndef data
	$(error data is required, e.g. make run-gmm data=massive_scenario k=100)
endif
	mkdir -p logs
	$(BIN)tc-gmm --data $(data) --gmm_k $(k) 2>&1 | tee logs/$(data)_gmm.log

# usage: make run-gmm-classify data=massive_scenario run=./runs/<run_dir>
run-gmm-classify:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260313_...)
endif
	mkdir -p logs
	nohup $(BIN)tc-classify --data $(data) --run_dir $(run) --representative_mode \
		>> logs/$(data)_gmm_classification.log 2>&1 &
	@echo "running in background — tail -f logs/$(data)_gmm_classification.log"

# usage: make run-gmm-propagate data=massive_scenario run=./runs/<run_dir>
run-gmm-propagate:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260313_...)
endif
	$(BIN)tc-gmm --data $(data) --run_dir $(run) --propagate

# ── SEAL-Clust full framework ───────────────────────────────────────────

# usage: make run-sealclust data=massive_scenario
#        make run-sealclust data=massive_scenario k0=200  (custom K₀)
#        make run-sealclust data=massive_scenario kstar=18 (manual K*)
#        make run-sealclust data=massive_scenario kmethod=ensemble
# Default: Stages 1–7 (Embed + PCA + Overcluster + Label Discovery + K* + Consolidate)
k0 ?= 300
kstar ?= 0
kmethod ?= silhouette
run-sealclust:
ifndef data
	$(error data is required, e.g. make run-sealclust data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-sealclust --data $(data) --k0 $(k0) --k_star $(kstar) --k_method $(kmethod) $(if $(reuse_labels),--reuse_labels,) 2>&1 | tee logs/$(data)_sealclust.log

# usage: make run-sealclust-full data=massive_scenario k0=300
#        make run-sealclust-full data=massive_scenario k0=300 kstar=18
# Runs the entire SEALClust pipeline end-to-end: Stages 1-9 + evaluation.
run-sealclust-full:
ifndef data
	$(error data is required, e.g. make run-sealclust-full data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-sealclust --data $(data) --k0 $(k0) --k_star $(kstar) --k_method $(kmethod) --full $(if $(reuse_labels),--reuse_labels,) 2>&1 | tee logs/$(data)_sealclust_full.log

# usage: make run-sealclust-classify data=massive_scenario run=./runs/<run_dir>
run-sealclust-classify:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260313_...)
endif
	mkdir -p logs
	nohup $(BIN)tc-classify --data $(data) --run_dir $(run) --medoid_mode \
		>> logs/$(data)_sealclust_classification.log 2>&1 &
	@echo "running in background — tail -f logs/$(data)_sealclust_classification.log"

# usage: make run-sealclust-propagate data=massive_scenario run=./runs/<run_dir>
run-sealclust-propagate:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260313_...)
endif
	$(BIN)tc-sealclust --data $(data) --run_dir $(run) --propagate

# ── Hybrid Pipeline (Mode F) ────────────────────────────────────────────

# usage: make run-hybrid data=massive_scenario
#        make run-hybrid data=massive_scenario step=4        (single step)
#        make run-hybrid data=massive_scenario cont=./runs/<run_dir>  (resume)
#        make run-hybrid data=massive_scenario p=0.15 k_min=3 k_max=40
# Default: Steps 1–5 (Label Generation + Embed + Reduce + Optimise K + Align)
hybrid_p         ?= 0.1
hybrid_k_min     ?= 2
hybrid_k_max     ?= 50
hybrid_batch     ?= 30
hybrid_cov       ?= full
run-hybrid:
ifndef data
	$(error data is required, e.g. make run-hybrid data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-hybrid --data $(data) \
		$(if $(step),--step $(step),) \
		$(if $(cont),--continue_from $(cont),) \
		--p $(hybrid_p) --k_min $(hybrid_k_min) --k_max $(hybrid_k_max) \
		--llm_batch_size $(hybrid_batch) --covariance_type $(hybrid_cov) \
		$(if $(reuse_labels),--reuse_labels,) \
		2>&1 | tee logs/$(data)_hybrid.log

# usage: make run-hybrid-full data=massive_scenario
#        make run-hybrid-full data=massive_scenario p=0.15 k_min=3 k_max=40
#        make run-hybrid-full data=massive_scenario target_k=18
# Runs the entire hybrid pipeline end-to-end: Steps 1-8 + evaluation.
run-hybrid-full:
ifndef data
	$(error data is required, e.g. make run-hybrid-full data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-hybrid --data $(data) --full \
		--p $(hybrid_p) --k_min $(hybrid_k_min) --k_max $(hybrid_k_max) \
		--llm_batch_size $(hybrid_batch) --covariance_type $(hybrid_cov) \
		$(if $(target_k),--target_k $(target_k),) \
		$(if $(reuse_labels),--reuse_labels,) \
		2>&1 | tee logs/$(data)_hybrid_full.log

# ── Baselines (Mode G) ──────────────────────────────────────────────────

# usage: make run-baseline-kmeans data=massive_scenario k=18
#        make run-baseline-kmeans data=massive_scenario auto_k=1 k_min=5 k_max=30
#        make run-baseline-kmeans data=massive_scenario k=18 pca=50
baseline_k_min   ?= 2
baseline_k_max   ?= 50
run-baseline-kmeans:
ifndef data
	$(error data is required, e.g. make run-baseline-kmeans data=massive_scenario k=18)
endif
	mkdir -p logs
	$(BIN)tc-baseline --data $(data) --method kmeans \
		$(if $(k),--k $(k),) \
		$(if $(auto_k),--auto_k,) \
		$(if $(pca),--pca_dims $(pca),) \
		--k_min $(baseline_k_min) --k_max $(baseline_k_max) \
		2>&1 | tee logs/$(data)_baseline_kmeans.log

# usage: make run-baseline-gmm data=massive_scenario k=18
#        make run-baseline-gmm data=massive_scenario auto_k=1 k_min=5 k_max=30
#        make run-baseline-gmm data=massive_scenario k=18 pca=50 cov=diag
baseline_cov     ?= full
run-baseline-gmm:
ifndef data
	$(error data is required, e.g. make run-baseline-gmm data=massive_scenario k=18)
endif
	mkdir -p logs
	$(BIN)tc-baseline --data $(data) --method gmm \
		$(if $(k),--k $(k),) \
		$(if $(auto_k),--auto_k,) \
		$(if $(pca),--pca_dims $(pca),) \
		--k_min $(baseline_k_min) --k_max $(baseline_k_max) \
		--covariance_type $(baseline_cov) \
		2>&1 | tee logs/$(data)_baseline_gmm.log

# ── Graph Community Clustering (Mode H) ─────────────────────────────

# usage: make run-graphclust data=massive_scenario
#        make run-graphclust data=massive_scenario target_k=18
# Default: Steps 1–2 (k-NN Graph + Louvain Community Detection)
graph_knn    ?= 15
min_sim      ?= 0.3
resolution   ?= 1.0
samples_per  ?= 8
run-graphclust:
ifndef data
	$(error data is required, e.g. make run-graphclust data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-graphclust --data $(data) \
		--knn $(graph_knn) --min_similarity $(min_sim) \
		--resolution $(resolution) \
		--samples_per_community $(samples_per) \
		$(if $(target_k),--target_k $(target_k),) \
		2>&1 | tee logs/$(data)_graphclust.log

# usage: make run-graphclust-full data=massive_scenario target_k=18
#        make run-graphclust-full data=massive_scenario knn=20 resolution=1.5
# Runs the entire pipeline end-to-end: Steps 1-3 + evaluation.
run-graphclust-full:
ifndef data
	$(error data is required, e.g. make run-graphclust-full data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-graphclust --data $(data) --full \
		--knn $(graph_knn) --min_similarity $(min_sim) \
		--resolution $(resolution) \
		--samples_per_community $(samples_per) \
		$(if $(target_k),--target_k $(target_k),) \
		2>&1 | tee logs/$(data)_graphclust_full.log

# ── SEAL-Clust v3 (Modes Y/Z) ──────────────────────────────────────

# usage: make run-sealclust-v3 data=massive_scenario
#        make run-sealclust-v3 data=massive_scenario k0=200 kstar=18
#        make run-sealclust-v3 data=massive_scenario cluster_method=gmm
#        make run-sealclust-v3 data=massive_scenario reduction=pca pca_dims=50
#        make run-sealclust-v3 data=massive_scenario label_source=representatives
# Default: Stages 1–7 (Embed + Overcluster + Labels + K* + Consolidate)
# No dimensionality reduction by default (raw 384D embeddings).
cluster_method ?= kmedoids
v3_classify_batch ?= 20
v3_reduction ?= none
v3_pca_dims ?= 50
label_source ?= all
run-sealclust-v3:
ifndef data
	$(error data is required, e.g. make run-sealclust-v3 data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-sealclust-v3 --data $(data) --k0 $(k0) --k_star $(kstar) --k_method $(kmethod) \
		--cluster_method $(cluster_method) --reduction $(v3_reduction) \
		$(if $(filter-out none,$(v3_reduction)),--pca_dims $(v3_pca_dims),) \
		--label_source $(label_source) \
		$(if $(reuse_labels),--reuse_labels,) \
		2>&1 | tee logs/$(data)_sealclust_v3.log

# usage: make run-sealclust-v3-full data=massive_scenario
#        make run-sealclust-v3-full data=massive_scenario k0=300 kstar=18 cluster_method=gmm
#        make run-sealclust-v3-full data=massive_scenario v3_reduction=pca v3_pca_dims=50
# Runs the entire v3 pipeline end-to-end: Stages 1-9 + evaluation.
run-sealclust-v3-full:
ifndef data
	$(error data is required, e.g. make run-sealclust-v3-full data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-sealclust-v3 --data $(data) --k0 $(k0) --k_star $(kstar) --k_method $(kmethod) \
		--cluster_method $(cluster_method) --classify_batch_size $(v3_classify_batch) \
		--reduction $(v3_reduction) \
		$(if $(filter-out none,$(v3_reduction)),--pca_dims $(v3_pca_dims),) \
		--label_source $(label_source) \
		--full \
		$(if $(reuse_labels),--reuse_labels,) \
		2>&1 | tee logs/$(data)_sealclust_v3_full.log

# usage: make run-sealclust-v3-classify data=massive_scenario run=./runs/<run_dir>
run-sealclust-v3-classify:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_v3_20260323_...)
endif
	mkdir -p logs
	$(BIN)tc-sealclust-v3 --data $(data) --run_dir $(run) --classify \
		--classify_batch_size $(v3_classify_batch) \
		2>&1 | tee logs/$(data)_sealclust_v3_classify.log

# usage: make run-sealclust-v3-propagate data=massive_scenario run=./runs/<run_dir>
run-sealclust-v3-propagate:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_v3_20260323_...)
endif
	$(BIN)tc-sealclust-v3 --data $(data) --run_dir $(run) --propagate

# ── Dataset Analysis ─────────────────────────────────────────────────────
# usage: make analyze-datasets
#        make analyze-datasets data=arxiv_fine     # single dataset
analyze-datasets:
	$(BIN)tc-analyze-datasets $(if $(data),--dataset $(data),)
