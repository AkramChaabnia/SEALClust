.DEFAULT_GOAL := help
.PHONY: help setup setup-conda lint branch release run-step0 run-step1 run-step2 run-step3 run-kmedoids run-kmedoids-classify run-kmedoids-propagate run-gmm run-gmm-classify run-gmm-propagate run-sealclust run-sealclust-classify run-sealclust-propagate

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
	@echo "  run-step0                          seed labels (run once)"
	@echo "  run-step1 data=<dataset>           label generation — prints the run directory"
	@echo "  run-step2 data=<d> run=<run_dir>   classification (background, resumes on restart)"
	@echo "  run-step3 data=<d> run=<run_dir>   evaluation → results.json"
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
	$(BIN)tc-label-gen --data $(data) 2>&1 | tee logs/$(data)_label_gen.log

# usage: make run-step2 data=massive_scenario run=./runs/massive_scenario_small_20260220_143012
# Runs in the background; resumes automatically if a checkpoint.json exists in run_dir.
run-step2:
ifndef data
	$(error data is required)
endif
ifndef run
	$(error run is required, e.g. run=./runs/massive_scenario_small_20260220_143012)
endif
	mkdir -p logs
	nohup $(BIN)tc-classify --data $(data) --run_dir $(run) \
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
	$(BIN)tc-sealclust --data $(data) --k0 $(k0) --k_star $(kstar) --k_method $(kmethod) 2>&1 | tee logs/$(data)_sealclust.log

# usage: make run-sealclust-full data=massive_scenario k0=300
#        make run-sealclust-full data=massive_scenario k0=300 kstar=18
# Runs the entire SEALClust pipeline end-to-end: Stages 1-9 + evaluation.
run-sealclust-full:
ifndef data
	$(error data is required, e.g. make run-sealclust-full data=massive_scenario)
endif
	mkdir -p logs
	$(BIN)tc-sealclust --data $(data) --k0 $(k0) --k_star $(kstar) --k_method $(kmethod) --full 2>&1 | tee logs/$(data)_sealclust_full.log

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
