.DEFAULT_GOAL := help
.PHONY: help setup lint branch release run-step0 run-step1 run-step2 run-step3 run-kmedoids run-kmedoids-classify run-kmedoids-propagate

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  setup                              venv + install + git hooks"
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
	@echo "  Example (K-Medoids pipeline):"
	@echo "    make run-kmedoids data=massive_scenario k=100"
	@echo "    make run-step1 data=massive_scenario"
	@echo "    make run-kmedoids-classify data=massive_scenario run=./runs/<run_dir>"
	@echo "    make run-kmedoids-propagate data=massive_scenario run=./runs/<run_dir>"
	@echo "    make run-step3 data=massive_scenario run=./runs/<run_dir>"
	@echo ""

setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"
	.venv/bin/pre-commit install

lint:
	.venv/bin/ruff check .

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
	.venv/bin/cz bump
	$(eval NEW_TAG := $(shell git describe --tags --abbrev=0))
	git checkout main
	git merge --no-ff develop -m "release: merge develop into main for $(NEW_TAG)"
	git checkout develop
	git push origin main
	git push origin develop
	git push origin --tags

run-step0:
	.venv/bin/tc-seed-labels

# usage: make run-step1 data=massive_scenario
# Prints the created run_dir — copy it for use in steps 2 and 3.
run-step1:
ifndef data
	$(error data is required, e.g. make run-step1 data=massive_scenario)
endif
	mkdir -p logs
	.venv/bin/tc-label-gen --data $(data) 2>&1 | tee logs/$(data)_label_gen.log

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
	nohup .venv/bin/tc-classify --data $(data) --run_dir $(run) \
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
	.venv/bin/tc-evaluate --data $(data) --run_dir $(run)

# ── K-Medoids accelerated pipeline ──────────────────────────────────────

# usage: make run-kmedoids data=massive_scenario k=100
# Prints the created run_dir — copy it for use in subsequent steps.
k ?= 100
run-kmedoids:
ifndef data
	$(error data is required, e.g. make run-kmedoids data=massive_scenario k=100)
endif
	mkdir -p logs
	.venv/bin/tc-kmedoids --data $(data) --kmedoids_k $(k) 2>&1 | tee logs/$(data)_kmedoids.log

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
	nohup .venv/bin/tc-classify --data $(data) --run_dir $(run) --medoid_mode \
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
	.venv/bin/tc-kmedoids --data $(data) --run_dir $(run) --propagate
