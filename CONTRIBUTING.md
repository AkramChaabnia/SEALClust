# Contributing Guide

## Branching Strategy

A simplified GitFlow adapted to a research project:

```
main ──────────────────────────────────────── (stable, tagged releases)
  └── develop ────────────────────────────── (integration branch)
        ├── feature/<description>
        ├── fix/<description>
        └── docs/<description>
```

### Branch rules

| Branch | Purpose | Merge into | PR required? |
|--------|---------|------------|:------------:|
| `main` | Stable snapshots, tagged releases only | — | Yes (from `develop` only) |
| `develop` | Integration — all feature/fix/docs branches merge here | `main` | Yes |
| `feature/<desc>` | New feature or experiment | `develop` | Yes |
| `fix/<desc>` | Bug fix | `develop` | Yes |
| `docs/<desc>` | Documentation only | `develop` | Yes |

> Branches are **not deleted after merge** — they stay visible on remote so the development history is browsable on GitHub.

---

## Commit Messages — Conventional Commits via Commitizen

We use [Conventional Commits](https://www.conventionalcommits.org/) enforced by [Commitizen](https://commitizen-tools.github.io/commitizen/).

### Interactive commit (recommended)

```bash
cz commit
```

### Manual format

```
<type>(<scope>): <short description>

[optional body]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or experiment step |
| `fix` | Bug fix |
| `build` | Dependencies, tooling, packaging |
| `ci` | CI/CD configuration |
| `docs` | Documentation only |
| `refactor` | Code change with no behavior change |
| `test` | Adding or fixing tests |
| `chore` | Misc housekeeping |
| `experiment` | Experimental run / results |

### Examples

```
feat(label-gen): add retry logic for API rate limits
fix(evaluate): handle missing predictions in scoring
experiment(baseline): run massive_scenario with trinity-large-preview
docs(findings): add probe results for all tested models
build(deps): upgrade openai to 1.30
```

---

## Versioning & Releases

Versioning follows [Semantic Versioning](https://semver.org/). The version in `pyproject.toml` is the single source of truth — git tags are derived from it.

| Commit type | Version bump |
|-------------|-------------|
| `fix:`, `docs:`, `build:`, `ci:`, `chore:` | patch (`1.0.0` → `1.0.1`) |
| `feat:`, `experiment:` | minor (`1.0.0` → `1.1.0`) |
| `feat!:` or `BREAKING CHANGE` in body | major (`1.0.0` → `2.0.0`) |

To cut a new release:

```bash
# 1. On develop, bump version — updates pyproject.toml, CHANGELOG.md, creates commit + tag
cz bump

# 2. Merge develop into main
git checkout main
git merge --no-ff develop -m "release: merge develop into main for vX.Y.Z"

# 3. Push everything
git push origin main && git push origin develop && git push origin --tags
```

Or with the Makefile shortcut (does all three steps):

```bash
make release
```

---

## Pull Request Process

1. Branch from `develop`:

```bash
git checkout develop && git pull origin develop
git checkout -b <type>/<name>      # e.g. fix/label-gen-retry
git push -u origin <type>/<name>
```

Or: `make branch name=label-gen-retry type=fix`

2. Fill in the PR template — include the **Experimental results** table if your PR contains a run
3. Check lint locally: `ruff check .` (or `make lint`)
4. Merge into `develop` with `--no-ff` to preserve the branch in history

---

## Environment Setup

### Option A — uv + venv (recommended for pure-pip workflows)

```bash
# 1. Clone
git clone https://github.com/AkramChaabnia/SEALClust.git
cd SEALClust

# 2. Create venv with uv
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3. Install all dependencies (includes ruff, commitizen, pre-commit)
uv pip install -e ".[dev]"

# 4. Install git hooks (runs ruff automatically on every commit)
pre-commit install

# 5. Configure env
cp .env.example .env
# Edit .env: set OPENAI_API_KEY to your OpenRouter key

# 6. Smoke test
python openrouter_adapter.py
```

Or steps 2–4 in one command: `make setup`

### Option B — Conda

```bash
# 1. Clone
git clone https://github.com/AkramChaabnia/SEALClust.git
cd SEALClust

# 2. Create and activate a conda environment
conda create -n ppd python=3.12 -y
conda activate ppd

# 3. Install all dependencies (includes ruff, commitizen, pre-commit)
pip install -e ".[dev]"

# 4. Install git hooks
pre-commit install

# 5. Configure env
cp .env.example .env
# Edit .env: set OPENAI_API_KEY to your OpenRouter key

# 6. Smoke test
python openrouter_adapter.py
```

Or steps 3–4 in one command: `make setup-conda`

> **Note:** The Makefile auto-detects your environment. If a conda environment is active (`$CONDA_PREFIX` set), it uses `$CONDA_PREFIX/bin/`; otherwise it falls back to `.venv/bin/` if present. Run `make help` to see which prefix is detected.

---

## Adding Dependencies

```bash
# Add the package to pyproject.toml, then:
uv pip install -e ".[dev]"
uv lock
git add uv.lock pyproject.toml
cz commit   # type: build
```

---

## Code Quality & Pre-commit Hooks

We use **Ruff** for linting/formatting and **Bandit** for security checks (configured via pre-commit):

```bash
# Manual checks before commit
ruff check . --fix          # Fix linting issues
ruff format .              # Format code

# Or use Makefile shortcut
make lint
```

Pre-commit hooks run automatically on `git commit`. To manually test all hooks:

```bash
pre-commit run --all-files
```

---

## Submitting Experimental Results

If your PR includes new baseline runs or experimental comparisons:

1. **Create `experiment/<description>` branch**
2. **Run pipeline and save results:**
   ```bash
   make run-step1 data=<dataset>
   make run-step2 data=<dataset> run=<run_dir>
   make run-step3 data=<dataset> run=<run_dir>
   # Results → runs/<dataset>_small_<timestamp>/
   ```
3. **Document in FINDINGS.md:**
   - Dataset, timestamp, model, metrics (ACC/NMI/ARI)
   - Key findings and observations

4. **Commit results:**
   ```bash
   git add runs/<dataset>_small_*/metrics.json FINDINGS.md
   cz commit --scope findings --message "add: experimental results for <dataset>"
   ```

---

## Documentation Standards

- **Code:** Add docstrings to public functions, include type hints
- **README & tutorials:** Keep examples runnable and up-to-date
- **FINDINGS.md:** Document timestamps, models, results, root causes

---

## Getting Help

- **Questions?** Check [README.md](./README.md) and [FINDINGS.md](./FINDINGS.md)
- **Bug?** Search [GitHub Issues](https://github.com/AkramChaabnia/SEALClust/issues)
- **New idea?** [Create an issue](https://github.com/AkramChaabnia/SEALClust/issues/new/choose)

---

**Happy contributing!**
