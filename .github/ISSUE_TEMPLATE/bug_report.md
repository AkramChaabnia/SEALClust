---
name: Bug Report
about: Report a bug or unexpected behavior in the pipeline or code
title: "[BUG] "
labels: bug
assignees: ""
---

## Description

<!-- Clear and concise description of the bug -->

## Steps to Reproduce

<!-- How to reproduce the issue -->

1. Step 1
2. Step 2
3. ...

## Expected Behavior

<!-- What should have happened -->

## Actual Behavior

<!-- What actually happened -->

## Environment

- **OS**: (e.g., Ubuntu 22.04, macOS 13.x)
- **Python Version**:
- **Package Version** (from `pip show sealclust` or `CHANGELOG.md`):
- **Installation Method**: (conda/venv/pip/uv)
- **LLM Provider**: (OpenAI/OpenRouter/local)

## Logs & Output

<!-- Paste relevant logs, error messages, or output -->

```
Paste error/log here
```

## Dataset Information

- **Dataset**: (massive_scenario / massive_intent / go_emotion / arxiv_fine / mtop_intent)
- **Dataset Size**: (small / large)
- **Custom Dataset**: (Yes/No) — if yes, describe

## Pipeline Details

- **Pipeline Used**: (baseline / sealclust / hybrid / graphclust / kmedoids / gmm)
- **Step(s) Affected**: (label-gen / classify / evaluate)
- **LLM Model Used**: (e.g., google/gemini-2.0-flash-001)

## Configuration

<!-- Relevant `.env` settings or command-line arguments (mask API keys!) -->

```bash
# Command used to trigger the bug
```

## Additional Context

<!-- Any other context that might help -->

## Checklist

- [ ] I have searched existing issues
- [ ] I have checked the [FINDINGS.md](./FINDINGS.md) for known issues
- [ ] I have checked [TROUBLESHOOTING](./README.md#13-troubleshooting) section
- [ ] I can reproduce this consistently
- [ ] I have provided all relevant logs
