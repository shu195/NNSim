# Contributing

Thanks for your interest in contributing to nnsim.

## Development Setup

1. Create and activate a Python virtual environment.
2. Install package and dev dependencies:

```powershell
python -m pip install -e .[dev]
```

3. Run quality checks before opening a PR:

```powershell
ruff check .
mypy src tests
pytest -q
```

## Branch and Commit Guidelines

1. Create a branch from main.
2. Keep each PR focused on a single concern.
3. Use descriptive commit messages in imperative mood.
4. Include tests for new features and bug fixes.

## Pull Request Checklist

- [ ] Tests added or updated
- [ ] Lint passes
- [ ] Type checks pass
- [ ] README/docs updated if behavior changed
- [ ] Changelog entry updated when relevant

## Reporting Bugs

Use the Bug Report issue template and include:
- Reproduction command
- Expected vs actual behavior
- Environment details (Python version, OS)
- Relevant logs or artifact files

## Feature Requests

Use the Feature Request template and include:
- Problem statement
- Proposed API/CLI experience
- Example command or usage flow
- Backward-compatibility notes
