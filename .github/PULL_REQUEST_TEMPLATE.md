## Description

Briefly summarize the changes made in this Pull Request and why they are necessary.

Fixes #(issue number if applicable)

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] 🌟 New feature (non-breaking change adding functionality, e.g., new safety checker or benchmark loader)
- [ ] ⚠️ Breaking change (fix or feature that changes existing API behavior)
- [ ] 📖 Documentation update
- [ ] 🧪 Tests / Refactoring

## Verification Checklist

Please verify that your PR passes all quality checks locally before requesting review:

- [ ] `pytest` passes with 0 failures
- [ ] `ruff check .` passes with 0 lint errors
- [ ] `ruff format --check .` passes with 0 formatting diffs
- [ ] `mypy medeval/` passes with 0 static type errors
- [ ] Added unit tests covering new logic or bug fixes
- [ ] Updated `README.md` or docstrings if API interface changed
