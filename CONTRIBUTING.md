# Contributing to medeval-framework

First off, thank you for considering contributing to `medeval-framework`! We welcome contributions from researchers, software engineers, and medical professionals.

---

## 🛠️ Local Development Setup

### 1. Fork and Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/medeval-framework.git
cd medeval-framework
```

### 2. Create a Virtual Environment and Install Dependencies

We recommend using Python 3.9+:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install editable package with all dev extras
python -m pip install --upgrade pip
pip install -e ".[all]"
```

---

## 🧪 Quality Standards & Code Checks

Before submitting a Pull Request, ensure your code satisfies all project quality gates:

```bash
# 1. Run the test suite
pytest

# 2. Run Ruff linter and formatter checks
ruff check .
ruff format --check .

# 3. Run Mypy static type checking
mypy medeval/
```

> **Note**: All 3 checks must pass cleanly. Our GitHub Actions CI pipeline enforces these checks automatically on every Pull Request.

---

## 🚀 How to Add a New Safety Checker

One of the best ways to contribute is adding new clinical safety rules or domain checkers:

1. Create a new module under `medeval/safety/your_domain.py`.
2. Inherit from `BaseSafetyChecker` (found in `medeval/safety/base.py`).
3. Implement `check_contraindications(self, text: str) -> list[str]` and `check_contraindications_detailed(self, text: str) -> list[SafetyViolation]`.
4. Add comprehensive unit tests in `tests/test_your_domain.py`.
5. Export your new checker in `medeval/safety/__init__.py`.

---

## 📥 Submitting a Pull Request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-checker
   ```
2. Commit your changes with descriptive commit messages:
   ```bash
   git commit -m "feat(safety): add nephrology contraindication checker"
   ```
3. Push to your fork and open a Pull Request against `TeslaInch/medeval-framework:main`.
4. Ensure CI checks pass and address any code review feedback.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [Apache License 2.0](LICENSE).
