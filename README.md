# medeval-framework

[![PyPI version](https://img.shields.io/pypi/v/medeval-framework.svg)](https://pypi.org/project/medeval-framework/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Support](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![CI Pipeline](https://github.com/TeslaInch/medeval-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/TeslaInch/medeval-framework/actions/workflows/ci.yml)

A rigorous, open-source Python evaluation framework designed to benchmark medical Large Language Models (LLMs) for clinical accuracy, hallucination rates, model calibration, and safety.

---

## 📖 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
  - [Install via Pip (Recommended)](#1-install-via-pip-recommended)
  - [Install from Source](#2-install-from-source)
- [Quickstart](#-quickstart)
  - [Command Line Interface (CLI)](#1-command-line-interface-cli)
  - [Python Orchestration API](#2-python-orchestration-api)
- [Repository Structure](#-repository-structure)
- [Development & Testing](#-development--testing)
- [License](#-license)

---

## 🌟 Key Features

- **Multi-Dataset Benchmarks**: Out-of-the-box loaders for standardized medical datasets (MedQA, PubMedQA).
- **Dual Clinical Safety Audit**:
  - *Deterministic Checker*: Fast regex scanning for explicit clinical contraindications in **Sickle Cell Disease** and **Cardiology**.
  - *Semantic Safety Net*: NLI-based hazard verification (`SemanticSafetyChecker`) to flag context-dependent medical hazards.
- **Unified Model & PEFT Connectors**: Modular drivers to query API models (OpenAI) or execute local PyTorch/Transformers weights (Hugging Face) and PEFT/LoRA adapters smoothly.
- **NLP & Semantic Accuracy Engines**: Exact Match comparison and BERTScore semantic similarity scoring (`SemanticSimilarityScorer`).
- **NLI Hallucination Detection**: Cross-encoder Natural Language Inference (`NLIHallucinationDetector`) evaluating predictions (`hypothesis`) against authoritative clinical facts (`ground_truth`).
- **Advanced Calibration Suite**: Vectorized calculation of **Expected Calibration Error (ECE)**, **Maximum Calibration Error (MCE)**, and **Brier Score**.
- **CLI & Report Generator**: Command-line `medeval` interface and structured JSON report exporter for auditability.

---

## ⚙️ Installation

### 1. Install via Pip (Recommended)

`medeval-framework` is available on PyPI. You can install the core framework or include optional ML/NLP extras:

```bash
# Core installation (numpy-only, lightweight)
pip install medeval-framework

# Full ML & NLP stack (Transformers, PyTorch, evaluate, datasets, bert_score, peft)
pip install medeval-framework[nlp]

# Complete installation including development and testing tools
pip install medeval-framework[all]
```

### 2. Install from Source

For development or contributing:

```bash
# Clone the repository
git clone https://github.com/TeslaInch/medeval-framework.git
cd medeval-framework

# Install editable package with all extras
pip install -e ".[all]"
```

---

## 🚀 Quickstart

### 1. Command Line Interface (CLI)

Run evaluations directly from your terminal using the `medeval` command:

```bash
# Get CLI usage help
medeval --help

# Run evaluation on MedQA using OpenAI GPT-4o with Sickle Cell safety audit
export OPENAI_API_KEY="your-api-key"
medeval --model gpt-4o --dataset medqa --limit 20 --output report.json

# Run evaluation on Hugging Face base model or PEFT adapter on GPU
medeval \
  --model "microsoft/Phi-3.5-mini-instruct" \
  --dataset medqa \
  --device "cuda:0" \
  --limit 50 \
  --output base_model_report.json
```

### 2. Python Orchestration API

Build customized evaluation pipelines using the `BenchmarkRunner` API. An executable demonstration is available in `example.py`:

```python
from medeval.benchmark import BenchmarkLoader
from medeval.models.huggingface import HuggingFaceConnector
from medeval.runner import BenchmarkRunner
from medeval.safety import SickleCellSafetyChecker, SemanticSafetyChecker, SafetySuite
from medeval.report import export_report_to_json

# 1. Load benchmark dataset
loader = BenchmarkLoader(split="test", max_samples=10)
samples = loader.load_medqa()

# 2. Instantiate Model Connector (Hugging Face base or PEFT adapter)
model = HuggingFaceConnector(
    model_name="microsoft/Phi-3.5-mini-instruct",
    device="cuda:0"
)

# 3. Setup Safety Suite (Deterministic + DeBERTa NLI Semantic Net)
safety_suite = SafetySuite([
    SickleCellSafetyChecker(),
    SemanticSafetyChecker(device=0)
])

# 4. Initialize and execute BenchmarkRunner
runner = BenchmarkRunner(
    model=model,
    safety_checker=safety_suite,
    ignore_errors=True
)
report = runner.run(samples)

# 5. Export structured JSON report
export_report_to_json(report, "evaluation_report.json")
```

---

## 📁 Repository Structure

```
medeval/
├── medeval/
│   ├── models/               # Model Connectors (Base, HF, PEFT, OpenAI, Mock)
│   ├── safety/               # Safety Checkers (SickleCell, Cardiology, Semantic, SafetySuite)
│   ├── accuracy.py           # Scorers (Exact Match, BERTScore F1)
│   ├── benchmark.py          # Benchmark Loaders (MedQA, PubMedQA)
│   ├── calibration.py        # Calibration Suite (ECE, MCE, Brier Score)
│   ├── hallucination.py      # NLI Cross-Encoder Hallucination Engine
│   ├── report.py             # Metric aggregation & JSON serialization
│   ├── runner.py             # BenchmarkRunner pipeline orchestrator
│   └── structures.py         # Data contracts (MedicalEvalSample & EvaluationReport)
├── tests/                    # 169 Unit & Integration Tests
├── pyproject.toml            # Ruff & Mypy configurations
├── setup.py                  # PyPI Packaging configuration
└── requirements.txt          # Package dependencies
```

---

## 🧪 Development & Testing

Ensure style alignment and type safety before submitting pull requests:

```bash
# Run full pytest test suite
pytest

# Run Ruff style & linting check
ruff check .

# Run Mypy static type verification
mypy medeval/
```

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
