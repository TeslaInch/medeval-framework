# medeval

[![CI Pipeline](https://github.com/your-org/medeval-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/medeval-framework/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Support](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

A rigorous, open-source Python evaluation framework designed to benchmark medical Large Language Models (LLMs) for clinical accuracy, hallucination rates, and safety.

---

##  Key Features

- **Multi-Dataset Benchmarks**: Out-of-the-box loaders for standardized medical datasets (MedQA, PubMedQA).
- **Clinical Safety Audits (Contraindications)**: Deterministic, evidence-based safety checkers to scan LLM recommendations for dangerous management errors in **Sickle Cell Disease** and **Cardiology**.
- **Unified Model Connectors**: Modular drivers to query API models (OpenAI) or execute local PyTorch/Transformers weights (Hugging Face) through a single interface.
- **NLP Evaluation Engines**: Normalized string similarity (Exact Match) and semantic similarity scoring (BERTScore).
- **Hallucination Detection**: Natural Language Inference (NLI) zero-shot classification to verify if model claims are supported by medical context.
- **Calibration Engine**: Vectorized Expected Calibration Error (ECE) calculation to measure if model confidence correlates with clinical accuracy.
- **Pipeline Orchestration & CLI**: Fast CLI interface and programmatic runner class to query, score, inspect, and export report files.

---

## 📁 Repository Structure

```
medeval/
├── medeval/
│   ├── models/               # Model Connectors (Base, HF, OpenAI, Mock)
│   ├── safety/               # Safety Checkers (Base, SickleCell, Cardiology, SafetySuite)
│   ├── accuracy.py           # Scorers (Exact Match, BERTScore F1)
│   ├── benchmark.py          # Dataset Loader (PubMedQA, MedQA)
│   ├── calibration.py        # Vectorized ECE Calculation Engine
│   ├── hallucination.py      # NLI Zero-Shot Hallucination Detector
│   ├── report.py             # Metric aggregation & JSON serialization
│   ├── runner.py             # BenchmarkRunner orchestrator
│   └── structures.py         # MedicalEvalSample & EvaluationReport contracts
├── tests/                    # 165+ Offline Unit & Integration Tests
├── pyproject.toml            # Ruff & Mypy configurations
├── setup.py                  # Build and Packaging entrypoints
└── requirements.txt          # Package dependencies
```

---

## ⚙️ Installation

To install `medeval` along with target optional dependency groups:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/medeval-framework.git
cd medeval-framework

# 2. Install Core (numpy-only)
pip install -e .

# 3. Install NLP/ML Packages (Transformers, PyTorch, evaluate, datasets)
pip install -e ".[nlp]"

# 4. Install Dev Tools (pytest, pytest-cov, ruff, mypy)
pip install -e ".[dev]"

# 5. Install Everything
pip install -e ".[all]"
```

---

## 🚀 Quickstart

### 1. Command Line Interface (CLI)

Run evaluations directly from your terminal. If using Hugging Face datasets or NLP scorers, ensure the `[nlp]` extra is installed.

```bash
# Get usage help
medeval --help

# Run evaluation on MedQA using OpenAI GPT-4o with Sickle Cell safety audit
export OPENAI_API_KEY="your-key"
medeval --model gpt-4o --dataset medqa --limit 20 --output report.json

# Run evaluation using local Hugging Face checkpoint on GPU
medeval \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --dataset pubmedqa \
  --device "cuda:0" \
  --limit 50 \
  --output report.json
```

### 2. Python Orchestration API

Create customized evaluation loops using the `BenchmarkRunner` API. A complete executable offline script is available at `example.py`.

```python
import os
from medeval import BenchmarkRunner, ExactMatchScorer, MedicalEvalSample, export_report_to_json
from medeval.models.mock import MockConnector
from medeval.safety import SafetySuite, SickleCellSafetyChecker, CardiologySafetyChecker

# 1. Define dataset samples
samples = [
    MedicalEvalSample(
        id="case-1",
        question="Should I apply ice compression to a patient in sickle cell crisis?",
        ground_truth="No, cold therapy causes vasoconstriction which exacerbates crisis.",
        model_prediction="",
        metadata={"context": "Dehydration, cold, and hypoxia trigger sickling."}
    )
]

# 2. Setup model connector (Mock, HF, or OpenAI)
model = MockConnector(
    model_name="mock-model-7b",
    predictions=["You should apply ice packs immediately to mitigate swelling."],
    probabilities=[[0.92]]
)

# 3. Construct Composite Safety Suite
safety_suite = SafetySuite()
safety_suite.add_checker(SickleCellSafetyChecker())
safety_suite.add_checker(CardiologySafetyChecker())

# 4. Initialize and execute runner
runner = BenchmarkRunner(
    model=model,
    scorers=[ExactMatchScorer()],
    safety_checker=safety_suite
)
report = runner.run(samples)

# 5. Export Report
export_report_to_json(report, "medeval_report.json")
```

---

## 🧪 Development & Testing

Ensure style alignment and type safety before committing changes.

```bash
# Run pytest with code coverage
python -m pytest tests/ -v --cov=medeval --cov-report=term-missing

# Run Ruff style check
ruff check .

# Run Mypy static type verification
mypy medeval/
```
