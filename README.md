# medeval-framework

[![PyPI version](https://img.shields.io/pypi/v/medeval-framework.svg)](https://pypi.org/project/medeval-framework/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Support](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![CI Pipeline](https://github.com/TeslaInch/medeval-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/TeslaInch/medeval-framework/actions/workflows/ci.yml)

A rigorous, open-source Python evaluation framework designed to benchmark medical Large Language Models (LLMs) and PEFT/LoRA adapters for clinical accuracy, hallucination rates, model calibration, and safety.

---

## 📖 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
  - [Install via Pip (Recommended)](#1-install-via-pip-recommended)
  - [Install from Source](#2-install-from-source)
- [Quickstart](#-quickstart)
  - [Command Line Interface (CLI)](#1-command-line-interface-cli)
  - [Python Orchestration API](#2-python-orchestration-api)
  - [Multi-Model Side-by-Side Comparison](#3-multi-model-side-by-side-comparison)
- [Repository Structure](#-repository-structure)
- [Development & Testing](#-development--testing)
- [License](#-license)

---

## 🌟 Key Features

- **Multi-Dataset Benchmarks**: Out-of-the-box loaders for standard medical datasets (**MedQA**, **PubMedQA**, **MedMCQA**, **MMLU-Medical**).
- **Dynamic Topic Filtering**: Filter any benchmark dataset by medical domain or disease keyword on the fly (e.g. `--topic "sickle"`, `--topic "cardiac"`, `--topic "renal"`).
- **Universal SOTA Connectors**: Query local PyTorch weights, PEFT/LoRA adapters, OpenAI APIs (`gpt-4o`), or SOTA routers (**OpenRouter**) driving Claude 3.5 Sonnet, Llama 3.1 70B/405B, DeepSeek, and Gemini.
- **Dual Clinical Safety Audit**:
  - *Deterministic Checker*: Pure-Python regex engine detecting explicit contraindications in **Sickle Cell Disease** and **Cardiology**.
  - *Semantic Safety Net*: Cross-encoder NLI hazard verification (`SemanticSafetyChecker`) flagging context-dependent medical risks.
- **NLI Hallucination Detection**: Cross-encoder Natural Language Inference (`NLIHallucinationDetector`) evaluating prediction hypotheses against ground-truth clinical facts.
- **Advanced Calibration Suite**: Vectorized Expected Calibration Error (**ECE**), Maximum Calibration Error (**MCE**), and **Brier Score**.
- **Multi-Format Export & Comparison**: Export structured reports to **JSON**, **Markdown** tables (`.md`), or interactive **HTML** dashboards (`.html`), with built-in side-by-side model comparison tools (`compare_reports`).

---

## ⚙️ Installation

### 1. Install via Pip (Recommended)

`medeval-framework` is available on PyPI:

```bash
# Core lightweight installation (NumPy & pure-Python engines)
pip install medeval-framework

# Full ML & NLP stack (Transformers, PyTorch, Datasets, PEFT, BERTScore)
pip install medeval-framework[nlp]

# Complete installation including development and test suites
pip install medeval-framework[all]
```

### 2. Install from Source

For development or contributing:

```bash
git clone https://github.com/TeslaInch/medeval-framework.git
cd medeval-framework
pip install -e ".[all]"
```

---

## 🚀 Quickstart

### 1. Command Line Interface (CLI)

Run evaluations directly from your terminal:

```bash
# Check version and available flags
medeval --version
medeval --help

# Evaluate OpenAI GPT-4o on MedQA
export OPENAI_API_KEY="your-api-key"
medeval --model gpt-4o --dataset medqa --limit 20 --output report.json

# Evaluate Claude 3.5 Sonnet or Llama 3.1 via AgentRouter with custom API flags
medeval \
  --model "agentrouter:claude-3-5-sonnet" \
  --api-key "your_agentrouter_key" \
  --api-base-url "https://agentrouter.ai/v1" \
  --dataset medqa \
  --output report_claude.html

# Evaluate a specialized PEFT adapter filtered specifically for Sickle Cell questions
medeval \
  --model "TeslaInch/scd-phi35-adapter-v8" \
  --dataset medqa \
  --topic "sickle" \
  --output sickle_cell_report.md
```

#### CLI Options Reference

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--model` *(required)* | Model ID (`HuggingFace repo`, PEFT adapter, `gpt-4o`, `agentrouter:...`, `openrouter:...`, `mock-...`). | - |
| `--dataset` *(required)* | Benchmark dataset: `medqa`, `pubmedqa`, `medmcqa`, or `mmlu_medical`. | - |
| `--output` *(required)* | Target report filepath (`.json`, `.md`, or `.html`). | - |
| `--topic` | Filter dataset by topic keyword (e.g. `sickle`, `cardiac`, `renal`, `diabetes`). | All |
| `--api-key` | API Key override for OpenAI, AgentRouter, or OpenRouter models. | Env var |
| `--api-base-url` | Base URL override (e.g. `https://agentrouter.ai/v1`). | Env var |
| `--safety` | Safety audit checker: `sickle_cell` (default) or `none`. | `sickle_cell` |
| `--device` | PyTorch execution device (`cpu`, `cuda:0`, etc.). | `cpu` |
| `--limit` | Maximum number of samples to evaluate. | All |
| `--trust-remote-code` | Enable `trust_remote_code=True` for Hugging Face models. | `False` |

---

### 2. Python Orchestration API

Build custom evaluation pipelines in Python:

```python
from medeval.benchmark import BenchmarkLoader
from medeval.models.openai_connector import OpenAIConnector
from medeval.runner import BenchmarkRunner
from medeval.safety import SickleCellSafetyChecker
from medeval.report import export_report_to_html, export_report_to_markdown

# 1. Load dataset with topic filtering (e.g. Cardiology questions)
loader = BenchmarkLoader(split="test", max_samples=20, topic="cardiac")
samples = loader.load_medqa()

# 2. Instantiate Model Connector (API Router or HuggingFace model)
model = OpenAIConnector(
    model_name="claude-3-5-sonnet",
    base_url="https://agentrouter.ai/v1"
)

# 3. Configure Safety Checker & Execute Benchmark
runner = BenchmarkRunner(model=model, safety_checker=SickleCellSafetyChecker(), ignore_errors=True)
report = runner.run(samples)

# 4. Export report as styled Markdown or HTML dashboard
export_report_to_markdown(report, "cardiology_report.md")
export_report_to_html(report, "cardiology_report.html")
```

---

### 3. Multi-Model Side-by-Side Comparison

Compare base models, fine-tuned adapters, and SOTA models in Python:

```python
from medeval.comparison import load_reports_from_files, export_comparison_to_markdown

# Load evaluation reports
reports = load_reports_from_files([
    "base_model_report.json",
    "sickle_cell_adapter_report.json",
    "sota_claude_report.json"
])

# Export side-by-side comparative Markdown matrix
export_comparison_to_markdown(reports, "model_comparison_matrix.md")
```

---

## 📁 Repository Structure

```
medeval/
├── medeval/
│   ├── models/               # Model Connectors (HF, PEFT, OpenAI, AgentRouter, Mock)
│   ├── safety/               # Safety Checkers (SickleCell, Cardiology, Semantic, SafetySuite)
│   ├── accuracy.py           # Scorers (Exact Match, BERTScore F1)
│   ├── benchmark.py          # Loaders (MedQA, PubMedQA, MedMCQA, MMLU-Medical)
│   ├── calibration.py        # Calibration Suite (ECE, MCE, Brier Score)
│   ├── comparison.py         # Multi-Model Side-by-Side Comparison Engine
│   ├── hallucination.py      # NLI Cross-Encoder Hallucination Engine
│   ├── report.py             # Metric aggregation (JSON, Markdown, HTML exporters)
│   ├── runner.py             # BenchmarkRunner pipeline orchestrator
│   └── structures.py         # Data contracts (MedicalEvalSample & EvaluationReport)
├── tests/                    # 46 Unit & Integration Tests
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

