"""
examples/kaggle_sota_benchmark.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ready-to-run Kaggle benchmark script for evaluating SOTA models via AgentRouter.

Supports:
    - Kaggle Secrets retrieval for AGENTROUTER_API_KEY
    - Evaluating Claude 3.5 Sonnet, Llama 3.1 70B, and GPT-4o
    - Clinical safety audit (Sickle Cell & Cardiology contraindication checks)
    - Exporting HTML dashboards and side-by-side comparative Markdown reports
"""

import os
import sys

# 1. Install medeval-framework if running inside Kaggle
try:
    import medeval  # noqa: F401
except ImportError:
    print("Installing medeval-framework...")
    os.system("pip install -q medeval-framework[all]")

from medeval.benchmark import BenchmarkLoader
from medeval.comparison import export_comparison_to_markdown
from medeval.models.openai_connector import OpenAIConnector
from medeval.report import export_report_to_html, export_report_to_json, export_report_to_markdown
from medeval.runner import BenchmarkRunner
from medeval.safety import SickleCellSafetyChecker

# 2. Retrieve AgentRouter API Key (Kaggle Secrets or Environment Variable)
api_key = os.environ.get("AGENTROUTER_API_KEY")
if not api_key:
    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("AGENTROUTER_API_KEY")
        print("Successfully retrieved AGENTROUTER_API_KEY from Kaggle User Secrets.")
    except Exception:
        pass

if not api_key:
    print("ERROR: AGENTROUTER_API_KEY is not set.")
    print(
        "Please add AGENTROUTER_API_KEY to your Kaggle Secrets (Add-ons -> Secrets) or os.environ."
    )
    sys.exit(1)

# Set base URL for AgentRouter
base_url = os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.ai/v1")

# 3. Target SOTA Models to Evaluate
MODELS_TO_TEST = [
    {"name": "claude-3-5-sonnet", "label": "Claude 3.5 Sonnet"},
    {"name": "meta-llama/llama-3.1-70b-instruct", "label": "Llama 3.1 70B"},
    {"name": "gpt-4o", "label": "GPT-4o"},
]

# 4. Load Benchmark Dataset (MedQA test split, 20 samples smoke test)
# To filter by topic, add topic="sickle" or topic="cardiac" to BenchmarkLoader
print("\n--- Loading Benchmark Dataset ---")
loader = BenchmarkLoader(split="test", max_samples=20)
samples = loader.load_medqa()
print(f"Loaded {len(samples)} evaluation samples.")

# 5. Run Evaluations across SOTA Models
reports = []
safety_checker = SickleCellSafetyChecker()

for target in MODELS_TO_TEST:
    model_id = target["name"]
    label = target["label"]
    print("\n==========================================")
    print(f" Evaluating SOTA Model: {label} ({model_id})")
    print("==========================================")

    try:
        connector = OpenAIConnector(
            model_name=model_id,
            api_key=api_key,
            base_url=base_url,
        )

        runner = BenchmarkRunner(
            model=connector,
            safety_checker=safety_checker,
            ignore_errors=True,
        )

        report = runner.run(samples)
        reports.append(report)

        # Export individual model reports (JSON, Markdown, and HTML)
        clean_filename = label.lower().replace(" ", "_").replace(".", "")
        export_report_to_json(report, f"report_{clean_filename}.json")
        export_report_to_markdown(report, f"report_{clean_filename}.md")
        export_report_to_html(report, f"report_{clean_filename}.html")

        print(f"Successfully evaluated {label}!")
        print(f"  - ECE: {report.metrics.get('ece', 'N/A')}")
        print(f"  - Safety Violations: {len(report.safety_violations)}")

    except Exception as exc:
        print(f"Skipping {label} due to evaluation error: {exc}")

# 6. Generate Side-by-Side Multi-Model Comparison
if len(reports) >= 2:
    print("\n--- Generating Multi-Model Side-by-Side Comparison ---")
    comp_matrix = export_comparison_to_markdown(reports, "sota_comparison_matrix.md")
    print("Multi-Model Comparison Matrix written to 'sota_comparison_matrix.md'!")

    # Display comparison matrix in console / notebook
    with open("sota_comparison_matrix.md", encoding="utf-8") as f:
        print("\n" + f.read())

print("\n✨ Benchmark evaluation complete! All reports and HTML dashboards generated.")
