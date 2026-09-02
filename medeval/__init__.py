"""
medeval/__init__.py
~~~~~~~~~~~~~~~~~~~
Public API surface for the medeval evaluation framework.

Importing from ``medeval`` directly exposes the core data contracts,
calibration engine, accuracy scorers, NLI hallucination detector, benchmark
dataset loaders, pipeline orchestrator runner, and reporting tools.

Examples:
    >>> from medeval import (
    ...     MedicalEvalSample, EvaluationReport,
    ...     calculate_ece,
    ...     ExactMatchScorer, SemanticSimilarityScorer,
    ...     NLIHallucinationDetector,
    ...     BenchmarkLoader,
    ...     BenchmarkRunner,
    ...     ReportGenerator, export_report_to_json,
    ... )
"""

from .accuracy import ExactMatchScorer, SemanticSimilarityScorer
from .benchmark import BenchmarkLoader, DatasetLoadError
from .calibration import calculate_ece
from .comparison import compare_reports, export_comparison_to_markdown, load_reports_from_files
from .hallucination import NLIHallucinationDetector, NLIResult
from .report import (
    ReportGenerator,
    export_report_to_html,
    export_report_to_json,
    export_report_to_markdown,
)
from .runner import BenchmarkRunner, default_prompt_formatter
from .structures import EvaluationReport, MedicalEvalSample

__version__ = "0.1.9"

__all__ = [
    # Data contracts
    "MedicalEvalSample",
    "EvaluationReport",
    # Calibration
    "calculate_ece",
    # Accuracy
    "ExactMatchScorer",
    "SemanticSimilarityScorer",
    # Hallucination
    "NLIHallucinationDetector",
    "NLIResult",
    # Benchmark loading
    "BenchmarkLoader",
    "DatasetLoadError",
    # Runner / Orchestration
    "BenchmarkRunner",
    "default_prompt_formatter",
    # Reporting & Exporting
    "ReportGenerator",
    "export_report_to_json",
    "export_report_to_markdown",
    "export_report_to_html",
    # Multi-model comparison
    "compare_reports",
    "export_comparison_to_markdown",
    "load_reports_from_files",
]
