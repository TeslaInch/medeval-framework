"""
medeval/__init__.py
~~~~~~~~~~~~~~~~~~~
Public API surface for the medeval evaluation framework.

Importing from ``medeval`` directly exposes the core data contracts,
calibration engine, accuracy scorers, hallucination detector, benchmark
loader, safety checker, and report generator.

Example:
    >>> from medeval import (
    ...     MedicalEvalSample, EvaluationReport,
    ...     calculate_ece,
    ...     ExactMatchScorer, SemanticSimilarityScorer,
    ...     NLIHallucinationDetector,
    ...     BenchmarkLoader,
    ...     ReportGenerator, export_report_to_json,
    ... )
    >>> from medeval.safety import SickleCellSafetyChecker
"""

from .accuracy import ExactMatchScorer, SemanticSimilarityScorer
from .benchmark import BenchmarkLoader, DatasetLoadError
from .calibration import calculate_ece
from .hallucination import NLIHallucinationDetector, NLIResult
from .report import ReportGenerator, export_report_to_json
from .structures import EvaluationReport, MedicalEvalSample

__version__ = "0.1.0"

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
    # Reporting
    "ReportGenerator",
    "export_report_to_json",
]