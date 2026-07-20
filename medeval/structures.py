"""
medeval/structures.py
~~~~~~~~~~~~~~~~~~~~~
Core data contracts for the medeval evaluation framework.

All structures use Python dataclasses to enforce a clean, explicit data model
and to avoid hidden state. ``MedicalEvalSample`` is frozen (immutable) to
guarantee that raw evaluation samples are never accidentally mutated during a
benchmark run.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class MedicalEvalSample:
    """Immutable atomic data structure representing a single medical evaluation case.

    Using ``frozen=True`` makes each sample hashable and immutable, preserving
    evaluation integrity — a sample cannot be altered after it is created.

    Attributes:
        id: Unique identifier for this evaluation case (e.g., "usmle_q_0042").
        question: The clinical question posed to the model.
        ground_truth: The clinically correct answer string.
        model_prediction: The raw answer string produced by the model.
        prediction_probabilities: Optional list of softmax probabilities or
            log-probs from the model, used for calibration calculations such as
            ECE. The ordering must correspond to the model's output classes.
        metadata: Arbitrary key-value metadata for filtering and grouping
            (e.g., ``{"domain": "cardiology", "dataset": "usmle"}``).

    Example:
        >>> sample = MedicalEvalSample(
        ...     id="q_001",
        ...     question="What is the first-line treatment for hypertension?",
        ...     ground_truth="Lifestyle modification",
        ...     model_prediction="Lifestyle modification",
        ...     prediction_probabilities=[0.85, 0.10, 0.05],
        ...     metadata={"domain": "cardiology", "difficulty": "easy"},
        ... )
    """

    id: str
    question: str
    ground_truth: str
    model_prediction: str
    # Softmax probabilities or log-probs for calibration (e.g., ECE) checks.
    prediction_probabilities: Optional[list[float]] = None
    # Arbitrary metadata for filtering (e.g., specialty, dataset, difficulty).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Mutable container for macro-level benchmark results produced by the framework.

    An ``EvaluationReport`` is built incrementally as samples are evaluated and
    is meant to be serialized (e.g., to JSON) at the end of a benchmark run.

    Attributes:
        model_name: Human-readable identifier of the model being evaluated
            (e.g., ``"gpt-4o"`` or ``"meditron-70b"``).
        framework_version: The version of medeval used to produce this report,
            enabling reproducibility tracking (e.g., ``"0.1.0"``).
        total_samples: Total number of ``MedicalEvalSample`` instances evaluated.
        metrics: Dictionary mapping metric names to their computed float values
            (e.g., ``{"accuracy": 0.82, "ece": 0.04}``).
        safety_violations: List of structured dicts, each describing a clinical
            safety violation detected during evaluation.

    Example:
        >>> report = EvaluationReport(
        ...     model_name="meditron-70b",
        ...     framework_version="0.1.0",
        ...     total_samples=500,
        ...     metrics={"accuracy": 0.82, "ece": 0.04},
        ...     safety_violations=[],
        ... )
        >>> report.metrics["accuracy"]
        0.82
    """

    model_name: str
    framework_version: str
    total_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    safety_violations: list[dict[str, Any]] = field(default_factory=list)
