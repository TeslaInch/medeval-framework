"""
medeval/report.py
~~~~~~~~~~~~~~~~~
Aggregation and export utilities for medeval evaluation results.

Provides:
    - ``ReportGenerator``: Consumes a list of processed ``MedicalEvalSample``
      objects (with metric values stored in their ``metadata`` dicts) and
      produces a fully populated ``EvaluationReport``.
    - ``export_report_to_json``: Serialises an ``EvaluationReport`` to a
      JSON file, handling non-serialisable types gracefully.

**Metadata contract**
``ReportGenerator`` reads the following optional keys from each sample's
``metadata`` dict (set by upstream scorers):

    ``"bert_score_f1"`` (float)  — BERTScore F1 for this sample.
    ``"is_hallucination"`` (bool) — Hallucination flag from NLI detector.
    ``"safety_violations"`` (List[str]) — Violation codes from safety checker.
    ``"y_true"`` (int, 0 or 1)  — Binary correctness label for ECE.
    ``"y_prob"`` (float)         — Model confidence for ECE.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .calibration import calculate_brier_score, calculate_ece, calculate_mce
from .structures import EvaluationReport, MedicalEvalSample

logger = logging.getLogger(__name__)

# Keys read from MedicalEvalSample.metadata by ReportGenerator.
_KEY_BERT_SCORE: str = "bert_score_f1"
_KEY_HALLUCINATION: str = "is_hallucination"
_KEY_SAFETY_VIOLATIONS: str = "safety_violations"
_KEY_Y_TRUE: str = "y_true"
_KEY_Y_PROB: str = "y_prob"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Aggregates per-sample metrics into a single ``EvaluationReport``.

    Reads pre-computed metric values from each ``MedicalEvalSample.metadata``
    dict (populated by the upstream scorer/detector classes) and computes
    framework-level aggregates.

    Args:
        model_name: Identifier of the model under evaluation.
        framework_version: Version of medeval used, for reproducibility.
        samples: List of ``MedicalEvalSample`` objects with populated
            ``metadata`` fields.

    Example:
        >>> from medeval.structures import MedicalEvalSample
        >>> samples = [
        ...     MedicalEvalSample(
        ...         id="q1", question="...", ground_truth="A",
        ...         model_prediction="A",
        ...         metadata={"bert_score_f1": 0.95, "is_hallucination": False,
        ...                   "safety_violations": [], "y_true": 1, "y_prob": 0.9},
        ...     )
        ... ]
        >>> gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        >>> report = gen.generate()
        >>> report.metrics["bert_score_mean_f1"]
        0.95
    """

    def __init__(
        self,
        model_name: str,
        framework_version: str,
        samples: list[MedicalEvalSample],
    ) -> None:
        """Initialise with model identity and the evaluated sample list.

        Args:
            model_name: Human-readable model identifier.
            framework_version: medeval version string.
            samples: Evaluated ``MedicalEvalSample`` objects. Must be
                non-empty.

        Raises:
            ValueError: If ``samples`` is empty.
        """
        if not samples:
            raise ValueError(
                "ReportGenerator requires at least one MedicalEvalSample. Received an empty list."
            )

        self._model_name: str = model_name
        self._framework_version: str = framework_version
        self._samples: list[MedicalEvalSample] = samples

    # ------------------------------------------------------------------
    # Private aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_bert_score(self) -> float | None:
        """Compute mean BERTScore F1 across samples that have the metric.

        Returns:
            Mean F1 float, or ``None`` if no samples carry the metric.
        """
        scores: list[float] = [
            float(s.metadata[_KEY_BERT_SCORE])
            for s in self._samples
            if _KEY_BERT_SCORE in s.metadata
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _aggregate_hallucination_rate(self) -> float | None:
        """Compute fraction of samples flagged as hallucinations.

        Returns:
            Hallucination rate in [0, 1], or ``None`` if no samples carry
            the metric.
        """
        flags: list[bool] = [
            bool(s.metadata[_KEY_HALLUCINATION])
            for s in self._samples
            if _KEY_HALLUCINATION in s.metadata
        ]
        if not flags:
            return None
        return sum(flags) / len(flags)

    def _aggregate_safety_violations(self) -> list[dict[str, Any]]:
        """Collect all safety violations across all samples.

        Returns:
            A list of violation record dicts, each containing ``"sample_id"``
            and ``"codes"`` (list of violation code strings).
        """
        all_violations: list[dict[str, Any]] = []
        for sample in self._samples:
            codes: list[str] = sample.metadata.get(_KEY_SAFETY_VIOLATIONS, [])
            if codes:
                all_violations.append({"sample_id": sample.id, "codes": codes})
        return all_violations

    def _aggregate_calibration(self) -> dict[str, float]:
        """Compute calibration metrics (ECE, MCE, Brier Score) from samples.

        Samples without both `y_true` and `y_prob` keys are silently skipped.
        Requires at least 2 eligible samples to produce meaningful metrics.

        Returns:
            A dictionary containing 'ece', 'mce', and 'brier_score'. Empty if
            fewer than 2 eligible samples exist.
        """
        y_true: list[int] = []
        y_prob: list[float] = []

        for sample in self._samples:
            if _KEY_Y_TRUE in sample.metadata and _KEY_Y_PROB in sample.metadata:
                y_true.append(int(sample.metadata[_KEY_Y_TRUE]))
                y_prob.append(float(sample.metadata[_KEY_Y_PROB]))

        if len(y_true) < 2:
            logger.warning(
                "Fewer than 2 samples have calibration data; skipping calibration metrics."
            )
            return {}

        return {
            "ece": calculate_ece(y_true, y_prob),
            "mce": calculate_mce(y_true, y_prob),
            "brier_score": calculate_brier_score(y_true, y_prob),
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self) -> EvaluationReport:
        """Aggregate all per-sample metrics and return an ``EvaluationReport``.

        Computed metrics (only included if data is available):
            - ``"bert_score_mean_f1"``: Mean BERTScore F1.
            - ``"hallucination_rate"``: Fraction of hallucinated samples.
            - ``"ece"``: Expected Calibration Error.
            - ``"safety_violation_count"``: Number of samples with violations.

        Returns:
            A fully populated :class:`~medeval.structures.EvaluationReport`.
        """
        metrics: dict[str, float] = {}

        bert_score = self._aggregate_bert_score()
        if bert_score is not None:
            metrics[_KEY_BERT_SCORE.replace("_f1", "_mean_f1")] = bert_score

        hallucination_rate = self._aggregate_hallucination_rate()
        if hallucination_rate is not None:
            metrics["hallucination_rate"] = hallucination_rate

        metrics.update(self._aggregate_calibration())

        safety_violations = self._aggregate_safety_violations()
        metrics["safety_violation_count"] = float(len(safety_violations))

        report = EvaluationReport(
            model_name=self._model_name,
            framework_version=self._framework_version,
            total_samples=len(self._samples),
            metrics=metrics,
            safety_violations=safety_violations,
        )

        logger.info(
            "Generated EvaluationReport for model='%s': %d samples, metrics=%s",
            self._model_name,
            report.total_samples,
            metrics,
        )
        return report


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class _MedevalJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for medeval report serialisation.

    Handles non-standard float values (``nan``, ``inf``, ``-inf``) by
    converting them to JSON ``null``, and dataclasses by converting them to
    dicts via ``dataclasses.asdict``.

    Subclassing ``JSONEncoder`` (rather than using the ``default`` kwarg) is
    necessary because the standard encoder special-cases Python ``float``
    objects *before* calling ``default``, emitting bare ``NaN`` tokens that
    are not valid JSON.
    """

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        """Serialise types not handled by the base encoder.

        Args:
            obj: Object that the standard encoder cannot serialise.

        Returns:
            A JSON-serialisable value.

        Raises:
            TypeError: If ``obj`` is of an unsupported type.
        """
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return super().default(obj)

    def iterencode(self, obj: Any, _one_shot: bool = False) -> Iterator[str]:
        """Iterate over JSON chunks, converting nan/inf floats to null.

        Args:
            obj: The Python object to encode.
            _one_shot: Internal CPython flag passed through.

        Yields:
            JSON string chunks.
        """
        for chunk in super().iterencode(obj, _one_shot=_one_shot):
            # The base encoder emits 'NaN', 'Infinity', '-Infinity' as bare
            # tokens; replace them with the JSON-compliant 'null'.
            if chunk in ("NaN", "Infinity", "-Infinity"):
                yield "null"
            else:
                yield chunk


def export_report_to_json(
    report: EvaluationReport,
    output_path: str,
    indent: int = 2,
) -> Path:
    """Serialise an ``EvaluationReport`` to a JSON file.

    Handles non-serialisable float values (``nan``, ``inf``) and nested
    dataclasses gracefully.

    Args:
        report: The ``EvaluationReport`` to serialise.
        output_path: Destination file path (created if it does not exist;
            parent directories must exist).
        indent: JSON indentation level. Defaults to 2.

    Returns:
        A ``pathlib.Path`` object pointing to the written file.

    Raises:
        TypeError: If ``report`` is not an ``EvaluationReport`` instance.
        OSError: If the file cannot be written (e.g. permissions error).

    Example:
        >>> from medeval.structures import EvaluationReport
        >>> report = EvaluationReport(
        ...     model_name="gpt-4o",
        ...     framework_version="0.1.0",
        ...     total_samples=100,
        ...     metrics={"accuracy": 0.82},
        ... )
        >>> path = export_report_to_json(report, "/tmp/report.json")
        >>> path.exists()
        True
    """
    if not isinstance(report, EvaluationReport):
        raise TypeError(
            f"'report' must be an EvaluationReport instance. Got: {type(report).__name__!r}."
        )

    output = Path(output_path)
    report_dict = dataclasses.asdict(report)

    with output.open("w", encoding="utf-8") as fh:
        json.dump(
            report_dict,
            fh,
            indent=indent,
            cls=_MedevalJSONEncoder,
            ensure_ascii=False,
        )

    logger.info("EvaluationReport exported to '%s'.", output)
    return output
