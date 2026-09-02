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
from collections.abc import Iterator, Sequence
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

    def _aggregate_hallucination_rate(self) -> tuple[float | None, tuple[float, float] | None]:
        """Compute fraction of samples flagged as hallucinations and its CI.

        Returns:
            A tuple of (rate, ci), where rate is in [0, 1] and ci is a tuple of
            (lower, upper) bounds. Returns (None, None) if no samples carry
            the metric.
        """
        flags: list[bool] = [
            bool(s.metadata[_KEY_HALLUCINATION])
            for s in self._samples
            if _KEY_HALLUCINATION in s.metadata
        ]
        if not flags:
            return None, None

        rate = sum(flags) / len(flags)
        ci = self._calculate_binary_ci(flags)
        return rate, ci

    def _calculate_binary_ci(self, values: Sequence[bool | int]) -> tuple[float, float] | None:
        """Calculate 95% CI for binary proportions using Wald's formula or Bootstrap."""
        if not values:
            return None

        n = len(values)
        if n == 0:
            return None

        p = sum(values) / n

        # Check normal approximation conditions (magic number = 5)
        if n >= 100 and (n * p >= 5) and (n * (1 - p) >= 5):
            import math

            z = 1.96  # For 95% confidence
            se = math.sqrt(p * (1 - p) / n)
            return max(0.0, p - z * se), min(1.0, p + z * se)

        # Fallback to bootstrap
        return self._calculate_continuous_ci(values)

    def _calculate_continuous_ci(
        self, values: Sequence[float | int | bool]
    ) -> tuple[float, float] | None:
        """Calculate 95% CI using Bootstrap resampling."""
        if not values or len(values) < 2:
            return None

        import numpy as np

        arr = np.array(values, dtype=float)
        n = len(arr)

        # Resample 10,000 times
        n_iterations = 10000
        # Use a vectorized bootstrap for speed
        # random.choice is fast, but np.random.choice inside a loop is also fine
        # For small n, this is virtually instantaneous
        boot_means = np.empty(n_iterations, dtype=float)
        for i in range(n_iterations):
            sample = np.random.choice(arr, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

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
            # Extract list for CI
            bert_values = [
                float(s.metadata[_KEY_BERT_SCORE])
                for s in self._samples
                if _KEY_BERT_SCORE in s.metadata
            ]
            ci = self._calculate_continuous_ci(bert_values)
            if ci:
                metrics["bert_score_mean_f1_ci_lower"] = ci[0]
                metrics["bert_score_mean_f1_ci_upper"] = ci[1]

        halluc_result = self._aggregate_hallucination_rate()
        if halluc_result[0] is not None:
            metrics["hallucination_rate"] = halluc_result[0]
            if halluc_result[1]:
                metrics["hallucination_rate_ci_lower"] = halluc_result[1][0]
                metrics["hallucination_rate_ci_upper"] = halluc_result[1][1]

            hallucination_count = sum(
                1 for s in self._samples if s.metadata.get(_KEY_HALLUCINATION) is True
            )
            metrics["hallucination_count"] = float(hallucination_count)

        # Accuracy
        y_true_vals = [int(s.metadata["y_true"]) for s in self._samples if "y_true" in s.metadata]
        if y_true_vals:
            acc = sum(y_true_vals) / len(y_true_vals)
            metrics["accuracy"] = acc
            ci = self._calculate_binary_ci(y_true_vals)
            if ci:
                metrics["accuracy_ci_lower"] = ci[0]
                metrics["accuracy_ci_upper"] = ci[1]

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


def export_report_to_markdown(
    report: EvaluationReport,
    output_path: str,
) -> Path:
    """Serialise an ``EvaluationReport`` to a GitHub-flavored Markdown file.

    Args:
        report: The ``EvaluationReport`` to serialise.
        output_path: Destination Markdown file path.

    Returns:
        A ``pathlib.Path`` object pointing to the written file.
    """
    if not isinstance(report, EvaluationReport):
        raise TypeError(
            f"'report' must be an EvaluationReport instance. Got: {type(report).__name__!r}."
        )

    output = Path(output_path)
    safety_count = len(report.safety_violations)
    status_badge = "🟢 PASSED" if safety_count == 0 else f"🔴 FAILED ({safety_count} violations)"

    lines = [
        f"# Medical LLM Evaluation Report: {report.model_name}",
        "",
        f"**Framework Version**: `medeval v{report.framework_version}`  ",
        f"**Total Samples Evaluated**: `{report.total_samples}`  ",
        f"**Safety Audit Status**: {status_badge}",
        "",
        "## 📊 Aggregate Evaluation Metrics",
        "",
        "| Metric | Value | Status / Description |",
        "| :--- | :--- | :--- |",
    ]

    for metric_name, val in report.metrics.items():
        clean_name = metric_name.replace("_", " ").title()
        formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(f"| **{clean_name}** | `{formatted_val}` | Computed across evaluated split |")

    lines.extend(
        [
            "",
            "## 🚨 Clinical Safety Violations",
            "",
        ]
    )

    if not report.safety_violations:
        lines.append("✨ **Zero clinical safety violations detected.**")
    else:
        lines.extend(
            [
                "| Sample ID | Violation Codes |",
                "| :--- | :--- |",
            ]
        )
        for v in report.safety_violations:
            codes_str = ", ".join(f"`{c}`" for c in v.get("codes", []))
            lines.append(f"| `{v.get('sample_id', 'unknown')}` | {codes_str} |")

    lines.extend(
        [
            "",
            "---",
            "*Generated automatically by [medeval-framework](https://github.com/TeslaInch/medeval-framework)*",
        ]
    )

    content = "\n".join(lines)
    with output.open("w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("EvaluationReport exported to Markdown '%s'.", output)
    return output


def export_report_to_html(
    report: EvaluationReport,
    output_path: str,
) -> Path:
    """Serialise an ``EvaluationReport`` to a self-contained styled HTML page.

    Args:
        report: The ``EvaluationReport`` to serialise.
        output_path: Destination HTML file path.

    Returns:
        A ``pathlib.Path`` object pointing to the written file.
    """
    if not isinstance(report, EvaluationReport):
        raise TypeError(
            f"'report' must be an EvaluationReport instance. Got: {type(report).__name__!r}."
        )

    output = Path(output_path)
    safety_count = len(report.safety_violations)
    badge_color = "#10B981" if safety_count == 0 else "#EF4444"
    badge_text = "SAFETY AUDIT PASSED" if safety_count == 0 else f"{safety_count} SAFETY VIOLATIONS"

    rows_html = ""
    for metric_name, val in report.metrics.items():
        clean_name = metric_name.replace("_", " ").title()
        formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
        rows_html += (
            f"<tr><td><strong>{clean_name}</strong></td><td><code>{formatted_val}</code></td></tr>"
        )

    violations_html = ""
    if not report.safety_violations:
        violations_html = "<p style='color: #10B981; font-weight: bold;'>✨ Zero clinical safety violations detected.</p>"
    else:
        violations_html = "<table class='violations-table'><thead><tr><th>Sample ID</th><th>Violation Codes</th></tr></thead><tbody>"
        for v in report.safety_violations:
            codes_str = ", ".join(
                f"<span class='badge-code'>{c}</span>" for c in v.get("codes", [])
            )
            violations_html += f"<tr><td><code>{v.get('sample_id', 'unknown')}</code></td><td>{codes_str}</td></tr>"
        violations_html += "</tbody></table>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>medeval Report - {report.model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1E293B; background: #F8FAFC; margin: 0; padding: 40px 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; background: #FFFFFF; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 32px; border: 1px solid #E2E8F0; }}
        .header {{ border-bottom: 2px solid #F1F5F9; padding-bottom: 20px; margin-bottom: 24px; }}
        h1 {{ margin: 0 0 10px 0; color: #0F172A; font-size: 24px; }}
        .status-badge {{ display: inline-block; padding: 6px 12px; border-radius: 20px; color: white; font-weight: 600; font-size: 12px; background: {badge_color}; }}
        .metadata-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 20px 0; background: #F8FAFC; padding: 16px; border-radius: 8px; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #E2E8F0; }}
        th {{ background: #F8FAFC; font-weight: 600; color: #475569; }}
        code {{ background: #F1F5F9; padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #0F172A; }}
        .badge-code {{ background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 4px; display: inline-block; margin-bottom: 4px; }}
        .footer {{ margin-top: 40px; text-align: center; font-size: 12px; color: #94A3B8; border-top: 1px solid #E2E8F0; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="status-badge">{badge_text}</span>
            <h1 style="margin-top: 12px;">Clinical LLM Evaluation Report</h1>
            <p style="margin: 0; color: #64748B;">Model: <strong>{report.model_name}</strong></p>
        </div>

        <div class="metadata-grid">
            <div><strong>Framework Version:</strong> <code>medeval v{report.framework_version}</code></div>
            <div><strong>Total Samples:</strong> <code>{report.total_samples}</code></div>
        </div>

        <h2>📊 Aggregate Metrics</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>

        <h2 style="margin-top: 32px;">🚨 Clinical Safety Violations</h2>
        {violations_html}

        <div class="footer">
            Generated automatically by <a href="https://github.com/TeslaInch/medeval-framework" style="color: #3B82F6; text-decoration: none;">medeval-framework</a>
        </div>
    </div>
</body>
</html>
"""

    with output.open("w", encoding="utf-8") as fh:
        fh.write(html_content)

    logger.info("EvaluationReport exported to HTML '%s'.", output)
    return output
