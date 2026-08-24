"""
medeval/comparison.py
~~~~~~~~~~~~~~~~~~~~~
Multi-model evaluation report comparison utilities.

Provides functions to compare multiple ``EvaluationReport`` objects or JSON files
and generate side-by-side comparative Markdown tables and HTML dashboards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .structures import EvaluationReport

logger = logging.getLogger(__name__)


def compare_reports(
    reports: list[EvaluationReport],
) -> dict[str, Any]:
    """Compare multiple ``EvaluationReport`` instances side-by-side.

    Args:
        reports: List of ``EvaluationReport`` objects (at least 2 required).

    Returns:
        A structured dictionary containing aligned models, metrics comparison,
        and safety violation counts.

    Raises:
        ValueError: If fewer than 2 reports are provided.
    """
    if len(reports) < 2:
        raise ValueError("compare_reports requires at least 2 EvaluationReport instances.")

    model_names = [r.model_name for r in reports]
    all_metric_keys: set[str] = set()
    for r in reports:
        all_metric_keys.update(r.metrics.keys())

    metrics_comparison: dict[str, dict[str, float | None]] = {}
    for metric in sorted(all_metric_keys):
        metrics_comparison[metric] = {r.model_name: r.metrics.get(metric) for r in reports}

    safety_summary: dict[str, int] = {r.model_name: len(r.safety_violations) for r in reports}

    return {
        "models": model_names,
        "metrics": metrics_comparison,
        "safety_violations": safety_summary,
        "total_samples": {r.model_name: r.total_samples for r in reports},
    }


def export_comparison_to_markdown(
    reports: list[EvaluationReport],
    output_path: str,
) -> Path:
    """Generate a side-by-side Markdown comparison report.

    Args:
        reports: List of ``EvaluationReport`` objects.
        output_path: Output file path.

    Returns:
        A ``pathlib.Path`` object pointing to the generated file.
    """
    comparison = compare_reports(reports)
    models = comparison["models"]
    output = Path(output_path)

    header = ["| Metric | " + " | ".join(f"**{m}**" for m in models) + " |"]
    separator = ["| :--- | " + " | ".join(":---" for _ in models) + " |"]

    lines = [
        "# Medical LLM Multi-Model Comparison Report",
        "",
        f"Comparing `{len(models)}` models across standardized evaluation metrics.",
        "",
        "## 📊 Side-by-Side Metrics Matrix",
        "",
        header[0],
        separator[0],
    ]

    for metric, values in comparison["metrics"].items():
        clean_name = metric.replace("_", " ").title()
        val_str = " | ".join(
            f"`{values[m]:.4f}`" if values[m] is not None else "`N/A`" for m in models
        )
        lines.append(f"| **{clean_name}** | {val_str} |")

    lines.extend(
        [
            "",
            "## 🚨 Clinical Safety Violations Summary",
            "",
            header[0],
            separator[0],
        ]
    )

    safety_str = " | ".join(f"`{comparison['safety_violations'][m]}`" for m in models)
    lines.append(f"| **Safety Violations Count** | {safety_str} |")

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

    logger.info("Comparison report written to Markdown '%s'.", output)
    return output


def load_reports_from_files(json_paths: list[str]) -> list[EvaluationReport]:
    """Load multiple ``EvaluationReport`` objects from JSON file paths.

    Args:
        json_paths: List of JSON file paths.

    Returns:
        List of ``EvaluationReport`` objects.
    """
    reports: list[EvaluationReport] = []
    for path_str in json_paths:
        p = Path(path_str)
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            report = EvaluationReport(
                model_name=data.get("model_name", p.stem),
                framework_version=data.get("framework_version", "unknown"),
                total_samples=data.get("total_samples", 0),
                metrics=data.get("metrics", {}),
                safety_violations=data.get("safety_violations", []),
            )
            reports.append(report)
    return reports
