"""
tests/test_report.py
~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.report`` — ReportGenerator and export_report_to_json.

Tests are fully deterministic. ``export_report_to_json`` uses Python's
``tmp_path`` pytest fixture to avoid touching the production filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medeval.report import ReportGenerator, export_report_to_json
from medeval.structures import EvaluationReport, MedicalEvalSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    sample_id: str = "q1",
    bert_score: float = 0.9,
    is_hallucination: bool = False,
    safety_violations: list[str] | None = None,
    y_true: int | None = 1,
    y_prob: float | None = 0.9,
) -> MedicalEvalSample:
    """Build a MedicalEvalSample with pre-populated metric metadata.

    Args:
        sample_id: Unique sample identifier.
        bert_score: BERTScore F1 to embed in metadata.
        is_hallucination: Hallucination flag to embed.
        safety_violations: List of violation codes to embed.
        y_true: Binary ground truth label for ECE (0 or 1).
        y_prob: Model confidence for ECE.

    Returns:
        A ``MedicalEvalSample`` with metadata ready for ``ReportGenerator``.
    """
    meta = {}
    if bert_score is not None:
        meta["bert_score_f1"] = bert_score
    if is_hallucination is not None:
        meta["is_hallucination"] = is_hallucination
    if safety_violations is not None:
        meta["safety_violations"] = safety_violations
    if y_true is not None:
        meta["y_true"] = y_true
    if y_prob is not None:
        meta["y_prob"] = y_prob

    return MedicalEvalSample(
        id=sample_id,
        question="What is the treatment?",
        ground_truth="Answer",
        model_prediction="Answer",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# ReportGenerator — constructor validation
# ---------------------------------------------------------------------------


class TestReportGeneratorInit:
    """Tests for ReportGenerator constructor validation."""

    def test_raises_on_empty_samples(self) -> None:
        """ValueError must be raised when an empty sample list is provided."""
        with pytest.raises(ValueError, match="at least one"):
            ReportGenerator("gpt-4o", "0.1.0", [])

    def test_accepts_single_sample(self) -> None:
        """A single sample must be accepted without error."""
        sample = _make_sample()
        gen = ReportGenerator("gpt-4o", "0.1.0", [sample])
        assert gen is not None


# ---------------------------------------------------------------------------
# ReportGenerator.generate() — metrics aggregation
# ---------------------------------------------------------------------------


class TestReportGeneratorGenerate:
    """Tests for the generate() method and its metric aggregations."""

    def test_returns_evaluation_report(self) -> None:
        """generate() must return an EvaluationReport instance."""
        gen = ReportGenerator("gpt-4o", "0.1.0", [_make_sample()])
        report = gen.generate()
        assert isinstance(report, EvaluationReport)

    def test_model_name_is_preserved(self) -> None:
        """Report model_name must match what was passed to the constructor."""
        gen = ReportGenerator("meditron-70b", "0.1.0", [_make_sample()])
        report = gen.generate()
        assert report.model_name == "meditron-70b"

    def test_total_samples_count(self) -> None:
        """Report total_samples must equal the number of samples provided."""
        samples = [_make_sample(f"q{i}") for i in range(5)]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        assert report.total_samples == 5

    def test_bert_score_mean_computed_correctly(self) -> None:
        """bert_score_mean_f1 must be the arithmetic mean of all sample F1s."""
        scores = [0.8, 0.9, 1.0]
        samples = [_make_sample(f"q{i}", bert_score=s) for i, s in enumerate(scores)]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        expected = sum(scores) / len(scores)
        assert abs(report.metrics["bert_score_mean_f1"] - expected) < 1e-9

    def test_hallucination_rate_computed_correctly(self) -> None:
        """hallucination_rate must be fraction of hallucinated samples."""
        samples = [
            _make_sample("q1", is_hallucination=True),
            _make_sample("q2", is_hallucination=False),
            _make_sample("q3", is_hallucination=False),
            _make_sample("q4", is_hallucination=True),
        ]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        # 2 out of 4 hallucinated → rate = 0.5
        assert abs(report.metrics["hallucination_rate"] - 0.5) < 1e-9

    def test_zero_hallucination_rate(self) -> None:
        """hallucination_rate must be 0.0 when no samples are flagged."""
        samples = [_make_sample(f"q{i}", is_hallucination=False) for i in range(3)]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        assert report.metrics["hallucination_rate"] == 0.0

    def test_ece_computed_from_calibration_metadata(self) -> None:
        """ECE must be computed from y_true/y_prob metadata when available."""
        # All correct at confidence 0.9 → perfect calibration → ECE = 0.0
        samples = [_make_sample(f"q{i}", y_true=1, y_prob=0.9) for i in range(10)]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        assert "ece" in report.metrics
        assert report.metrics["ece"] >= 0.0

    def test_ece_skipped_when_fewer_than_two_calibration_samples(self) -> None:
        """ECE must not be included in metrics when only 1 sample has calibration data."""
        # Only 1 sample has y_true/y_prob; the other has no such keys.
        samples = [
            _make_sample("q1", y_true=1, y_prob=0.9),
            MedicalEvalSample(id="q2", question="Q", ground_truth="A", model_prediction="A"),
        ]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        # Only 1 sample has calibration data → ECE skipped
        assert "ece" not in report.metrics

    def test_safety_violations_collected(self) -> None:
        """safety_violations must list sample IDs with their violation codes."""
        samples = [
            _make_sample("q1", safety_violations=["CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION"]),
            _make_sample("q2", safety_violations=[]),
            _make_sample("q3", safety_violations=["WARNING:NSAID_RENAL_RISK_IN_SCD"]),
        ]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        assert report.metrics["safety_violation_count"] == 2.0
        violation_ids = {v["sample_id"] for v in report.safety_violations}
        assert "q1" in violation_ids
        assert "q3" in violation_ids
        assert "q2" not in violation_ids

    def test_no_safety_violations_gives_count_zero(self) -> None:
        """safety_violation_count must be 0.0 when no samples have violations."""
        samples = [_make_sample(f"q{i}", safety_violations=[]) for i in range(3)]
        gen = ReportGenerator("gpt-4o", "0.1.0", samples)
        report = gen.generate()
        assert report.metrics["safety_violation_count"] == 0.0
        assert report.safety_violations == []

    def test_metrics_omitted_when_data_absent(self) -> None:
        """Metrics without corresponding data must not appear in the report."""
        # Sample with no bert_score_f1, no is_hallucination, no y_true/y_prob.
        sample = MedicalEvalSample(
            id="q1",
            question="Q?",
            ground_truth="A",
            model_prediction="A",
            metadata={"safety_violations": []},
        )
        gen = ReportGenerator("gpt-4o", "0.1.0", [sample])
        report = gen.generate()
        assert "bert_score_mean_f1" not in report.metrics
        assert "hallucination_rate" not in report.metrics
        assert "ece" not in report.metrics


# ---------------------------------------------------------------------------
# export_report_to_json
# ---------------------------------------------------------------------------


class TestExportReportToJson:
    """Tests for the export_report_to_json standalone function."""

    def _make_report(self) -> EvaluationReport:
        """Create a simple EvaluationReport for serialisation tests."""
        return EvaluationReport(
            model_name="test-model",
            framework_version="0.1.0",
            total_samples=10,
            metrics={"accuracy": 0.85, "ece": 0.04, "hallucination_rate": 0.1},
            safety_violations=[{"sample_id": "q1", "codes": ["CRITICAL"]}],
        )

    def test_creates_file(self, tmp_path: Path) -> None:
        """export_report_to_json must create the output file."""
        report = self._make_report()
        out_path = tmp_path / "report.json"
        result = export_report_to_json(report, str(out_path))
        assert result.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        """The function must return a pathlib.Path pointing to the written file."""
        report = self._make_report()
        result = export_report_to_json(report, str(tmp_path / "report.json"))
        assert isinstance(result, Path)

    def test_json_is_valid_and_parseable(self, tmp_path: Path) -> None:
        """The written file must be valid JSON."""
        report = self._make_report()
        out_path = tmp_path / "report.json"
        export_report_to_json(report, str(out_path))

        with out_path.open() as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_json_contains_expected_fields(self, tmp_path: Path) -> None:
        """The JSON output must contain model_name, metrics, and safety_violations."""
        report = self._make_report()
        out_path = tmp_path / "report.json"
        export_report_to_json(report, str(out_path))

        with out_path.open() as f:
            data = json.load(f)

        assert data["model_name"] == "test-model"
        assert data["total_samples"] == 10
        assert data["metrics"]["accuracy"] == pytest.approx(0.85)
        assert len(data["safety_violations"]) == 1

    def test_handles_nan_metrics_gracefully(self, tmp_path: Path) -> None:
        """NaN metric values must be serialised as null without raising."""
        report = EvaluationReport(
            model_name="model",
            framework_version="0.1.0",
            total_samples=1,
            metrics={"ece": float("nan")},
        )
        out_path = tmp_path / "report_nan.json"
        export_report_to_json(report, str(out_path))

        with out_path.open() as f:
            data = json.load(f)

        assert data["metrics"]["ece"] is None

    def test_raises_on_non_report_input(self, tmp_path: Path) -> None:
        """TypeError must be raised when a non-EvaluationReport is passed."""
        with pytest.raises(TypeError, match="EvaluationReport"):
            export_report_to_json({"not": "a report"}, str(tmp_path / "out.json"))  # type: ignore[arg-type]
