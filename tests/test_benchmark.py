"""
tests/test_benchmark.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.benchmark`` — BenchmarkLoader and DatasetLoadError.

The HuggingFace ``datasets.load_dataset`` is fully mocked so no network
request or dataset download occurs during the test run.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from medeval.benchmark import BenchmarkLoader, DatasetLoadError
from medeval.structures import MedicalEvalSample

# ---------------------------------------------------------------------------
# Helpers — fake HuggingFace dataset
# ---------------------------------------------------------------------------


def _make_fake_hf_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    """Build a minimal mock that behaves like a HuggingFace ``Dataset``.

    Args:
        rows: The list of row dicts the dataset should iterate over.

    Returns:
        A ``MagicMock`` with ``column_names``, ``__iter__``, ``__len__``,
        and ``select`` attributes set appropriately.
    """
    columns = list(rows[0].keys()) if rows else []
    mock_ds = MagicMock()
    mock_ds.column_names = columns
    mock_ds.__iter__ = MagicMock(return_value=iter(rows))
    mock_ds.__len__ = MagicMock(return_value=len(rows))
    mock_ds.select = MagicMock(return_value=mock_ds)  # cap returns self for simplicity
    return mock_ds


# Sample rows matching the expected MedQA schema.
_MEDQA_ROWS: list[dict[str, Any]] = [
    {
        "id": "medqa_0001",
        "question": "Which enzyme is deficient in PKU?",
        "choices": [
            {"key": "A", "value": "Phenylalanine hydroxylase"},
            {"key": "B", "value": "Tyrosinase"},
            {"key": "C", "value": "Homogentisate oxidase"},
            {"key": "D", "value": "DOPA decarboxylase"},
        ],
        "answer_idx": "A",
    },
    {
        "id": "medqa_0002",
        "question": "What is the first-line treatment for Type 2 DM?",
        "choices": [
            {"key": "A", "value": "Insulin"},
            {"key": "B", "value": "Metformin"},
        ],
        "answer_idx": "B",
    },
]

# Sample rows matching the expected PubMedQA schema.
_PUBMEDQA_ROWS: list[dict[str, Any]] = [
    {
        "pubid": "12345678",
        "question": "Does aspirin reduce cardiovascular events?",
        "context": {
            "contexts": ["Aspirin inhibits platelet aggregation.", "Studies show reduced MI risk."]
        },
        "final_decision": "yes",
    },
    {
        "pubid": "87654321",
        "question": "Is high-dose vitamin C effective against cancer?",
        "context": {"contexts": ["Evidence is inconclusive."]},
        "final_decision": "maybe",
    },
]


# ---------------------------------------------------------------------------
# BenchmarkLoader — constructor validation
# ---------------------------------------------------------------------------


class TestBenchmarkLoaderInit:
    """Tests for BenchmarkLoader constructor validation."""

    def test_raises_on_invalid_max_samples(self) -> None:
        """ValueError must be raised when max_samples is 0 or negative."""
        with pytest.raises(ValueError, match="max_samples"):
            BenchmarkLoader(max_samples=0)

        with pytest.raises(ValueError, match="max_samples"):
            BenchmarkLoader(max_samples=-10)

    def test_valid_construction(self) -> None:
        """Valid arguments must construct the loader without error."""
        loader = BenchmarkLoader(split="test", max_samples=100)
        assert loader._split == "test"
        assert loader._max_samples == 100

    def test_defaults(self) -> None:
        """Default arguments must produce a sensible loader configuration."""
        loader = BenchmarkLoader()
        assert loader._split == "test"
        assert loader._max_samples is None


# ---------------------------------------------------------------------------
# BenchmarkLoader.load_medqa
# ---------------------------------------------------------------------------


def _make_mock_datasets_module(rows: list[dict[str, Any]]) -> MagicMock:
    """Build a mock ``datasets`` module whose ``load_dataset`` returns fake data.

    Args:
        rows: The rows the fake dataset should contain.

    Returns:
        A ``MagicMock`` configured to mimic ``datasets.load_dataset``.
    """
    fake_ds = _make_fake_hf_dataset(rows)
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = fake_ds
    return mock_datasets


class TestLoadMedQA:
    """Tests for BenchmarkLoader.load_medqa()."""

    def test_returns_medical_eval_samples(self) -> None:
        """load_medqa must return a list of MedicalEvalSample objects."""
        loader = BenchmarkLoader(split="test")
        mock_datasets = _make_mock_datasets_module(_MEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_medqa()

        assert len(samples) == len(_MEDQA_ROWS)
        assert all(isinstance(s, MedicalEvalSample) for s in samples)

    def test_ground_truth_extracted_from_choices(self) -> None:
        """Ground truth must be the value for the correct answer key."""
        loader = BenchmarkLoader(split="test")
        mock_datasets = _make_mock_datasets_module(_MEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_medqa()

        assert samples[0].ground_truth == "Phenylalanine hydroxylase"
        assert samples[1].ground_truth == "Metformin"

    def test_model_prediction_is_empty_string(self) -> None:
        """model_prediction must be an empty string (to be filled by evaluation)."""
        loader = BenchmarkLoader()
        mock_datasets = _make_mock_datasets_module(_MEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_medqa()

        assert all(s.model_prediction == "" for s in samples)

    def test_metadata_contains_dataset_key(self) -> None:
        """Each sample's metadata must include 'dataset': 'medqa'."""
        loader = BenchmarkLoader()
        mock_datasets = _make_mock_datasets_module(_MEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_medqa()

        assert all(s.metadata.get("dataset") == "medqa" for s in samples)

    def test_raises_dataset_load_error_on_missing_columns(self) -> None:
        """DatasetLoadError must be raised when the dataset schema is wrong."""
        bad_rows = [{"wrong_col": "value"}]
        loader = BenchmarkLoader()
        mock_datasets = _make_mock_datasets_module(bad_rows)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            with pytest.raises(DatasetLoadError):
                loader.load_medqa()

    def test_raises_when_datasets_not_installed(self) -> None:
        """DatasetLoadError must be raised with install hint when datasets is missing."""
        loader = BenchmarkLoader()

        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises((DatasetLoadError, TypeError)):
                loader.load_medqa()


# ---------------------------------------------------------------------------
# BenchmarkLoader.load_pubmedqa
# ---------------------------------------------------------------------------


class TestLoadPubMedQA:
    """Tests for BenchmarkLoader.load_pubmedqa()."""

    def test_returns_medical_eval_samples(self) -> None:
        """load_pubmedqa must return a list of MedicalEvalSample objects."""
        loader = BenchmarkLoader(split="train")
        mock_datasets = _make_mock_datasets_module(_PUBMEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_pubmedqa()

        assert len(samples) == len(_PUBMEDQA_ROWS)
        assert all(isinstance(s, MedicalEvalSample) for s in samples)

    def test_ground_truth_is_final_decision(self) -> None:
        """Ground truth must be the 'final_decision' string."""
        loader = BenchmarkLoader()
        mock_datasets = _make_mock_datasets_module(_PUBMEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_pubmedqa()

        assert samples[0].ground_truth == "yes"
        assert samples[1].ground_truth == "maybe"

    def test_context_merged_into_metadata(self) -> None:
        """Context passages must be joined and stored in metadata['context']."""
        loader = BenchmarkLoader()
        mock_datasets = _make_mock_datasets_module(_PUBMEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_pubmedqa()

        context = samples[0].metadata.get("context", "")
        assert "Aspirin inhibits platelet aggregation." in context

    def test_metadata_contains_dataset_key(self) -> None:
        """Each sample's metadata must include 'dataset': 'pubmedqa'."""
        loader = BenchmarkLoader()
        mock_datasets = _make_mock_datasets_module(_PUBMEDQA_ROWS)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            samples = loader.load_pubmedqa()

        assert all(s.metadata.get("dataset") == "pubmedqa" for s in samples)

    def test_raises_dataset_load_error_on_missing_columns(self) -> None:
        """DatasetLoadError must be raised when the PubMedQA schema is wrong."""
        bad_rows = [{"only_wrong": "col"}]
        loader = BenchmarkLoader()
        mock_datasets = _make_mock_datasets_module(bad_rows)

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            with pytest.raises(DatasetLoadError):
                loader.load_pubmedqa()


# ---------------------------------------------------------------------------
# DatasetLoadError
# ---------------------------------------------------------------------------


class TestDatasetLoadError:
    """Tests for the DatasetLoadError custom exception."""

    def test_carries_dataset_name(self) -> None:
        """The exception must expose the dataset_name attribute."""
        error = DatasetLoadError("medqa", "Something went wrong.")
        assert error.dataset_name == "medqa"

    def test_str_includes_dataset_name_and_message(self) -> None:
        """str(error) must include both the dataset name and the message."""
        error = DatasetLoadError("pubmed_qa", "Missing column 'final_decision'.")
        assert "pubmed_qa" in str(error)
        assert "Missing column" in str(error)
