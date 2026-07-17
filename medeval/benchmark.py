"""
medeval/benchmark.py
~~~~~~~~~~~~~~~~~~~~
Dataset loading utilities that adapt HuggingFace ``datasets`` into the
``MedicalEvalSample`` data contract.

Provides:
    - ``DatasetLoadError``: Custom exception for schema or load failures.
    - ``BenchmarkLoader``: Wraps ``datasets.load_dataset`` with typed loaders
      for MedQA and PubMedQA, returning ``List[MedicalEvalSample]``.

All loaders perform schema validation and raise ``DatasetLoadError`` with
actionable messages if the expected columns are absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .structures import MedicalEvalSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DatasetLoadError(Exception):
    """Raised when a benchmark dataset cannot be loaded or has an unexpected schema.

    Attributes:
        dataset_name: The name of the dataset that caused the error.
        message: A human-readable description of what went wrong.
    """

    def __init__(self, dataset_name: str, message: str) -> None:
        """Initialise with the dataset name and a descriptive error message.

        Args:
            dataset_name: Identifier of the dataset (e.g. ``"medqa"``).
            message: Explanation of the failure.
        """
        self.dataset_name = dataset_name
        super().__init__(f"[{dataset_name}] {message}")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class BenchmarkLoader:
    """Loads standard medical benchmark datasets into ``MedicalEvalSample`` objects.

    Wraps ``datasets.load_dataset`` (HuggingFace) and applies schema-specific
    mapping logic for each supported benchmark. The resulting
    ``List[MedicalEvalSample]`` can be passed directly to
    :class:`~medeval.report.ReportGenerator`.

    Supported datasets:
        - **MedQA** (``"bigbio/med_qa"``): US medical licensing exam questions.
        - **PubMedQA** (``"pubmed_qa"``): biomedical research question answering.

    Args:
        split: HuggingFace dataset split to load (e.g. ``"train"``,
            ``"validation"``, ``"test"``). Defaults to ``"test"`` since we
            are running evaluation, not training.
        max_samples: If provided, truncates the loaded dataset to at most this
            many samples. Useful for fast smoke-testing. ``None`` means load
            all samples.
        cache_dir: Optional path to override the HuggingFace cache directory.

    Example:
        >>> loader = BenchmarkLoader(split="test", max_samples=50)
        >>> samples = loader.load_medqa()
        >>> len(samples) <= 50
        True
    """

    def __init__(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialise the loader with split, sample cap, and cache configuration.

        Args:
            split: Dataset split to load.
            max_samples: Maximum number of samples to return. ``None`` = no cap.
            cache_dir: HuggingFace datasets cache directory override.

        Raises:
            ValueError: If ``max_samples`` is provided but is not a positive integer.
        """
        if max_samples is not None and max_samples < 1:
            raise ValueError(
                f"max_samples must be a positive integer. Got: {max_samples!r}."
            )

        self._split: str = split
        self._max_samples: Optional[int] = max_samples
        self._cache_dir: Optional[str] = cache_dir

    def _load_hf_dataset(self, dataset_name: str, config: Optional[str] = None) -> Any:
        """Load a HuggingFace dataset, raising ``DatasetLoadError`` on failure.

        Args:
            dataset_name: HuggingFace dataset identifier.
            config: Optional dataset configuration name (for datasets with
                multiple sub-configs).

        Returns:
            A HuggingFace ``Dataset`` object for the configured split.

        Raises:
            DatasetLoadError: If the ``datasets`` package is not installed or
                if the dataset cannot be fetched.
        """
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError as exc:
            raise DatasetLoadError(
                dataset_name,
                "The 'datasets' package is required. Install it with: pip install datasets",
            ) from exc

        try:
            load_kwargs: Dict[str, Any] = {"split": self._split}
            if self._cache_dir is not None:
                load_kwargs["cache_dir"] = self._cache_dir

            args = (dataset_name,) if config is None else (dataset_name, config)
            dataset = load_dataset(*args, **load_kwargs)
        except Exception as exc:
            raise DatasetLoadError(
                dataset_name,
                f"Failed to load dataset: {exc}",
            ) from exc

        return dataset

    def _cap(self, dataset: Any) -> Any:
        """Apply the ``max_samples`` cap to a loaded HuggingFace dataset.

        Args:
            dataset: A HuggingFace ``Dataset`` object.

        Returns:
            The same dataset, or a ``select()``-truncated slice if
            ``max_samples`` is configured.
        """
        if self._max_samples is not None:
            n = min(self._max_samples, len(dataset))
            dataset = dataset.select(range(n))
            logger.info("Capped dataset to %d samples.", n)
        return dataset

    # ------------------------------------------------------------------
    # MedQA
    # ------------------------------------------------------------------

    def load_medqa(self) -> List[MedicalEvalSample]:
        """Load MedQA and map to ``MedicalEvalSample`` objects.

        Uses the ``bigbio/med_qa`` HuggingFace dataset. Each row is a
        multiple-choice question with a ``question``, a list of answer
        ``options`` (dicts with ``key`` and ``value``), and a
        ``answer_idx`` indicating the correct option key.

        The ground truth is extracted from the options list by matching the
        ``answer_idx`` key.

        Returns:
            A list of ``MedicalEvalSample`` objects. ``model_prediction`` is
            set to an empty string (to be filled by downstream evaluation).

        Raises:
            DatasetLoadError: If the dataset cannot be loaded or the expected
                columns are missing.
        """
        dataset_name = "bigbio/med_qa"
        raw = self._load_hf_dataset(dataset_name, config="med_qa_en_bigbio_qa")
        raw = self._cap(raw)

        required_columns = {"id", "question", "choices", "answer_idx"}
        actual_columns = set(raw.column_names)
        # Fall back gracefully if schema differs between dataset versions.
        missing = required_columns - actual_columns
        if missing:
            raise DatasetLoadError(
                dataset_name,
                f"Expected columns {required_columns!r} but missing: {missing!r}. "
                f"Available columns: {actual_columns!r}.",
            )

        samples: List[MedicalEvalSample] = []
        for i, row in enumerate(raw):
            # choices is a list of dicts: [{"key": "A", "value": "..."}, ...]
            choice_map: Dict[str, str] = {
                c["key"]: c["value"] for c in row["choices"]
            }
            answer_key: str = row["answer_idx"]
            ground_truth: str = choice_map.get(answer_key, answer_key)

            sample = MedicalEvalSample(
                id=str(row.get("id", f"medqa_{i}")),
                question=str(row["question"]),
                ground_truth=ground_truth,
                model_prediction="",
                metadata={
                    "dataset": "medqa",
                    "split": self._split,
                    "choices": choice_map,
                    "answer_key": answer_key,
                },
            )
            samples.append(sample)

        logger.info("Loaded %d MedQA samples from split='%s'.", len(samples), self._split)
        return samples

    # ------------------------------------------------------------------
    # PubMedQA
    # ------------------------------------------------------------------

    def load_pubmedqa(self) -> List[MedicalEvalSample]:
        """Load PubMedQA and map to ``MedicalEvalSample`` objects.

        Uses the ``pubmed_qa`` HuggingFace dataset with the
        ``pqa_labeled`` configuration, which contains human-labeled examples.
        Each row contains a ``question``, a ``context`` (list of passages),
        and a ``final_decision`` (``"yes"``, ``"no"``, or ``"maybe"``).

        Returns:
            A list of ``MedicalEvalSample`` objects. The ``question`` field
            contains the research question; ``ground_truth`` is the
            ``final_decision`` string.

        Raises:
            DatasetLoadError: If the dataset cannot be loaded or the expected
                columns are missing.
        """
        dataset_name = "pubmed_qa"
        raw = self._load_hf_dataset(dataset_name, config="pqa_labeled")
        raw = self._cap(raw)

        required_columns = {"pubid", "question", "context", "final_decision"}
        actual_columns = set(raw.column_names)
        missing = required_columns - actual_columns
        if missing:
            raise DatasetLoadError(
                dataset_name,
                f"Expected columns {required_columns!r} but missing: {missing!r}. "
                f"Available columns: {actual_columns!r}.",
            )

        samples: List[MedicalEvalSample] = []
        for i, row in enumerate(raw):
            # context is a dict with "contexts" (list of passage strings)
            context_obj = row.get("context", {})
            context_passages: List[str] = (
                context_obj.get("contexts", [])
                if isinstance(context_obj, dict)
                else []
            )
            context_text: str = " ".join(context_passages)

            sample = MedicalEvalSample(
                id=str(row.get("pubid", f"pubmedqa_{i}")),
                question=str(row["question"]),
                ground_truth=str(row["final_decision"]),
                model_prediction="",
                metadata={
                    "dataset": "pubmedqa",
                    "split": self._split,
                    "context": context_text,
                },
            )
            samples.append(sample)

        logger.info(
            "Loaded %d PubMedQA samples from split='%s'.", len(samples), self._split
        )
        return samples
