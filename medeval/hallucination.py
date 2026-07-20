"""
medeval/hallucination.py
~~~~~~~~~~~~~~~~~~~~~~~~
Hallucination detection via Natural Language Inference (NLI).

Clinical hallucination — a model asserting a clinical fact that contradicts
or is unsupported by the provided context — is detected by framing the
problem as NLI:

    Premise  → The authoritative clinical context / source text.
    Hypothesis → The model's generated claim or response.

An NLI model classifies the relationship as one of:
    - ``entailment``    → claim is supported by context (NOT a hallucination)
    - ``neutral``       → claim is neither confirmed nor denied (uncertain)
    - ``contradiction`` → claim conflicts with context (definite hallucination)

We flag a hallucination if the combined (neutral + contradiction) probability
exceeds a configurable threshold (default 0.5), erring on the side of caution
for clinical safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Label names returned by the zero-shot pipeline.
_ENTAILMENT_LABEL: str = "entailment"
_NEUTRAL_LABEL: str = "neutral"
_CONTRADICTION_LABEL: str = "contradiction"

_NLI_CANDIDATE_LABELS: list[str] = [
    _ENTAILMENT_LABEL,
    _NEUTRAL_LABEL,
    _CONTRADICTION_LABEL,
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NLIResult:
    """Structured result produced by ``NLIHallucinationDetector.detect()``.

    Attributes:
        is_hallucination: ``True`` if the model's claim is classified as a
            hallucination given the context.
        entailment_score: Probability assigned to the *entailment* label.
        neutral_score: Probability assigned to the *neutral* label.
        contradiction_score: Probability assigned to the *contradiction* label.
        threshold: The threshold used for the hallucination decision.
        raw_output: Full dictionary returned by the underlying NLI pipeline,
            preserved for debugging.
    """

    is_hallucination: bool
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    threshold: float
    raw_output: dict[str, Any] = field(default_factory=dict)

    @property
    def non_entailment_score(self) -> float:
        """Combined probability of neutral + contradiction.

        Returns:
            Sum of neutral and contradiction scores. Used for the threshold
            comparison that drives the hallucination flag.
        """
        return self.neutral_score + self.contradiction_score


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class NLIHallucinationDetector:
    """Hallucination detector using a Transformer NLI / zero-shot pipeline.

    Loads a ``transformers.pipeline("zero-shot-classification")`` lazily at
    first use so the class can be instantiated without downloading the model.

    The model is queried with three candidate labels: *entailment*, *neutral*,
    and *contradiction*. A hallucination is flagged when::

        P(neutral) + P(contradiction) > threshold

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``"microsoft/deberta-v3-large-mnli"``, which achieves strong NLI
            performance on clinical text.
        threshold: Decision threshold for hallucination classification.
            Higher values are more permissive (fewer false positives).
            Defaults to ``0.5``.
        device: Torch device index for GPU acceleration (e.g. ``0``).
            ``-1`` forces CPU. Defaults to ``-1`` (CPU).

    Example:
        >>> detector = NLIHallucinationDetector(threshold=0.5)
        >>> result = detector.detect(
        ...     premise="Aspirin is an NSAID used to reduce fever and pain.",
        ...     hypothesis="Aspirin is an ACE inhibitor used to treat hypertension.",
        ... )
        >>> result.is_hallucination
        True
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large-mnli",
        threshold: float = 0.5,
        device: int = -1,
    ) -> None:
        """Initialise the detector with model, threshold, and device settings.

        Args:
            model_name: Hugging Face model identifier for the NLI pipeline.
            threshold: Combined (neutral + contradiction) probability above
                which a hallucination is flagged. Must be in (0, 1).
            device: Integer device index. ``-1`` = CPU.

        Raises:
            ValueError: If ``threshold`` is not strictly between 0 and 1.
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in the open interval (0, 1). Got: {threshold!r}.")

        self._model_name: str = model_name
        self._threshold: float = threshold
        self._device: int = device
        # Pipeline is loaded lazily on first detect() call.
        self._pipeline: Any = None

    def _load_pipeline(self) -> None:
        """Lazily load the zero-shot classification pipeline.

        Raises:
            ImportError: If ``transformers`` is not installed.
        """
        try:
            from transformers import pipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for NLIHallucinationDetector. "
                "Install it with: pip install transformers"
            ) from exc

        if self._pipeline is None:
            logger.info(
                "Loading NLI pipeline: model='%s', device=%d.",
                self._model_name,
                self._device,
            )
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=self._device,
            )

    def detect(self, premise: str, hypothesis: str) -> NLIResult:
        """Detect whether ``hypothesis`` is a hallucination given ``premise``.

        Frames the problem as NLI: the pipeline is asked whether the premise
        *entails*, is *neutral* toward, or *contradicts* the hypothesis.

        Args:
            premise: Authoritative clinical context (e.g. a retrieved passage
                from a medical knowledge base or patient record).
            hypothesis: The model's generated claim or answer to check.

        Returns:
            An :class:`NLIResult` containing the per-label probabilities and
            the final hallucination flag.

        Raises:
            ValueError: If ``premise`` or ``hypothesis`` is empty.
            ImportError: If ``transformers`` is not installed.
        """
        if not premise.strip():
            raise ValueError("'premise' must not be an empty string.")
        if not hypothesis.strip():
            raise ValueError("'hypothesis' must not be an empty string.")

        self._load_pipeline()

        # The pipeline template makes the model evaluate:
        # "Does this passage imply: <hypothesis>?"
        raw: dict[str, Any] = self._pipeline(
            premise,
            candidate_labels=_NLI_CANDIDATE_LABELS,
            hypothesis_template="{}",
        )

        # Map label → score from the pipeline's parallel lists.
        label_to_score: dict[str, float] = dict(zip(raw["labels"], raw["scores"]))

        entailment_score: float = label_to_score.get(_ENTAILMENT_LABEL, 0.0)
        neutral_score: float = label_to_score.get(_NEUTRAL_LABEL, 0.0)
        contradiction_score: float = label_to_score.get(_CONTRADICTION_LABEL, 0.0)

        non_entailment: float = neutral_score + contradiction_score
        is_hallucination: bool = non_entailment > self._threshold

        logger.debug(
            "NLI scores — entailment=%.3f, neutral=%.3f, contradiction=%.3f → "
            "hallucination=%s (threshold=%.2f)",
            entailment_score,
            neutral_score,
            contradiction_score,
            is_hallucination,
            self._threshold,
        )

        return NLIResult(
            is_hallucination=is_hallucination,
            entailment_score=entailment_score,
            neutral_score=neutral_score,
            contradiction_score=contradiction_score,
            threshold=self._threshold,
            raw_output=raw,
        )

    def detect_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
    ) -> list[NLIResult]:
        """Run hallucination detection over a parallel batch of pairs.

        Args:
            premises: List of authoritative context strings.
            hypotheses: List of model-generated claims to check.

        Returns:
            A list of :class:`NLIResult` objects, one per input pair.

        Raises:
            ValueError: If the input lists have different lengths or are empty.
        """
        if not premises or not hypotheses:
            raise ValueError("Both 'premises' and 'hypotheses' must be non-empty lists.")
        if len(premises) != len(hypotheses):
            raise ValueError(
                "Lengths of 'premises' and 'hypotheses' must match. "
                f"Got premises={len(premises)}, hypotheses={len(hypotheses)}."
            )

        return [self.detect(premise=p, hypothesis=h) for p, h in zip(premises, hypotheses)]
