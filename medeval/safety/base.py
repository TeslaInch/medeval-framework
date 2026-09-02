"""
medeval/safety/base.py
~~~~~~~~~~~~~~~~~~~~~~
Abstract base class definition and data structures for clinical safety checkers.

Enforces a common interface for rule-based, programmatic, or model-based safety checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyViolation:
    """Immutable record of a single clinical safety violation.

    Attributes:
        code: Machine-readable violation identifier, e.g.
            ``"CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION"``.
        severity: Severity tier — either ``"CRITICAL"`` or ``"WARNING"``.
        matched_term: The exact substring in the input text that triggered
            the rule.
        rationale: A clinician-facing explanation of why this is dangerous.
    """

    code: str
    severity: str
    matched_term: str
    rationale: str


class BaseSafetyChecker(ABC):
    """Abstract base class that all safety checkers must implement.

    Ensures that custom domain-specific checkers (e.g. Cardiology, Hematology,
    Pediatrics) share the same execution contract.
    """

    @abstractmethod
    def check_contraindications(self, text: str) -> list[str]:
        """Scan text for clinical safety violations and return violation codes.

        Args:
            text: Free-text clinical recommendation or LLM output.

        Returns:
            A list of violation code strings (e.g. ["CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION"]).
        """

    @abstractmethod
    def check_contraindications_detailed(self, text: str) -> list[SafetyViolation]:
        """Scan text and return detailed structured SafetyViolation records.

        Args:
            text: Free-text clinical recommendation or LLM output.

        Returns:
            A list of SafetyViolation instances.
        """

    def is_negated(self, text: str, match_start: int, match_end: int, window: int = 50) -> bool:
        """Check if a matched term is accompanied by a negation cue.

        Scans the text immediately preceding, inside, and following the match
        (up to `window` characters) for common clinical negation or avoidance phrasing.

        Args:
            text: The full text being evaluated.
            match_start: The start index of the matched term in `text`.
            match_end: The end index of the matched term in `text`.
            window: Number of characters before and after the match to scan.

        Returns:
            True if a negation cue is found in the surrounding window, False otherwise.
        """
        import re

        negation_cues = re.compile(
            r"\b(avoid|do\s+not|don'?t|never|contraindicated|should\s+not|"
            r"must\s+not|refrain\s+from|"
            r"not\s+(?:use|give|administer|recommend|prescribe))\b",
            re.IGNORECASE,
        )

        start = max(0, match_start - window)
        end = min(len(text), match_end + window)
        surrounding_text = text[start:end]
        return bool(negation_cues.search(surrounding_text))
