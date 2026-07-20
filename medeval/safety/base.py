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
