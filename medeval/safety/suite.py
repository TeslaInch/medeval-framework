"""
medeval/safety/suite.py
~~~~~~~~~~~~~~~~~~~~~~~
Composite safety checker suite registry.

Allows multiple domain-specific safety checkers to be registered and run
together as a single composite safety checker.
"""

from __future__ import annotations

import logging

from .base import BaseSafetyChecker, SafetyViolation

logger = logging.getLogger(__name__)


class SafetySuite(BaseSafetyChecker):
    """Composite safety checker wrapping a collection of sub-checkers.

    Implements the ``BaseSafetyChecker`` interface. When executed, it runs
    all registered checkers and merges their clinical safety violations.

    Example:
        >>> suite = SafetySuite()
        >>> suite.add_checker(SickleCellSafetyChecker())
        >>> suite.add_checker(CardiologySafetyChecker())
        >>> violations = suite.check_contraindications("some patient treatment")
    """

    def __init__(self, checkers: list[BaseSafetyChecker] | None = None) -> None:
        """Initialise the suite with optional initial list of checkers."""
        self._checkers: list[BaseSafetyChecker] = list(checkers) if checkers is not None else []

    def add_checker(self, checker: BaseSafetyChecker) -> None:
        """Register a new safety checker to the suite.

        Args:
            checker: An instance of a class implementing ``BaseSafetyChecker``.

        Raises:
            TypeError: If the checker does not implement ``BaseSafetyChecker``.
        """
        if not isinstance(checker, BaseSafetyChecker):
            raise TypeError("checker must be an instance of BaseSafetyChecker.")
        self._checkers.append(checker)
        logger.info("Registered safety checker: %s", checker.__class__.__name__)

    @property
    def checkers(self) -> list[BaseSafetyChecker]:
        """Get the list of registered safety checkers.

        Returns:
            A list of registered safety checkers.
        """
        return list(self._checkers)

    def check_contraindications(self, text: str) -> list[str]:
        """Run all registered safety checkers and return unique violation codes.

        Args:
            text: Free-text recommendation.

        Returns:
            Merged list of violation codes.
        """
        merged_codes: list[str] = []
        for checker in self._checkers:
            try:
                codes = checker.check_contraindications(text)
                for code in codes:
                    # Maintain order but avoid duplicates
                    if code not in merged_codes:
                        merged_codes.append(code)
            except Exception as exc:
                logger.error(
                    "Error running safety checker %s: %s",
                    checker.__class__.__name__,
                    exc,
                )
        return merged_codes

    def check_contraindications_detailed(self, text: str) -> list[SafetyViolation]:
        """Run all registered safety checkers and return aggregated SafetyViolation records.

        Args:
            text: Free-text recommendation.

        Returns:
            Aggregated list of SafetyViolation records.
        """
        merged_violations: list[SafetyViolation] = []
        # Track already added codes to avoid duplicate violation objects
        added_codes = set()
        for checker in self._checkers:
            try:
                violations = checker.check_contraindications_detailed(text)
                for violation in violations:
                    if violation.code not in added_codes:
                        merged_violations.append(violation)
                        added_codes.add(violation.code)
            except Exception as exc:
                logger.error(
                    "Error running safety checker %s: %s",
                    checker.__class__.__name__,
                    exc,
                )
        return merged_violations
