"""
medeval/safety/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Public API for the medeval safety sub-package.

Exposes the domain-specific clinical safety checkers so users can import
directly from ``medeval.safety`` without navigating sub-modules.

Example:
    >>> from medeval.safety import SickleCellSafetyChecker
    >>> checker = SickleCellSafetyChecker()
    >>> checker.check_contraindications("Apply ice packs to the affected limb.")
    ['CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION']
"""

from .sickle_cell import SickleCellSafetyChecker, SafetyViolation

__all__ = [
    "SickleCellSafetyChecker",
    "SafetyViolation",
]
