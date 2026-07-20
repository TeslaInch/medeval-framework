"""
medeval/safety/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Public API for the medeval safety package.

Exposes base abstract classes, the composite SafetySuite, and domain-specific
safety checkers (Sickle Cell, Cardiology).
"""

from .base import BaseSafetyChecker, SafetyViolation
from .cardiology import CardiologySafetyChecker
from .sickle_cell import SickleCellSafetyChecker
from .suite import SafetySuite

__all__ = [
    "BaseSafetyChecker",
    "SafetyViolation",
    "SafetySuite",
    "SickleCellSafetyChecker",
    "CardiologySafetyChecker",
]
