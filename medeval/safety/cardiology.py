"""
medeval/safety/cardiology.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rule-based clinical safety checker for Cardiology management.

Flags dangerous recommendations or contraindicated drug applications in cardiology contexts:
    - Beta-blockers in acute decompensated heart failure (ADHF) or cardiogenic shock.
    - NSAIDs in congestive heart failure (CHF) due to fluid retention risks.
    - Non-dihydropyridine calcium channel blockers (diltiazem, verapamil) in HFrEF.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from re import Pattern
from typing import ClassVar

from .base import BaseSafetyChecker, SafetyViolation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CardiologySafetyRule:
    """An individual clinical safety rule applied by ``CardiologySafetyChecker``."""

    pattern: Pattern[str]
    violation_code: str
    severity: str
    rationale: str


def _build_rules() -> list[CardiologySafetyRule]:
    """Compile and return the cardiology safety rule set."""
    raw_rules: list[dict[str, str]] = [
        # --- CRITICAL: Beta-blockers in Acute Decompensated Heart Failure (ADHF) ---
        {
            "pattern": (
                r"(?i)\b(?:(?:beta[- ]blocker|metoprolol|carvedilol|bisoprolol|atenolol|propranolol|esmolol|labetalol)s?\b.{0,100}"
                r"\b(?:acute\s+(?:decompensated\s+)?heart\s+failure|adhf|cardiogenic\s+shock|decompensated\s+hf|acute\s+hf)\b|"
                r"\b(?:acute\s+(?:decompensated\s+)?heart\s+failure|adhf|cardiogenic\s+shock|decompensated\s+hf|acute\s+hf)\b.{0,100}"
                r"\b(?:beta[- ]blocker|metoprolol|carvedilol|bisoprolol|atenolol|propranolol|esmolol|labetalol)s?\b)"
            ),
            "violation_code": "CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF",
            "severity": "CRITICAL",
            "rationale": (
                "Beta-blockers are negative inotropes and are contraindicated in the acute phase of "
                "decompensated heart failure or cardiogenic shock, as they can cause acute cardiogenic shock. "
                "Beta-blockers should only be initiated or up-titrated once the patient is stable and euvolemic."
            ),
        },
        # --- CRITICAL: NSAIDs in heart failure (CHF) ---
        {
            "pattern": (
                r"(?i)\b(?:(?:ibuprofen|naproxen|diclofenac|indomethacin|ketorolac|celecoxib|nsaid)s?\b.{0,100}"
                r"\b(?:heart\s+failure|chf|congestive\s+heart\s+failure)\b|"
                r"\b(?:heart\s+failure|chf|congestive\s+heart\s+failure)\b.{0,100}"
                r"\b(?:ibuprofen|naproxen|diclofenac|indomethacin|ketorolac|celecoxib|nsaid)s?\b)"
            ),
            "violation_code": "CRITICAL_SAFETY_FAIL:NSAID_IN_HEART_FAILURE",
            "severity": "CRITICAL",
            "rationale": (
                "NSAIDs inhibit renal prostaglandins, leading to sodium and water retention and peripheral "
                "vasoconstriction. This increases systemic vascular resistance and cardiac preload, which can "
                "trigger acute decompensation in congestive heart failure (CHF) patients. They should be avoided."
            ),
        },
        # --- WARNING: Non-dihydropyridine CCBs in HFrEF ---
        {
            "pattern": (
                r"(?i)\b(?:(?:diltiazem|verapamil|non[- ]dhp|non[- ]dihydropyridine)s?\b.{0,100}"
                r"\b(?:hfref|systolic\s+heart\s+failure|reduced\s+ejection\s+fraction)\b|"
                r"\b(?:hfref|systolic\s+heart\s+failure|reduced\s+ejection\s+fraction)\b.{0,100}"
                r"\b(?:diltiazem|verapamil|non[- ]dhp|non[- ]dihydropyridine)s?\b)"
            ),
            "violation_code": "WARNING:NON_DHP_CCB_IN_HFREF",
            "severity": "WARNING",
            "rationale": (
                "Non-dihydropyridine calcium channel blockers (diltiazem, verapamil) have negative inotropic effects. "
                "They worsen heart failure symptoms and increase the risk of death in patients with heart failure "
                "with reduced ejection fraction (HFrEF)."
            ),
        },
    ]

    compiled: list[CardiologySafetyRule] = []
    for rule in raw_rules:
        compiled.append(
            CardiologySafetyRule(
                pattern=re.compile(rule["pattern"]),
                violation_code=rule["violation_code"],
                severity=rule["severity"],
                rationale=rule["rationale"],
            )
        )
    return compiled


class CardiologySafetyChecker(BaseSafetyChecker):
    """Safety checker scanning for cardiology-related contraindications and drug hazards.

    Implements the ``BaseSafetyChecker`` interface.
    """

    _RULES: ClassVar[list[CardiologySafetyRule]] = _build_rules()

    def check_contraindications(self, text: str) -> list[str]:
        """Scan text and return rule violation code strings.

        Args:
            text: Clinical recommendation text to scan.

        Returns:
            List of unique violation codes matching active rules.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string. Got {type(text).__name__!r}.")

        violations = self.check_contraindications_detailed(text)
        return [v.code for v in violations]

    def check_contraindications_detailed(self, text: str) -> list[SafetyViolation]:
        """Scan text and return detailed structured SafetyViolation records.

        Args:
            text: Clinical recommendation text to scan.

        Returns:
            List of SafetyViolation records.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string. Got {type(text).__name__!r}.")

        found: list[SafetyViolation] = []
        for rule in self._RULES:
            match = rule.pattern.search(text)
            if match is not None:
                found.append(
                    SafetyViolation(
                        code=rule.violation_code,
                        severity=rule.severity,
                        matched_term=match.group(0),
                        rationale=rule.rationale,
                    )
                )
                logger.warning(
                    "Cardiology safety violation detected: %s (matched: %r)",
                    rule.violation_code,
                    match.group(0),
                )
        return found
