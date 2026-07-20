"""
tests/test_safety_suite.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the generalised ``SafetySuite`` composite class.

Tests verify composite behavior, violation merging/deduplication, error handling,
and custom checker registration.
"""

from __future__ import annotations

import pytest

from medeval.safety import BaseSafetyChecker, SafetySuite, SafetyViolation

# ---------------------------------------------------------------------------
# Mock Safety Checkers for Testing
# ---------------------------------------------------------------------------


class MockCardiologyChecker(BaseSafetyChecker):
    """Mock cardiology safety checker."""

    def check_contraindications(self, text: str) -> list[str]:
        """Mock check returning a single cardiolody fail code."""
        if "beta-blocker" in text:
            return ["CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF"]
        return []

    def check_contraindications_detailed(self, text: str) -> list[SafetyViolation]:
        """Mock detailed check."""
        if "beta-blocker" in text:
            return [
                SafetyViolation(
                    code="CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF",
                    severity="CRITICAL",
                    matched_term="beta-blocker",
                    rationale="Avoid beta-blockers in acute decompensated heart failure.",
                )
            ]
        return []


class MockPharmacologyChecker(BaseSafetyChecker):
    """Mock pharmacology safety checker."""

    def check_contraindications(self, text: str) -> list[str]:
        """Mock check returning duplicate and distinct codes."""
        res: list[str] = []
        if "NSAID" in text:
            res.append("WARNING:NSAID_RENAL_RISK_IN_SCD")
        if "beta-blocker" in text:
            # Duplicate code to test merging behavior of the composite suite
            res.append("CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF")
        return res

    def check_contraindications_detailed(self, text: str) -> list[SafetyViolation]:
        """Mock detailed check."""
        res: list[SafetyViolation] = []
        if "NSAID" in text:
            res.append(
                SafetyViolation(
                    code="WARNING:NSAID_RENAL_RISK_IN_SCD",
                    severity="WARNING",
                    matched_term="NSAID",
                    rationale="NSAIDs carry renal risk in vulnerable patients.",
                )
            )
        if "beta-blocker" in text:
            res.append(
                SafetyViolation(
                    code="CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF",
                    severity="CRITICAL",
                    matched_term="beta-blocker",
                    rationale="Alternative rationale description.",
                )
            )
        return res


class MockFailingChecker(BaseSafetyChecker):
    """Mock safety checker that throws exceptions to verify suite resilience."""

    def check_contraindications(self, text: str) -> list[str]:
        """Throw a RuntimeError."""
        raise RuntimeError("Fatal database connection error.")

    def check_contraindications_detailed(self, text: str) -> list[SafetyViolation]:
        """Throw a RuntimeError."""
        raise RuntimeError("Fatal database connection error.")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestSafetySuiteComposite:
    """Tests verify the Composite Pattern behaviour of SafetySuite."""

    def test_empty_suite_returns_empty_list(self) -> None:
        """SafetySuite with no checkers must return empty lists without error."""
        suite = SafetySuite()
        assert suite.check_contraindications("beta-blocker and NSAID") == []
        assert suite.check_contraindications_detailed("beta-blocker and NSAID") == []

    def test_register_non_checker_raises_type_error(self) -> None:
        """Registering a class that doesn't inherit from BaseSafetyChecker must raise TypeError."""
        suite = SafetySuite()
        with pytest.raises(TypeError, match="BaseSafetyChecker"):
            suite.add_checker("not a checker")  # type: ignore[arg-type]

    def test_adds_checkers_correctly(self) -> None:
        """SafetySuite.checkers property must reflect registered sub-checkers."""
        cardio = MockCardiologyChecker()
        pharm = MockPharmacologyChecker()
        suite = SafetySuite(checkers=[cardio])
        suite.add_checker(pharm)

        assert len(suite.checkers) == 2
        assert cardio in suite.checkers
        assert pharm in suite.checkers

    def test_suite_merges_and_deduplicates_violation_codes(self) -> None:
        """SafetySuite must merge and deduplicate identical codes from different checkers."""
        suite = SafetySuite()
        suite.add_checker(MockCardiologyChecker())
        suite.add_checker(MockPharmacologyChecker())

        # String triggers both MockCardiologyChecker and MockPharmacologyChecker.
        # Both return 'CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF'.
        # MockPharmacologyChecker also returns 'WARNING:NSAID_RENAL_RISK_IN_SCD'.
        text = "Administer beta-blocker and NSAID."
        codes = suite.check_contraindications(text)

        assert len(codes) == 2
        assert "CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF" in codes
        assert "WARNING:NSAID_RENAL_RISK_IN_SCD" in codes
        # Confirm list size is exactly 2 (no duplicate 'BETA_BLOCKER_IN_ACUTE_HF' code)
        assert codes.count("CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF") == 1

    def test_suite_merges_and_deduplicates_detailed_violations(self) -> None:
        """SafetySuite must merge and deduplicate detailed violations."""
        suite = SafetySuite()
        suite.add_checker(MockCardiologyChecker())
        suite.add_checker(MockPharmacologyChecker())

        text = "Administer beta-blocker and NSAID."
        violations = suite.check_contraindications_detailed(text)

        assert len(violations) == 2
        codes = [v.code for v in violations]
        assert "CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF" in codes
        assert "WARNING:NSAID_RENAL_RISK_IN_SCD" in codes

    def test_suite_resilience_to_subchecker_exceptions(self) -> None:
        """SafetySuite must log exceptions from failing sub-checkers and continue executing."""
        suite = SafetySuite()
        suite.add_checker(MockFailingChecker())  # will raise RuntimeError
        suite.add_checker(MockCardiologyChecker())  # should execute successfully

        text = "beta-blocker is administered."

        # check_contraindications must run and successfully catch the cardiolody violation
        codes = suite.check_contraindications(text)
        assert codes == ["CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF"]

        # check_contraindications_detailed must run and successfully return detailed violation
        violations = suite.check_contraindications_detailed(text)
        assert len(violations) == 1
        assert violations[0].code == "CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF"
