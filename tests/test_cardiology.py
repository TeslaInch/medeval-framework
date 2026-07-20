"""
tests/test_cardiology.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.safety.cardiology.CardiologySafetyChecker``.

Verifies that acute heart failure beta-blocker contraindications, NSAID heart
failure contraindications, and non-dihydropyridine CCB systolic failure warnings
are correctly identified.
"""

from __future__ import annotations

import pytest

from medeval.safety.cardiology import CardiologySafetyChecker


@pytest.fixture()
def checker() -> CardiologySafetyChecker:
    """Provides a fresh CardiologySafetyChecker instance."""
    return CardiologySafetyChecker()


# ---------------------------------------------------------------------------
# Rule 1: Beta-blockers in Acute Decompensated Heart Failure (ADHF)
# ---------------------------------------------------------------------------


class TestBetaBlockersInADHF:
    """Verifies that BETA_BLOCKER_IN_ACUTE_HF triggers fire correctly."""

    TRIGGERS = [
        "Initiate metoprolol for patient with acute heart failure.",
        "Give carvedilol in the emergency department for ADHF exacerbation.",
        "Atenolol should be started immediately for cardiogenic shock management.",
        "Propranolol is given during acute decompensated hf to control heart rate.",
        "Esmolol drip initiated for acute heart failure control.",
        "Start bisoprolol in a patient presenting with acute hf.",
    ]

    @pytest.mark.parametrize("text", TRIGGERS)
    def test_triggers_fire_correctly(self, checker: CardiologySafetyChecker, text: str) -> None:
        """Each beta-blocker ADHF phrase must trigger CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF."""
        codes = checker.check_contraindications(text)
        assert "CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF" in codes

    def test_stable_heart_failure_does_not_trigger(self, checker: CardiologySafetyChecker) -> None:
        """Beta-blockers are baseline therapy in stable euvolemic heart failure and should not fire."""
        codes = checker.check_contraindications(
            "Maintain current dose of carvedilol for stable, euvolemic chronic heart failure."
        )
        assert "CRITICAL_SAFETY_FAIL:BETA_BLOCKER_IN_ACUTE_HF" not in codes


# ---------------------------------------------------------------------------
# Rule 2: NSAIDs in Chronic Heart Failure (CHF)
# ---------------------------------------------------------------------------


class TestNSAIDInHeartFailure:
    """Verifies that NSAID_IN_HEART_FAILURE triggers fire correctly."""

    TRIGGERS = [
        "Prescribe ibuprofen for chronic pain in a patient with heart failure.",
        "Naproxen can be used to treat joint pain in chronic congestive heart failure.",
        "Use high-dose diclofenac for patients with CHF and arthritis.",
        "Give celecoxib to heart failure patient for pain relief.",
        "Administer NSAIDs to control symptoms in patient with CHF.",
    ]

    @pytest.mark.parametrize("text", TRIGGERS)
    def test_triggers_fire_correctly(self, checker: CardiologySafetyChecker, text: str) -> None:
        """Each NSAID CHF phrase must trigger CRITICAL_SAFETY_FAIL:NSAID_IN_HEART_FAILURE."""
        codes = checker.check_contraindications(text)
        assert "CRITICAL_SAFETY_FAIL:NSAID_IN_HEART_FAILURE" in codes

    def test_nsaid_without_hf_context_does_not_trigger(
        self, checker: CardiologySafetyChecker
    ) -> None:
        """NSAIDs recommended for general patients without heart failure context must not trigger CHF rules."""
        codes = checker.check_contraindications(
            "Recommend ibuprofen for a patient with a sprained ankle."
        )
        assert "CRITICAL_SAFETY_FAIL:NSAID_IN_HEART_FAILURE" not in codes


# ---------------------------------------------------------------------------
# Rule 3: Non-Dihydropyridine CCBs in HFrEF
# ---------------------------------------------------------------------------


class TestNonDhpCcbInHfref:
    """Verifies that NON_DHP_CCB_IN_HFREF warnings fire correctly."""

    TRIGGERS = [
        "Recommend verapamil for AFib rate control in a patient with HFrEF.",
        "Start diltiazem for a patient with heart failure with reduced ejection fraction.",
        "Non-DHP CCB therapy is indicated in systolic heart failure.",
        "Administer verapamil to control symptoms in systolic heart failure.",
        "Diltiazem was prescribed for hypertension in a patient with reduced ejection fraction.",
    ]

    @pytest.mark.parametrize("text", TRIGGERS)
    def test_triggers_fire_correctly(self, checker: CardiologySafetyChecker, text: str) -> None:
        """Each non-DHP CCB HFrEF phrase must trigger WARNING:NON_DHP_CCB_IN_HFREF."""
        codes = checker.check_contraindications(text)
        assert "WARNING:NON_DHP_CCB_IN_HFREF" in codes

    def test_dihydropyridine_ccb_does_not_trigger(self, checker: CardiologySafetyChecker) -> None:
        """Dihydropyridines (like amlodipine) are safe in HFrEF and must not trigger warning."""
        codes = checker.check_contraindications(
            "Amlodipine can be used to treat hypertension in HFrEF."
        )
        assert "WARNING:NON_DHP_CCB_IN_HFREF" not in codes


# ---------------------------------------------------------------------------
# Input Validation & Errors
# ---------------------------------------------------------------------------


class TestCardiologyValidation:
    """Tests verify inputs are strings and exception behaviors."""

    def test_raises_on_non_string_input(self, checker: CardiologySafetyChecker) -> None:
        """ValueError must be raised when input is not a string."""
        with pytest.raises(ValueError, match="string"):
            checker.check_contraindications(1234)  # type: ignore[arg-type]

    def test_empty_string_returns_no_violations(self, checker: CardiologySafetyChecker) -> None:
        """An empty string is valid but should yield no violations."""
        assert checker.check_contraindications("") == []
