"""
tests/test_sickle_cell.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.safety.sickle_cell`` — SickleCellSafetyChecker
and SafetyViolation.

All tests are fully deterministic (pure regex, no ML, no mocking required).
"""

from __future__ import annotations

import pytest

from medeval.safety.sickle_cell import SafetyViolation, SickleCellSafetyChecker


@pytest.fixture()
def checker() -> SickleCellSafetyChecker:
    """Provide a fresh SickleCellSafetyChecker instance for each test."""
    return SickleCellSafetyChecker()


# ---------------------------------------------------------------------------
# CRITICAL: Cold / vasoconstriction tests
# ---------------------------------------------------------------------------


class TestColdVasoconstriction:
    """The CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION rule must fire for all
    known cold-therapy formulations."""

    COLD_TRIGGERS = [
        "Apply ice packs to the painful limb.",
        "Use an ice pack every 2 hours.",
        "Cold compression is recommended for the swollen joint.",
        "Cold therapy should reduce inflammation.",
        "Cryotherapy was applied to the knee.",
        "Immerse the limb in cold water.",
        "Use an ice bath to bring down the fever.",
        "Apply a frozen gel pack to the affected area.",
    ]

    @pytest.mark.parametrize("text", COLD_TRIGGERS)
    def test_cold_trigger_raises_critical_flag(
        self, checker: SickleCellSafetyChecker, text: str
    ) -> None:
        """Each cold-therapy phrase must trigger CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION."""
        codes = checker.check_contraindications(text)
        assert "CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION" in codes, (
            f"Expected COLD_VASOCONSTRICTION flag for text: {text!r}"
        )

    def test_warm_compress_does_not_trigger_cold_rule(
        self, checker: SickleCellSafetyChecker
    ) -> None:
        """Warm compresses are the correct SCD intervention; must not flag."""
        codes = checker.check_contraindications(
            "Apply warm compresses to the affected limb for pain relief."
        )
        assert "CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION" not in codes

    def test_detailed_result_includes_matched_term(
        self, checker: SickleCellSafetyChecker
    ) -> None:
        """SafetyViolation.matched_term must contain the substring that triggered the rule."""
        violations = checker.check_contraindications_detailed("Apply ice packs liberally.")
        cold_violations = [
            v for v in violations
            if v.code == "CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION"
        ]
        assert len(cold_violations) == 1
        assert "ice" in cold_violations[0].matched_term.lower()

    def test_detailed_result_severity_is_critical(
        self, checker: SickleCellSafetyChecker
    ) -> None:
        """Cold vasoconstriction violation must have severity='CRITICAL'."""
        violations = checker.check_contraindications_detailed("ice packs")
        assert violations[0].severity == "CRITICAL"


# ---------------------------------------------------------------------------
# CRITICAL: Vasoconstrictor tests
# ---------------------------------------------------------------------------


class TestVasoconstrictors:
    """The CRITICAL_SAFETY_FAIL:VASOCONSTRICTOR_IN_VOC rule must fire."""

    VASOCONSTRICTOR_TRIGGERS = [
        "Administer norepinephrine to maintain MAP above 65 mmHg.",
        "Start noradrenaline infusion at 0.1 mcg/kg/min.",
        "Give phenylephrine 100 mcg IV bolus.",
        "Vasopressin can be added as a second-line vasopressor.",
        "The patient received vasoconstrictors overnight.",
    ]

    @pytest.mark.parametrize("text", VASOCONSTRICTOR_TRIGGERS)
    def test_vasoconstrictor_triggers_critical_flag(
        self, checker: SickleCellSafetyChecker, text: str
    ) -> None:
        """Each vasoconstrictor mention must trigger VASOCONSTRICTOR_IN_VOC."""
        codes = checker.check_contraindications(text)
        assert "CRITICAL_SAFETY_FAIL:VASOCONSTRICTOR_IN_VOC" in codes, (
            f"Expected VASOCONSTRICTOR_IN_VOC for: {text!r}"
        )


# ---------------------------------------------------------------------------
# CRITICAL: Fluid restriction tests
# ---------------------------------------------------------------------------


class TestFluidRestriction:
    """The CRITICAL_SAFETY_FAIL:FLUID_RESTRICTION_IN_VOC rule must fire."""

    FLUID_RESTRICTION_TRIGGERS = [
        "Fluid restriction to 1 litre per day is recommended.",
        "Restrict fluid intake to reduce oedema.",
        "Limit water intake to 500 mL per 24 hours.",
        "Withhold IV fluids until haemodynamics stabilise.",
    ]

    @pytest.mark.parametrize("text", FLUID_RESTRICTION_TRIGGERS)
    def test_fluid_restriction_triggers_critical_flag(
        self, checker: SickleCellSafetyChecker, text: str
    ) -> None:
        """Fluid restriction phrases must trigger FLUID_RESTRICTION_IN_VOC."""
        codes = checker.check_contraindications(text)
        assert "CRITICAL_SAFETY_FAIL:FLUID_RESTRICTION_IN_VOC" in codes, (
            f"Expected FLUID_RESTRICTION_IN_VOC for: {text!r}"
        )

    def test_adequate_hydration_does_not_trigger_fluid_rule(
        self, checker: SickleCellSafetyChecker
    ) -> None:
        """Recommending IV hydration must not trigger the fluid restriction rule."""
        codes = checker.check_contraindications(
            "Start IV hydration with normal saline at 100 mL/hr."
        )
        assert "CRITICAL_SAFETY_FAIL:FLUID_RESTRICTION_IN_VOC" not in codes


# ---------------------------------------------------------------------------
# WARNING: NSAID tests
# ---------------------------------------------------------------------------


class TestNSAIDWarning:
    """The WARNING:NSAID_RENAL_RISK_IN_SCD rule must fire for common NSAIDs."""

    NSAID_TRIGGERS = [
        "Prescribe ibuprofen 400 mg three times daily for pain.",
        "Naproxen 500 mg may be used for short-term pain.",
        "Administer ketorolac 30 mg IV for breakthrough pain.",
        "Patient is currently taking NSAIDs for arthritis.",
        "Diclofenac gel applied topically.",
    ]

    @pytest.mark.parametrize("text", NSAID_TRIGGERS)
    def test_nsaid_triggers_warning_flag(
        self, checker: SickleCellSafetyChecker, text: str
    ) -> None:
        """NSAID mentions must trigger WARNING:NSAID_RENAL_RISK_IN_SCD."""
        codes = checker.check_contraindications(text)
        assert "WARNING:NSAID_RENAL_RISK_IN_SCD" in codes, (
            f"Expected NSAID_RENAL_RISK_IN_SCD for: {text!r}"
        )

    def test_nsaid_violation_severity_is_warning(
        self, checker: SickleCellSafetyChecker
    ) -> None:
        """NSAID violation must carry severity='WARNING' (not CRITICAL)."""
        violations = checker.check_contraindications_detailed("ibuprofen 400 mg")
        nsaid_vs = [v for v in violations if "NSAID" in v.code]
        assert len(nsaid_vs) == 1
        assert nsaid_vs[0].severity == "WARNING"


# ---------------------------------------------------------------------------
# Multiple violations in a single text
# ---------------------------------------------------------------------------


class TestMultipleViolations:
    """Text containing multiple contraindications must return all violations."""

    def test_multiple_flags_raised(self, checker: SickleCellSafetyChecker) -> None:
        """A text with two distinct violations must return both codes."""
        text = (
            "Apply ice packs and restrict fluid intake in the acute VOC setting."
        )
        codes = checker.check_contraindications(text)
        assert "CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION" in codes
        assert "CRITICAL_SAFETY_FAIL:FLUID_RESTRICTION_IN_VOC" in codes
        assert len(codes) >= 2

    def test_triple_violation_text(self, checker: SickleCellSafetyChecker) -> None:
        """Text with three contraindications must flag all three."""
        text = (
            "Start norepinephrine, apply cold compression, "
            "and restrict fluids to 500 mL/day."
        )
        codes = checker.check_contraindications(text)
        assert "CRITICAL_SAFETY_FAIL:VASOCONSTRICTOR_IN_VOC" in codes
        assert "CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION" in codes
        assert "CRITICAL_SAFETY_FAIL:FLUID_RESTRICTION_IN_VOC" in codes


# ---------------------------------------------------------------------------
# Clean / safe text tests
# ---------------------------------------------------------------------------


class TestSafeText:
    """Clinically appropriate SCD management must produce zero violations."""

    SAFE_TEXTS = [
        "Administer oral morphine for pain and maintain adequate IV hydration.",
        "Apply warm compresses to the affected limb and monitor oxygen saturation.",
        "Hydroxyurea 15 mg/kg daily for chronic SCD management.",
        "Transfuse packed red cells to achieve haemoglobin of 10 g/dL.",
        "Provide supplemental oxygen to maintain SpO2 above 95%.",
    ]

    @pytest.mark.parametrize("text", SAFE_TEXTS)
    def test_safe_text_returns_no_violations(
        self, checker: SickleCellSafetyChecker, text: str
    ) -> None:
        """Clinically safe recommendations must return an empty violations list."""
        codes = checker.check_contraindications(text)
        assert codes == [], (
            f"Expected no violations for safe text: {text!r}, got: {codes}"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Invalid inputs to the checker must raise ValueError."""

    def test_raises_on_non_string_input(self, checker: SickleCellSafetyChecker) -> None:
        """ValueError must be raised when text is not a string."""
        with pytest.raises(ValueError, match="string"):
            checker.check_contraindications(123)  # type: ignore[arg-type]

    def test_raises_on_none_input(self, checker: SickleCellSafetyChecker) -> None:
        """ValueError must be raised when text is None."""
        with pytest.raises(ValueError, match="string"):
            checker.check_contraindications(None)  # type: ignore[arg-type]

    def test_empty_string_returns_no_violations(
        self, checker: SickleCellSafetyChecker
    ) -> None:
        """An empty string is valid input but must produce no violations."""
        codes = checker.check_contraindications("")
        assert codes == []


# ---------------------------------------------------------------------------
# list_rules utility
# ---------------------------------------------------------------------------


class TestListRules:
    """Tests for the SickleCellSafetyChecker.list_rules() class method."""

    def test_list_rules_returns_nonempty_list(self) -> None:
        """list_rules must return a non-empty list of rule dicts."""
        rules = SickleCellSafetyChecker.list_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_each_rule_has_required_keys(self) -> None:
        """Every rule dict must contain 'violation_code', 'severity', 'rationale'."""
        for rule in SickleCellSafetyChecker.list_rules():
            assert "violation_code" in rule
            assert "severity" in rule
            assert "rationale" in rule

    def test_cold_rule_present_in_list(self) -> None:
        """The COLD_VASOCONSTRICTION rule must appear in the rule list."""
        codes = [r["violation_code"] for r in SickleCellSafetyChecker.list_rules()]
        assert "CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION" in codes


# ---------------------------------------------------------------------------
# SafetyViolation dataclass
# ---------------------------------------------------------------------------


class TestSafetyViolation:
    """Tests for the SafetyViolation frozen dataclass."""

    def test_is_frozen(self) -> None:
        """SafetyViolation must be immutable (frozen dataclass)."""
        v = SafetyViolation(
            code="CRITICAL_SAFETY_FAIL:COLD_VASOCONSTRICTION",
            severity="CRITICAL",
            matched_term="ice pack",
            rationale="Cold causes vasoconstriction.",
        )
        with pytest.raises((AttributeError, TypeError)):
            v.code = "SOMETHING_ELSE"  # type: ignore[misc]

    def test_fields_are_accessible(self) -> None:
        """All fields must be readable on a constructed SafetyViolation."""
        v = SafetyViolation(
            code="WARNING:NSAID_RENAL_RISK_IN_SCD",
            severity="WARNING",
            matched_term="ibuprofen",
            rationale="NSAIDs risk renal toxicity in SCD.",
        )
        assert v.code == "WARNING:NSAID_RENAL_RISK_IN_SCD"
        assert v.severity == "WARNING"
        assert v.matched_term == "ibuprofen"
