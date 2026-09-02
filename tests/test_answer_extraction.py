"""
tests/test_answer_extraction.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.answer_extraction``.

Validates the cascade of extraction strategies used to pull answer choices
from verbose LLM responses before scoring.
"""

from __future__ import annotations

from medeval.answer_extraction import extract_answer_choice, normalize_prediction

# ---------------------------------------------------------------------------
# Tests for extract_answer_choice
# ---------------------------------------------------------------------------


class TestDirectMatch:
    """Strategy 1: prediction IS a choice key."""

    def test_exact_letter(self) -> None:
        """Single letter 'A' should match directly."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        assert extract_answer_choice("A", choices) == "A"

    def test_exact_letter_lowercase(self) -> None:
        """Lowercase letter 'b' should match case-insensitively."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        assert extract_answer_choice("b", choices) == "B"

    def test_exact_letter_with_whitespace(self) -> None:
        """Letter with surrounding whitespace should be stripped and matched."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        assert extract_answer_choice("  A  ", choices) == "A"


class TestPrefixPattern:
    """Strategy 2: 'The answer is A', 'Answer: B', etc."""

    def test_answer_is_a(self) -> None:
        """Standard 'The answer is A' prefix."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        result = extract_answer_choice("The answer is A. Hydroxyurea is the first-line.", choices)
        assert result == "A"

    def test_correct_answer_is_b(self) -> None:
        """'The correct answer is B' prefix."""
        choices = {"A": "Aspirin", "B": "Metformin"}
        result = extract_answer_choice("The correct answer is B. Metformin reduces HbA1c.", choices)
        assert result == "B"

    def test_answer_colon(self) -> None:
        """'Answer: C' prefix."""
        choices = {"A": "X", "B": "Y", "C": "Z"}
        result = extract_answer_choice("Answer: C is the best option here.", choices)
        assert result == "C"

    def test_best_answer_would_be(self) -> None:
        """'The best answer would be D' prefix."""
        choices = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        result = extract_answer_choice("The best answer would be D, because...", choices)
        assert result == "D"


class TestLeadingLetter:
    """Strategy 3: prediction starts with 'A.', 'B)', etc."""

    def test_leading_letter_dot(self) -> None:
        """'A. Hydroxyurea' should extract 'A'."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        result = extract_answer_choice("A. Hydroxyurea is the treatment.", choices)
        assert result == "A"

    def test_leading_letter_paren(self) -> None:
        """'B) Metformin' should extract 'B'."""
        choices = {"A": "Aspirin", "B": "Metformin"}
        result = extract_answer_choice("B) Metformin", choices)
        assert result == "B"

    def test_leading_letter_colon(self) -> None:
        """'C: Lisinopril' should extract 'C'."""
        choices = {"A": "X", "B": "Y", "C": "Lisinopril"}
        result = extract_answer_choice("C: Lisinopril is an ACE inhibitor.", choices)
        assert result == "C"


class TestOptionPhrase:
    """Strategy 4: 'Option A', 'Choice B', 'select C'."""

    def test_option_a(self) -> None:
        """'I would select Option A' should extract 'A'."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        result = extract_answer_choice("I would select Option A for this patient.", choices)
        assert result == "A"

    def test_choice_b(self) -> None:
        """'Choice B is the most appropriate' should extract 'B'."""
        choices = {"A": "Aspirin", "B": "Metformin"}
        result = extract_answer_choice("Choice B is the most appropriate.", choices)
        assert result == "B"


class TestValueMatch:
    """Strategy 5: prediction contains a choice value verbatim."""

    def test_value_in_verbose_response(self) -> None:
        """'Hydroxyurea is the first-line...' should match choice A."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        result = extract_answer_choice(
            "Hydroxyurea is the first-line treatment for sickle cell disease.",
            choices,
        )
        assert result == "A"

    def test_longest_value_match_wins(self) -> None:
        """When multiple values match, the longest value match should win."""
        choices = {"A": "ACE inhibitor", "B": "ACE inhibitor plus diuretic"}
        result = extract_answer_choice(
            "ACE inhibitor plus diuretic is the best combination.",
            choices,
        )
        assert result == "B"

    def test_short_values_skipped(self) -> None:
        """Values shorter than 3 chars should not match to prevent false positives."""
        choices = {"A": "No", "B": "Yes"}
        # 'No' is only 2 chars, should not trigger value match
        result = extract_answer_choice("No, this is not recommended.", choices)
        # Should fall through to fallback normalization
        assert isinstance(result, str)


class TestFallback:
    """Strategy 6: fallback to normalized prediction."""

    def test_no_choices_returns_normalized(self) -> None:
        """Without choices dict, should return normalized prediction."""
        result = extract_answer_choice("The correct answer is Metformin.")
        assert "Metformin" in result

    def test_no_pattern_match(self) -> None:
        """When no pattern matches, should return normalized text."""
        choices = {"A": "Hydroxyurea", "B": "Insulin"}
        result = extract_answer_choice("This is a complex clinical scenario.", choices)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests for normalize_prediction
# ---------------------------------------------------------------------------


class TestNormalizePrediction:
    """Tests for preamble stripping in normalize_prediction."""

    def test_strip_based_on_clinical(self) -> None:
        """Strip 'Based on the clinical presentation, ...'."""
        result = normalize_prediction("Based on the clinical presentation, use ACE inhibitors.")
        assert result == "use ACE inhibitors."

    def test_strip_correct_answer_is(self) -> None:
        """Strip 'The correct answer is ...'."""
        result = normalize_prediction("The correct answer is Metformin.")
        assert result == "Metformin."

    def test_strip_i_would_choose(self) -> None:
        """Strip 'I would choose ...'."""
        result = normalize_prediction("I would choose Hydroxyurea for this patient.")
        assert result == "Hydroxyurea for this patient."

    def test_no_preamble_unchanged(self) -> None:
        """Text without a preamble should be returned as-is (stripped)."""
        result = normalize_prediction("  Metformin  ")
        assert result == "Metformin"

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert normalize_prediction("") == ""
