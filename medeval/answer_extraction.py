"""
medeval/answer_extraction.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Robust answer extraction from verbose LLM responses.

Generative models often return verbose explanations instead of a single
answer choice letter.  For example::

    "Based on the clinical presentation, the best answer is A. Hydroxyurea
     is the first-line treatment for sickle cell disease because..."

This module provides utilities to extract the core answer (``"A"``) from
such responses before scoring, so that ``y_true`` (and consequently ECE,
MCE, Brier Score) are computed correctly.

Two main functions are exposed:

* :func:`extract_answer_choice` — extracts a letter choice from a verbose
  prediction when a ``choices`` dict is available.
* :func:`normalize_prediction` — strips common LLM preamble boilerplate
  to reveal the core answer string.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for answer extraction (ordered by specificity)
# ---------------------------------------------------------------------------

# Matches patterns like "The answer is A", "Answer: B", "answer is (C)",
# "The correct answer is D.", "my answer: A"
_PREFIX_PATTERN = re.compile(
    r"(?:the\s+)?(?:correct\s+|best\s+|most\s+likely\s+)?"
    r"answer\s*(?:is|:|would\s+be)\s*\(?([A-E])\)?"
    r"[\.\,\:\s\*]",
    re.IGNORECASE,
)

# Matches a standalone letter at the very start: "A.", "A)", "A ", "A\n"
_LEADING_LETTER_PATTERN = re.compile(
    r"^\s*\(?([A-E])\)?[\.\)\:\s\*]",
    re.IGNORECASE,
)

# Matches "Option A", "Choice B", "select A"
_OPTION_PATTERN = re.compile(
    r"(?:option|choice|select|pick|chose|choose)\s*\(?([A-E])\)?",
    re.IGNORECASE,
)

# Common LLM preamble phrases to strip for normalization
_PREAMBLE_PATTERNS = [
    re.compile(
        r"^(?:based\s+on\s+(?:the\s+)?(?:clinical\s+)?(?:presentation|information|data|findings|history)[\s,]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:the\s+)?(?:correct|best|most\s+(?:likely|appropriate))\s+answer\s+(?:is|would\s+be)\s*:?\s*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:I\s+(?:would|will)\s+(?:choose|select|pick|go\s+with)\s+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:In\s+this\s+case|Given\s+the\s+(?:above|information))[\s,]*)",
        re.IGNORECASE,
    ),
]


def extract_answer_choice(
    prediction: str,
    choices: dict[str, str] | None = None,
) -> str:
    """Extract a concise answer choice from a verbose LLM prediction.

    Uses a cascade of extraction strategies, from most specific to least:

    1. **Direct match** — prediction is exactly a choice key (``"A"``).
    2. **Prefix pattern** — ``"The answer is A"``, ``"Answer: B"``, etc.
    3. **Leading letter** — prediction starts with ``"A."`` or ``"B)"``.
    4. **Option phrase** — ``"Option A"``, ``"Choice B"``, ``"select C"``.
    5. **Value match** — prediction contains a choice *value* verbatim
       (e.g. ``"Hydroxyurea"`` → key ``"A"`` if ``choices["A"] == "Hydroxyurea"``).
    6. **Fallback** — returns ``normalize_prediction(prediction)``.

    Args:
        prediction: Raw model-generated response text.
        choices: Optional mapping of choice keys to choice values
            (e.g. ``{"A": "Hydroxyurea", "B": "Insulin"}``).

    Returns:
        The extracted answer string. For multiple-choice tasks this will
        typically be a single letter (``"A"``, ``"B"``, etc.). For
        open-ended tasks it returns the normalized prediction.

    Examples:
        >>> extract_answer_choice("The answer is A. Hydroxyurea is...", {"A": "Hydroxyurea", "B": "Insulin"})
        'A'
        >>> extract_answer_choice("B) Metformin", {"A": "Aspirin", "B": "Metformin"})
        'B'
        >>> extract_answer_choice("Hydroxyurea is the first-line treatment.", {"A": "Hydroxyurea", "B": "Insulin"})
        'A'
    """
    if not prediction or not isinstance(prediction, str):
        return prediction or ""

    # Strip markdown asterisks to prevent regex matching failures
    prediction_clean = prediction.replace("**", "").replace("*", "")
    pred_stripped = prediction_clean.strip()

    # Strategy 1: Direct match — prediction IS a choice key
    if choices:
        keys_upper = {str(k).upper(): str(k) for k in choices.keys()}
        if pred_stripped.upper() in keys_upper:
            extracted = keys_upper[pred_stripped.upper()]
            logger.debug("Answer extraction: direct match → %r", extracted)
            return extracted

    # Strategy 2: Prefix pattern — "The answer is A", "Answer: B"
    match = _PREFIX_PATTERN.search(prediction_clean)
    if match:
        letter = match.group(1).upper()
        if choices is None or letter in {str(k).upper() for k in choices.keys()}:
            logger.debug("Answer extraction: prefix pattern → %r", letter)
            return letter

    # Strategy 3: Leading letter — "A.", "B)", "C: ..."
    match = _LEADING_LETTER_PATTERN.match(prediction_clean)
    if match:
        letter = match.group(1).upper()
        if choices is None or letter in {str(k).upper() for k in choices.keys()}:
            logger.debug("Answer extraction: leading letter → %r", letter)
            return letter

    # Strategy 4: Option phrase — "Option A", "Choice B"
    match = _OPTION_PATTERN.search(prediction_clean)
    if match:
        letter = match.group(1).upper()
        if choices is None or letter in {str(k).upper() for k in choices.keys()}:
            logger.debug("Answer extraction: option phrase → %r", letter)
            return letter

    # Strategy 5: Value match — prediction contains a choice value verbatim
    if choices:
        pred_lower = prediction_clean.lower()
        first_index = len(pred_lower)
        best_length = 0
        best_key = None
        for key, value in choices.items():
            val_lower = value.lower().strip()
            if len(val_lower) > 0:
                # Use regex to find whole-word matches to avoid matching "x" inside "exchange"
                pattern = r"\b" + re.escape(val_lower) + r"\b"
                match = re.search(pattern, pred_lower)
                if match:
                    idx = match.start()
                    # Prefer earlier match. If same start index, prefer longer match.
                    if idx < first_index or (idx == first_index and len(val_lower) > best_length):
                        first_index = idx
                        best_length = len(val_lower)
                        best_key = str(key)
        if best_key is not None:
            logger.debug("Answer extraction: value match → %r", best_key)
            return best_key

    # Strategy 6: Fallback — return normalized prediction
    normalized = normalize_prediction(prediction)
    logger.debug("Answer extraction: fallback to normalized → %r", normalized[:80])
    return normalized


def normalize_prediction(prediction: str) -> str:
    """Strip common LLM preamble boilerplate to reveal the core answer.

    Removes conversational filler phrases like ``"Based on the clinical
    presentation, ..."`` and ``"The correct answer is ..."`` that models
    prepend before the actual answer.

    Args:
        prediction: Raw model-generated response text.

    Returns:
        The prediction with leading preamble stripped and whitespace
        trimmed. If no preamble is detected, the original string is
        returned (stripped).

    Examples:
        >>> normalize_prediction("The correct answer is Metformin.")
        'Metformin.'
        >>> normalize_prediction("Based on the clinical presentation, use ACE inhibitors.")
        'use ACE inhibitors.'
    """
    if not prediction:
        return ""

    result = prediction.strip()
    for pattern in _PREAMBLE_PATTERNS:
        result = pattern.sub("", result).strip()

    return result
