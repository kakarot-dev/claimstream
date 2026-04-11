"""
preprocess.py — NLP Stage 1: Text Sanitization

Cleans up raw Whisper output before sending to Gemma for verification.
"""

import re

# Filler words to strip
FILLERS = re.compile(
    r'\b(um+|uh+|ah+|er+|like|you know|i mean|basically|literally|'
    r'so+|well|right|okay|ok)\b',
    re.IGNORECASE
)

# Common Whisper artifacts
ARTIFACTS = re.compile(r'(\.\.\.+|---+|\[.*?\]|\(.*?\))')


def sanitize(text: str) -> str:
    """Stage 1: Clean raw Whisper output.

    - Remove filler words (um, uh, like, you know)
    - Remove Whisper artifacts ([inaudible], ...)
    - Fix double spaces
    - Fix capitalization after periods
    - Merge broken fragments
    """
    if not text:
        return ""

    # Remove artifacts
    text = ARTIFACTS.sub('', text)

    # Remove fillers (but keep sentence structure)
    text = FILLERS.sub('', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # fix space before punctuation
    text = re.sub(r'([.,!?])\s*([.,!?])', r'\1', text)  # fix double punctuation

    # Capitalize after sentence endings
    def cap_after_period(m):
        return m.group(1) + ' ' + m.group(2).upper()
    text = re.sub(r'([.!?])\s+([a-z])', cap_after_period, text)

    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text.strip()


def is_filler(text: str) -> bool:
    """Check if entire text is filler (not worth processing)."""
    stripped = FILLERS.sub('', text).strip()
    return len(stripped.split()) < 3
