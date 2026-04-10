"""
preprocess.py — Data preprocessing and text handling for ConvoAI.

Handles:
1. Sentence splitting from transcribed speech
2. Filler/noise filtering (removes non-factual utterances)
3. Dataset export to CSV for sample submission
"""

import re
import json
import csv


# ============================================================
# Filler Detection
# ============================================================

FILLER_PATTERNS = re.compile(
    r'^(ok|okay|um+|uh+|ah+|hey|hi|hello|yeah|yes|no|sure|right|'
    r'thank you|thanks|test.*|let me|look at|so+|well|bye|'
    r'can you hear|is this on|checking|one two|alright)[\s.,!?]*$',
    re.IGNORECASE,
)


def is_filler(text: str) -> bool:
    """Return True if text is conversational filler, not a factual claim."""
    stripped = text.strip().rstrip('.!?,')
    if len(stripped.split()) < 4:
        return True
    if FILLER_PATTERNS.match(stripped):
        return True
    return False


# ============================================================
# Sentence Splitting
# ============================================================

def split_sentences(text: str) -> list[str]:
    """Split transcribed text into individual sentences/claims."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) == 1 and not any(text.endswith(p) for p in ".!?"):
        return [text]
    return [s.strip() for s in sentences if s.strip()]


def extract_claims(text: str, min_length: int = 10) -> list[str]:
    """Extract factual claims from transcribed speech.

    Splits into sentences, filters out filler and short utterances.
    """
    sentences = split_sentences(text)
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) >= min_length and not is_filler(s):
            claims.append(s)
    return claims


# ============================================================
# Dataset Export
# ============================================================

def export_dataset_csv(json_path: str = "data/space_facts.json",
                       csv_path: str = "dataset_sample.csv",
                       max_rows: int = 100):
    """Export a sample of the fact dataset as CSV for project submission."""
    with open(json_path) as f:
        data = json.load(f)

    facts = data["facts"][:max_rows]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "verdict", "source", "category", "correction"])
        for fact in facts:
            writer.writerow([
                fact.get("claim", ""),
                fact.get("verdict", ""),
                fact.get("source", ""),
                fact.get("category", ""),
                fact.get("correction", ""),
            ])

    print(f"Exported {len(facts)} facts to {csv_path}")


if __name__ == "__main__":
    export_dataset_csv()
