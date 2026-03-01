import argparse
import logging
import re
from collections import Counter
from sqlalchemy import text
from db import get_local_engine, init_local_db, get_session, Label

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Outcome labeling patterns
# ---------------------------------------------------------------------------

PLAINTIFF_WIN_PATTERNS = [
    (re.compile(r"judgment\s+for\s+plaintiff", re.IGNORECASE), 2),
    (re.compile(r"grant(s|ed)?\s+plaintiff'?s?\s+motion\s+for\s+summary\s+judgment", re.IGNORECASE), 2),
    (re.compile(r"plaintiff'?s?\s+motion\s+for\s+summary\s+judgment\s+is\s+granted", re.IGNORECASE), 2),
    (re.compile(r"defendant\s+is\s+liable", re.IGNORECASE), 2),
    (re.compile(r"damages\s+(are\s+)?awarded", re.IGNORECASE), 1),
    (re.compile(r"finds?\s+(for|in\s+favor\s+of)\s+plaintiff", re.IGNORECASE), 2),
    (re.compile(r"verdict\s+(for|in\s+favor\s+of)\s+plaintiff", re.IGNORECASE), 2),
]

DEFENDANT_WIN_PATTERNS = [
    (re.compile(r"judgment\s+for\s+defendant", re.IGNORECASE), 2),
    (re.compile(r"grant(s|ed)?\s+defendant'?s?\s+motion\s+for\s+summary\s+judgment", re.IGNORECASE), 2),
    (re.compile(r"defendant'?s?\s+motion\s+for\s+summary\s+judgment\s+is\s+granted", re.IGNORECASE), 2),
    (re.compile(r"plaintiff'?s?\s+complaint\s+is\s+dismissed", re.IGNORECASE), 2),
    (re.compile(r"case\s+(is\s+)?dismissed", re.IGNORECASE), 1),
    (re.compile(r"dismissed\s+with\s+prejudice", re.IGNORECASE), 1),
    (re.compile(r"finds?\s+(for|in\s+favor\s+of)\s+defendant", re.IGNORECASE), 2),
    (re.compile(r"verdict\s+(for|in\s+favor\s+of)\s+defendant", re.IGNORECASE), 2),
]

MIXED_PATTERNS = [
    (re.compile(r"granted\s+in\s+part\s+and\s+denied\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"denied\s+in\s+part\s+and\s+granted\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"partial\s+summary\s+judgment", re.IGNORECASE), 2),
    (re.compile(r"granted\s+in\s+part", re.IGNORECASE), 2),
]


def label_outcome(text_content: str) -> dict:
    """Classify an opinion's outcome using regex pattern matching.

    Returns a dict with 'label' (str) and 'confidence' (float).
    """
    # Check mixed patterns first (most specific)
    for pattern, weight in MIXED_PATTERNS:
        if pattern.search(text_content):
            return {"label": "mixed", "confidence": min(0.5 + weight * 0.1, 1.0)}

    # Score plaintiff and defendant signals
    p_score = 0
    for pattern, weight in PLAINTIFF_WIN_PATTERNS:
        if pattern.search(text_content):
            p_score += weight

    d_score = 0
    for pattern, weight in DEFENDANT_WIN_PATTERNS:
        if pattern.search(text_content):
            d_score += weight

    # If both have strong signals, return mixed
    if p_score >= 2 and d_score >= 2:
        return {"label": "mixed", "confidence": 0.5}

    # No signals
    if p_score == 0 and d_score == 0:
        return {"label": "unlabeled", "confidence": 0.0}

    # Determine winner
    total = p_score + d_score
    if p_score > d_score:
        return {"label": "plaintiff_win", "confidence": min(p_score / max(total, 1), 1.0)}
    elif d_score > p_score:
        return {"label": "defendant_win", "confidence": min(d_score / max(total, 1), 1.0)}
    else:
        # Tie with some signals -- default to mixed
        return {"label": "mixed", "confidence": 0.5}
