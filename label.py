import argparse
import logging
import re
import json as _json
from collections import Counter
import requests
import config
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
    (re.compile(r"motion\s+to\s+dismiss.*(?:is\s+)?denied", re.IGNORECASE), 2),
    (re.compile(r"order\s+denying.*motion\s+to\s+dismiss", re.IGNORECASE), 2),
    (re.compile(r"default\s+judgment.*(?:is\s+)?entered", re.IGNORECASE), 2),
    (re.compile(r"\breversed\b", re.IGNORECASE), 1),
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
    (re.compile(r"motion\s+to\s+dismiss.*(?:is\s+)?(?:hereby\s+)?granted", re.IGNORECASE), 2),
    (re.compile(r"order\s+granting.*motion\s+to\s+dismiss", re.IGNORECASE), 2),
    (re.compile(r"\baffirmed\b", re.IGNORECASE), 1),
]

MIXED_PATTERNS = [
    (re.compile(r"granted\s+in\s+part\s+and\s+denied\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"denied\s+in\s+part\s+and\s+granted\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"partial\s+summary\s+judgment", re.IGNORECASE), 2),
    (re.compile(r"granted\s+in\s+part", re.IGNORECASE), 2),
    (re.compile(r"affirmed\s+in\s+part.*reversed\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"reversed\s+in\s+part.*affirmed\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"vacated\s+and\s+remanded", re.IGNORECASE), 2),
]

SETTLEMENT_PATTERNS = [
    (re.compile(r"parties\s+have\s+reached\s+a\s+settlement", re.IGNORECASE), 3),
    (re.compile(r"stipulat(?:ed|ion)\s+(?:of\s+)?(?:voluntary\s+)?dismissal", re.IGNORECASE), 3),
    (re.compile(r"voluntary\s+dismissal", re.IGNORECASE), 2),
    (re.compile(r"dismissed\s+pursuant\s+to\s+(?:the\s+)?(?:parties'?\s+)?stipulation", re.IGNORECASE), 3),
    (re.compile(r"settled\s+(?:out\s+of\s+court|between\s+the\s+parties)", re.IGNORECASE), 3),
]


def label_outcome(text_content: str) -> dict:
    """Classify an opinion's outcome using regex pattern matching.

    Returns a dict with 'label' (str) and 'confidence' (float).
    """
    # Check mixed patterns first (most specific)
    for pattern, weight in MIXED_PATTERNS:
        if pattern.search(text_content):
            return {"label": "mixed", "confidence": min(0.5 + weight * 0.1, 1.0)}

    # Check settlement patterns
    settlement_score = 0
    for pattern, weight in SETTLEMENT_PATTERNS:
        if pattern.search(text_content):
            settlement_score += weight

    if settlement_score >= 2:
        return {"label": "settled", "confidence": min(0.5 + settlement_score * 0.1, 0.95)}

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


# ---------------------------------------------------------------------------
# Claim type labeling patterns
# ---------------------------------------------------------------------------

SECTION_PATTERNS = [
    re.compile(r'\u00a7\s*(1692[a-p](?:-\d+)?)', re.IGNORECASE),
    re.compile(r'\u00a7\s*(227)', re.IGNORECASE),
    re.compile(r'\u00a7\s*(1681[a-x](?:-\d+)?)', re.IGNORECASE),
    re.compile(r'section\s+(1692[a-p](?:-\d+)?)', re.IGNORECASE),
    re.compile(r'section\s+(227)', re.IGNORECASE),
    re.compile(r'section\s+(1681[a-x](?:-\d+)?)', re.IGNORECASE),
    re.compile(r'\d+\s+U\.?S\.?C\.?\s+\u00a7?\s*(1692[a-p](?:-\d+)?)', re.IGNORECASE),
    re.compile(r'\d+\s+U\.?S\.?C\.?\s+\u00a7?\s*(227)', re.IGNORECASE),
    re.compile(r'\d+\s+U\.?S\.?C\.?\s+\u00a7?\s*(1681[a-x](?:-\d+)?)', re.IGNORECASE),
]


def label_claim_types(text_content: str) -> list:
    """Extract statutory section references from opinion text.

    Returns a sorted, deduplicated list of section identifiers.
    """
    found = set()
    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text_content):
            section = match.group(1).lower()
            found.add(section)
    return sorted(found)


# ---------------------------------------------------------------------------
# LLM-assisted labeling (Ollama)
# ---------------------------------------------------------------------------

LLM_PROMPT_TEMPLATE = """You are a legal analyst. Read this federal court opinion and classify its outcome.

Respond with ONLY a JSON object, no other text:
{{"outcome": "<LABEL>", "confidence": <0.0-1.0>}}

Where LABEL is one of:
- "plaintiff_win" - plaintiff prevailed (judgment for plaintiff, damages awarded, defendant liable)
- "defendant_win" - defendant prevailed (case dismissed, summary judgment for defendant)
- "mixed" - split decision (granted in part, denied in part)
- "settled" - case settled or voluntarily dismissed
- "unclear" - cannot determine from the text

OPINION TEXT:
{text}"""


def _parse_llm_response(response_text: str) -> dict:
    """Parse LLM response, extracting JSON from potentially noisy output."""
    try:
        data = _json.loads(response_text.strip())
    except _json.JSONDecodeError:
        import re as _re
        match = _re.search(r'\{[^}]+\}', response_text)
        if match:
            try:
                data = _json.loads(match.group())
            except _json.JSONDecodeError:
                return {"label": "unclear", "confidence": 0.0}
        else:
            return {"label": "unclear", "confidence": 0.0}

    outcome = data.get("outcome", "unclear").lower().strip()
    confidence = float(data.get("confidence", 0.5))

    valid_labels = {"plaintiff_win", "defendant_win", "mixed", "settled", "unclear"}
    if outcome not in valid_labels:
        return {"label": "unclear", "confidence": 0.0}

    return {"label": outcome, "confidence": confidence}


def label_with_llm(text_content: str) -> dict:
    """Label an opinion using Ollama LLM.

    Returns {"label": str, "confidence": float}
    On error: {"label": "error", "confidence": 0.0}
    """
    truncated = text_content[:config.LLM_TEXT_LIMIT]
    prompt = LLM_PROMPT_TEMPLATE.format(text=truncated)

    try:
        response = requests.post(
            config.OLLAMA_URL,
            json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": True},
            timeout=600,
            stream=True,
        )
        response.raise_for_status()
        result_text = ""
        for line in response.iter_lines():
            if line:
                chunk = _json.loads(line)
                result_text += chunk.get("response", "")
                if chunk.get("done", False):
                    break
        return _parse_llm_response(result_text)
    except requests.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return {"label": "error", "confidence": 0.0}
    except requests.Timeout:
        logger.warning("Ollama request timed out")
        return {"label": "error", "confidence": 0.0}
    except Exception as e:
        logger.error(f"LLM labeling error: {e}")
        return {"label": "error", "confidence": 0.0}


# ---------------------------------------------------------------------------
# Batch labeling pipeline
# ---------------------------------------------------------------------------

def run_llm_labeling(engine=None, limit=None):
    """Label unlabeled opinions using Ollama LLM.

    Only processes opinions that have no outcome label at all.
    Saves after each opinion for crash safety.

    Args:
        engine: SQLAlchemy engine
        limit: Max opinions to process (None = all)

    Returns:
        dict with stats
    """
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    session = get_session(engine)

    query = (
        "SELECT o.id, o.plain_text FROM opinions o "
        "WHERE o.plain_text IS NOT NULL AND o.plain_text != '' "
        "AND o.id NOT IN (SELECT DISTINCT opinion_id FROM labels WHERE label_type = 'outcome')"
    )
    if limit:
        query += f" LIMIT {int(limit)}"

    opinions = session.execute(text(query)).fetchall()
    logger.info(f"LLM labeling: {len(opinions)} unlabeled opinions to process")

    if not opinions:
        session.close()
        return {"total": 0}

    counts = Counter()
    errors = 0

    for i, (opinion_id, plain_text) in enumerate(opinions):
        result = label_with_llm(plain_text)

        if result["label"] == "error":
            errors += 1
            if errors >= 5:
                logger.error("Too many consecutive errors. Is Ollama running? Stopping.")
                break
            continue

        errors = 0  # reset consecutive error count on success
        counts[result["label"]] += 1

        if result["label"] != "unclear" and result["confidence"] >= 0.5:
            session.add(Label(
                opinion_id=opinion_id,
                label_type="outcome",
                label_value=result["label"],
                source="llm",
                confidence=result["confidence"],
            ))
            session.commit()

        if (i + 1) % 100 == 0:
            logger.info(f"  LLM progress: {i + 1}/{len(opinions)} — {dict(counts)}")

    session.close()

    stats = {
        "total": len(opinions),
        "processed": sum(counts.values()),
        "labeled": sum(v for k, v in counts.items() if k not in ("unclear", "error")),
        "distribution": dict(counts),
    }
    logger.info(f"LLM labeling complete: {stats}")
    return stats


def run_labeling(engine=None, relabel=False):
    """Label all opinions in the database."""
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    session = get_session(engine)

    if relabel:
        session.execute(text("DELETE FROM labels"))
        session.commit()

    # Get opinions to label (skip already labeled unless relabeling)
    if relabel:
        opinions = session.execute(text(
            "SELECT id, plain_text FROM opinions WHERE plain_text IS NOT NULL AND plain_text != ''"
        )).fetchall()
    else:
        opinions = session.execute(text(
            "SELECT o.id, o.plain_text FROM opinions o "
            "WHERE o.plain_text IS NOT NULL AND o.plain_text != '' "
            "AND o.id NOT IN (SELECT DISTINCT opinion_id FROM labels WHERE label_type = 'outcome')"
        )).fetchall()

    logger.info(f"Opinions to label: {len(opinions)}")

    outcome_counts = Counter()
    claim_type_counts = Counter()
    batch_labels = []

    for opinion_id, plain_text in opinions:
        outcome = label_outcome(plain_text)
        if outcome["label"] != "unlabeled":
            batch_labels.append(Label(
                opinion_id=opinion_id, label_type="outcome",
                label_value=outcome["label"], source="regex",
                confidence=outcome["confidence"],
            ))
        outcome_counts[outcome["label"]] += 1

        sections = label_claim_types(plain_text)
        for section in sections:
            batch_labels.append(Label(
                opinion_id=opinion_id, label_type="claim_type",
                label_value=section, source="regex", confidence=0.95,
            ))
            claim_type_counts[section] += 1

        if len(batch_labels) >= 5000:
            session.add_all(batch_labels)
            session.commit()
            batch_labels = []

    if batch_labels:
        session.add_all(batch_labels)
        session.commit()

    session.close()

    stats = {
        "outcome": {
            "total": len(opinions),
            "plaintiff_win": outcome_counts.get("plaintiff_win", 0),
            "defendant_win": outcome_counts.get("defendant_win", 0),
            "mixed": outcome_counts.get("mixed", 0),
            "settled": outcome_counts.get("settled", 0),
            "unlabeled": outcome_counts.get("unlabeled", 0),
        },
        "claim_type": {
            "total_labels": sum(claim_type_counts.values()),
            "unique_sections": len(claim_type_counts),
            "top_sections": claim_type_counts.most_common(10),
        },
    }
    logger.info(f"Labeling complete: {stats}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Label opinions")
    parser.add_argument("--relabel", action="store_true", help="Re-label all opinions")
    parser.add_argument("--stats", action="store_true", help="Show label distribution")
    parser.add_argument("--llm", action="store_true", help="Use Ollama LLM for unlabeled opinions")
    parser.add_argument("--llm-limit", type=int, default=None, help="Max opinions to LLM-label")
    args = parser.parse_args()

    if args.stats:
        engine = get_local_engine()
        session = get_session(engine)
        outcome = session.execute(text(
            "SELECT label_value, COUNT(*) FROM labels WHERE label_type='outcome' GROUP BY label_value"
        )).fetchall()
        claim = session.execute(text(
            "SELECT label_value, COUNT(*) FROM labels WHERE label_type='claim_type' GROUP BY label_value ORDER BY COUNT(*) DESC LIMIT 20"
        )).fetchall()
        session.close()
        print("\nOutcome distribution:")
        for val, count in outcome:
            print(f"  {val}: {count}")
        print("\nTop claim type sections:")
        for val, count in claim:
            print(f"  \u00a7{val}: {count}")
    elif args.llm:
        stats = run_llm_labeling(limit=args.llm_limit)
        print(f"\nLLM labeling: {stats}")
    else:
        stats = run_labeling(relabel=args.relabel)
        print(f"\nOutcome: {stats['outcome']}")
        print(f"Claim types: {stats['claim_type']['total_labels']} labels across {stats['claim_type']['unique_sections']} sections")
