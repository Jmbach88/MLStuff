"""
Named Entity Recognition: extract legal entities from opinions.

Usage:
    python ner.py                          # extract all entities
    python ner.py --types judge,plaintiff  # extract specific types only
    python ner.py --no-spacy               # disable spaCy, regex only
    python ner.py --info                   # print entity summary
"""
import argparse
import logging
import re

from sqlalchemy import text

import config
from db import (
    get_local_engine, init_local_db, get_session,
    Opinion, Entity,
)

logger = logging.getLogger(__name__)

CONTEXT_CHARS = 100

# --- Judge patterns (applied to first 1000 chars) ---
JUDGE_PATTERNS = [
    re.compile(r'JUDGE\s+([A-Z][A-Za-z\'-]+(?:\s+[A-Z][A-Za-z\'-]+)*)'),
    re.compile(r'before\s+(?:the\s+Honorable\s+)?Judge\s+([A-Z][A-Za-z\'-]+(?:\s+[A-Z][A-Za-z\'-]+)*)', re.IGNORECASE),
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s+(?:United States\s+)?(?:District|Circuit|Magistrate)\s+Judge'),
]

# --- Dollar pattern ---
DOLLAR_PATTERN = re.compile(r'\$\s?[\d,]+(?:\.\d{2})?')

# --- Debt type keywords ---
DEBT_TYPES = [
    "credit card", "medical debt", "medical bill", "student loan",
    "auto loan", "automobile loan", "car loan", "mortgage",
    "payday loan", "utility", "utility bill", "personal loan",
    "consumer debt", "phone bill", "cell phone", "telecommunications",
]

DEBT_TYPE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(dt) for dt in DEBT_TYPES) + r')\b',
    re.IGNORECASE,
)

# --- Original creditor context patterns ---
CREDITOR_CONTEXTS = [
    re.compile(r'original\s+creditor\s+(?:was\s+|is\s+)?([A-Z][A-Za-z\s,\.]+?)(?:\.|,|\s+and\s|\s+to\s)', re.IGNORECASE),
    re.compile(r'assignee\s+of\s+([A-Z][A-Za-z\s,\.]+?)(?:\.|,|\s+and\s)', re.IGNORECASE),
    re.compile(r'on\s+behalf\s+of\s+([A-Z][A-Za-z\s,\.]+?)(?:\.|,|\s+and\s)', re.IGNORECASE),
    re.compile(r'successor\s+to\s+([A-Z][A-Za-z\s,\.]+?)(?:\.|,|\s+and\s)', re.IGNORECASE),
]

# --- Defense type patterns ---
DEFENSE_TYPES = [
    "bona fide error",
    "statute of limitations",
    "standing",
    "rooker-feldman",
    "res judicata",
    "claim preclusion",
    "collateral estoppel",
    "issue preclusion",
    "preemption",
    "sovereign immunity",
    "qualified immunity",
    "failure to state a claim",
    "mootness",
    "ripeness",
    "abstention",
    "arbitration",
    "good faith",
    "fair use",
    "consent",
    "established business relationship",
    "prior express consent",
]

DEFENSE_TYPE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(dt) for dt in DEFENSE_TYPES) + r')\b',
    re.IGNORECASE,
)


def extract_parties_from_title(title):
    """Extract plaintiff and defendant from opinion title.

    Splits on ' v. ' or ' vs. ', strips 'et al'.
    Returns (plaintiff, defendant) or (None, None).
    """
    if not title:
        return None, None

    for sep in [' v. ', ' vs. ', ' V. ', ' VS. ']:
        if sep in title:
            parts = title.split(sep, 1)
            plaintiff = parts[0].strip()
            defendant = parts[1].strip()
            for suffix in [' et al', ' et al.', ' Et Al', ' Et Al.', ' ET AL', ' ET AL.']:
                if defendant.endswith(suffix):
                    defendant = defendant[:-len(suffix)].strip().rstrip(',')
                if plaintiff.endswith(suffix):
                    plaintiff = plaintiff[:-len(suffix)].strip().rstrip(',')
            return plaintiff, defendant

    return None, None


def extract_judges(text_content, max_chars=1000):
    """Extract judge names from opinion text header.

    Returns list of entity dicts with entity_type='JUDGE'.
    """
    if not text_content:
        return []

    header = text_content[:max_chars]
    results = []
    seen = set()

    for pattern in JUDGE_PATTERNS:
        for match in pattern.finditer(header):
            name = match.group(1).strip().rstrip('.')
            if len(name) < 3:
                continue
            name_lower = name.lower()
            if name_lower in seen:
                continue
            seen.add(name_lower)

            start = max(0, match.start() - CONTEXT_CHARS)
            end = min(len(text_content), match.end() + CONTEXT_CHARS)
            context = text_content[start:end].strip()

            results.append({
                "entity_type": "JUDGE",
                "entity_value": name,
                "context_snippet": context,
                "start_char": match.start(),
                "end_char": match.end(),
            })

    return results


def extract_dollar_amounts(text_content):
    """Extract and classify dollar amounts from text.

    Classification by surrounding context (+-50 chars):
    - attorney + fee -> ATTORNEY_FEES
    - damages/awarded/judgment for -> DAMAGES_AWARDED
    - otherwise -> DOLLAR_AMOUNT
    """
    if not text_content:
        return []

    results = []
    for match in DOLLAR_PATTERN.finditer(text_content):
        value = match.group(0).strip()

        ctx_start = max(0, match.start() - 50)
        ctx_end = min(len(text_content), match.end() + 50)
        context_window = text_content[ctx_start:ctx_end].lower()

        has_attorney = 'attorney' in context_window and ('fee' in context_window or 'fees' in context_window)
        has_damages = any(kw in context_window for kw in ['damages', 'awarded', 'judgment for'])

        if has_attorney and has_damages:
            # Both keywords present -- classify by which is closer
            match_pos = match.start() - ctx_start
            atty_pos = context_window.find('attorney')
            dmg_pos = min(
                (context_window.find(kw) for kw in ['damages', 'awarded', 'judgment for']
                 if context_window.find(kw) != -1),
                default=999,
            )
            if abs(atty_pos - match_pos) <= abs(dmg_pos - match_pos):
                entity_type = "ATTORNEY_FEES"
            else:
                entity_type = "DAMAGES_AWARDED"
        elif has_attorney:
            entity_type = "ATTORNEY_FEES"
        elif has_damages:
            entity_type = "DAMAGES_AWARDED"
        else:
            entity_type = "DOLLAR_AMOUNT"

        snip_start = max(0, match.start() - CONTEXT_CHARS)
        snip_end = min(len(text_content), match.end() + CONTEXT_CHARS)
        context = text_content[snip_start:snip_end].strip()

        results.append({
            "entity_type": entity_type,
            "entity_value": value,
            "context_snippet": context,
            "start_char": match.start(),
            "end_char": match.end(),
        })

    return results


def extract_debt_types(text_content, max_chars=3000):
    """Extract debt type keywords from opinion text.

    Returns list of entity dicts with entity_type='DEBT_TYPE'.
    """
    if not text_content:
        return []

    search_text = text_content[:max_chars]
    results = []
    seen = set()

    for match in DEBT_TYPE_PATTERN.finditer(search_text):
        value = match.group(1).lower()
        if value in seen:
            continue
        seen.add(value)

        snip_start = max(0, match.start() - CONTEXT_CHARS)
        snip_end = min(len(search_text), match.end() + CONTEXT_CHARS)
        context = search_text[snip_start:snip_end].strip()

        results.append({
            "entity_type": "DEBT_TYPE",
            "entity_value": value,
            "context_snippet": context,
            "start_char": match.start(),
            "end_char": match.end(),
        })

    return results


def extract_original_creditors(text_content):
    """Extract original creditor names using context patterns.

    Returns list of entity dicts with entity_type='ORIGINAL_CREDITOR'.
    """
    if not text_content:
        return []

    results = []
    seen = set()

    for pattern in CREDITOR_CONTEXTS:
        for match in pattern.finditer(text_content):
            name = match.group(1).strip().rstrip('.')
            if len(name) < 3 or len(name) > 100:
                continue
            name_lower = name.lower()
            if name_lower in seen:
                continue
            seen.add(name_lower)

            snip_start = max(0, match.start() - CONTEXT_CHARS)
            snip_end = min(len(text_content), match.end() + CONTEXT_CHARS)
            context = text_content[snip_start:snip_end].strip()

            results.append({
                "entity_type": "ORIGINAL_CREDITOR",
                "entity_value": name,
                "context_snippet": context,
                "start_char": match.start(),
                "end_char": match.end(),
            })

    return results


def extract_defense_types(text_content):
    """Extract defense type keywords from opinion text.
    Searches full text. Returns list of entity dicts with entity_type='DEFENSE_TYPE'.
    Deduplicates per opinion.
    """
    if not text_content:
        return []
    results = []
    seen = set()
    for match in DEFENSE_TYPE_PATTERN.finditer(text_content):
        value = match.group(1).lower()
        if value in seen:
            continue
        seen.add(value)
        snip_start = max(0, match.start() - CONTEXT_CHARS)
        snip_end = min(len(text_content), match.end() + CONTEXT_CHARS)
        context = text_content[snip_start:snip_end].strip()
        results.append({
            "entity_type": "DEFENSE_TYPE",
            "entity_value": value,
            "context_snippet": context,
            "start_char": match.start(),
            "end_char": match.end(),
        })
    return results


def extract_entities_from_opinion(title, text_content, use_spacy=False, nlp=None):
    """Extract all entity types from one opinion.

    Args:
        title: opinion title string
        text_content: opinion plain_text
        use_spacy: whether to use spaCy for org/person enrichment
        nlp: pre-loaded spaCy model (optional)

    Returns list of entity dicts.
    """
    entities = []

    plaintiff, defendant = extract_parties_from_title(title)
    if plaintiff:
        entities.append({
            "entity_type": "PLAINTIFF",
            "entity_value": plaintiff,
            "context_snippet": title,
            "start_char": None,
            "end_char": None,
        })
    if defendant:
        entities.append({
            "entity_type": "DEFENDANT",
            "entity_value": defendant,
            "context_snippet": title,
            "start_char": None,
            "end_char": None,
        })

    entities.extend(extract_judges(text_content))
    entities.extend(extract_dollar_amounts(text_content))
    entities.extend(extract_debt_types(text_content))
    entities.extend(extract_original_creditors(text_content))
    entities.extend(extract_defense_types(text_content))

    if use_spacy and text_content:
        if nlp is None:
            import spacy
            nlp = spacy.load(config.SPACY_MODEL)
        doc = nlp(text_content[:5000])
        for ent in doc.ents:
            if ent.label_ == "ORG":
                ctx_start = max(0, ent.start_char - 100)
                surrounding = text_content[ctx_start:ent.end_char + 100].lower()
                if any(kw in surrounding for kw in ['original creditor', 'assignee', 'on behalf of', 'successor']):
                    name_lower = ent.text.lower()
                    existing = {e["entity_value"].lower() for e in entities if e["entity_type"] == "ORIGINAL_CREDITOR"}
                    if name_lower not in existing:
                        entities.append({
                            "entity_type": "ORIGINAL_CREDITOR",
                            "entity_value": ent.text,
                            "context_snippet": text_content[ctx_start:ent.end_char + 100].strip(),
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                        })

    return entities


def store_entities(engine, opinion_id, entities):
    """Store extracted entities for one opinion. Clears existing entities first."""
    session = get_session(engine)
    try:
        session.query(Entity).filter_by(opinion_id=opinion_id).delete()
        session.commit()

        for ent in entities:
            session.add(Entity(
                opinion_id=opinion_id,
                entity_type=ent["entity_type"],
                entity_value=ent["entity_value"],
                context_snippet=ent.get("context_snippet"),
                start_char=ent.get("start_char"),
                end_char=ent.get("end_char"),
            ))
        session.commit()
    finally:
        session.close()


def run_ner_extraction(engine=None, entity_types=None, use_spacy=True):
    """Full NER extraction pipeline."""
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    nlp = None
    if use_spacy:
        try:
            import spacy
            nlp = spacy.load(config.SPACY_MODEL)
            logger.info(f"Loaded spaCy model: {config.SPACY_MODEL}")
        except (ImportError, OSError) as e:
            logger.warning(f"spaCy not available, using regex only: {e}")
            use_spacy = False

    session = get_session(engine)
    opinions = session.execute(
        text("SELECT id, title, plain_text FROM opinions "
             "WHERE plain_text IS NOT NULL AND plain_text != ''")
    ).fetchall()
    session.close()

    logger.info(f"Extracting entities from {len(opinions)} opinions...")
    total_entities = 0

    for i, (opinion_id, title, plain_text) in enumerate(opinions):
        entities = extract_entities_from_opinion(
            title, plain_text, use_spacy=use_spacy, nlp=nlp,
        )

        if entity_types:
            allowed = {t.upper() for t in entity_types}
            entities = [e for e in entities if e["entity_type"] in allowed]

        if entities:
            store_entities(engine, opinion_id, entities)
            total_entities += len(entities)

        if (i + 1) % 5000 == 0:
            logger.info(f"  Progress: {i + 1}/{len(opinions)} opinions, "
                        f"{total_entities} entities stored")

    logger.info(f"Extraction complete: {total_entities} entities from "
                f"{len(opinions)} opinions")


def get_entity_summary(engine):
    """Get entity summary stats."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT entity_type, COUNT(*) FROM entities "
            "GROUP BY entity_type ORDER BY COUNT(*) DESC"
        )).fetchall()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Named Entity Recognition")
    parser.add_argument("--types", type=str, default=None,
                        help="Comma-separated entity types to extract")
    parser.add_argument("--no-spacy", action="store_true",
                        help="Disable spaCy, use regex only")
    parser.add_argument("--info", action="store_true",
                        help="Print entity summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    engine = get_local_engine()
    init_local_db(engine)

    if args.info:
        rows = get_entity_summary(engine)
        if not rows:
            print("No entities found.")
        else:
            print(f"{'Entity Type':<25} {'Count':>8}")
            print("-" * 35)
            for row in rows:
                print(f"{row[0]:<25} {row[1]:>8}")
        return

    entity_types = args.types.split(",") if args.types else None
    run_ner_extraction(engine, entity_types=entity_types, use_spacy=not args.no_spacy)


if __name__ == "__main__":
    main()
