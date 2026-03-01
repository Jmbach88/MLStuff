# Text Classification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Classify ~30K court opinions by outcome (plaintiff win / defendant win / mixed) and claim types (statutory sections cited) using TF-IDF + sklearn classifiers, with analytics dashboard and search integration.

**Architecture:** Regex labeler extracts training labels from opinion text (outcome signals + statutory section references). TF-IDF + LogisticRegression classifiers trained on labeled data. Predictions stored in SQLite and surfaced in FAISS metadata, Streamlit search filters, and a new analytics dashboard.

**Tech Stack:** scikit-learn (TF-IDF, LogisticRegression, OneVsRestClassifier), joblib (model persistence), SQLAlchemy (DB), Streamlit (UI), existing FAISS index infrastructure.

---

### Task 1: Add Dependencies and New DB Tables

**Files:**
- Modify: `requirements.txt`
- Modify: `db.py:1-79`
- Create: `tests/test_label.py`

**Step 1: Update requirements.txt**

Add to the end of `requirements.txt`:

```
scikit-learn>=1.3.0
joblib>=1.3.0
```

**Step 2: Run pip install**

Run: `pip install scikit-learn joblib`
Expected: Successfully installed

**Step 3: Write the failing test for new DB tables**

Create `tests/test_classify_db.py`:

```python
import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Label, Prediction, Model


def test_label_table_created():
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    label = Label(
        opinion_id=1,
        label_type="outcome",
        label_value="plaintiff_win",
        source="regex",
        confidence=0.9,
    )
    session.add(label)
    session.commit()

    result = session.query(Label).filter_by(opinion_id=1).first()
    assert result.label_value == "plaintiff_win"
    assert result.source == "regex"
    session.close()


def test_prediction_table_created():
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    pred = Prediction(
        opinion_id=1,
        model_name="outcome_logreg_v1",
        label_type="outcome",
        predicted_value="defendant_win",
        confidence=0.85,
        created_at="2026-03-01T00:00:00",
    )
    session.add(pred)
    session.commit()

    result = session.query(Prediction).filter_by(opinion_id=1).first()
    assert result.predicted_value == "defendant_win"
    session.close()


def test_model_table_created():
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    model = Model(
        name="outcome_logreg_v1",
        label_type="outcome",
        accuracy=0.82,
        f1_score=0.78,
        trained_at="2026-03-01T00:00:00",
        params_json='{"max_iter": 1000}',
    )
    session.add(model)
    session.commit()

    result = session.query(Model).filter_by(name="outcome_logreg_v1").first()
    assert result.accuracy == 0.82
    session.close()
```

**Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_classify_db.py -v`
Expected: FAIL with `ImportError: cannot import name 'Label' from 'db'`

**Step 5: Add ORM models to db.py**

Add these classes after the `FDCPASection` class (after line 58) in `db.py`:

```python
class Label(Base):
    __tablename__ = "labels"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    label_type = Column(Text, nullable=False)  # "outcome" or "claim_type"
    label_value = Column(Text, nullable=False)  # e.g. "plaintiff_win", "§1692e"
    source = Column(Text, nullable=False)       # "regex", "llm", "manual"
    confidence = Column(Integer)                # 0.0-1.0 (stored as real)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    model_name = Column(Text, nullable=False)
    label_type = Column(Text, nullable=False)   # "outcome" or "claim_type"
    predicted_value = Column(Text, nullable=False)
    confidence = Column(Integer)
    created_at = Column(Text)


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    label_type = Column(Text, nullable=False)
    accuracy = Column(Integer)
    f1_score = Column(Integer)
    trained_at = Column(Text)
    params_json = Column(Text)
```

Note: Use `Column(Integer)` for the `confidence`, `accuracy`, `f1_score` fields — SQLAlchemy maps Python floats to SQLite REAL via Integer type. Actually, use the `Float` type import. Update the import line at the top of `db.py` to include `Float`:

```python
from sqlalchemy import (
    Column, Integer, Float, Text, ForeignKey, UniqueConstraint,
    create_engine, inspect
)
```

Then use `Float` instead of `Integer` for `confidence`, `accuracy`, `f1_score`:

```python
class Label(Base):
    __tablename__ = "labels"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    label_type = Column(Text, nullable=False)
    label_value = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    confidence = Column(Float)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    model_name = Column(Text, nullable=False)
    label_type = Column(Text, nullable=False)
    predicted_value = Column(Text, nullable=False)
    confidence = Column(Float)
    created_at = Column(Text)


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    label_type = Column(Text, nullable=False)
    accuracy = Column(Float)
    f1_score = Column(Float)
    trained_at = Column(Text)
    params_json = Column(Text)
```

Also update the import in `db.py` line 18 to export these new models. No explicit export is needed — they just need to be importable.

**Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_classify_db.py -v`
Expected: 3 PASS

**Step 7: Run all existing tests to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All 17 existing tests + 3 new tests PASS

**Step 8: Commit**

```bash
git add requirements.txt db.py tests/test_classify_db.py
git commit -m "feat: add Label, Prediction, Model tables for classification"
```

---

### Task 2: Outcome Labeling — Regex Labeler

**Files:**
- Create: `label.py`
- Create: `tests/test_label.py`

**Step 1: Write the failing tests**

Create `tests/test_label.py`:

```python
import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from label import label_outcome, label_claim_types


class TestOutcomeLabeling:
    def test_plaintiff_win_judgment(self):
        text = "The court grants judgment for plaintiff. Damages are awarded in the amount of $1,000."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"
        assert result["confidence"] > 0.5

    def test_defendant_win_dismissed(self):
        text = "Plaintiff's complaint is dismissed with prejudice. Judgment for defendant."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"
        assert result["confidence"] > 0.5

    def test_mixed_partial(self):
        text = "Defendant's motion for summary judgment is granted in part and denied in part."
        result = label_outcome(text)
        assert result["label"] == "mixed"

    def test_unlabeled_no_signals(self):
        text = "This is a procedural order regarding scheduling of the pre-trial conference."
        result = label_outcome(text)
        assert result["label"] == "unlabeled"

    def test_plaintiff_win_summary_judgment(self):
        text = "Plaintiff's motion for summary judgment is GRANTED. The defendant is liable under the FDCPA."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_defendant_win_summary_judgment(self):
        text = "The court hereby grants defendant's motion for summary judgment. The case is dismissed."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_label.py::TestOutcomeLabeling -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'label'` or `ImportError`

**Step 3: Implement the outcome labeler**

Create `label.py`:

```python
"""
Labeling pipeline: extract outcome and claim type labels from opinion text.

Usage:
    python label.py                    # label all unlabeled opinions
    python label.py --relabel          # re-label everything
    python label.py --stats            # show label distribution
"""
import argparse
import logging
import re
from collections import Counter

from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session, Label

logger = logging.getLogger(__name__)

# --- Outcome regex patterns ---
# Each tuple: (compiled_regex, weight)
# Positive weight = plaintiff win signal, negative = defendant win signal

PLAINTIFF_WIN_PATTERNS = [
    (re.compile(r"judgment\s+(?:is\s+)?(?:entered\s+)?(?:for|in\s+favor\s+of)\s+(?:the\s+)?plaintiff", re.IGNORECASE), 2),
    (re.compile(r"grant(?:s|ed)\s+plaintiff'?s?\s+motion\s+for\s+summary\s+judgment", re.IGNORECASE), 2),
    (re.compile(r"defendant\s+is\s+(?:found\s+)?liable", re.IGNORECASE), 2),
    (re.compile(r"damages\s+(?:are\s+)?awarded", re.IGNORECASE), 1),
    (re.compile(r"plaintiff\s+(?:is\s+)?entitled\s+to\s+(?:summary\s+)?judgment", re.IGNORECASE), 2),
    (re.compile(r"judgment\s+(?:is\s+)?entered\s+against\s+(?:the\s+)?defendant", re.IGNORECASE), 2),
    (re.compile(r"(?:the\s+)?(?:court|jury)\s+finds?\s+(?:for|in\s+favor\s+of)\s+(?:the\s+)?plaintiff", re.IGNORECASE), 2),
    (re.compile(r"defendant'?s?\s+motion\s+(?:for\s+summary\s+judgment\s+)?is\s+denied", re.IGNORECASE), 1),
]

DEFENDANT_WIN_PATTERNS = [
    (re.compile(r"judgment\s+(?:is\s+)?(?:entered\s+)?(?:for|in\s+favor\s+of)\s+(?:the\s+)?defendant", re.IGNORECASE), 2),
    (re.compile(r"grant(?:s|ed)\s+defendant'?s?\s+motion\s+for\s+summary\s+judgment", re.IGNORECASE), 2),
    (re.compile(r"plaintiff'?s?\s+(?:complaint|case|action)\s+is\s+dismissed", re.IGNORECASE), 2),
    (re.compile(r"case\s+(?:is\s+)?dismissed\s+with\s+prejudice", re.IGNORECASE), 2),
    (re.compile(r"plaintiff\s+(?:has\s+)?fail(?:s|ed)\s+to\s+(?:state|establish|show|prove)", re.IGNORECASE), 1),
    (re.compile(r"(?:the\s+)?(?:court|jury)\s+finds?\s+(?:for|in\s+favor\s+of)\s+(?:the\s+)?defendant", re.IGNORECASE), 2),
    (re.compile(r"plaintiff'?s?\s+motion\s+(?:for\s+summary\s+judgment\s+)?is\s+denied", re.IGNORECASE), 1),
]

MIXED_PATTERNS = [
    (re.compile(r"granted\s+in\s+part\s+and\s+denied\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"partial\s+summary\s+judgment", re.IGNORECASE), 2),
    (re.compile(r"grant(?:s|ed)\s+in\s+part", re.IGNORECASE), 2),
    (re.compile(r"denied\s+in\s+part", re.IGNORECASE), 2),
]


def label_outcome(text_content: str) -> dict:
    """Label an opinion's outcome based on regex signals.

    Returns:
        {"label": "plaintiff_win"|"defendant_win"|"mixed"|"unlabeled",
         "confidence": float 0-1}
    """
    if not text_content:
        return {"label": "unlabeled", "confidence": 0.0}

    # Check for mixed signals first (they're most specific)
    mixed_score = 0
    for pattern, weight in MIXED_PATTERNS:
        matches = pattern.findall(text_content)
        mixed_score += len(matches) * weight

    if mixed_score >= 2:
        return {"label": "mixed", "confidence": min(0.5 + mixed_score * 0.1, 0.95)}

    # Score plaintiff and defendant signals
    plaintiff_score = 0
    for pattern, weight in PLAINTIFF_WIN_PATTERNS:
        matches = pattern.findall(text_content)
        plaintiff_score += len(matches) * weight

    defendant_score = 0
    for pattern, weight in DEFENDANT_WIN_PATTERNS:
        matches = pattern.findall(text_content)
        defendant_score += len(matches) * weight

    # If both have strong signals, it's mixed
    if plaintiff_score >= 2 and defendant_score >= 2:
        return {"label": "mixed", "confidence": 0.6}

    total = plaintiff_score + defendant_score
    if total == 0:
        return {"label": "unlabeled", "confidence": 0.0}

    if plaintiff_score > defendant_score:
        confidence = min(0.5 + (plaintiff_score - defendant_score) * 0.1, 0.95)
        return {"label": "plaintiff_win", "confidence": confidence}
    elif defendant_score > plaintiff_score:
        confidence = min(0.5 + (defendant_score - plaintiff_score) * 0.1, 0.95)
        return {"label": "defendant_win", "confidence": confidence}
    else:
        # Equal scores — ambiguous, label as mixed
        return {"label": "mixed", "confidence": 0.4}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_label.py::TestOutcomeLabeling -v`
Expected: 6 PASS

**Step 5: Commit**

```bash
git add label.py tests/test_label.py
git commit -m "feat: add outcome regex labeler with tests"
```

---

### Task 3: Claim Type Labeling — Statutory Section Extraction

**Files:**
- Modify: `label.py`
- Modify: `tests/test_label.py`

**Step 1: Write the failing tests**

Add to `tests/test_label.py`:

```python
class TestClaimTypeLabeling:
    def test_fdcpa_section_extraction(self):
        text = "The defendant violated §1692e by making false representations about the debt."
        result = label_claim_types(text)
        assert "1692e" in result

    def test_fdcpa_multiple_sections(self):
        text = "Claims under §1692e and §1692f of the FDCPA were both asserted."
        result = label_claim_types(text)
        assert "1692e" in result
        assert "1692f" in result

    def test_fdcpa_section_with_subsection(self):
        text = "Section 1692e(5) prohibits threats to take action that cannot legally be taken."
        result = label_claim_types(text)
        assert "1692e" in result

    def test_tcpa_section(self):
        text = "The plaintiff alleged violations of 47 U.S.C. §227(b)(1)(A)."
        result = label_claim_types(text)
        assert "227" in result

    def test_fcra_section(self):
        text = "Defendant failed to conduct a reasonable investigation under 15 U.S.C. §1681s-2(b)."
        result = label_claim_types(text)
        assert "1681s-2" in result

    def test_no_sections_found(self):
        text = "This is a procedural order about scheduling."
        result = label_claim_types(text)
        assert result == []

    def test_usc_format(self):
        text = "15 U.S.C. § 1692g requires debt validation."
        result = label_claim_types(text)
        assert "1692g" in result

    def test_text_format_section(self):
        text = "section 1692d of the Fair Debt Collection Practices Act"
        result = label_claim_types(text)
        assert "1692d" in result

    def test_deduplication(self):
        text = "§1692e was violated. The §1692e violation is clear. Under section 1692e the defendant..."
        result = label_claim_types(text)
        assert result.count("1692e") == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_label.py::TestClaimTypeLabeling -v`
Expected: FAIL (label_claim_types not implemented)

**Step 3: Implement claim type labeler**

Add to `label.py` after the `label_outcome` function:

```python
# --- Claim type regex patterns ---
# Matches statutory section references in various formats

# FDCPA: 15 U.S.C. §§ 1692-1692p
# TCPA: 47 U.S.C. § 227
# FCRA: 15 U.S.C. §§ 1681-1681x

SECTION_PATTERNS = [
    # §1692e, § 1692e, §1692e(5), etc.
    re.compile(r'§\s*(1692[a-p](?:-\d+)?)', re.IGNORECASE),
    # §227, § 227(b), etc.
    re.compile(r'§\s*(227)', re.IGNORECASE),
    # §1681a through §1681x, including §1681s-2
    re.compile(r'§\s*(1681[a-x](?:-\d+)?)', re.IGNORECASE),
    # "section 1692e", "Section 1692f(1)"
    re.compile(r'section\s+(1692[a-p](?:-\d+)?)', re.IGNORECASE),
    re.compile(r'section\s+(227)', re.IGNORECASE),
    re.compile(r'section\s+(1681[a-x](?:-\d+)?)', re.IGNORECASE),
    # "15 U.S.C. § 1692e" or "47 U.S.C. § 227"
    re.compile(r'\d+\s+U\.?S\.?C\.?\s+§?\s*(1692[a-p](?:-\d+)?)', re.IGNORECASE),
    re.compile(r'\d+\s+U\.?S\.?C\.?\s+§?\s*(227)', re.IGNORECASE),
    re.compile(r'\d+\s+U\.?S\.?C\.?\s+§?\s*(1681[a-x](?:-\d+)?)', re.IGNORECASE),
]


def label_claim_types(text_content: str) -> list[str]:
    """Extract statutory section references from opinion text.

    Returns:
        Sorted, deduplicated list of section identifiers.
        e.g., ["1692e", "1692f", "227"]
    """
    if not text_content:
        return []

    sections = set()
    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text_content):
            section = match.group(1).lower()
            # Normalize: strip any trailing subsection parens that leaked
            sections.add(section)

    return sorted(sections)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_label.py::TestClaimTypeLabeling -v`
Expected: 9 PASS

**Step 5: Commit**

```bash
git add label.py tests/test_label.py
git commit -m "feat: add claim type regex labeler for FDCPA/TCPA/FCRA sections"
```

---

### Task 4: Batch Labeling Pipeline

**Files:**
- Modify: `label.py`
- Modify: `tests/test_label.py`

**Step 1: Write the failing test**

Add to `tests/test_label.py`:

```python
from db import get_local_engine, init_local_db, get_session, Label, Opinion, Statute, OpinionStatute


class TestBatchLabeling:
    def _setup_opinions(self):
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)

        # Clear any existing data
        session.execute(text("DELETE FROM labels"))
        session.execute(text("DELETE FROM opinions"))
        session.commit()

        opinions = [
            Opinion(id=1, package_id="p1", title="Smith v. Collector",
                    plain_text="Judgment for plaintiff. Damages awarded of $1000 under §1692e."),
            Opinion(id=2, package_id="p2", title="Jones v. Agency",
                    plain_text="Plaintiff's complaint is dismissed. Judgment for defendant."),
            Opinion(id=3, package_id="p3", title="Brown v. Corp",
                    plain_text="This is a scheduling order for the pretrial conference."),
        ]
        session.add_all(opinions)
        session.commit()
        return engine

    def test_run_labeling_creates_labels(self):
        from label import run_labeling
        engine = self._setup_opinions()
        stats = run_labeling(engine)
        session = get_session(engine)

        outcome_labels = session.query(Label).filter_by(label_type="outcome").all()
        # Opinion 1 = plaintiff_win, Opinion 2 = defendant_win, Opinion 3 = unlabeled (no label stored)
        labeled = [l for l in outcome_labels if l.label_value != "unlabeled"]
        assert len(labeled) >= 2
        session.close()

    def test_run_labeling_creates_claim_labels(self):
        from label import run_labeling
        engine = self._setup_opinions()
        stats = run_labeling(engine)
        session = get_session(engine)

        claim_labels = session.query(Label).filter_by(label_type="claim_type").all()
        # Opinion 1 mentions §1692e
        values = [l.label_value for l in claim_labels]
        assert "1692e" in values
        session.close()

    def test_run_labeling_returns_stats(self):
        from label import run_labeling
        engine = self._setup_opinions()
        stats = run_labeling(engine)
        assert "outcome" in stats
        assert "claim_type" in stats
        assert stats["outcome"]["total"] == 3
```

Note: add `from sqlalchemy import text` at the top of the test file if not already imported.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_label.py::TestBatchLabeling -v`
Expected: FAIL with `ImportError: cannot import name 'run_labeling'`

**Step 3: Implement batch labeling**

Add to `label.py`:

```python
def run_labeling(engine=None, relabel=False):
    """Label all opinions in the database.

    Args:
        engine: SQLAlchemy engine (uses default if None)
        relabel: If True, delete existing labels and re-label all

    Returns:
        dict with stats: {"outcome": {...}, "claim_type": {...}}
    """
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    session = get_session(engine)

    if relabel:
        session.execute(text("DELETE FROM labels"))
        session.commit()
        logger.info("Cleared existing labels for re-labeling")

    # Get opinions to label
    if relabel:
        opinions = session.execute(text(
            "SELECT id, plain_text FROM opinions WHERE plain_text IS NOT NULL AND plain_text != ''"
        )).fetchall()
    else:
        # Only label opinions that don't already have outcome labels
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
        # Outcome labeling
        outcome = label_outcome(plain_text)
        if outcome["label"] != "unlabeled":
            batch_labels.append(Label(
                opinion_id=opinion_id,
                label_type="outcome",
                label_value=outcome["label"],
                source="regex",
                confidence=outcome["confidence"],
            ))
        outcome_counts[outcome["label"]] += 1

        # Claim type labeling
        sections = label_claim_types(plain_text)
        for section in sections:
            batch_labels.append(Label(
                opinion_id=opinion_id,
                label_type="claim_type",
                label_value=section,
                source="regex",
                confidence=0.95,  # regex extraction is high-confidence
            ))
            claim_type_counts[section] += 1

        # Batch insert every 1000 opinions
        if len(batch_labels) >= 5000:
            session.add_all(batch_labels)
            session.commit()
            batch_labels = []

    # Final batch
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
```

Also add the needed imports at the top of `label.py` if not already present:

```python
from db import get_local_engine, init_local_db, get_session, Label
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_label.py::TestBatchLabeling -v`
Expected: 3 PASS

**Step 5: Add CLI to label.py**

Add at the bottom of `label.py`:

```python
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Label opinions")
    parser.add_argument("--relabel", action="store_true", help="Re-label all opinions")
    parser.add_argument("--stats", action="store_true", help="Show label distribution")
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
            print(f"  §{val}: {count}")
    else:
        stats = run_labeling(relabel=args.relabel)
        print(f"\nOutcome: {stats['outcome']}")
        print(f"Claim types: {stats['claim_type']['total_labels']} labels across {stats['claim_type']['unique_sections']} sections")
```

**Step 6: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add label.py tests/test_label.py
git commit -m "feat: add batch labeling pipeline with CLI"
```

---

### Task 5: Run Labeling on Real Data

**Files:**
- No code changes — run the labeler on the full dataset

**Step 1: Run the labeling pipeline**

Run: `cd /c/PythonProject/ml && python label.py`
Expected: Processes ~30K opinions, outputs stats

**Step 2: Check label distribution**

Run: `python label.py --stats`
Expected: Shows outcome distribution and top claim type sections. Review the numbers to verify they look reasonable:
- Expect some percentage of unlabeled (opinions without clear outcome signals)
- Expect §1692e and §1692f to be among the most common FDCPA sections
- Expect §227 to appear for TCPA opinions

**Step 3: Commit (no code changes, just verification)**

No commit needed — this is a validation step.

---

### Task 6: Classification Models — Train & Evaluate

**Files:**
- Create: `classify.py`
- Create: `tests/test_classify.py`

**Step 1: Write the failing tests**

Create `tests/test_classify.py`:

```python
import os
import pytest
import numpy as np

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Label
from sqlalchemy import text


def _setup_labeled_data():
    """Create opinions with labels for training."""
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    session.execute(text("DELETE FROM labels"))
    session.execute(text("DELETE FROM opinions"))
    session.commit()

    # Create enough opinions for train/test split (need at least ~20 per class)
    opinions = []
    labels = []
    oid = 1
    for outcome in ["plaintiff_win", "defendant_win", "mixed"]:
        for i in range(30):
            if outcome == "plaintiff_win":
                txt = f"The court grants judgment for plaintiff {oid}. Defendant is liable under §1692e for debt collection abuse."
            elif outcome == "defendant_win":
                txt = f"Defendant's motion is granted {oid}. Plaintiff's complaint is dismissed under §227 of the TCPA."
            else:
                txt = f"Motion granted in part and denied in part {oid}. Claims under §1681s-2 partially succeed."
            opinions.append(Opinion(id=oid, package_id=f"p{oid}", title=f"Case {oid}", plain_text=txt))
            labels.append(Label(opinion_id=oid, label_type="outcome", label_value=outcome, source="regex", confidence=0.9))
            oid += 1

    session.add_all(opinions)
    session.add_all(labels)
    session.commit()
    return engine


class TestOutcomeClassifier:
    def test_train_outcome_model(self):
        from classify import train_outcome_model
        engine = _setup_labeled_data()
        result = train_outcome_model(engine)
        assert "accuracy" in result
        assert "f1_score" in result
        assert result["accuracy"] > 0.3  # Should be better than random (0.33)
        assert result["model_name"] is not None

    def test_predict_outcomes(self):
        from classify import train_outcome_model, predict_outcomes
        engine = _setup_labeled_data()
        train_outcome_model(engine)
        count = predict_outcomes(engine)
        assert count > 0


class TestClaimTypeClassifier:
    def test_train_claim_type_model(self):
        from classify import train_claim_type_model
        engine = _setup_labeled_data()

        # Add claim type labels
        session = get_session(engine)
        from label import label_claim_types
        opinions = session.execute(text("SELECT id, plain_text FROM opinions")).fetchall()
        for oid, txt in opinions:
            sections = label_claim_types(txt)
            for s in sections:
                session.add(Label(opinion_id=oid, label_type="claim_type", label_value=s, source="regex", confidence=0.95))
        session.commit()
        session.close()

        result = train_claim_type_model(engine)
        assert "sections_trained" in result

    def test_predict_claim_types(self):
        from classify import train_claim_type_model, predict_claim_types
        engine = _setup_labeled_data()

        session = get_session(engine)
        from label import label_claim_types
        opinions = session.execute(text("SELECT id, plain_text FROM opinions")).fetchall()
        for oid, txt in opinions:
            sections = label_claim_types(txt)
            for s in sections:
                session.add(Label(opinion_id=oid, label_type="claim_type", label_value=s, source="regex", confidence=0.95))
        session.commit()
        session.close()

        train_claim_type_model(engine)
        count = predict_claim_types(engine)
        assert count > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'classify'`

**Step 3: Implement classify.py**

Create `classify.py`:

```python
"""
Train, evaluate, and predict with text classifiers.

Usage:
    python classify.py                  # train all models and predict
    python classify.py --train-only     # just train, don't predict
    python classify.py --predict-only   # predict with existing models
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import text

import config
from db import get_local_engine, init_local_db, get_session, Label, Prediction, Model as ModelRecord

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(config.PROJECT_ROOT, "data", "models")


def train_outcome_model(engine=None):
    """Train an outcome classifier (3-class: plaintiff_win, defendant_win, mixed).

    Returns:
        dict with model_name, accuracy, f1_score, classification_report
    """
    if engine is None:
        engine = get_local_engine()

    session = get_session(engine)

    # Load labeled data
    rows = session.execute(text(
        "SELECT o.id, o.plain_text, l.label_value "
        "FROM opinions o "
        "JOIN labels l ON o.id = l.opinion_id "
        "WHERE l.label_type = 'outcome' AND l.label_value != 'unlabeled' "
        "AND o.plain_text IS NOT NULL AND o.plain_text != ''"
    )).fetchall()
    session.close()

    if len(rows) < 10:
        logger.warning(f"Not enough labeled data for training: {len(rows)} samples")
        return {"error": "insufficient data", "count": len(rows)}

    texts = [r[1] for r in rows]
    labels = [r[2] for r in rows]

    logger.info(f"Training outcome model with {len(texts)} samples")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)

    logger.info(f"Outcome model — Accuracy: {acc:.3f}, Macro F1: {f1:.3f}")
    logger.info(f"\n{report}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_name = "outcome_logreg_v1"
    joblib.dump({"tfidf": tfidf, "clf": clf}, os.path.join(MODELS_DIR, "outcome_model.pkl"))

    # Save model record to DB
    session = get_session(engine)
    existing = session.query(ModelRecord).filter_by(name=model_name).first()
    if existing:
        existing.accuracy = float(acc)
        existing.f1_score = float(f1)
        existing.trained_at = datetime.now(timezone.utc).isoformat()
        existing.params_json = json.dumps({"max_features": 50000, "max_iter": 1000})
    else:
        session.add(ModelRecord(
            name=model_name,
            label_type="outcome",
            accuracy=float(acc),
            f1_score=float(f1),
            trained_at=datetime.now(timezone.utc).isoformat(),
            params_json=json.dumps({"max_features": 50000, "max_iter": 1000}),
        ))
    session.commit()
    session.close()

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "f1_score": float(f1),
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def predict_outcomes(engine=None):
    """Predict outcomes for all opinions using the trained model.

    Returns:
        Number of predictions made.
    """
    if engine is None:
        engine = get_local_engine()

    model_path = os.path.join(MODELS_DIR, "outcome_model.pkl")
    if not os.path.exists(model_path):
        logger.error("No trained outcome model found. Run train first.")
        return 0

    model_data = joblib.load(model_path)
    tfidf = model_data["tfidf"]
    clf = model_data["clf"]
    model_name = "outcome_logreg_v1"

    session = get_session(engine)

    # Get all opinions (or just those without predictions)
    rows = session.execute(text(
        "SELECT o.id, o.plain_text FROM opinions o "
        "WHERE o.plain_text IS NOT NULL AND o.plain_text != '' "
        "AND o.id NOT IN (SELECT DISTINCT opinion_id FROM predictions WHERE model_name = :model_name)"
    ), {"model_name": model_name}).fetchall()

    if not rows:
        logger.info("All opinions already have outcome predictions")
        session.close()
        return 0

    logger.info(f"Predicting outcomes for {len(rows)} opinions")

    BATCH_SIZE = 5000
    total = 0

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start:batch_start + BATCH_SIZE]
        opinion_ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        X_tfidf = tfidf.transform(texts)
        predictions = clf.predict(X_tfidf)
        probas = clf.predict_proba(X_tfidf)

        now = datetime.now(timezone.utc).isoformat()
        pred_objects = []
        for oid, pred, proba in zip(opinion_ids, predictions, probas):
            confidence = float(max(proba))
            pred_objects.append(Prediction(
                opinion_id=oid,
                model_name=model_name,
                label_type="outcome",
                predicted_value=pred,
                confidence=confidence,
                created_at=now,
            ))

        session.add_all(pred_objects)
        session.commit()
        total += len(batch)
        logger.info(f"  Predicted {total}/{len(rows)} opinions")

    session.close()
    return total


def train_claim_type_model(engine=None):
    """Train a multi-label claim type classifier.

    Returns:
        dict with model info and per-section metrics.
    """
    if engine is None:
        engine = get_local_engine()

    session = get_session(engine)

    # Get sections with enough examples (>= 50, or >= 10 for testing)
    MIN_EXAMPLES = 10  # Use 50 for production, 10 for testing
    section_counts = session.execute(text(
        "SELECT label_value, COUNT(DISTINCT opinion_id) as cnt "
        "FROM labels WHERE label_type = 'claim_type' "
        "GROUP BY label_value HAVING cnt >= :min_count"
    ), {"min_count": MIN_EXAMPLES}).fetchall()

    if not section_counts:
        logger.warning("Not enough claim type labels for training")
        session.close()
        return {"error": "insufficient data", "sections_trained": 0}

    valid_sections = sorted([r[0] for r in section_counts])
    logger.info(f"Training claim type model for {len(valid_sections)} sections: {valid_sections}")

    # Build dataset: opinion_id → set of sections
    claim_rows = session.execute(text(
        "SELECT opinion_id, label_value FROM labels "
        "WHERE label_type = 'claim_type' AND label_value IN ({})".format(
            ",".join(f"'{s}'" for s in valid_sections)
        )
    )).fetchall()

    opinion_sections = {}
    for oid, section in claim_rows:
        opinion_sections.setdefault(oid, set()).add(section)

    # Get texts for labeled opinions
    opinion_ids = list(opinion_sections.keys())
    placeholders = ",".join(str(oid) for oid in opinion_ids)
    text_rows = session.execute(text(
        f"SELECT id, plain_text FROM opinions WHERE id IN ({placeholders}) "
        "AND plain_text IS NOT NULL AND plain_text != ''"
    )).fetchall()
    session.close()

    texts = []
    multi_labels = []
    for oid, plain_text in text_rows:
        texts.append(plain_text)
        multi_labels.append(list(opinion_sections.get(oid, set())))

    if len(texts) < 10:
        return {"error": "insufficient data", "count": len(texts), "sections_trained": 0}

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=valid_sections)
    y = mlb.fit_transform(multi_labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42
    )

    # TF-IDF + OneVsRest LogisticRegression
    tfidf = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_tfidf)
    per_section = {}
    for i, section in enumerate(valid_sections):
        if y_test[:, i].sum() > 0:
            acc = accuracy_score(y_test[:, i], y_pred[:, i])
            f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            per_section[section] = {"accuracy": float(acc), "f1": float(f1)}

    logger.info(f"Claim type model — per-section F1: {per_section}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_name = "claim_type_logreg_v1"
    joblib.dump(
        {"tfidf": tfidf, "clf": clf, "mlb": mlb, "sections": valid_sections},
        os.path.join(MODELS_DIR, "claim_type_model.pkl"),
    )

    # Save model record
    session = get_session(engine)
    avg_f1 = np.mean([v["f1"] for v in per_section.values()]) if per_section else 0
    existing = session.query(ModelRecord).filter_by(name=model_name).first()
    if existing:
        existing.f1_score = float(avg_f1)
        existing.trained_at = datetime.now(timezone.utc).isoformat()
        existing.params_json = json.dumps({"sections": valid_sections, "per_section": per_section})
    else:
        session.add(ModelRecord(
            name=model_name,
            label_type="claim_type",
            f1_score=float(avg_f1),
            trained_at=datetime.now(timezone.utc).isoformat(),
            params_json=json.dumps({"sections": valid_sections, "per_section": per_section}),
        ))
    session.commit()
    session.close()

    return {
        "model_name": model_name,
        "sections_trained": len(valid_sections),
        "sections": valid_sections,
        "per_section_metrics": per_section,
        "avg_f1": float(avg_f1),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def predict_claim_types(engine=None):
    """Predict claim types for all opinions using the trained model.

    Returns:
        Number of opinions processed.
    """
    if engine is None:
        engine = get_local_engine()

    model_path = os.path.join(MODELS_DIR, "claim_type_model.pkl")
    if not os.path.exists(model_path):
        logger.error("No trained claim type model found. Run train first.")
        return 0

    model_data = joblib.load(model_path)
    tfidf = model_data["tfidf"]
    clf = model_data["clf"]
    mlb = model_data["mlb"]
    model_name = "claim_type_logreg_v1"

    session = get_session(engine)

    rows = session.execute(text(
        "SELECT o.id, o.plain_text FROM opinions o "
        "WHERE o.plain_text IS NOT NULL AND o.plain_text != '' "
        "AND o.id NOT IN (SELECT DISTINCT opinion_id FROM predictions WHERE model_name = :model_name)"
    ), {"model_name": model_name}).fetchall()

    if not rows:
        logger.info("All opinions already have claim type predictions")
        session.close()
        return 0

    logger.info(f"Predicting claim types for {len(rows)} opinions")

    BATCH_SIZE = 5000
    total = 0

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start:batch_start + BATCH_SIZE]
        opinion_ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        X_tfidf = tfidf.transform(texts)
        y_pred = clf.predict(X_tfidf)

        # Get decision function scores for confidence
        try:
            decision_scores = clf.decision_function(X_tfidf)
        except AttributeError:
            decision_scores = None

        now = datetime.now(timezone.utc).isoformat()
        pred_objects = []
        for i, (oid, pred_row) in enumerate(zip(opinion_ids, y_pred)):
            predicted_sections = mlb.inverse_transform(pred_row.reshape(1, -1))[0]
            for section in predicted_sections:
                confidence = 0.5  # default
                if decision_scores is not None:
                    sec_idx = list(mlb.classes_).index(section)
                    confidence = float(min(max(decision_scores[i, sec_idx] * 0.5 + 0.5, 0), 1))

                pred_objects.append(Prediction(
                    opinion_id=oid,
                    model_name=model_name,
                    label_type="claim_type",
                    predicted_value=section,
                    confidence=confidence,
                    created_at=now,
                ))

        session.add_all(pred_objects)
        session.commit()
        total += len(batch)
        logger.info(f"  Predicted {total}/{len(rows)} opinions")

    session.close()
    return total
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classify.py -v`
Expected: 4 PASS

**Step 5: Add CLI to classify.py**

Add at the bottom of `classify.py`:

```python
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train and predict with classifiers")
    parser.add_argument("--train-only", action="store_true", help="Only train models")
    parser.add_argument("--predict-only", action="store_true", help="Only predict (requires trained models)")
    args = parser.parse_args()

    engine = get_local_engine()
    init_local_db(engine)

    if not args.predict_only:
        print("\n=== Training Outcome Model ===")
        outcome_result = train_outcome_model(engine)
        print(f"Accuracy: {outcome_result.get('accuracy', 'N/A')}")
        print(f"Macro F1: {outcome_result.get('f1_score', 'N/A')}")

        print("\n=== Training Claim Type Model ===")
        claim_result = train_claim_type_model(engine)
        print(f"Sections trained: {claim_result.get('sections_trained', 0)}")
        print(f"Avg F1: {claim_result.get('avg_f1', 'N/A')}")

    if not args.train_only:
        print("\n=== Predicting Outcomes ===")
        n = predict_outcomes(engine)
        print(f"Predicted {n} opinions")

        print("\n=== Predicting Claim Types ===")
        n = predict_claim_types(engine)
        print(f"Predicted {n} opinions")
```

**Step 6: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add classify.py tests/test_classify.py
git commit -m "feat: add outcome and claim type classifiers with train/predict pipeline"
```

---

### Task 7: Train and Predict on Real Data

**Files:**
- No code changes — run on the full dataset

**Step 1: Run labeling (if not done in Task 5)**

Run: `cd /c/PythonProject/ml && python label.py`

**Step 2: Train models**

Run: `python classify.py --train-only`
Expected: Prints accuracy and F1 for both models. Review the metrics:
- Outcome model: Expect 50-80% accuracy depending on how distinctive the language is
- Claim type model: Expect variable per-section F1 (common sections like §1692e should be higher)

**Step 3: Predict all opinions**

Run: `python classify.py --predict-only`
Expected: Predicts outcomes and claim types for all ~30K opinions

**Step 4: Verify predictions**

Run:
```python
python -c "
from db import get_local_engine, get_session
from sqlalchemy import text
engine = get_local_engine()
session = get_session(engine)
outcome_preds = session.execute(text(\"SELECT predicted_value, COUNT(*) FROM predictions WHERE label_type='outcome' GROUP BY predicted_value\")).fetchall()
print('Outcome predictions:')
for val, cnt in outcome_preds:
    print(f'  {val}: {cnt}')
claim_preds = session.execute(text(\"SELECT predicted_value, COUNT(*) FROM predictions WHERE label_type='claim_type' GROUP BY predicted_value ORDER BY COUNT(*) DESC LIMIT 15\")).fetchall()
print('Top claim type predictions:')
for val, cnt in claim_preds:
    print(f'  §{val}: {cnt}')
session.close()
"
```

**Step 5: Commit (no code changes)**

No commit needed — validation step.

---

### Task 8: Update FAISS Metadata with Predictions

**Files:**
- Modify: `index.py:60-89` (add_to_index function)
- Modify: `search.py:85-114` (_passes_filters function)
- Modify: `pipeline.py`
- Modify: `tests/test_search.py`

**Step 1: Write the failing test**

Add to `tests/test_search.py`:

```python
def test_search_with_outcome_filter(search_index):
    from search import search_opinions
    index, chunk_map = search_index
    # Add predicted_outcome to test chunks
    for entry in chunk_map:
        entry["predicted_outcome"] = "plaintiff_win"
    chunk_map[2]["predicted_outcome"] = "defendant_win"  # opinion 2

    results = search_opinions(
        index, chunk_map, "violation", filters={"predicted_outcome": "plaintiff_win"}
    )
    for r in results:
        assert r.get("predicted_outcome") == "plaintiff_win"


def test_search_with_claim_section_filter(search_index):
    from search import search_opinions
    index, chunk_map = search_index
    # Add claim_sections to test chunks
    chunk_map[0]["claim_sections"] = "1692e,1692f"
    chunk_map[1]["claim_sections"] = "1692e,1692f"
    chunk_map[2]["claim_sections"] = "227"

    results = search_opinions(
        index, chunk_map, "violation", filters={"claim_section": "1692e"}
    )
    for r in results:
        assert "1692e" in r.get("claim_sections", "")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search.py::test_search_with_outcome_filter tests/test_search.py::test_search_with_claim_section_filter -v`
Expected: FAIL (filter not implemented)

**Step 3: Update search.py filters**

In `search.py`, update the `_passes_filters` function (lines 85-114) to handle new filters. Add these checks after the `opinion_ids` filter (before `return True`):

```python
    if filters.get("predicted_outcome"):
        if meta.get("predicted_outcome") != filters["predicted_outcome"]:
            return False

    if filters.get("claim_section"):
        claim_sections = meta.get("claim_sections", "")
        if filters["claim_section"] not in claim_sections:
            return False
```

Also update the `search_opinions` function result building (around line 66-78) to include the new fields:

```python
            opinion_best[opinion_id] = {
                "opinion_id": opinion_id,
                "title": meta.get("title", ""),
                "court_name": meta.get("court_name", ""),
                "court_type": meta.get("court_type", ""),
                "circuit": meta.get("circuit", ""),
                "date_issued": meta.get("date_issued", ""),
                "statutes": meta.get("statutes", ""),
                "predicted_outcome": meta.get("predicted_outcome", ""),
                "claim_sections": meta.get("claim_sections", ""),
                "similarity_score": similarity,
                "best_passage": meta.get("text", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk_id": meta.get("chunk_id", ""),
            }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_search.py -v`
Expected: All 6 search tests PASS

**Step 5: Update index.py add_to_index**

In `index.py`, update the `add_to_index` function (line 75-87) to include new metadata fields in the chunk_map entries:

```python
    for c in chunks:
        chunk_map.append({
            "chunk_id": c["chunk_id"],
            "opinion_id": c["opinion_id"],
            "chunk_index": c["chunk_index"],
            "title": c.get("title", ""),
            "court_name": c.get("court_name", ""),
            "court_type": c.get("court_type", ""),
            "circuit": c.get("circuit", ""),
            "date_issued": c.get("date_issued", ""),
            "statutes": c.get("statutes", ""),
            "text": c.get("text", ""),
            "predicted_outcome": c.get("predicted_outcome", ""),
            "claim_sections": c.get("claim_sections", ""),
        })
```

**Step 6: Create a script to backfill existing chunk_map with predictions**

Create a new function in `classify.py`:

```python
def update_chunk_map_with_predictions(engine=None):
    """Add predicted_outcome and claim_sections to existing FAISS chunk_map."""
    if engine is None:
        engine = get_local_engine()

    from index import load_index, save_index

    index, chunk_map = load_index()
    if index is None:
        logger.error("No FAISS index found")
        return 0

    session = get_session(engine)

    # Build lookup: opinion_id → predicted outcome
    outcome_rows = session.execute(text(
        "SELECT opinion_id, predicted_value FROM predictions "
        "WHERE model_name = 'outcome_logreg_v1' AND label_type = 'outcome'"
    )).fetchall()
    outcome_map = {r[0]: r[1] for r in outcome_rows}

    # Build lookup: opinion_id → comma-separated claim sections
    claim_rows = session.execute(text(
        "SELECT opinion_id, predicted_value FROM predictions "
        "WHERE model_name = 'claim_type_logreg_v1' AND label_type = 'claim_type'"
    )).fetchall()
    claim_map = {}
    for oid, section in claim_rows:
        claim_map.setdefault(oid, []).append(section)

    session.close()

    updated = 0
    for entry in chunk_map:
        oid = entry["opinion_id"]
        entry["predicted_outcome"] = outcome_map.get(oid, "")
        sections = claim_map.get(oid, [])
        entry["claim_sections"] = ",".join(sorted(sections))
        updated += 1

    save_index(index, chunk_map)
    logger.info(f"Updated {updated} chunk_map entries with predictions")
    return updated
```

Also add `--update-index` to the CLI at the bottom of `classify.py`:

```python
    parser.add_argument("--update-index", action="store_true", help="Update FAISS chunk_map with predictions")
```

And in the main block:

```python
    if args.update_index:
        print("\n=== Updating FAISS Index Metadata ===")
        n = update_chunk_map_with_predictions(engine)
        print(f"Updated {n} chunk_map entries")
```

**Step 7: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add search.py index.py classify.py tests/test_search.py
git commit -m "feat: add prediction metadata to FAISS and search filters"
```

---

### Task 9: Analytics Dashboard

**Files:**
- Modify: `pages/3_Analytics.py` (replace placeholder)

**Step 1: Implement the analytics page**

Replace the contents of `pages/3_Analytics.py` with:

```python
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session

st.set_page_config(page_title="Analytics", page_icon="⚖️", layout="wide")
st.title("Trends & Analytics")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


engine = get_db_engine()
session = get_session(engine)

# Check if predictions exist
pred_count = session.execute(text("SELECT COUNT(*) FROM predictions WHERE label_type='outcome'")).fetchone()[0]

if pred_count == 0:
    st.warning("No predictions found. Run `python classify.py` to train models and generate predictions.")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    statute_options = ["All", "FDCPA", "TCPA", "FCRA"]
    selected_statute = st.selectbox("Statute", statute_options)

    circuits = sorted([
        r[0] for r in session.execute(
            text("SELECT DISTINCT circuit FROM opinions WHERE circuit IS NOT NULL AND circuit != ''")
        ).fetchall()
    ])
    selected_circuit = st.selectbox("Circuit", ["All"] + circuits)

    court_types = ["All", "district", "circuit"]
    selected_court_type = st.selectbox("Court Type", court_types)

# --- Build filter SQL clause ---
filter_clauses = []
if selected_statute != "All":
    filter_clauses.append(
        f"o.id IN (SELECT os.opinion_id FROM opinion_statutes os "
        f"JOIN statutes s ON os.statute_id = s.id WHERE UPPER(s.key) = '{selected_statute.upper()}')"
    )
if selected_circuit != "All":
    filter_clauses.append(f"o.circuit = '{selected_circuit}'")
if selected_court_type != "All":
    filter_clauses.append(f"o.court_type = '{selected_court_type}'")

where_clause = " AND ".join(filter_clauses) if filter_clauses else "1=1"

# --- Outcome Distribution ---
st.subheader("Outcome Distribution")

outcome_data = session.execute(text(
    f"SELECT p.predicted_value, COUNT(*) "
    f"FROM predictions p JOIN opinions o ON p.opinion_id = o.id "
    f"WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1' "
    f"AND {where_clause} "
    f"GROUP BY p.predicted_value ORDER BY COUNT(*) DESC"
)).fetchall()

if outcome_data:
    col1, col2 = st.columns([2, 1])
    with col1:
        chart_data = {r[0]: r[1] for r in outcome_data}
        st.bar_chart(chart_data)
    with col2:
        total = sum(r[1] for r in outcome_data)
        for label, count in outcome_data:
            pct = count / total * 100
            display_label = label.replace("_", " ").title()
            st.metric(display_label, f"{count:,}", f"{pct:.1f}%")

# --- Outcome by Statute ---
st.subheader("Outcome by Statute")

statute_outcome = session.execute(text(
    "SELECT UPPER(s.key), p.predicted_value, COUNT(*) "
    "FROM predictions p "
    "JOIN opinions o ON p.opinion_id = o.id "
    "JOIN opinion_statutes os ON o.id = os.opinion_id "
    "JOIN statutes s ON os.statute_id = s.id "
    "WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1' "
    "GROUP BY UPPER(s.key), p.predicted_value "
    "ORDER BY UPPER(s.key), p.predicted_value"
)).fetchall()

if statute_outcome:
    import pandas as pd
    df = pd.DataFrame(statute_outcome, columns=["Statute", "Outcome", "Count"])
    pivot = df.pivot_table(index="Statute", columns="Outcome", values="Count", fill_value=0)
    st.bar_chart(pivot)

# --- Outcome by Circuit ---
st.subheader("Outcome by Circuit")

circuit_outcome = session.execute(text(
    f"SELECT o.circuit, p.predicted_value, COUNT(*) "
    f"FROM predictions p JOIN opinions o ON p.opinion_id = o.id "
    f"WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1' "
    f"AND o.circuit IS NOT NULL AND o.circuit != '' "
    f"AND {where_clause} "
    f"GROUP BY o.circuit, p.predicted_value "
    f"ORDER BY o.circuit"
)).fetchall()

if circuit_outcome:
    import pandas as pd
    df = pd.DataFrame(circuit_outcome, columns=["Circuit", "Outcome", "Count"])
    pivot = df.pivot_table(index="Circuit", columns="Outcome", values="Count", fill_value=0)
    st.bar_chart(pivot)

# --- Top Claim Sections ---
st.subheader("Top Statutory Sections Cited")

claim_data = session.execute(text(
    f"SELECT p.predicted_value, COUNT(*) "
    f"FROM predictions p JOIN opinions o ON p.opinion_id = o.id "
    f"WHERE p.label_type = 'claim_type' AND p.model_name = 'claim_type_logreg_v1' "
    f"AND {where_clause} "
    f"GROUP BY p.predicted_value ORDER BY COUNT(*) DESC LIMIT 20"
)).fetchall()

if claim_data:
    chart_data = {f"§{r[0]}": r[1] for r in claim_data}
    st.bar_chart(chart_data)

# --- Model Performance ---
st.subheader("Model Performance")

models = session.execute(text(
    "SELECT name, label_type, accuracy, f1_score, trained_at FROM models ORDER BY trained_at DESC"
)).fetchall()

if models:
    for name, ltype, acc, f1, trained in models:
        with st.expander(f"{name} ({ltype})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if acc is not None:
                    st.metric("Accuracy", f"{acc:.3f}")
            with col2:
                if f1 is not None:
                    st.metric("Macro F1", f"{f1:.3f}")
            with col3:
                st.metric("Trained", trained[:10] if trained else "N/A")

# --- Summary Stats ---
st.subheader("Dataset Summary")

total_opinions = session.execute(text("SELECT COUNT(*) FROM opinions")).fetchone()[0]
labeled_outcomes = session.execute(text(
    "SELECT COUNT(DISTINCT opinion_id) FROM labels WHERE label_type='outcome' AND label_value != 'unlabeled'"
)).fetchone()[0]
predicted_outcomes = session.execute(text(
    "SELECT COUNT(DISTINCT opinion_id) FROM predictions WHERE label_type='outcome'"
)).fetchone()[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Opinions", f"{total_opinions:,}")
with col2:
    st.metric("Labeled (regex)", f"{labeled_outcomes:,}")
with col3:
    st.metric("Predicted", f"{predicted_outcomes:,}")

session.close()
```

**Step 2: Run the Streamlit app to verify**

Run: `python -m streamlit run app.py --server.headless true`
Navigate to the Analytics tab and verify charts render.

**Step 3: Commit**

```bash
git add pages/3_Analytics.py
git commit -m "feat: implement analytics dashboard with outcome/claim charts"
```

---

### Task 10: Search Page Integration

**Files:**
- Modify: `pages/1_Search.py`

**Step 1: Add prediction filters to sidebar**

In `pages/1_Search.py`, add after the existing FDCPA subsection filter (around line 89), before `results_per_page`:

```python
    # Predicted outcome filter
    outcome_options = ["All", "Plaintiff Win", "Defendant Win", "Mixed"]
    selected_outcome = st.selectbox("Predicted Outcome", outcome_options)

    # Claim section filter
    claim_sections_available = sorted(set(
        entry.get("claim_sections", "").split(",")
        for entry in (chunk_map or [])
        if entry.get("claim_sections")
    ))
    # Flatten and deduplicate
    all_sections = set()
    for entry in (chunk_map or []):
        cs = entry.get("claim_sections", "")
        if cs:
            for s in cs.split(","):
                if s.strip():
                    all_sections.add(s.strip())
    all_sections = sorted(all_sections)
    selected_claim_sections = st.multiselect("Claim Sections", [f"§{s}" for s in all_sections])
```

**Step 2: Wire up filters in the search execution**

In the filter building section (around lines 105-128), add after the subsection filter logic:

```python
    # Predicted outcome filter
    if selected_outcome != "All":
        outcome_value = selected_outcome.lower().replace(" ", "_")
        filters["predicted_outcome"] = outcome_value

    # Claim section filter
    if selected_claim_sections and len(selected_claim_sections) == 1:
        section = selected_claim_sections[0].replace("§", "")
        filters["claim_section"] = section
```

**Step 3: Display predictions in results**

In the results display (around line 155-189), add after the statute badges (line 171) and before the progress bar:

```python
            # Prediction badges
            badges = []
            pred_outcome = r.get("predicted_outcome", "")
            if pred_outcome:
                outcome_display = pred_outcome.replace("_", " ").title()
                badges.append(f"`{outcome_display}`")
            claim_secs = r.get("claim_sections", "")
            if claim_secs:
                for s in claim_secs.split(","):
                    if s.strip():
                        badges.append(f"`§{s.strip()}`")
            if badges:
                st.markdown("Predictions: " + " ".join(badges))
```

**Step 4: Handle multi-select claim section filter client-side**

After the existing multi-select filtering (around line 145), add:

```python
    if selected_claim_sections and len(selected_claim_sections) > 1:
        filter_sections = [s.replace("§", "") for s in selected_claim_sections]
        results = [r for r in results if any(
            s in r.get("claim_sections", "") for s in filter_sections
        )]
```

**Step 5: Test in Streamlit**

Run: `python -m streamlit run app.py --server.headless true`
Verify: new filters appear in sidebar, predictions display on results.

**Step 6: Commit**

```bash
git add pages/1_Search.py
git commit -m "feat: add prediction filters and badges to search page"
```

---

### Task 11: Pipeline Integration

**Files:**
- Modify: `pipeline.py`

**Step 1: Add classification flags to pipeline**

In `pipeline.py`, add imports at the top (after line 22):

```python
from label import run_labeling
from classify import train_outcome_model, train_claim_type_model, predict_outcomes, predict_claim_types, update_chunk_map_with_predictions
```

**Step 2: Add classification step to run_pipeline**

After the main processing loop (after line 161 `logger.info(f"FAISS index total vectors: {index.ntotal}")`), add:

```python
    # Step 4: Classification (if requested)
    if classify:
        logger.info("Running labeling pipeline...")
        stats = run_labeling(engine)
        logger.info(f"Labeling stats: {stats}")

        logger.info("Training outcome model...")
        outcome_result = train_outcome_model(engine)
        logger.info(f"Outcome model: {outcome_result.get('accuracy', 'N/A')} accuracy")

        logger.info("Training claim type model...")
        claim_result = train_claim_type_model(engine)
        logger.info(f"Claim type model: {claim_result.get('sections_trained', 0)} sections")

        logger.info("Predicting all opinions...")
        predict_outcomes(engine)
        predict_claim_types(engine)

        logger.info("Updating FAISS index metadata...")
        update_chunk_map_with_predictions(engine)

    if predict_new:
        logger.info("Predicting new opinions with existing models...")
        n1 = predict_outcomes(engine)
        n2 = predict_claim_types(engine)
        if n1 > 0 or n2 > 0:
            update_chunk_map_with_predictions(engine)
        logger.info(f"Predicted {n1} outcomes, {n2} claim types")
```

**Step 3: Update function signature and argparse**

Update `run_pipeline` signature to:

```python
def run_pipeline(sync_only=False, reindex=False, classify=False, predict_new=False):
```

Add to argparse (after line 176):

```python
    parser.add_argument("--classify", action="store_true", help="Run labeling + training + prediction")
    parser.add_argument("--predict-only", action="store_true", help="Predict new opinions with existing models")
```

Update the `run_pipeline` call at the end:

```python
    run_pipeline(
        sync_only=args.sync_only,
        reindex=args.reindex,
        classify=args.classify,
        predict_new=args.predict_only,
    )
```

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add pipeline.py
git commit -m "feat: integrate classification into pipeline with --classify and --predict-only flags"
```

---

### Task 12: End-to-End Validation & Final Commit

**Files:**
- No new code — validation and final push

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run the full classification pipeline on real data**

Run: `python label.py && python classify.py`
Expected: Labels ~30K opinions, trains models, predicts all

**Step 3: Update FAISS index metadata**

Run: `python classify.py --update-index`
Expected: Updates all chunk_map entries with predictions

**Step 4: Verify in Streamlit**

Run: `python -m streamlit run app.py --server.headless true`
Check:
- Analytics tab shows charts with data
- Search page has prediction filters in sidebar
- Search results display prediction badges

**Step 5: Verify label stats**

Run: `python label.py --stats`
Expected: Shows reasonable distributions

**Step 6: Run pipeline with --classify to verify integration**

Run: `python pipeline.py --sync-only` (just verify it starts without errors, then Ctrl+C)

**Step 7: Final commit and push**

```bash
git add -A
git commit -m "feat: complete text classification system (Project 2a/2b)"
git push origin main
```

---

## Summary

| Task | Description | New Files | Modified Files |
|------|-------------|-----------|----------------|
| 1 | DB tables + dependencies | tests/test_classify_db.py | requirements.txt, db.py |
| 2 | Outcome regex labeler | label.py, tests/test_label.py | — |
| 3 | Claim type regex labeler | — | label.py, tests/test_label.py |
| 4 | Batch labeling pipeline | — | label.py, tests/test_label.py |
| 5 | Run labeling on real data | — | — |
| 6 | Classification models | classify.py, tests/test_classify.py | — |
| 7 | Train/predict on real data | — | — |
| 8 | FAISS metadata + search filters | — | index.py, search.py, classify.py, tests/test_search.py |
| 9 | Analytics dashboard | — | pages/3_Analytics.py |
| 10 | Search page integration | — | pages/1_Search.py |
| 11 | Pipeline integration | — | pipeline.py |
| 12 | End-to-end validation | — | — |

## Commands Reference

```bash
# Label opinions
python label.py                     # label all new opinions
python label.py --relabel           # re-label everything
python label.py --stats             # show distributions

# Train and predict
python classify.py                  # train + predict
python classify.py --train-only     # just train
python classify.py --predict-only   # just predict
python classify.py --update-index   # update FAISS metadata

# Full pipeline with classification
python pipeline.py --classify       # sync + embed + label + train + predict
python pipeline.py --predict-only   # predict new opinions only

# Tests
python -m pytest tests/ -v          # all tests
```
