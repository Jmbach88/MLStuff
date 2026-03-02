# Trend Analysis & Judicial Analytics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add time-series trend analysis, judicial analytics, circuit comparisons, defense extraction/effectiveness, and statistical significance testing with a dedicated 5-tab Trends dashboard.

**Architecture:** Pure query/aggregation layer (`trends.py`) on top of existing `opinions`, `predictions`, `entities`, and `labels` tables. No new DB tables. Defense types added as a new entity type (`DEFENSE_TYPE`) in existing NER pipeline. Statistical testing via `scipy.stats`. Dashboard as `pages/6_Trends.py` with Plotly charts and significance badges.

**Tech Stack:** pandas, scipy.stats, plotly, streamlit, sqlalchemy

---

### Task 1: Add scipy dependency

**Files:**
- Modify: `requirements.txt:15` (append after spacy line)

**Step 1: Add scipy to requirements.txt**

Add this line at the end of `requirements.txt`:
```
scipy>=1.10.0
```

**Step 2: Install dependency**

Run: `pip install scipy>=1.10.0`
Expected: Successfully installed scipy

**Step 3: Verify import**

Run: `python -c "import scipy; print(scipy.__version__)"`
Expected: Prints version >= 1.10.0

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add scipy dependency for statistical testing"
```

---

### Task 2: Add defense type extraction to NER

**Files:**
- Modify: `ner.py` (add DEFENSE_TYPES patterns, `extract_defense_types()`, wire into `extract_entities_from_opinion()`)
- Modify: `tests/test_ner.py` (add TestDefenseTypeExtraction class)

**Step 1: Write failing tests**

Add to `tests/test_ner.py` after the `TestDebtTypeExtraction` class:

```python
class TestDefenseTypeExtraction:
    def test_bona_fide_error(self):
        from ner import extract_defense_types
        text = "Defendant asserts the bona fide error defense under 15 U.S.C. § 1692k(c)."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("bona fide error" in r["entity_value"].lower() for r in results)

    def test_statute_of_limitations(self):
        from ner import extract_defense_types
        text = "The defendant argues this claim is barred by the statute of limitations."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("statute of limitations" in r["entity_value"].lower() for r in results)

    def test_res_judicata(self):
        from ner import extract_defense_types
        text = "Defendant moves to dismiss on grounds of res judicata."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("res judicata" in r["entity_value"].lower() for r in results)

    def test_claim_preclusion(self):
        from ner import extract_defense_types
        text = "The doctrine of claim preclusion bars this action."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("claim preclusion" in r["entity_value"].lower() for r in results)

    def test_collateral_estoppel(self):
        from ner import extract_defense_types
        text = "Defendant raises collateral estoppel as a bar to relitigation."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("collateral estoppel" in r["entity_value"].lower() for r in results)

    def test_failure_to_state_a_claim(self):
        from ner import extract_defense_types
        text = "Defendant moves to dismiss for failure to state a claim upon which relief can be granted."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("failure to state a claim" in r["entity_value"].lower() for r in results)

    def test_qualified_immunity(self):
        from ner import extract_defense_types
        text = "The officer asserts qualified immunity."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("qualified immunity" in r["entity_value"].lower() for r in results)

    def test_arbitration(self):
        from ner import extract_defense_types
        text = "Defendant moves to compel arbitration under the agreement."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("arbitration" in r["entity_value"].lower() for r in results)

    def test_no_defense(self):
        from ner import extract_defense_types
        text = "The court grants summary judgment for the plaintiff."
        results = extract_defense_types(text)
        assert len(results) == 0

    def test_multiple_defenses(self):
        from ner import extract_defense_types
        text = (
            "Defendant asserts the bona fide error defense and argues "
            "the claim is barred by the statute of limitations. "
            "Additionally, defendant moves to compel arbitration."
        )
        results = extract_defense_types(text)
        assert len(results) >= 3

    def test_deduplication(self):
        from ner import extract_defense_types
        text = (
            "The bona fide error defense applies here. "
            "As stated, bona fide error is established."
        )
        results = extract_defense_types(text)
        bfe_count = sum(1 for r in results if "bona fide error" in r["entity_value"].lower())
        assert bfe_count == 1

    def test_consent_tcpa(self):
        from ner import extract_defense_types
        text = "Defendant argues plaintiff gave prior express consent to receive calls."
        results = extract_defense_types(text)
        assert len(results) >= 1
        assert any("prior express consent" in r["entity_value"].lower() for r in results)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ner.py::TestDefenseTypeExtraction -v`
Expected: FAIL — `extract_defense_types` not found

**Step 3: Implement defense type extraction in ner.py**

Add after the `CREDITOR_CONTEXTS` list (around line 55) in `ner.py`:

```python
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
```

Add new function after `extract_original_creditors()`:

```python
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
```

Wire into `extract_entities_from_opinion()` — add after the `extract_original_creditors` call (line 279):

```python
    entities.extend(extract_defense_types(text_content))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ner.py -v`
Expected: All tests pass (existing 30 + 12 new = 42 tests)

**Step 5: Commit**

```bash
git add ner.py tests/test_ner.py
git commit -m "feat: add defense type extraction to NER pipeline"
```

---

### Task 3: Create trends.py with statistical functions

**Files:**
- Create: `trends.py`
- Create: `tests/test_trends.py`

**Step 1: Write failing tests**

Create `tests/test_trends.py`:

```python
import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Entity
from sqlalchemy import text as sql_text


@pytest.fixture
def engine():
    """Create in-memory DB with test data for trend queries."""
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    # Clean slate
    for table in ["entities", "predictions", "labels", "opinion_statutes", "opinions", "statutes"]:
        session.execute(sql_text(f"DELETE FROM {table}"))
    session.commit()

    # Create statutes
    session.execute(sql_text("INSERT INTO statutes (id, key, name) VALUES (1, 'fdcpa', 'FDCPA')"))
    session.execute(sql_text("INSERT INTO statutes (id, key, name) VALUES (2, 'tcpa', 'TCPA')"))
    session.commit()

    # Create opinions with varied dates, circuits
    opinions = [
        (1, "pkg1", "Smith v. Jones", "district", "3rd", "2020-01-15"),
        (2, "pkg2", "Doe v. Acme", "district", "3rd", "2020-06-20"),
        (3, "pkg3", "Roe v. Bank", "district", "9th", "2021-03-10"),
        (4, "pkg4", "Lee v. Corp", "district", "9th", "2021-07-05"),
        (5, "pkg5", "Kim v. LLC", "district", "3rd", "2022-01-01"),
        (6, "pkg6", "Park v. Inc", "district", "9th", "2022-08-15"),
        (7, "pkg7", "Chen v. Co", "district", "5th", "2020-04-01"),
        (8, "pkg8", "Wang v. Ltd", "district", "5th", "2021-11-01"),
        (9, "pkg9", "Ali v. Firm", "district", "5th", "2022-05-01"),
        (10, "pkg10", "Cruz v. Grp", "district", "3rd", "2022-12-01"),
        (11, "pkg11", "Test v. Case", "district", "3rd", "2020-03-01"),
        (12, "pkg12", "More v. Data", "district", "9th", "2021-09-01"),
    ]
    for oid, pkg, title, ct, circ, date in opinions:
        session.add(Opinion(id=oid, package_id=pkg, title=title,
                            court_type=ct, circuit=circ, date_issued=date))
    session.commit()

    # Link opinions to statutes
    for oid in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        session.execute(sql_text(
            f"INSERT INTO opinion_statutes (opinion_id, statute_id) VALUES ({oid}, 1)"
        ))
    for oid in [3, 4, 6, 12]:
        session.execute(sql_text(
            f"INSERT INTO opinion_statutes (opinion_id, statute_id) VALUES ({oid}, 2)"
        ))
    session.commit()

    # Add outcome predictions
    outcomes = [
        (1, "plaintiff_win"), (2, "defendant_win"), (3, "plaintiff_win"),
        (4, "defendant_win"), (5, "plaintiff_win"), (6, "plaintiff_win"),
        (7, "defendant_win"), (8, "plaintiff_win"), (9, "defendant_win"),
        (10, "plaintiff_win"), (11, "defendant_win"), (12, "plaintiff_win"),
    ]
    for oid, outcome in outcomes:
        session.execute(sql_text(
            f"INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            f"VALUES ({oid}, 'outcome_logreg_v1', 'outcome', '{outcome}', 0.85)"
        ))
    session.commit()

    # Add entities: DAMAGES_AWARDED, ATTORNEY_FEES, JUDGE, DEFENSE_TYPE
    entities = [
        (1, "DAMAGES_AWARDED", "$5,000"),
        (1, "ATTORNEY_FEES", "$2,000"),
        (1, "JUDGE", "Smith"),
        (2, "DAMAGES_AWARDED", "$3,000"),
        (2, "JUDGE", "Smith"),
        (3, "DAMAGES_AWARDED", "$10,000"),
        (3, "ATTORNEY_FEES", "$4,000"),
        (3, "JUDGE", "Jones"),
        (4, "JUDGE", "Jones"),
        (5, "DAMAGES_AWARDED", "$8,000"),
        (5, "ATTORNEY_FEES", "$3,000"),
        (5, "JUDGE", "Smith"),
        (6, "DAMAGES_AWARDED", "$12,000"),
        (6, "ATTORNEY_FEES", "$5,000"),
        (6, "JUDGE", "Jones"),
        (7, "DAMAGES_AWARDED", "$2,000"),
        (7, "JUDGE", "Williams"),
        (8, "DAMAGES_AWARDED", "$7,000"),
        (8, "ATTORNEY_FEES", "$3,500"),
        (8, "JUDGE", "Williams"),
        (9, "JUDGE", "Williams"),
        (10, "DAMAGES_AWARDED", "$6,000"),
        (10, "ATTORNEY_FEES", "$2,500"),
        (10, "JUDGE", "Smith"),
        (11, "JUDGE", "Smith"),
        (12, "DAMAGES_AWARDED", "$9,000"),
        (12, "ATTORNEY_FEES", "$4,500"),
        (12, "JUDGE", "Jones"),
        # Defense types
        (1, "DEFENSE_TYPE", "bona fide error"),
        (2, "DEFENSE_TYPE", "statute of limitations"),
        (2, "DEFENSE_TYPE", "bona fide error"),
        (3, "DEFENSE_TYPE", "arbitration"),
        (4, "DEFENSE_TYPE", "bona fide error"),
        (5, "DEFENSE_TYPE", "statute of limitations"),
        (7, "DEFENSE_TYPE", "bona fide error"),
        (9, "DEFENSE_TYPE", "arbitration"),
        (9, "DEFENSE_TYPE", "bona fide error"),
    ]
    for oid, etype, evalue in entities:
        session.add(Entity(opinion_id=oid, entity_type=etype, entity_value=evalue,
                           context_snippet="test context"))
    session.commit()
    session.close()

    return engine


class TestOutcomeTrends:
    def test_outcome_trends_returns_dataframe(self, engine):
        from trends import outcome_trends_by_year
        df = outcome_trends_by_year(engine)
        assert not df.empty
        assert "year" in df.columns
        assert "plaintiff_win_rate" in df.columns
        assert "total" in df.columns

    def test_outcome_trends_filter_by_circuit(self, engine):
        from trends import outcome_trends_by_year
        df = outcome_trends_by_year(engine, circuit="3rd")
        assert not df.empty
        assert all(df["total"] > 0)

    def test_outcome_trends_filter_by_statute(self, engine):
        from trends import outcome_trends_by_year
        df = outcome_trends_by_year(engine, statute="FDCPA")
        assert not df.empty


class TestDamagesTrends:
    def test_damages_trends_returns_dataframe(self, engine):
        from trends import damages_trends_by_year
        df = damages_trends_by_year(engine)
        assert not df.empty
        assert "year" in df.columns
        assert "mean_damages" in df.columns
        assert "median_damages" in df.columns

    def test_damages_trends_filter_by_circuit(self, engine):
        from trends import damages_trends_by_year
        df = damages_trends_by_year(engine, circuit="9th")
        assert not df.empty


class TestJudgeStats:
    def test_judge_stats_returns_dataframe(self, engine):
        from trends import judge_stats
        df = judge_stats(engine, min_opinions=2)
        assert not df.empty
        assert "judge" in df.columns
        assert "opinion_count" in df.columns
        assert "plaintiff_win_rate" in df.columns

    def test_judge_stats_filters_minimum(self, engine):
        from trends import judge_stats
        df = judge_stats(engine, min_opinions=100)
        assert df.empty


class TestCircuitComparison:
    def test_circuit_comparison_returns_dataframe(self, engine):
        from trends import circuit_comparison
        df = circuit_comparison(engine)
        assert not df.empty
        assert "circuit" in df.columns
        assert "opinion_count" in df.columns
        assert "plaintiff_win_rate" in df.columns

    def test_circuit_outcome_heatmap_returns_dataframe(self, engine):
        from trends import circuit_outcome_heatmap
        df = circuit_outcome_heatmap(engine)
        assert not df.empty
        assert "circuit" in df.columns
        assert "statute" in df.columns
        assert "plaintiff_win_rate" in df.columns


class TestDefenseAnalysis:
    def test_defense_frequency(self, engine):
        from trends import defense_frequency
        df = defense_frequency(engine)
        assert not df.empty
        assert "defense_type" in df.columns
        assert "count" in df.columns

    def test_defense_effectiveness(self, engine):
        from trends import defense_effectiveness
        df = defense_effectiveness(engine, min_count=1)
        assert not df.empty
        assert "defense_type" in df.columns
        assert "times_raised" in df.columns
        assert "defendant_win_rate_when_raised" in df.columns

    def test_defense_by_circuit(self, engine):
        from trends import defense_by_circuit
        df = defense_by_circuit(engine)
        assert not df.empty
        assert "defense_type" in df.columns
        assert "circuit" in df.columns


class TestStatisticalFunctions:
    def test_compare_outcome_rates_significant(self):
        from trends import compare_outcome_rates
        result = compare_outcome_rates(0.8, 100, 0.3, 100)
        assert "p_value" in result
        assert "significant" in result
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_compare_outcome_rates_not_significant(self):
        from trends import compare_outcome_rates
        result = compare_outcome_rates(0.5, 10, 0.55, 10)
        assert result["significant"] is False

    def test_compare_outcome_rates_badges(self):
        from trends import compare_outcome_rates
        result = compare_outcome_rates(0.9, 500, 0.1, 500)
        assert "badge" in result
        assert result["badge"] in ("*", "**", "***")

    def test_compare_damages_returns_result(self):
        from trends import compare_damages
        group1 = [1000, 2000, 3000, 4000, 5000]
        group2 = [10000, 20000, 30000, 40000, 50000]
        result = compare_damages(group1, group2)
        assert "p_value" in result
        assert "significant" in result

    def test_compare_damages_empty_group(self):
        from trends import compare_damages
        result = compare_damages([], [1000, 2000])
        assert result["significant"] is False


class TestClaimFrequency:
    def test_claim_frequency_returns_dataframe(self, engine):
        from trends import claim_frequency_by_year
        # Add some claim type predictions
        session = get_session(engine)
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            "VALUES (1, 'claim_type_logreg_v1', 'claim_type', '1692e', 0.8)"
        ))
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            "VALUES (3, 'claim_type_logreg_v1', 'claim_type', '1692e', 0.7)"
        ))
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            "VALUES (5, 'claim_type_logreg_v1', 'claim_type', '1692g', 0.9)"
        ))
        session.commit()
        session.close()

        df = claim_frequency_by_year(engine)
        assert not df.empty
        assert "year" in df.columns
        assert "claim_section" in df.columns
        assert "count" in df.columns
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trends.py -v`
Expected: FAIL — `trends` module not found

**Step 3: Implement trends.py**

Create `trends.py`:

```python
"""
Trend analysis and judicial analytics: query/aggregation layer.

Pure query functions returning pandas DataFrames. No new DB tables.

Usage:
    import trends
    df = trends.outcome_trends_by_year(engine, statute="FDCPA")
"""
import re
import logging

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _parse_dollar(value):
    """Parse dollar string like '$1,000.00' to float."""
    if not value:
        return None
    cleaned = re.sub(r'[,$\s]', '', value)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _build_filter_clause(statute=None, circuit=None, court_type=None,
                          year_min=None, year_max=None):
    """Build SQL WHERE clauses for common filters."""
    clauses = []
    if statute:
        clauses.append(
            f"o.id IN (SELECT os.opinion_id FROM opinion_statutes os "
            f"JOIN statutes s ON os.statute_id = s.id "
            f"WHERE UPPER(s.key) = '{statute.upper()}')"
        )
    if circuit:
        clauses.append(f"o.circuit = '{circuit}'")
    if court_type:
        clauses.append(f"o.court_type = '{court_type}'")
    if year_min is not None:
        clauses.append(f"CAST(SUBSTR(o.date_issued, 1, 4) AS INTEGER) >= {year_min}")
    if year_max is not None:
        clauses.append(f"CAST(SUBSTR(o.date_issued, 1, 4) AS INTEGER) <= {year_max}")
    return " AND ".join(clauses) if clauses else "1=1"


# --- Time-series functions ---

def outcome_trends_by_year(engine, statute=None, circuit=None,
                            court_type=None, year_min=None, year_max=None):
    """Plaintiff win rate by year with counts.

    Returns DataFrame with columns: year, plaintiff_wins, total, plaintiff_win_rate
    """
    where = _build_filter_clause(statute, circuit, court_type, year_min, year_max)
    query = f"""
        SELECT CAST(SUBSTR(o.date_issued, 1, 4) AS INTEGER) as year,
               SUM(CASE WHEN p.predicted_value = 'plaintiff_win' THEN 1 ELSE 0 END) as plaintiff_wins,
               COUNT(*) as total
        FROM predictions p
        JOIN opinions o ON p.opinion_id = o.id
        WHERE p.label_type = 'outcome'
          AND p.model_name = 'outcome_logreg_v1'
          AND o.date_issued IS NOT NULL AND o.date_issued != ''
          AND {where}
        GROUP BY year
        ORDER BY year
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["year", "plaintiff_wins", "total", "plaintiff_win_rate"])

    df = pd.DataFrame(rows, columns=["year", "plaintiff_wins", "total"])
    df["plaintiff_win_rate"] = df["plaintiff_wins"] / df["total"]
    return df


def damages_trends_by_year(engine, statute=None, circuit=None,
                            court_type=None, year_min=None, year_max=None):
    """Mean/median DAMAGES_AWARDED and ATTORNEY_FEES by year.

    Returns DataFrame with columns: year, mean_damages, median_damages,
    mean_fees, median_fees, damages_count, fees_count
    """
    where = _build_filter_clause(statute, circuit, court_type, year_min, year_max)
    query = f"""
        SELECT CAST(SUBSTR(o.date_issued, 1, 4) AS INTEGER) as year,
               e.entity_type, e.entity_value
        FROM entities e
        JOIN opinions o ON e.opinion_id = o.id
        WHERE e.entity_type IN ('DAMAGES_AWARDED', 'ATTORNEY_FEES')
          AND o.date_issued IS NOT NULL AND o.date_issued != ''
          AND {where}
        ORDER BY year
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "year", "mean_damages", "median_damages",
            "mean_fees", "median_fees", "damages_count", "fees_count",
        ])

    data = []
    for year, etype, evalue in rows:
        amount = _parse_dollar(evalue)
        if amount is not None and amount > 0:
            data.append({"year": year, "type": etype, "amount": amount})

    if not data:
        return pd.DataFrame(columns=[
            "year", "mean_damages", "median_damages",
            "mean_fees", "median_fees", "damages_count", "fees_count",
        ])

    raw_df = pd.DataFrame(data)
    years = sorted(raw_df["year"].unique())
    results = []
    for yr in years:
        yr_data = raw_df[raw_df["year"] == yr]
        damages = yr_data[yr_data["type"] == "DAMAGES_AWARDED"]["amount"]
        fees = yr_data[yr_data["type"] == "ATTORNEY_FEES"]["amount"]
        results.append({
            "year": yr,
            "mean_damages": damages.mean() if len(damages) > 0 else None,
            "median_damages": damages.median() if len(damages) > 0 else None,
            "mean_fees": fees.mean() if len(fees) > 0 else None,
            "median_fees": fees.median() if len(fees) > 0 else None,
            "damages_count": len(damages),
            "fees_count": len(fees),
        })
    return pd.DataFrame(results)


def claim_frequency_by_year(engine, statute=None, top_n=10):
    """Top N claim sections by year.

    Returns DataFrame with columns: year, claim_section, count
    """
    where = _build_filter_clause(statute)
    query = f"""
        SELECT CAST(SUBSTR(o.date_issued, 1, 4) AS INTEGER) as year,
               p.predicted_value as claim_section,
               COUNT(*) as count
        FROM predictions p
        JOIN opinions o ON p.opinion_id = o.id
        WHERE p.label_type = 'claim_type'
          AND p.model_name = 'claim_type_logreg_v1'
          AND o.date_issued IS NOT NULL AND o.date_issued != ''
          AND {where}
        GROUP BY year, claim_section
        ORDER BY year, count DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["year", "claim_section", "count"])

    df = pd.DataFrame(rows, columns=["year", "claim_section", "count"])

    # Keep only top N sections overall
    top_sections = df.groupby("claim_section")["count"].sum().nlargest(top_n).index
    df = df[df["claim_section"].isin(top_sections)]
    return df


# --- Judicial analytics ---

def judge_stats(engine, min_opinions=10):
    """Per-judge statistics.

    Returns DataFrame: judge, opinion_count, plaintiff_win_rate,
    avg_damages, avg_fees
    """
    query = """
        SELECT e.entity_value as judge,
               COUNT(DISTINCT e.opinion_id) as opinion_count,
               SUM(CASE WHEN p.predicted_value = 'plaintiff_win' THEN 1 ELSE 0 END) as plaintiff_wins,
               COUNT(CASE WHEN p.predicted_value IS NOT NULL THEN 1 END) as predicted_count
        FROM entities e
        JOIN predictions p ON e.opinion_id = p.opinion_id
            AND p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1'
        WHERE e.entity_type = 'JUDGE'
        GROUP BY e.entity_value
        HAVING opinion_count >= :min_opinions
        ORDER BY opinion_count DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query), {"min_opinions": min_opinions}).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "judge", "opinion_count", "plaintiff_win_rate", "avg_damages", "avg_fees",
        ])

    judges_df = pd.DataFrame(rows, columns=[
        "judge", "opinion_count", "plaintiff_wins", "predicted_count",
    ])
    judges_df["plaintiff_win_rate"] = judges_df["plaintiff_wins"] / judges_df["predicted_count"]

    # Get avg damages and fees per judge
    dmg_query = """
        SELECT j.entity_value as judge,
               AVG(CAST(REPLACE(REPLACE(REPLACE(d.entity_value, '$', ''), ',', ''), ' ', '') AS REAL)) as avg_damages
        FROM entities j
        JOIN entities d ON j.opinion_id = d.opinion_id AND d.entity_type = 'DAMAGES_AWARDED'
        WHERE j.entity_type = 'JUDGE'
        GROUP BY j.entity_value
    """
    fees_query = """
        SELECT j.entity_value as judge,
               AVG(CAST(REPLACE(REPLACE(REPLACE(f.entity_value, '$', ''), ',', ''), ' ', '') AS REAL)) as avg_fees
        FROM entities j
        JOIN entities f ON j.opinion_id = f.opinion_id AND f.entity_type = 'ATTORNEY_FEES'
        WHERE j.entity_type = 'JUDGE'
        GROUP BY j.entity_value
    """
    with engine.connect() as conn:
        dmg_rows = conn.execute(text(dmg_query)).fetchall()
        fees_rows = conn.execute(text(fees_query)).fetchall()

    dmg_map = {r[0]: r[1] for r in dmg_rows}
    fees_map = {r[0]: r[1] for r in fees_rows}

    judges_df["avg_damages"] = judges_df["judge"].map(dmg_map)
    judges_df["avg_fees"] = judges_df["judge"].map(fees_map)

    return judges_df[["judge", "opinion_count", "plaintiff_win_rate", "avg_damages", "avg_fees"]]


# --- Circuit comparison ---

def circuit_comparison(engine):
    """Per-circuit statistics.

    Returns DataFrame: circuit, opinion_count, plaintiff_win_rate,
    avg_damages, avg_fees
    """
    query = """
        SELECT o.circuit,
               COUNT(*) as opinion_count,
               SUM(CASE WHEN p.predicted_value = 'plaintiff_win' THEN 1 ELSE 0 END) as plaintiff_wins
        FROM predictions p
        JOIN opinions o ON p.opinion_id = o.id
        WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1'
          AND o.circuit IS NOT NULL AND o.circuit != ''
        GROUP BY o.circuit
        ORDER BY opinion_count DESC
    """
    dmg_query = """
        SELECT o.circuit,
               AVG(CAST(REPLACE(REPLACE(REPLACE(e.entity_value, '$', ''), ',', ''), ' ', '') AS REAL)) as avg_damages
        FROM entities e
        JOIN opinions o ON e.opinion_id = o.id
        WHERE e.entity_type = 'DAMAGES_AWARDED'
          AND o.circuit IS NOT NULL AND o.circuit != ''
        GROUP BY o.circuit
    """
    fees_query = """
        SELECT o.circuit,
               AVG(CAST(REPLACE(REPLACE(REPLACE(e.entity_value, '$', ''), ',', ''), ' ', '') AS REAL)) as avg_fees
        FROM entities e
        JOIN opinions o ON e.opinion_id = o.id
        WHERE e.entity_type = 'ATTORNEY_FEES'
          AND o.circuit IS NOT NULL AND o.circuit != ''
        GROUP BY o.circuit
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
        dmg_rows = conn.execute(text(dmg_query)).fetchall()
        fees_rows = conn.execute(text(fees_query)).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "circuit", "opinion_count", "plaintiff_win_rate", "avg_damages", "avg_fees",
        ])

    df = pd.DataFrame(rows, columns=["circuit", "opinion_count", "plaintiff_wins"])
    df["plaintiff_win_rate"] = df["plaintiff_wins"] / df["opinion_count"]

    dmg_map = {r[0]: r[1] for r in dmg_rows}
    fees_map = {r[0]: r[1] for r in fees_rows}
    df["avg_damages"] = df["circuit"].map(dmg_map)
    df["avg_fees"] = df["circuit"].map(fees_map)

    return df[["circuit", "opinion_count", "plaintiff_win_rate", "avg_damages", "avg_fees"]]


def circuit_outcome_heatmap(engine):
    """Plaintiff win rate by circuit x statute.

    Returns DataFrame: circuit, statute, plaintiff_win_rate, count
    """
    query = """
        SELECT o.circuit, UPPER(s.key) as statute,
               SUM(CASE WHEN p.predicted_value = 'plaintiff_win' THEN 1 ELSE 0 END) as plaintiff_wins,
               COUNT(*) as total
        FROM predictions p
        JOIN opinions o ON p.opinion_id = o.id
        JOIN opinion_statutes os ON o.id = os.opinion_id
        JOIN statutes s ON os.statute_id = s.id
        WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1'
          AND o.circuit IS NOT NULL AND o.circuit != ''
        GROUP BY o.circuit, UPPER(s.key)
        ORDER BY o.circuit, statute
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["circuit", "statute", "plaintiff_win_rate", "count"])

    df = pd.DataFrame(rows, columns=["circuit", "statute", "plaintiff_wins", "count"])
    df["plaintiff_win_rate"] = df["plaintiff_wins"] / df["count"]
    return df[["circuit", "statute", "plaintiff_win_rate", "count"]]


# --- Defense analysis ---

def defense_frequency(engine):
    """Defense type counts.

    Returns DataFrame: defense_type, count
    """
    query = """
        SELECT entity_value as defense_type, COUNT(*) as count
        FROM entities
        WHERE entity_type = 'DEFENSE_TYPE'
        GROUP BY entity_value
        ORDER BY count DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["defense_type", "count"])

    return pd.DataFrame(rows, columns=["defense_type", "count"])


def defense_effectiveness(engine, min_count=20):
    """Per defense: times raised, defendant win rate when raised vs. overall.

    Returns DataFrame: defense_type, times_raised,
    defendant_win_rate_when_raised, overall_defendant_win_rate
    """
    # Overall defendant win rate
    overall_query = """
        SELECT SUM(CASE WHEN predicted_value = 'defendant_win' THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
        FROM predictions
        WHERE label_type = 'outcome' AND model_name = 'outcome_logreg_v1'
    """
    # Per-defense stats
    defense_query = """
        SELECT e.entity_value as defense_type,
               COUNT(DISTINCT e.opinion_id) as times_raised,
               SUM(CASE WHEN p.predicted_value = 'defendant_win' THEN 1 ELSE 0 END) as defendant_wins,
               COUNT(*) as total
        FROM entities e
        JOIN predictions p ON e.opinion_id = p.opinion_id
            AND p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1'
        WHERE e.entity_type = 'DEFENSE_TYPE'
        GROUP BY e.entity_value
        HAVING times_raised >= :min_count
        ORDER BY times_raised DESC
    """
    with engine.connect() as conn:
        overall_rate = conn.execute(text(overall_query)).scalar() or 0.0
        rows = conn.execute(text(defense_query), {"min_count": min_count}).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "defense_type", "times_raised",
            "defendant_win_rate_when_raised", "overall_defendant_win_rate",
        ])

    df = pd.DataFrame(rows, columns=["defense_type", "times_raised", "defendant_wins", "total"])
    df["defendant_win_rate_when_raised"] = df["defendant_wins"] / df["total"]
    df["overall_defendant_win_rate"] = overall_rate
    return df[["defense_type", "times_raised", "defendant_win_rate_when_raised", "overall_defendant_win_rate"]]


def defense_by_circuit(engine):
    """Defense frequency and effectiveness by circuit.

    Returns DataFrame: defense_type, circuit, count, defendant_win_rate
    """
    query = """
        SELECT e.entity_value as defense_type,
               o.circuit,
               COUNT(DISTINCT e.opinion_id) as count,
               SUM(CASE WHEN p.predicted_value = 'defendant_win' THEN 1 ELSE 0 END) as defendant_wins,
               COUNT(*) as total
        FROM entities e
        JOIN opinions o ON e.opinion_id = o.id
        JOIN predictions p ON e.opinion_id = p.opinion_id
            AND p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1'
        WHERE e.entity_type = 'DEFENSE_TYPE'
          AND o.circuit IS NOT NULL AND o.circuit != ''
        GROUP BY e.entity_value, o.circuit
        ORDER BY defense_type, circuit
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["defense_type", "circuit", "count", "defendant_win_rate"])

    df = pd.DataFrame(rows, columns=["defense_type", "circuit", "count", "defendant_wins", "total"])
    df["defendant_win_rate"] = df["defendant_wins"] / df["total"]
    return df[["defense_type", "circuit", "count", "defendant_win_rate"]]


# --- Statistical testing ---

def compare_outcome_rates(rate1, n1, rate2, n2):
    """Chi-squared test for two proportions.

    Args:
        rate1: proportion in group 1
        n1: sample size of group 1
        rate2: proportion in group 2
        n2: sample size of group 2

    Returns dict with p_value, significant, badge
    """
    from scipy.stats import chi2_contingency
    import numpy as np

    if n1 < 5 or n2 < 5:
        return {"p_value": 1.0, "significant": False, "badge": ""}

    success1 = int(round(rate1 * n1))
    success2 = int(round(rate2 * n2))
    fail1 = n1 - success1
    fail2 = n2 - success2

    table = np.array([[success1, fail1], [success2, fail2]])

    if table.min() < 0:
        return {"p_value": 1.0, "significant": False, "badge": ""}

    try:
        chi2, p_value, dof, expected = chi2_contingency(table)
    except ValueError:
        return {"p_value": 1.0, "significant": False, "badge": ""}

    if p_value < 0.001:
        badge = "***"
    elif p_value < 0.01:
        badge = "**"
    elif p_value < 0.05:
        badge = "*"
    else:
        badge = ""

    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "badge": badge,
    }


def compare_damages(group1, group2):
    """Mann-Whitney U test for two damage distributions.

    Args:
        group1: list/array of dollar amounts
        group2: list/array of dollar amounts

    Returns dict with p_value, significant, badge
    """
    from scipy.stats import mannwhitneyu

    if len(group1) < 2 or len(group2) < 2:
        return {"p_value": 1.0, "significant": False, "badge": ""}

    try:
        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    except ValueError:
        return {"p_value": 1.0, "significant": False, "badge": ""}

    if p_value < 0.001:
        badge = "***"
    elif p_value < 0.01:
        badge = "**"
    elif p_value < 0.05:
        badge = "*"
    else:
        badge = ""

    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "badge": badge,
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trends.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add trends.py tests/test_trends.py
git commit -m "feat: add trend analysis query/aggregation module"
```

---

### Task 4: Run defense extraction on real data

**Files:** None (CLI execution only)

**Step 1: Run defense type extraction on the corpus**

Run: `python ner.py --types defense_type`
Expected: Extracts DEFENSE_TYPE entities from ~30K opinions. Takes ~5 minutes.

**Step 2: Verify extraction results**

Run: `python ner.py --info`
Expected: Shows DEFENSE_TYPE in the entity summary with counts.

**Step 3: Commit** (no file changes — extraction writes to DB)

No commit needed for this step.

---

### Task 5: Create Trends Dashboard

**Files:**
- Create: `pages/6_Trends.py`

**Step 1: Create the 5-tab Trends dashboard**

Create `pages/6_Trends.py`:

```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session
from trends import (
    outcome_trends_by_year, damages_trends_by_year, claim_frequency_by_year,
    judge_stats, circuit_comparison, circuit_outcome_heatmap,
    defense_frequency, defense_effectiveness, defense_by_circuit,
    compare_outcome_rates,
)

st.set_page_config(page_title="Trends", page_icon="⚖️", layout="wide")
st.title("Trend Analysis & Judicial Analytics")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


engine = get_db_engine()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    statute_options = ["All", "FDCPA", "TCPA", "FCRA"]
    selected_statute = st.selectbox("Statute", statute_options)

    with engine.connect() as conn:
        circuits = sorted([
            r[0] for r in conn.execute(
                text("SELECT DISTINCT circuit FROM opinions WHERE circuit IS NOT NULL AND circuit != ''")
            ).fetchall()
        ])
    selected_circuit = st.selectbox("Circuit", ["All"] + circuits)

    court_types = ["All", "district", "circuit"]
    selected_court_type = st.selectbox("Court Type", court_types)

    # Year range
    with engine.connect() as conn:
        year_range = conn.execute(text(
            "SELECT MIN(CAST(SUBSTR(date_issued, 1, 4) AS INTEGER)), "
            "MAX(CAST(SUBSTR(date_issued, 1, 4) AS INTEGER)) "
            "FROM opinions WHERE date_issued IS NOT NULL AND date_issued != ''"
        )).fetchone()

    if year_range and year_range[0] and year_range[1]:
        min_year, max_year = year_range
        selected_years = st.slider("Year Range", min_year, max_year, (min_year, max_year))
    else:
        selected_years = (None, None)

# Build filter kwargs
filter_kwargs = {}
if selected_statute != "All":
    filter_kwargs["statute"] = selected_statute
if selected_circuit != "All":
    filter_kwargs["circuit"] = selected_circuit
if selected_court_type != "All":
    filter_kwargs["court_type"] = selected_court_type
if selected_years[0] is not None:
    filter_kwargs["year_min"] = selected_years[0]
    filter_kwargs["year_max"] = selected_years[1]

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Outcome Trends", "Damages Trends", "Judicial Analytics",
    "Circuit Comparison", "Defense Analysis",
])

# === Tab 1: Outcome Trends ===
with tab1:
    df_outcome = outcome_trends_by_year(engine, **filter_kwargs)

    if df_outcome.empty:
        st.info("No outcome data available for the selected filters.")
    else:
        # Line chart: plaintiff win rate by year
        fig = px.line(
            df_outcome, x="year", y="plaintiff_win_rate",
            markers=True,
            labels={"year": "Year", "plaintiff_win_rate": "Plaintiff Win Rate"},
            title="Plaintiff Win Rate Over Time",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        display_df = df_outcome.copy()
        display_df["plaintiff_win_rate"] = display_df["plaintiff_win_rate"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Circuit vs overall comparison
        if selected_circuit != "All":
            df_overall = outcome_trends_by_year(
                engine,
                statute=filter_kwargs.get("statute"),
                court_type=filter_kwargs.get("court_type"),
                year_min=filter_kwargs.get("year_min"),
                year_max=filter_kwargs.get("year_max"),
            )
            if not df_overall.empty:
                st.subheader(f"{selected_circuit} vs. Overall")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df_outcome["year"], y=df_outcome["plaintiff_win_rate"],
                    name=selected_circuit, mode="lines+markers",
                ))
                fig2.add_trace(go.Scatter(
                    x=df_overall["year"], y=df_overall["plaintiff_win_rate"],
                    name="Overall", mode="lines+markers", line=dict(dash="dash"),
                ))
                fig2.update_yaxes(tickformat=".0%", range=[0, 1])
                fig2.update_layout(title="Circuit vs. Overall Plaintiff Win Rate")
                st.plotly_chart(fig2, use_container_width=True)

                # Significance test
                circ_total = df_outcome["total"].sum()
                circ_wins = df_outcome["plaintiff_wins"].sum()
                overall_total = df_overall["total"].sum()
                overall_wins = df_overall["plaintiff_wins"].sum()
                if circ_total > 0 and overall_total > 0:
                    result = compare_outcome_rates(
                        circ_wins / circ_total, circ_total,
                        overall_wins / overall_total, overall_total,
                    )
                    if result["significant"]:
                        st.success(
                            f"Statistically significant difference {result['badge']} "
                            f"(p={result['p_value']:.4f})"
                        )
                    else:
                        st.info(f"No significant difference (p={result['p_value']:.4f})")

# === Tab 2: Damages Trends ===
with tab2:
    df_damages = damages_trends_by_year(engine, **filter_kwargs)

    if df_damages.empty:
        st.info("No damages data available for the selected filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig_dmg = px.line(
                df_damages, x="year", y="median_damages",
                markers=True,
                labels={"year": "Year", "median_damages": "Median Damages ($)"},
                title="Median Damages Awarded Over Time",
            )
            fig_dmg.update_yaxes(tickprefix="$", tickformat=",")
            st.plotly_chart(fig_dmg, use_container_width=True)

        with col2:
            fees_data = df_damages.dropna(subset=["median_fees"])
            if not fees_data.empty:
                fig_fees = px.line(
                    fees_data, x="year", y="median_fees",
                    markers=True,
                    labels={"year": "Year", "median_fees": "Median Attorney Fees ($)"},
                    title="Median Attorney Fees Over Time",
                )
                fig_fees.update_yaxes(tickprefix="$", tickformat=",")
                st.plotly_chart(fig_fees, use_container_width=True)

        # Data table
        display_df = df_damages.copy()
        for col in ["mean_damages", "median_damages", "mean_fees", "median_fees"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# === Tab 3: Judicial Analytics ===
with tab3:
    df_judges = judge_stats(engine, min_opinions=10)

    if df_judges.empty:
        st.info("No judges found with 10+ opinions. Try lowering the threshold.")
    else:
        st.subheader(f"Judges with 10+ Opinions ({len(df_judges)} judges)")

        # Get overall plaintiff win rate for significance testing
        with engine.connect() as conn:
            overall = conn.execute(text(
                "SELECT SUM(CASE WHEN predicted_value = 'plaintiff_win' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), "
                "COUNT(*) "
                "FROM predictions WHERE label_type = 'outcome' AND model_name = 'outcome_logreg_v1'"
            )).fetchone()
        overall_rate = overall[0] or 0.5
        overall_n = overall[1] or 1

        # Add significance badges
        badges = []
        for _, row in df_judges.iterrows():
            result = compare_outcome_rates(
                row["plaintiff_win_rate"], row["opinion_count"],
                overall_rate, overall_n,
            )
            badges.append(result["badge"])
        df_judges["sig"] = badges

        display_df = df_judges.copy()
        display_df["plaintiff_win_rate"] = display_df["plaintiff_win_rate"].apply(lambda x: f"{x:.1%}")
        display_df["avg_damages"] = display_df["avg_damages"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        display_df["avg_fees"] = display_df["avg_fees"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        st.dataframe(
            display_df[["judge", "opinion_count", "plaintiff_win_rate", "avg_damages", "avg_fees", "sig"]],
            use_container_width=True, hide_index=True,
            column_config={"sig": st.column_config.TextColumn("Sig.", width="small")},
        )

# === Tab 4: Circuit Comparison ===
with tab4:
    df_circuits = circuit_comparison(engine)

    if df_circuits.empty:
        st.info("No circuit data available.")
    else:
        # Heatmap: plaintiff win rate by circuit x statute
        df_heatmap = circuit_outcome_heatmap(engine)
        if not df_heatmap.empty:
            st.subheader("Plaintiff Win Rate: Circuit x Statute")
            pivot = df_heatmap.pivot_table(
                index="circuit", columns="statute",
                values="plaintiff_win_rate", fill_value=0,
            )
            fig_heat = px.imshow(
                pivot, text_auto=".0%",
                labels=dict(x="Statute", y="Circuit", color="Win Rate"),
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=1,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Bar charts: avg damages and fees by circuit
        col1, col2 = st.columns(2)

        with col1:
            dmg_data = df_circuits.dropna(subset=["avg_damages"])
            if not dmg_data.empty:
                fig_dmg = px.bar(
                    dmg_data.sort_values("avg_damages", ascending=False),
                    x="circuit", y="avg_damages",
                    labels={"circuit": "Circuit", "avg_damages": "Avg Damages ($)"},
                    title="Average Damages by Circuit",
                )
                fig_dmg.update_yaxes(tickprefix="$", tickformat=",")
                st.plotly_chart(fig_dmg, use_container_width=True)

        with col2:
            fees_data = df_circuits.dropna(subset=["avg_fees"])
            if not fees_data.empty:
                fig_fees = px.bar(
                    fees_data.sort_values("avg_fees", ascending=False),
                    x="circuit", y="avg_fees",
                    labels={"circuit": "Circuit", "avg_fees": "Avg Attorney Fees ($)"},
                    title="Average Attorney Fees by Circuit",
                )
                fig_fees.update_yaxes(tickprefix="$", tickformat=",")
                st.plotly_chart(fig_fees, use_container_width=True)

        # Summary table with significance
        st.subheader("Circuit Summary")
        with engine.connect() as conn:
            overall = conn.execute(text(
                "SELECT SUM(CASE WHEN predicted_value = 'plaintiff_win' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), "
                "COUNT(*) "
                "FROM predictions WHERE label_type = 'outcome' AND model_name = 'outcome_logreg_v1'"
            )).fetchone()
        overall_rate = overall[0] or 0.5
        overall_n = overall[1] or 1

        badges = []
        for _, row in df_circuits.iterrows():
            result = compare_outcome_rates(
                row["plaintiff_win_rate"], row["opinion_count"],
                overall_rate, overall_n,
            )
            badges.append(result["badge"])
        df_circuits["sig"] = badges

        display_df = df_circuits.copy()
        display_df["plaintiff_win_rate"] = display_df["plaintiff_win_rate"].apply(lambda x: f"{x:.1%}")
        display_df["avg_damages"] = display_df["avg_damages"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        display_df["avg_fees"] = display_df["avg_fees"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        st.dataframe(
            display_df[["circuit", "opinion_count", "plaintiff_win_rate", "avg_damages", "avg_fees", "sig"]],
            use_container_width=True, hide_index=True,
            column_config={"sig": st.column_config.TextColumn("Sig.", width="small")},
        )

# === Tab 5: Defense Analysis ===
with tab5:
    df_defense_freq = defense_frequency(engine)

    if df_defense_freq.empty:
        st.warning("No defense types found. Run `python ner.py --types defense_type` to extract defenses.")
    else:
        # Bar chart: defense frequency (top 20)
        st.subheader("Defense Type Frequency")
        fig_def = px.bar(
            df_defense_freq.head(20),
            x="count", y="defense_type", orientation="h",
            labels={"count": "Times Raised", "defense_type": "Defense Type"},
            title="Most Common Defenses",
        )
        fig_def.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_def, use_container_width=True)

        # Table: defense effectiveness with significance
        st.subheader("Defense Effectiveness")
        df_eff = defense_effectiveness(engine, min_count=20)
        if df_eff.empty:
            st.info("Not enough data for effectiveness analysis (need 20+ occurrences per defense).")
        else:
            overall_rate = df_eff["overall_defendant_win_rate"].iloc[0]
            badges = []
            for _, row in df_eff.iterrows():
                result = compare_outcome_rates(
                    row["defendant_win_rate_when_raised"], row["times_raised"],
                    overall_rate, 1000,
                )
                badges.append(result["badge"])
            df_eff["sig"] = badges

            display_df = df_eff.copy()
            display_df["defendant_win_rate_when_raised"] = display_df["defendant_win_rate_when_raised"].apply(
                lambda x: f"{x:.1%}"
            )
            display_df["overall_defendant_win_rate"] = display_df["overall_defendant_win_rate"].apply(
                lambda x: f"{x:.1%}"
            )
            st.dataframe(
                display_df[["defense_type", "times_raised", "defendant_win_rate_when_raised",
                            "overall_defendant_win_rate", "sig"]],
                use_container_width=True, hide_index=True,
                column_config={"sig": st.column_config.TextColumn("Sig.", width="small")},
            )

        # Heatmap: defense frequency by circuit
        st.subheader("Defense Frequency by Circuit")
        df_def_circuit = defense_by_circuit(engine)
        if not df_def_circuit.empty:
            # Pivot for heatmap — top 10 defenses
            top_defenses = df_defense_freq.head(10)["defense_type"].tolist()
            df_filtered = df_def_circuit[df_def_circuit["defense_type"].isin(top_defenses)]
            if not df_filtered.empty:
                pivot = df_filtered.pivot_table(
                    index="defense_type", columns="circuit",
                    values="count", fill_value=0,
                )
                fig_heat = px.imshow(
                    pivot, text_auto=True,
                    labels=dict(x="Circuit", y="Defense Type", color="Count"),
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_heat, use_container_width=True)
```

**Step 2: Verify dashboard loads**

Run: `python -m streamlit run pages/6_Trends.py --server.headless true`
Expected: Dashboard loads without errors, all 5 tabs render.

**Step 3: Commit**

```bash
git add pages/6_Trends.py
git commit -m "feat: add 5-tab Trends dashboard"
```

---

### Task 6: Run all tests and final validation

**Files:** None

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (existing + new defense + trends tests)

**Step 2: Verify dashboard integration**

Run: `python -m streamlit run app.py --server.headless true`
Expected: Trends page appears in sidebar navigation and loads correctly.

**Step 3: Commit** (if any fixes needed)

Only commit if fixes were required. Otherwise, no action.
