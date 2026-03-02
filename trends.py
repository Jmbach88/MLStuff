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
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
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
