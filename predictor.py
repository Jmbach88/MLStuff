"""
Case Evaluation Predictor: predict outcome, damages, and recommendations
from similar precedents.

Usage:
    from predictor import evaluate_case
    result = evaluate_case(engine, index, chunk_map, "case summary text")
"""
import re
import logging
from collections import Counter

import numpy as np
from sqlalchemy import text

from search import search_opinions

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


def _weighted_percentiles(values, weights, percentiles):
    """Compute weighted percentiles.

    Args:
        values: list of numeric values
        weights: list of corresponding weights (same length)
        percentiles: list of percentiles to compute (0-100)

    Returns list of percentile values.
    """
    if not values or not weights:
        return [None] * len(percentiles)

    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)

    # Sort by values
    sorted_idx = np.argsort(values)
    values = values[sorted_idx]
    weights = weights[sorted_idx]

    # Cumulative weight
    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]
    if total_weight == 0:
        return [None] * len(percentiles)

    cum_pct = cum_weights / total_weight * 100

    results = []
    for p in percentiles:
        idx = np.searchsorted(cum_pct, p)
        idx = min(idx, len(values) - 1)
        results.append(float(values[idx]))
    return results


def evaluate_case(engine, index, chunk_map, query_text,
                  statute=None, circuit=None, court_type=None, top_k=50):
    """Evaluate a case by finding similar precedents and computing predictions.

    Args:
        engine: SQLAlchemy engine
        index: FAISS index
        chunk_map: chunk metadata list
        query_text: case summary text
        statute: optional statute filter (FDCPA, TCPA, FCRA)
        circuit: optional circuit filter
        court_type: optional court type filter
        top_k: number of similar cases to retrieve

    Returns dict with:
        predicted_outcome, confidence, damages_median, damages_25th,
        damages_75th, fees_median, fees_25th, fees_75th,
        similar_cases, risk_factors, claim_recommendations, case_count
    """
    empty_result = {
        "predicted_outcome": None,
        "confidence": 0.0,
        "damages_median": None,
        "damages_25th": None,
        "damages_75th": None,
        "fees_median": None,
        "fees_25th": None,
        "fees_75th": None,
        "similar_cases": [],
        "risk_factors": [],
        "claim_recommendations": [],
        "case_count": 0,
    }

    if not query_text or not query_text.strip():
        return empty_result

    # Build filters
    filters = {}
    if statute:
        filters["statute"] = statute.upper()
    if circuit:
        filters["circuit"] = circuit
    if court_type:
        filters["court_type"] = court_type

    # Search for similar opinions
    similar = search_opinions(index, chunk_map, query_text,
                              top_k=top_k, filters=filters or None)

    if not similar:
        return empty_result

    opinion_ids = [s["opinion_id"] for s in similar]
    scores = {s["opinion_id"]: s["similarity_score"] for s in similar}

    # Load outcome predictions for similar opinions
    id_list = ",".join(str(oid) for oid in opinion_ids)
    with engine.connect() as conn:
        outcome_rows = conn.execute(text(
            f"SELECT opinion_id, predicted_value, confidence "
            f"FROM predictions "
            f"WHERE opinion_id IN ({id_list}) "
            f"AND label_type = 'outcome' AND model_name = 'outcome_logreg_v1'"
        )).fetchall()

        entity_rows = conn.execute(text(
            f"SELECT opinion_id, entity_type, entity_value "
            f"FROM entities "
            f"WHERE opinion_id IN ({id_list}) "
            f"AND entity_type IN ('DAMAGES_AWARDED', 'ATTORNEY_FEES', 'DEFENSE_TYPE')"
        )).fetchall()

        claim_rows = conn.execute(text(
            f"SELECT opinion_id, predicted_value "
            f"FROM predictions "
            f"WHERE opinion_id IN ({id_list}) "
            f"AND label_type = 'claim_type' AND model_name = 'claim_type_logreg_v1'"
        )).fetchall()

    # Build lookup maps
    outcomes = {r[0]: r[1] for r in outcome_rows}
    confidences = {r[0]: r[2] for r in outcome_rows}

    damages = {}
    fees = {}
    defenses = {}
    for oid, etype, evalue in entity_rows:
        if etype == "DAMAGES_AWARDED":
            amount = _parse_dollar(evalue)
            if amount is not None and amount > 0:
                damages[oid] = amount
        elif etype == "ATTORNEY_FEES":
            amount = _parse_dollar(evalue)
            if amount is not None and amount > 0:
                fees[oid] = amount
        elif etype == "DEFENSE_TYPE":
            defenses.setdefault(oid, []).append(evalue)

    claims = {}
    for oid, section in claim_rows:
        claims.setdefault(oid, []).append(section)

    # --- Weighted outcome prediction ---
    outcome_weights = {}  # outcome_value -> total weight
    for oid in opinion_ids:
        if oid in outcomes:
            outcome = outcomes[oid]
            weight = scores[oid]
            outcome_weights[outcome] = outcome_weights.get(outcome, 0) + weight

    if outcome_weights:
        total_weight = sum(outcome_weights.values())
        predicted_outcome = max(outcome_weights, key=outcome_weights.get)
        confidence = outcome_weights[predicted_outcome] / total_weight
    else:
        predicted_outcome = None
        confidence = 0.0

    # --- Weighted damages/fees estimation ---
    dmg_values = []
    dmg_weights = []
    for oid in opinion_ids:
        if oid in damages:
            dmg_values.append(damages[oid])
            dmg_weights.append(scores[oid])

    damages_25, damages_50, damages_75 = _weighted_percentiles(
        dmg_values, dmg_weights, [25, 50, 75]
    )

    fee_values = []
    fee_weights = []
    for oid in opinion_ids:
        if oid in fees:
            fee_values.append(fees[oid])
            fee_weights.append(scores[oid])

    fees_25, fees_50, fees_75 = _weighted_percentiles(
        fee_values, fee_weights, [25, 50, 75]
    )

    # --- Risk factors: defenses in defendant-win cases ---
    defense_in_losses = Counter()
    defense_total = Counter()
    for oid in opinion_ids:
        if oid in defenses:
            for d in defenses[oid]:
                defense_total[d] += 1
                if outcomes.get(oid) == "defendant_win":
                    defense_in_losses[d] += 1

    risk_factors = []
    for defense, loss_count in defense_in_losses.most_common():
        risk_factors.append({
            "defense_type": defense,
            "count": defense_total[defense],
            "loss_count": loss_count,
            "loss_rate": loss_count / defense_total[defense] if defense_total[defense] > 0 else 0,
        })

    # --- Claim recommendations: sections ranked by plaintiff win rate ---
    claim_stats = {}
    for oid in opinion_ids:
        if oid in claims:
            outcome = outcomes.get(oid)
            for section in claims[oid]:
                if section not in claim_stats:
                    claim_stats[section] = {"wins": 0, "total": 0, "damages_sum": 0, "damages_count": 0}
                claim_stats[section]["total"] += 1
                if outcome == "plaintiff_win":
                    claim_stats[section]["wins"] += 1
                if oid in damages:
                    claim_stats[section]["damages_sum"] += damages[oid]
                    claim_stats[section]["damages_count"] += 1

    claim_recommendations = []
    for section, stats in claim_stats.items():
        if stats["total"] >= 2:  # minimum occurrences
            avg_damages = (stats["damages_sum"] / stats["damages_count"]
                          if stats["damages_count"] > 0 else None)
            claim_recommendations.append({
                "claim_section": section,
                "count": stats["total"],
                "plaintiff_win_rate": stats["wins"] / stats["total"],
                "avg_damages": avg_damages,
            })

    claim_recommendations.sort(key=lambda x: x["plaintiff_win_rate"], reverse=True)

    # --- Build similar cases list ---
    similar_cases = []
    for s in similar:
        oid = s["opinion_id"]
        similar_cases.append({
            "opinion_id": oid,
            "title": s["title"],
            "circuit": s["circuit"],
            "court_type": s["court_type"],
            "date_issued": s["date_issued"],
            "similarity_score": s["similarity_score"],
            "predicted_outcome": outcomes.get(oid, "unknown"),
            "damages": damages.get(oid),
            "attorney_fees": fees.get(oid),
        })

    return {
        "predicted_outcome": predicted_outcome,
        "confidence": float(confidence),
        "damages_median": damages_50,
        "damages_25th": damages_25,
        "damages_75th": damages_75,
        "fees_median": fees_50,
        "fees_25th": fees_25,
        "fees_75th": fees_75,
        "similar_cases": similar_cases,
        "risk_factors": risk_factors,
        "claim_recommendations": claim_recommendations,
        "case_count": len(similar),
    }
