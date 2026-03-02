# Advanced Outcome Prediction — Design Document

## Goal

Build a case evaluation tool where an attorney inputs a case summary and receives predicted outcome, damages range, similar precedents, risk factors, and claim recommendations. Retrieval-augmented prediction using similarity-weighted averaging over existing FAISS index and entity data.

## Architecture

1. **Input:** Attorney enters case summary (free text) plus optional filters (statute, circuit, court type)
2. **Retrieval:** Embed summary with all-MiniLM-L6-v2, search FAISS for top 50 similar opinions
3. **Aggregation:** Compute weighted predictions using similarity scores as weights
4. **Analysis:** Extract risk factors and claim recommendations from similar case pool
5. **Output:** Structured results on dedicated dashboard page

No new DB tables. No model training. Queries existing `predictions`, `entities`, `opinion_statutes` tables.

## Prediction Engine (`predictor.py`)

**Core function:** `evaluate_case(engine, query_text, statute=None, circuit=None, top_k=50)`

**Flow:**

1. **Embed query** — sentence-transformers encode to 384-dim vector
2. **Search FAISS** — Find top_k similar chunks, group by opinion_id, keep best similarity score per opinion
3. **Apply filters** — Filter to matching statute/circuit. Oversample 3x to ensure enough results after filtering
4. **Load opinion data** — For each similar opinion fetch: outcome prediction, DAMAGES_AWARDED, ATTORNEY_FEES, DEFENSE_TYPE entities, claim type predictions, metadata (circuit, court_type, date_issued, judge)
5. **Compute weighted predictions:**
   - Outcome: weighted vote across plaintiff_win/defendant_win/mixed, confidence = winning proportion
   - Damages: weighted median, 25th/75th percentiles
   - Attorney fees: same weighted percentile approach
6. **Extract risk factors** — Defense types appearing disproportionately in defendant-win similar cases
7. **Recommend claims** — Claim sections ranked by plaintiff win rate in similar cases (min 3 occurrences)

**Returns:** Dict with predicted_outcome, confidence, damages_median, damages_25th, damages_75th, fees_median, fees_25th, fees_75th, similar_cases (list with scores), risk_factors, claim_recommendations.

## Dashboard (`pages/7_Predictor.py`)

**Input section (top):**
- Large text area for case summary
- Sidebar filters: Statute, Circuit, Court Type
- "Evaluate Case" button

**Summary metrics row:**
- Predicted Outcome with confidence %
- Predicted Damages (median + 25th-75th range)
- Predicted Attorney Fees (median + 25th-75th range)
- Number of similar cases found

**Tab 1 — Similar Precedents:**
- Sortable table: title, circuit, date, outcome, damages, similarity score
- Top 20 most similar cases

**Tab 2 — Risk Factors:**
- Defense types common in losing similar cases with frequency counts
- Bar chart: defense prevalence in wins vs losses

**Tab 3 — Claim Recommendations:**
- Table: claim section, times seen, plaintiff win rate, avg damages
- Sorted by plaintiff win rate (min 3 occurrences)

**Disclaimer banner:** "For research and prioritization only. Not legal advice. Always review underlying precedents."

## Testing

Unit tests in `tests/test_predictor.py` using in-memory SQLite DB + small synthetic FAISS index.

- TestWeightedOutcome (3-4 tests): all plaintiff wins, all defendant wins, mixed with varying weights
- TestDamagesEstimation (3 tests): uniform weights, skewed weights, no damages
- TestRiskFactors (2 tests): disproportionate defense ranking, empty defenses
- TestClaimRecommendations (2 tests): sorted by win rate, min occurrence filter
- TestEndToEnd (1 test): full evaluate_case returns all expected keys
- TestEdgeCases (2 tests): empty query, no similar cases

~13-15 tests total.

## New/Modified Files

| File | Change |
|------|--------|
| `predictor.py` | New: prediction engine |
| `tests/test_predictor.py` | New: unit tests |
| `pages/7_Predictor.py` | New: case evaluation dashboard |

## Dependencies

None new. Uses existing sentence-transformers, faiss-cpu, numpy, pandas, streamlit, plotly.
