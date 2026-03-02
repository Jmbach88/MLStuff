# Trend Analysis & Judicial Analytics — Design Document

## Goal

Add time-series trend analysis, judicial analytics, circuit comparisons, defense extraction/effectiveness, and statistical significance testing. New dedicated Trends dashboard page with 5 tabs.

## Components

### 1. Defense Extraction

Add `DEFENSE_TYPE` entity to existing `ner.py` using keyword/regex matching. ~20 common defense patterns:

- bona fide error
- statute of limitations
- standing
- Rooker-Feldman
- res judicata / claim preclusion
- collateral estoppel / issue preclusion
- preemption
- sovereign immunity
- qualified immunity
- failure to state a claim
- mootness
- ripeness
- abstention
- arbitration
- good faith
- fair use
- consent (TCPA)
- established business relationship
- prior express consent

Store as `entity_type = "DEFENSE_TYPE"` in existing `entities` table. Deduplicate per opinion. Search full `plain_text`.

### 2. Trends Module (`trends.py`)

Pure query/aggregation layer. No new DB tables. Returns pandas DataFrames.

**Time-series functions:**
- `outcome_trends_by_year(engine, statute, circuit)` — Plaintiff win rate by year with counts
- `damages_trends_by_year(engine, statute, circuit)` — Mean/median DAMAGES_AWARDED and ATTORNEY_FEES by year
- `claim_frequency_by_year(engine, statute)` — Top 10 claim sections by year

**Judicial analytics:**
- `judge_stats(engine, min_opinions=10)` — Per-judge: count, plaintiff win %, avg damages, avg fees. Filter ≥10 opinions.

**Circuit comparison:**
- `circuit_comparison(engine)` — Per-circuit: count, plaintiff win %, avg damages, avg fees, top claims
- `circuit_outcome_heatmap(engine)` — Matrix: plaintiff win rate by circuit × statute

**Defense analysis:**
- `defense_frequency(engine)` — Defense type counts
- `defense_effectiveness(engine, min_count=20)` — Per defense: times raised, defendant win rate when raised vs. overall
- `defense_by_circuit(engine)` — Defense frequency and effectiveness by circuit

**Statistical testing (scipy.stats):**
- `compare_outcome_rates(rate1, n1, rate2, n2)` — Chi-squared, returns p-value + significance (*, **, ***)
- `compare_damages(group1, group2)` — Mann-Whitney U for damages distributions

### 3. Trends Dashboard (`pages/6_Trends.py`)

**Sidebar filters:** Statute, Circuit, Court Type, Date Range (year slider).

**Tab 1 — Outcome Trends:**
- Line chart: plaintiff win rate by year
- Circuit vs. overall average comparison

**Tab 2 — Damages Trends:**
- Line chart: median damages by year
- Line chart: median attorney fees by year
- Damages distribution histogram

**Tab 3 — Judicial Analytics:**
- Sortable table: judges ≥10 opinions, win %, avg damages, avg fees
- Significance badges vs. corpus average

**Tab 4 — Circuit Comparison:**
- Heatmap: plaintiff win rate by circuit × statute
- Bar charts: avg damages and fees by circuit with significance badges

**Tab 5 — Defense Analysis:**
- Bar chart: defense frequency (top 20)
- Table: defense effectiveness with significance
- Heatmap: defense frequency by circuit

## New/Modified Files

| File | Change |
|------|--------|
| `ner.py` | Add DEFENSE_TYPE patterns |
| `tests/test_ner.py` | Add defense extraction tests |
| `trends.py` | New: query/aggregation functions |
| `tests/test_trends.py` | New: unit tests for all trend functions |
| `pages/6_Trends.py` | New: 5-tab trends dashboard |
| `requirements.txt` | Add `scipy>=1.10.0` |

## Dependencies

- `scipy>=1.10.0` (chi-squared, Mann-Whitney U)

## Expected Runtime

- Defense extraction: ~5 minutes (regex on 30K texts)
- Trend queries: seconds (SQL aggregations)
- Dashboard load: seconds (cached queries)
