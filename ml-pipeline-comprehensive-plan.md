# Machine Learning & NLP Pipeline for 30K Federal Opinions — Comprehensive Plan

## Overview

This document covers the full set of ML/NLP projects that can be built on top of a corpus of ~30,000 federal court opinions across FDCPA, TCPA, and FCRA. Each project builds on a shared foundation and can be implemented independently or composed together into a unified litigation intelligence platform.

All processing is local. No data leaves the machine.

---

## Shared Foundation

Every project below depends on a common data layer. Build this first.

### Data Extraction & Normalization

1. Connect to the existing database and extract all opinions with metadata (case name, citation, court, date, statute, judge, docket number, full text).
2. Normalize court names to a consistent taxonomy (e.g., "S.D. Tex." and "Southern District of Texas" → single canonical form). Build a lookup table covering all 94 federal district courts and 13 circuit courts.
3. Normalize date formats to ISO 8601.
4. Tag each opinion with its primary statute (FDCPA/TCPA/FCRA). Some opinions may involve multiple statutes — flag these as multi-statute.
5. Generate a unique opinion ID if one doesn't exist.
6. Store the normalized dataset in a SQLite database as the single source of truth for all downstream pipelines.

### Text Preprocessing

1. Strip headers/footers that come from court filing systems (e.g., PACER headers, page numbers, Westlaw/Lexis artifacts if present).
2. Normalize whitespace, fix encoding issues, remove non-printable characters.
3. Detect and tag opinion sections where possible: caption, procedural history, facts, analysis/discussion, holding, damages/relief. This can start as regex-based heuristics (look for "BACKGROUND", "DISCUSSION", "CONCLUSION", "ORDER" headers) and be refined later with ML.
4. Build a sentence-tokenized version of each opinion using spaCy or NLTK for downstream tasks that need sentence-level granularity.

---

## Project 1: Semantic Search

**See the separate Semantic Search Implementation Plan for full details.**

Summary: Chunk opinions → embed with sentence-transformers → store in ChromaDB → query with natural language → Streamlit UI with metadata filters.

This is the foundational project. Build it first. Every other project benefits from having semantic search available.

---

## Project 2: Text Classification & Outcome Prediction

**Goal:** Train models to predict case outcomes and classify opinions by claim type, defense, and procedural posture.

### 2a. Outcome Classification

#### Data Labeling

This is the hardest part. You need labeled outcomes for a training set.

1. **Automated labeling (first pass):** Use regex and keyword patterns to extract outcomes from opinion text:
   - Plaintiff win signals: "judgment for plaintiff", "defendant is liable", "plaintiff is entitled to", "damages are awarded", "motion for summary judgment is granted" (when plaintiff moves)
   - Defendant win signals: "defendant's motion for summary judgment is granted", "complaint is dismissed", "judgment for defendant", "plaintiff fails to state a claim"
   - Partial win signals: "granted in part and denied in part"
   - Settlement/other: "stipulation of dismissal", "voluntary dismissal"
2. **LLM-assisted labeling (second pass):** For opinions where regex is ambiguous, use a local LLM (Llama 3, Mistral, or similar via Ollama) to read the conclusion/holding section and classify the outcome. Prompt template:
   ```
   Read the following court opinion conclusion. Classify the outcome as one of:
   - PLAINTIFF_WIN (plaintiff prevails on primary claims)
   - DEFENDANT_WIN (defendant prevails, case dismissed or summary judgment for defendant)
   - MIXED (split decision, partial win for each side)
   - PROCEDURAL (ruling on motion that doesn't resolve the case)
   - SETTLEMENT_DISMISSAL (case resolved by agreement)
   - UNCLEAR
   
   Conclusion text: {text}
   ```
3. **Manual review:** Spot-check 200-300 labels across all categories for accuracy. Correct errors and use corrections to refine the automated/LLM labeling.
4. **Target:** Label at least 5,000 opinions with reliable outcomes for training. More is better.

#### Feature Engineering

Two approaches, use both and compare:

1. **TF-IDF + traditional ML:**
   - Extract TF-IDF features from opinion text (or just the facts/analysis sections if section detection works).
   - Add metadata features: court, circuit, statute, year, judge (one-hot or label encoded).
   - Train logistic regression, random forest, and gradient boosting (XGBoost) classifiers.
   - This is fast, interpretable, and serves as a strong baseline.

2. **Fine-tuned transformer:**
   - Fine-tune `legal-bert-base-uncased` or `all-MiniLM-L6-v2` on the labeled outcome data.
   - Input: truncated opinion text (first 512 tokens, or facts + analysis sections).
   - Output: outcome class.
   - Use Hugging Face `Trainer` with class weights to handle imbalanced outcomes.

#### Evaluation

- Train/validation/test split: 70/15/15, stratified by outcome class.
- Metrics: accuracy, F1 per class, confusion matrix.
- **Practical threshold:** If the model achieves >70% accuracy on the test set, it's useful for case assessment. Even 60% with well-calibrated confidence scores is valuable — you're not replacing judgment, you're prioritizing review.

#### Output

A function: `predict_outcome(opinion_text, metadata) → {outcome: str, confidence: float, similar_cases: list}`

### 2b. Claim Type Classification

#### Taxonomy

Build a taxonomy of claim types for each statute. For FDCPA, this includes:

- Harassment/abuse (§1692d): repeated calls, threats of violence, obscene language
- False/misleading representations (§1692e): misrepresenting debt amount, false threats of legal action, impersonating attorneys
- Unfair practices (§1692f): collecting unauthorized fees, depositing post-dated checks early
- Validation of debt violations (§1692g): failure to provide written notice, continuing collection during dispute period
- Communication violations (§1692c): calling at inconvenient times, contacting represented consumers, third-party disclosure
- Initial communication requirements (§1692e(11)): failure to disclose collector identity

Build similar taxonomies for TCPA (autodialer, prerecorded voice, Do Not Call) and FCRA (inaccurate reporting, failure to investigate, permissible purpose).

#### Labeling

1. Use keyword/regex matching on statutory section citations within opinions to auto-label claim types. Opinions that cite §1692d are likely harassment claims, etc.
2. Many opinions involve multiple claims — use multi-label classification (an opinion can be tagged with multiple claim types).
3. LLM-assisted labeling for opinions where section citations are ambiguous or absent.

#### Model

- Multi-label classifier using fine-tuned BERT or a simpler TF-IDF + multi-label logistic regression.
- Evaluate with per-label F1 and Hamming loss.

### 2c. Defense Classification

Same approach as claim types, but for defenses raised:

- Bona fide error defense
- Statute of limitations
- Standing/real party in interest
- Not a "debt collector" under the statute
- Consent (for TCPA)
- Reasonableness of procedures (for FCRA)

Extract from defendant's arguments section of opinions. This is harder because defenses are discussed in more varied language than statutory citations.

---

## Project 3: Named Entity Recognition (NER)

**Goal:** Extract structured data from every opinion automatically.

### Entity Types

| Entity | Description | Example |
|--------|-------------|---------|
| JUDGE | Presiding judge | "Judge Sarah Hughes" |
| PLAINTIFF | Plaintiff name(s) | "John Smith" |
| DEFENDANT | Defendant name(s) | "ABC Collections, LLC" |
| ORIGINAL_CREDITOR | Original creditor | "Chase Bank" |
| DEBT_TYPE | Type of debt at issue | "credit card debt", "medical debt" |
| DOLLAR_AMOUNT | Any dollar figure | "$1,000 statutory damages" |
| STATUTORY_SECTION | Statute section cited | "15 U.S.C. §1692e(5)" |
| DATE | Relevant dates | "March 15, 2020" |
| CASE_CITATION | Cases cited in the opinion | "Jerman v. Carlisle, 559 U.S. 573" |
| DAMAGES_AWARDED | Specific damages amount | "$2,500 in actual damages" |
| ATTORNEY_FEES | Attorney fee award | "$15,000 in attorney's fees" |

### Approach

1. **Rule-based extraction (Phase 1):**
   - Dollar amounts: regex for `\$[\d,]+(\.\d{2})?` with context window to classify as damages, fees, debt amount, etc.
   - Statutory sections: regex for `§\s*\d+[a-z]*(\(\d+\))*` and `15 U.S.C. §...` patterns.
   - Case citations: regex for standard citation formats (`\d+ F\.\d+ \d+`, `\d+ U\.S\. \d+`, `\d+ F\. Supp\. \d+ \d+`, etc.).
   - Judge names: extract from opinion header/caption or from "before [Judge Name]" patterns.

2. **spaCy NER model (Phase 2):**
   - Start with spaCy's `en_core_web_trf` (transformer-based) model for base NER (person names, organizations, dates, money).
   - Train a custom NER model for legal-specific entities (STATUTORY_SECTION, CASE_CITATION, DAMAGES_AWARDED vs. general DOLLAR_AMOUNT).
   - Training data: manually annotate 200-300 opinions using Prodigy or Label Studio, then train spaCy NER on top.
   - Alternatively, use few-shot prompting with a local LLM to generate training annotations at scale, then manually verify a subset.

3. **Structured output storage:**
   - Create a `case_entities` table in the SQLite database:
     ```
     opinion_id | entity_type | entity_value | context_snippet | start_char | end_char
     ```
   - Create derived tables: `damages_awards`, `attorney_fees`, `citations_graph` for easy querying.

### Practical Value

Once NER is complete, you can answer questions like:
- "What's the average damages award in FDCPA cases in the 5th Circuit?"
- "Which debt collectors are most frequently sued?"
- "Which judges award the highest attorney's fees?"
- "What's the typical debt amount in TCPA cases?"

---

## Project 4: Topic Modeling

**Goal:** Discover natural clusters and themes across the corpus without manual labeling.

### Approach

Use BERTopic, which combines transformer embeddings with clustering:

```
pip install bertopic
```

### Pipeline

1. **Embed opinions:** Reuse the embeddings from the semantic search pipeline (all-MiniLM-L6-v2). Use one embedding per opinion — either the mean of all chunk embeddings or the embedding of the first 512 tokens.
2. **Dimensionality reduction:** BERTopic uses UMAP to reduce 384-dim vectors to 5-dim for clustering.
3. **Clustering:** HDBSCAN finds natural clusters without specifying the number of topics.
4. **Topic representation:** BERTopic uses c-TF-IDF to extract representative words for each cluster, then (optionally) an LLM to generate human-readable topic labels.

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_model = BERTopic(
    embedding_model=embedding_model,
    nr_topics="auto",       # let HDBSCAN decide
    min_topic_size=30,       # minimum opinions per topic
    verbose=True
)

topics, probs = topic_model.fit_transform(opinion_texts, embeddings)
```

### Analysis

1. **Topic overview:** `topic_model.get_topic_info()` — see all discovered topics with representative words and sizes.
2. **Topic evolution over time:** `topic_model.topics_over_time(opinion_texts, timestamps)` — how topics rise and fall across decades.
3. **Per-statute topics:** Run topic modeling separately for FDCPA, TCPA, and FCRA to find statute-specific themes.
4. **Topic-court heatmap:** Cross-tabulate topics with courts/circuits to see regional patterns.
5. **Outlier detection:** Opinions that don't fit any topic well may be novel legal theories or unusual fact patterns — worth flagging for manual review.

### Visualization

BERTopic has built-in Plotly visualizations:
- `topic_model.visualize_topics()` — interactive 2D topic map
- `topic_model.visualize_barchart()` — top words per topic
- `topic_model.visualize_heatmap()` — topic similarity matrix
- `topic_model.visualize_topics_over_time()` — trend lines

Embed these in the Streamlit app as a "Topics" tab.

---

## Project 5: Citation Network Analysis

**Goal:** Map how opinions cite each other to find influential cases, emerging trends, and circuit splits.

### Extraction

1. Parse all case citations from every opinion (reuse the regex NER from Project 3).
2. Resolve citations to opinion IDs where possible (match citations against the 30K opinions in the corpus). Not all cited cases will be in the corpus — that's fine; track external citations separately.
3. Build an edge list: `citing_opinion_id → cited_opinion_id` (or `cited_citation` for external cases).

### Graph Construction

```python
import networkx as nx

G = nx.DiGraph()
for edge in citation_edges:
    G.add_edge(edge['citing_id'], edge['cited_id'])
```

### Analysis

1. **Most-cited opinions (in-degree):** The cases other courts rely on most. These are the foundational precedents for each statute.
2. **PageRank:** Like Google's algorithm — opinions cited by other highly-cited opinions rank higher. This finds cases that are influential in the network sense, not just frequently cited.
3. **Hub and authority scores (HITS):** Hubs are opinions that cite many important cases (good survey opinions). Authorities are opinions that are cited by many hubs (foundational law).
4. **Community detection:** Use Louvain or Girvan-Newman to find clusters of opinions that cite each other heavily. These likely represent distinct legal doctrines or issue areas.
5. **Circuit split detection:** Look for cases where Circuit A cites a proposition and Circuit B cites a contradictory proposition, especially when both cite the same upstream authority. This is sophisticated but potentially very high-value.
6. **Temporal citation analysis:** Track which opinions gain citations over time. A case that's gaining citations rapidly is an emerging precedent. A case that stops being cited may have been superseded.
7. **Depth-of-reliance score:** For any given opinion, trace its citation chain backward — how many layers of authority support its holding? Deeply supported holdings are more durable.

### Visualization

- Interactive network graph (use pyvis or Gephi for the full network, d3.js in the Streamlit app for subsets).
- Citation timeline for individual cases.
- Circuit-level citation flow (Sankey diagram: which circuits cite which other circuits).

### Storage

- `citations` table: `citing_opinion_id, cited_opinion_id, cited_citation_string, context_snippet`
- `opinion_metrics` table: `opinion_id, in_degree, pagerank, hub_score, authority_score, community_id`

---

## Project 6: Trend Analysis & Judicial Analytics

**Goal:** Track how outcomes, damages, and legal reasoning change over time, by court, and by judge.

### Metrics to Track

This project depends on structured data from Projects 2 and 3 (classification labels and NER extractions).

1. **Outcome rates over time:**
   - Plaintiff win rate by year, by statute, by circuit.
   - How did specific Supreme Court decisions (e.g., Henson v. Santander, Jerman v. Carlisle) change win rates?

2. **Damages trends:**
   - Median and mean statutory damages, actual damages, and attorney's fees by year, circuit, and statute.
   - Distribution of damages awards (histogram).
   - Are courts awarding more or less over time?

3. **Claim type frequency:**
   - Which FDCPA violations are most commonly alleged? Is this shifting?
   - Which claim types have the highest success rates?

4. **Judicial analytics:**
   - Per-judge outcome rates (plaintiff win % for each judge with >10 opinions in the corpus).
   - Per-judge average damages awards.
   - Per-judge average time to decision (if filing date is available).
   - Caveat: present these as descriptive statistics, not predictive claims about individual judges. Note sample sizes.

5. **Defense effectiveness:**
   - Which defenses succeed most often?
   - Does defense effectiveness vary by circuit?

6. **Circuit analysis:**
   - Circuit-by-circuit comparison of all metrics above.
   - Identify plaintiff-friendly vs. defendant-friendly circuits per statute.

### Implementation

1. Aggregate the structured data from Projects 2 and 3 into an analytics-ready format (likely a pandas DataFrame or a set of SQL views).
2. Build charts and dashboards in the Streamlit app:
   - Time-series line charts for outcome rates, damages, claim frequency.
   - Bar charts for circuit and judge comparisons.
   - Heatmaps for claim type × circuit success rates.
3. Add statistical significance testing where appropriate (chi-squared for outcome rate differences, Mann-Whitney U for damages comparisons).
4. Export capabilities: let the user download any chart or the underlying data as CSV.

### Visualization Tab

Add a "Trends & Analytics" tab to the Streamlit app with:
- Date range selector
- Statute filter
- Circuit/court filter
- Toggle between chart types
- Data table below each chart for the underlying numbers

---

## Project 7: Outcome Prediction Model (Advanced)

**Goal:** Given a new fact pattern, predict the likely outcome, damages range, and identify the most relevant precedent.

This is distinct from Project 2's classification (which labels existing opinions). This is a forward-looking tool for case evaluation.

### Architecture

1. **Input:** A summary of the facts and claims for a new case (written by the attorney).
2. **Semantic similarity:** Find the 20 most similar opinions using the semantic search engine (Project 1).
3. **Feature extraction:** From the similar cases, extract:
   - Outcome distribution (what % plaintiff wins among similar cases)
   - Damages distribution (median, 25th percentile, 75th percentile)
   - Most common claim types and defenses
   - Jurisdictional breakdown
4. **Prediction:** Combine similarity-weighted outcomes into a prediction:
   ```
   predicted_outcome = weighted_average(similar_case_outcomes, weights=similarity_scores)
   predicted_damages = weighted_median(similar_case_damages, weights=similarity_scores)
   ```
5. **Output:**
   - Predicted outcome with confidence interval
   - Predicted damages range (25th–75th percentile of similar cases)
   - Top 10 most relevant precedent cases with links
   - Key risk factors identified from losing cases with similar facts
   - Recommended claims to assert based on success rates

### Important Caveats

- This is a research and prioritization tool, not legal advice.
- Always show the underlying similar cases so the attorney can evaluate the comparison.
- Display confidence metrics and sample sizes prominently.
- Flag when the query falls outside the distribution of training data (novel fact patterns).

---

## Integration: Unified Litigation Intelligence Platform

All seven projects feed into a single Streamlit application with these tabs:

| Tab | Source Projects | Function |
|-----|----------------|----------|
| **Search** | Project 1 | Natural-language search with filters |
| **Case Explorer** | Projects 3, 5 | View any opinion with extracted entities, citation graph, related cases |
| **Topics** | Project 4 | Browse topic clusters, explore topic evolution |
| **Analytics** | Projects 2, 3, 6 | Charts, trends, judicial analytics, circuit comparisons |
| **Case Evaluator** | Project 7 | Input facts → get outcome prediction with supporting cases |
| **Citation Map** | Project 5 | Interactive citation network, influential case finder |

### Database Schema (SQLite)

```
opinions
├── opinion_id (PK)
├── case_name
├── citation
├── court
├── circuit
├── judge
├── date_decided
├── statute (FDCPA, TCPA, FCRA)
├── full_text
├── outcome_label (from Project 2)
├── claim_types (JSON array, from Project 2b)
├── defenses (JSON array, from Project 2c)
├── topic_id (from Project 4)
├── pagerank (from Project 5)
└── community_id (from Project 5)

case_entities
├── id (PK)
├── opinion_id (FK)
├── entity_type
├── entity_value
├── context_snippet
├── start_char
└── end_char

citations
├── id (PK)
├── citing_opinion_id (FK)
├── cited_opinion_id (FK, nullable)
├── cited_citation_string
└── context_snippet

chunks (for semantic search)
├── chunk_id (PK)
├── opinion_id (FK)
├── chunk_index
├── text
└── embedding (stored in ChromaDB, not SQLite)

damages_awards
├── id (PK)
├── opinion_id (FK)
├── award_type (statutory, actual, punitive, attorney_fees, costs)
├── amount
└── context_snippet
```

---

## Implementation Order

| Priority | Project | Dependencies | Estimated Effort | Value |
|----------|---------|-------------|-----------------|-------|
| 1 | Shared Foundation | None | 1-2 days | Required for everything |
| 2 | Semantic Search (Project 1) | Foundation | 1-2 days | Immediate practical use |
| 3 | NER Extraction (Project 3) | Foundation | 2-3 days | Unlocks analytics |
| 4 | Citation Network (Project 5) | NER (for citation extraction) | 2-3 days | High research value |
| 5 | Outcome Classification (Project 2a) | Foundation + NER | 3-5 days | Case assessment tool |
| 6 | Trend Analysis (Project 6) | NER + Outcome Classification | 2-3 days | Strategic insights |
| 7 | Topic Modeling (Project 4) | Semantic Search embeddings | 1 day | Discovery tool |
| 8 | Claim/Defense Classification (2b, 2c) | Outcome Classification | 2-3 days | Refined analytics |
| 9 | Outcome Prediction (Project 7) | All above | 3-5 days | Case evaluation tool |

**Total estimated effort: 4-6 weeks of part-time work, or 2-3 weeks focused.**

---

## Hardware & Dependencies

### Minimum Requirements
- Python 3.10+
- 16GB RAM (some steps can work with 8GB but 16GB avoids swapping)
- 10GB free disk space
- CPU is sufficient for everything; GPU accelerates embedding and fine-tuning

### Key Python Packages
```
# Core
sentence-transformers>=2.2.0
chromadb>=0.4.0
streamlit>=1.28.0
nltk>=3.8
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# NER
spacy>=3.7.0

# Topic Modeling
bertopic>=0.15.0
hdbscan>=0.8.33
umap-learn>=0.5.4

# Citation Network
networkx>=3.1
pyvis>=0.3.2

# Classification (if fine-tuning transformers)
transformers>=4.35.0
torch>=2.1.0

# Visualization
plotly>=5.18.0
matplotlib>=3.8.0

# Optional
rank-bm25>=0.2.2          # hybrid search
xgboost>=2.0.0            # gradient boosting classifier
```

---

## Notes for Claude Code

- Start with the Shared Foundation and Semantic Search. Get those working end-to-end before branching into other projects.
- Each project should be a separate Python module that imports from a shared `db.py` and `config.py`.
- The user's database schema will determine how Phase 1 extraction works — ask for details before writing code.
- The user values data privacy — everything runs locally, no external API calls for embeddings or inference.
- Keep the Streamlit app modular — each tab should be a separate file imported by the main `app.py`.
- The corpus spans 1977 to present across three statutes and all federal courts. Date and jurisdiction filtering should be prominent throughout the UI.
- The user may want to add opinions over time as new ones are published via the GPO API. Design all pipelines to support incremental updates, not just one-time batch processing.
- Store intermediate results (embeddings checkpoint, NER extractions, classification labels) so reprocessing isn't required if a downstream step needs adjustment.
