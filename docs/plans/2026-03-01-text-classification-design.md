# Project 2: Text Classification — Design Document

## Goal

Classify ~30K court opinions by **outcome** (plaintiff win / defendant win / mixed) and **claim types** (statutory sections cited), using TF-IDF + traditional ML classifiers. Designed for future upgrade to transformer fine-tuning.

## Scope

- **2a: Outcome Classification** — 3-class single-label (plaintiff win, defendant win, mixed)
- **2b: Claim Type Classification** — Multi-label (statutory sections cited per opinion)
- All three statutes: FDCPA (§1692), TCPA (§227), FCRA (§1681)

## Architecture

```
                    ┌─────────────────┐
                    │  Label Pipeline  │
                    │  (regex + LLM)  │
                    └────────┬────────┘
                             │ labeled opinions
                    ┌────────▼────────┐
                    │   Train/Eval    │
                    │  TF-IDF + ML    │
                    └────────┬────────┘
                             │ trained models
                    ┌────────▼────────┐
                    │   Predict All   │
                    │  (batch infer)  │
                    └────────┬────────┘
                             │ predictions → DB
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼──────┐ ┌─────▼─────┐
     │  Search UI  │  │ Analytics   │ │  FAISS    │
     │  (filters)  │  │ (charts)    │ │ (metadata)│
     └─────────────┘  └─────────────┘ └───────────┘
```

## New Database Tables

### `labels`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| opinion_id | INTEGER FK | References opinions.id |
| label_type | TEXT | "outcome" or "claim_type" |
| label_value | TEXT | e.g., "plaintiff_win", "§1692e" |
| source | TEXT | "regex", "llm", or "manual" |
| confidence | REAL | 0.0-1.0 |

### `predictions`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| opinion_id | INTEGER FK | References opinions.id |
| model_name | TEXT | e.g., "outcome_logreg_v1" |
| label_type | TEXT | "outcome" or "claim_type" |
| predicted_value | TEXT | Predicted label |
| confidence | REAL | Model confidence |
| created_at | TEXT | ISO timestamp |

### `models`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| name | TEXT UNIQUE | Model identifier |
| label_type | TEXT | "outcome" or "claim_type" |
| accuracy | REAL | Overall accuracy |
| f1_score | REAL | Macro F1 |
| trained_at | TEXT | ISO timestamp |
| params_json | TEXT | Hyperparameters as JSON |

## New Files

| File | Purpose |
|------|---------|
| `label.py` | Regex + optional LLM labeling pipeline |
| `classify.py` | Train, evaluate, predict with sklearn |
| `pages/3_Analytics.py` | Analytics dashboard (replaces placeholder) |

## Labeling Pipeline (`label.py`)

### Outcome Labeling (regex-based)

Scan opinion text for outcome indicators:

**Plaintiff win signals:**
- "judgment for plaintiff"
- "granted plaintiff's motion for summary judgment"
- "defendant is liable"
- "judgment is entered in favor of plaintiff"
- "damages awarded"

**Defendant win signals:**
- "judgment for defendant"
- "plaintiff's complaint is dismissed"
- "granted defendant's motion"
- "summary judgment for defendant"
- "case dismissed with prejudice"

**Mixed signals:**
- "granted in part and denied in part"
- "partial summary judgment"
- Presence of both plaintiff and defendant win signals

**Logic:** Score each opinion by counting/weighting signals. If only plaintiff signals → `plaintiff_win`. If only defendant → `defendant_win`. If both or explicit "in part" language → `mixed`. If no clear signals → `unlabeled` (skip for training).

### Claim Type Labeling (regex-based)

Extract statutory section references from text:

- **FDCPA:** §1692b through §1692p, common aliases like "section 1692e(5)"
- **TCPA:** §227(b), §227(c), §227(d), TCPA subsections
- **FCRA:** §1681b, §1681e, §1681g, §1681i, §1681s-2, etc.

Pattern: `§\s*(\d{3,4}[a-z]?)(?:\(([a-z0-9]+)\))?` plus text variants ("section 1692e", "15 U.S.C. § 1692f")

Each opinion gets a set of section labels (multi-label).

### Optional LLM Enhancement (Ollama)

- For opinions where regex finds no outcome signals → send to LLM
- Configurable: `USE_LLM_LABELING = False` by default
- LLM prompt asks for structured JSON: `{"outcome": "plaintiff_win", "confidence": 0.85}`

## Classification Models

### Outcome Classifier (single-label, 3-class)

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
```

- **Models to compare:** Logistic Regression (baseline), Random Forest, XGBoost
- **Train/test split:** 80/20 stratified by outcome label
- **Evaluation:** Accuracy, per-class precision/recall/F1, confusion matrix
- **Selection:** Auto-select best model by macro F1 score

### Claim Type Classifier (multi-label)

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words='english')),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
])
```

- One classifier per section — each predicts present/absent
- Per-label precision/recall/F1, overall Hamming loss
- Minimum 50 examples per label to train

### Model Persistence

- Save with `joblib` to `data/models/outcome_model.pkl` and `data/models/claim_type_model.pkl`
- Store metadata in `models` table

### Batch Prediction

- After training, predict all 30K opinions → `predictions` table
- New opinions from future pipeline runs get auto-predicted

## UI Integration

### Search Page Enhancements (`pages/1_Search.py`)

- **New sidebar filters:**
  - Predicted Outcome: Dropdown — All / Plaintiff Win / Defendant Win / Mixed
  - Claim Sections: Multi-select of statutory sections
- Search results show predicted outcome badge and claim type tags
- Filters apply post-FAISS (Python filtering, same pattern as existing)

### Analytics Page (`pages/3_Analytics.py`)

- Outcome distribution bar chart
- Outcome by statute (grouped bars)
- Outcome by circuit (heatmap or grouped bars)
- Top claim sections (horizontal bars)
- Claim co-occurrence matrix
- Model performance display (accuracy, F1, confusion matrix)
- All charts filterable by statute, circuit, court type, date range

### Pipeline Integration (`pipeline.py`)

- `--classify` flag: run labeling + training + prediction after embedding
- `--predict-only` flag: predict new opinions with existing models
- Auto-predict new opinions if trained model exists

### FAISS Metadata Update

- Add `predicted_outcome` and `claim_sections` to chunk_map entries
- Enables filtering within FAISS search results

## Future Upgrade Path (Transformer Fine-tuning)

When hardware is available, swap the sklearn pipeline for:
- Fine-tuned `all-MiniLM-L6-v2` or `legal-bert` for classification
- Same labeled dataset, same evaluation framework
- ~20 lines of code change in `classify.py`
- All infrastructure (labeling, DB, UI, pipeline) carries over unchanged

## Dependencies

New packages:
- `scikit-learn` — TF-IDF, classifiers, evaluation
- `joblib` — Model serialization
- `xgboost` (optional) — Additional classifier
