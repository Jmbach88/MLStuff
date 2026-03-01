# Labeling Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve outcome labeling coverage from 38.5% to 80%+ through better regex patterns and Ollama LLM-assisted labeling, then retrain models and measure improvement.

**Architecture:** Phase 1 adds new regex patterns (appellate, motions, default judgments, settlements) to label.py. Phase 2 adds Ollama integration for LLM-assisted labeling of remaining unlabeled opinions. Phase 3 retrains models as v2 with expanded data and compares against v1 baseline.

**Tech Stack:** Ollama (Mistral 7B), requests (HTTP API), existing scikit-learn/SQLAlchemy/FAISS stack.

---

### Task 1: Expand Regex Patterns

**Files:**
- Modify: `label.py:14-40` (pattern lists)
- Modify: `tests/test_label.py` (add new test class)

**Step 1: Write the failing tests**

Add to `tests/test_label.py` after the `TestOutcomeLabeling` class:

```python
class TestExpandedOutcomePatterns:
    """Tests for new regex patterns added in Phase 1."""

    def test_motion_to_dismiss_granted(self):
        text = "Defendant's motion to dismiss is hereby granted."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"

    def test_motion_to_dismiss_denied(self):
        text = "The motion to dismiss filed by defendant is denied."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_order_granting_motion_to_dismiss(self):
        text = "ORDER GRANTING DEFENDANT'S MOTION TO DISMISS."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"

    def test_order_denying_motion_to_dismiss(self):
        text = "ORDER DENYING MOTION TO DISMISS."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_default_judgment(self):
        text = "Default judgment is entered against the defendant in the amount of $5,000."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_affirmed_low_weight(self):
        text = "The judgment of the district court is AFFIRMED."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"

    def test_reversed_low_weight(self):
        text = "The district court's order is REVERSED."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_affirmed_in_part_reversed_in_part(self):
        text = "The judgment is AFFIRMED IN PART and REVERSED IN PART."
        result = label_outcome(text)
        assert result["label"] == "mixed"

    def test_vacated_and_remanded(self):
        text = "The order is VACATED and REMANDED for further proceedings."
        result = label_outcome(text)
        assert result["label"] == "mixed"

    def test_settled(self):
        text = "The parties have reached a settlement. The case is dismissed pursuant to stipulation."
        result = label_outcome(text)
        assert result["label"] == "settled"

    def test_voluntary_dismissal(self):
        text = "Pursuant to the parties' stipulation of voluntary dismissal, this case is closed."
        result = label_outcome(text)
        assert result["label"] == "settled"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_label.py::TestExpandedOutcomePatterns -v`
Expected: FAIL — new patterns not yet implemented

**Step 3: Add new patterns to label.py**

In `label.py`, add to `PLAINTIFF_WIN_PATTERNS` list (after line 22, before the closing `]`):

```python
    (re.compile(r"motion\s+to\s+dismiss.*(?:is\s+)?denied", re.IGNORECASE), 2),
    (re.compile(r"order\s+denying.*motion\s+to\s+dismiss", re.IGNORECASE), 2),
    (re.compile(r"default\s+judgment.*(?:is\s+)?entered", re.IGNORECASE), 2),
    (re.compile(r"\breversed\b", re.IGNORECASE), 1),
```

Add to `DEFENDANT_WIN_PATTERNS` list (after line 33, before the closing `]`):

```python
    (re.compile(r"motion\s+to\s+dismiss.*(?:is\s+)?(?:hereby\s+)?granted", re.IGNORECASE), 2),
    (re.compile(r"order\s+granting.*motion\s+to\s+dismiss", re.IGNORECASE), 2),
    (re.compile(r"\baffirmed\b", re.IGNORECASE), 1),
```

Add to `MIXED_PATTERNS` list (after line 40, before the closing `]`):

```python
    (re.compile(r"affirmed\s+in\s+part.*reversed\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"reversed\s+in\s+part.*affirmed\s+in\s+part", re.IGNORECASE), 3),
    (re.compile(r"vacated\s+and\s+remanded", re.IGNORECASE), 2),
```

Add a new `SETTLEMENT_PATTERNS` list after `MIXED_PATTERNS`:

```python
SETTLEMENT_PATTERNS = [
    (re.compile(r"parties\s+have\s+reached\s+a\s+settlement", re.IGNORECASE), 3),
    (re.compile(r"stipulat(?:ed|ion)\s+(?:of\s+)?(?:voluntary\s+)?dismissal", re.IGNORECASE), 3),
    (re.compile(r"voluntary\s+dismissal", re.IGNORECASE), 2),
    (re.compile(r"dismissed\s+pursuant\s+to\s+(?:the\s+)?(?:parties'?\s+)?stipulation", re.IGNORECASE), 3),
    (re.compile(r"settled\s+(?:out\s+of\s+court|between\s+the\s+parties)", re.IGNORECASE), 3),
]
```

**Step 4: Update label_outcome() to handle settlements**

In `label.py`, update the `label_outcome()` function. Add settlement check after the mixed patterns check (after line 51) and before the plaintiff/defendant scoring:

```python
    # Check settlement patterns
    settlement_score = 0
    for pattern, weight in SETTLEMENT_PATTERNS:
        if pattern.search(text_content):
            settlement_score += weight

    if settlement_score >= 2:
        return {"label": "settled", "confidence": min(0.5 + settlement_score * 0.1, 0.95)}
```

Also update `run_labeling()` to handle the "settled" label. In the outcome labeling block (around line 149), change `if outcome["label"] != "unlabeled":` to also store "settled" labels but note that we want to track them. Actually, "settled" is a valid label value — just store it. The classifier training already filters by `label_value IN ('plaintiff_win', 'defendant_win', 'mixed')`, so settled opinions won't pollute training.

Wait — actually the current train_outcome_model query is:
```sql
WHERE l.label_type = 'outcome' AND o.plain_text IS NOT NULL
```
It doesn't filter by label_value. We need to exclude 'settled' from training. We'll handle that in Task 5.

Update `run_labeling()` stats to include "settled":

In the stats dict (around line 176-183), add:

```python
            "settled": outcome_counts.get("settled", 0),
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_label.py::TestExpandedOutcomePatterns -v`
Expected: 11 PASS

**Step 6: Run all tests to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All 55 tests PASS (44 existing + 11 new)

**Step 7: Commit**

```bash
git add label.py tests/test_label.py
git commit -m "feat: expand outcome regex patterns (motions, appellate, default judgments, settlements)"
```

---

### Task 2: Re-run Regex Labeling on Real Data

**Files:**
- No code changes — run labeling and measure coverage improvement

**Step 1: Re-run labeling with new patterns**

Run: `cd /c/PythonProject/ml && python label.py --relabel`

This will delete existing labels and re-label all 30,609 opinions with the expanded regex patterns.

Expected: More labeled opinions than before (was 11,796, should be ~15,000-17,000 now).

**Step 2: Check stats**

Run: `python label.py --stats`

Expected: Shows new distribution including settled category. Note the new counts for comparison with Phase 2.

**Step 3: No commit needed** — validation step only.

---

### Task 3: Add Ollama Config Settings

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`

**Step 1: Add Ollama settings to config.py**

Add at the end of `config.py`:

```python
# LLM Labeling (Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"
LLM_TEXT_LIMIT = 3000  # chars to send to LLM
USE_LLM_LABELING = False  # toggle when Ollama is available
```

**Step 2: Add requests to requirements.txt**

Add to `requirements.txt`:

```
requests>=2.28.0
```

**Step 3: Run pip install**

Run: `pip install requests`

**Step 4: Commit**

```bash
git add config.py requirements.txt
git commit -m "feat: add Ollama config settings and requests dependency"
```

---

### Task 4: Implement LLM Labeling Function

**Files:**
- Modify: `label.py`
- Modify: `tests/test_label.py`

**Step 1: Write the failing tests**

Add to `tests/test_label.py`:

```python
from unittest.mock import patch, MagicMock
import json


class TestLLMLabeling:
    """Tests for Ollama LLM-assisted labeling."""

    def test_parse_llm_response_valid(self):
        from label import _parse_llm_response
        response = '{"outcome": "plaintiff_win", "confidence": 0.85}'
        result = _parse_llm_response(response)
        assert result["label"] == "plaintiff_win"
        assert result["confidence"] == 0.85

    def test_parse_llm_response_with_text_around_json(self):
        from label import _parse_llm_response
        response = 'Based on my analysis:\n{"outcome": "defendant_win", "confidence": 0.9}\nThat is my conclusion.'
        result = _parse_llm_response(response)
        assert result["label"] == "defendant_win"

    def test_parse_llm_response_invalid(self):
        from label import _parse_llm_response
        response = "I cannot determine the outcome of this case."
        result = _parse_llm_response(response)
        assert result["label"] == "unclear"
        assert result["confidence"] == 0.0

    def test_parse_llm_response_unclear(self):
        from label import _parse_llm_response
        response = '{"outcome": "unclear", "confidence": 0.3}'
        result = _parse_llm_response(response)
        assert result["label"] == "unclear"

    def test_parse_llm_response_settled(self):
        from label import _parse_llm_response
        response = '{"outcome": "settled", "confidence": 0.8}'
        result = _parse_llm_response(response)
        assert result["label"] == "settled"

    @patch("label.requests.post")
    def test_label_with_llm_success(self, mock_post):
        from label import label_with_llm
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"outcome": "plaintiff_win", "confidence": 0.9}'
        }
        mock_post.return_value = mock_response

        result = label_with_llm("The court grants judgment for plaintiff.")
        assert result["label"] == "plaintiff_win"
        assert result["confidence"] == 0.9

    @patch("label.requests.post")
    def test_label_with_llm_connection_error(self, mock_post):
        from label import label_with_llm
        import requests as req
        mock_post.side_effect = req.ConnectionError("Ollama not running")

        result = label_with_llm("Some opinion text")
        assert result["label"] == "error"
        assert result["confidence"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_label.py::TestLLMLabeling -v`
Expected: FAIL with `ImportError: cannot import name '_parse_llm_response'`

**Step 3: Implement LLM labeling functions**

Add to `label.py` after the imports at the top:

```python
import json as _json
import requests
import config
```

Add these functions after `label_claim_types()` and before `run_labeling()`:

```python
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
    """Parse LLM response, extracting JSON from potentially noisy output.

    Returns:
        {"label": str, "confidence": float}
    """
    # Try to find JSON in the response
    try:
        # Direct parse
        data = _json.loads(response_text.strip())
    except _json.JSONDecodeError:
        # Try to extract JSON from surrounding text
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

    Returns:
        {"label": str, "confidence": float}
        On error: {"label": "error", "confidence": 0.0}
    """
    truncated = text_content[:config.LLM_TEXT_LIMIT]
    prompt = LLM_PROMPT_TEMPLATE.format(text=truncated)

    try:
        response = requests.post(
            config.OLLAMA_URL,
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        result_text = response.json().get("response", "")
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_label.py::TestLLMLabeling -v`
Expected: 7 PASS

**Step 5: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add label.py tests/test_label.py
git commit -m "feat: add Ollama LLM labeling function with response parsing"
```

---

### Task 5: Batch LLM Labeling Pipeline

**Files:**
- Modify: `label.py`

**Step 1: Add run_llm_labeling() function**

Add to `label.py` after `label_with_llm()` and before `run_labeling()`:

```python
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
```

**Step 2: Add --llm and --llm-limit flags to CLI**

In the CLI section of `label.py`, add after the `--stats` argument:

```python
    parser.add_argument("--llm", action="store_true", help="Use Ollama LLM for unlabeled opinions")
    parser.add_argument("--llm-limit", type=int, default=None, help="Max opinions to LLM-label")
```

Add handler in the CLI block, after the `args.stats` block and before the `else`:

Change the if/else to if/elif/elif/else:

```python
    if args.stats:
        # ... existing stats code ...
    elif args.llm:
        stats = run_llm_labeling(limit=args.llm_limit)
        print(f"\nLLM labeling: {stats}")
    else:
        stats = run_labeling(relabel=args.relabel)
        # ... existing output ...
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add label.py
git commit -m "feat: add batch LLM labeling pipeline with crash-safe progress"
```

---

### Task 6: Install and Test Ollama

**Files:**
- No code changes — system setup and validation

**Step 1: Install Ollama**

Download and install Ollama from https://ollama.com/download/windows

**Step 2: Pull the model**

Run: `ollama pull mistral:7b-instruct-v0.3-q4_K_M`

If that specific tag isn't available, try: `ollama pull mistral`

Expected: Downloads ~4 GB model

**Step 3: Test Ollama is running**

Run: `curl http://localhost:11434/api/tags`
Expected: JSON response listing available models

**Step 4: Test LLM labeling on a small batch**

Run: `cd /c/PythonProject/ml && python label.py --llm --llm-limit 5`

Expected: Processes 5 opinions, prints stats. Verify:
- No connection errors
- Labels are reasonable (plaintiff_win, defendant_win, mixed, settled)
- Each opinion takes ~5-10 seconds

**Step 5: Update config.py if model tag differs**

If you used a different model tag (e.g., `mistral` instead of `mistral:7b-instruct-v0.3-q4_K_M`), update `OLLAMA_MODEL` in `config.py`.

**Step 6: No commit needed** — system setup step.

---

### Task 7: Run Full LLM Labeling

**Files:**
- No code changes — run on full unlabeled dataset

**Step 1: Check remaining unlabeled count**

Run:
```python
python -c "
from db import get_local_engine, get_session
from sqlalchemy import text
engine = get_local_engine()
session = get_session(engine)
count = session.execute(text(
    \"SELECT COUNT(*) FROM opinions o WHERE o.plain_text IS NOT NULL AND o.plain_text != '' \"
    \"AND o.id NOT IN (SELECT DISTINCT opinion_id FROM labels WHERE label_type='outcome')\"
)).fetchone()[0]
print(f'Remaining unlabeled: {count}')
session.close()
"
```

**Step 2: Run LLM labeling on all remaining**

Run: `cd /c/PythonProject/ml && python label.py --llm`

This will take ~18-36 hours depending on model speed. Run overnight.

**Step 3: Check final coverage**

Run: `python label.py --stats`

Expected: Significantly improved coverage (target: 80%+ of opinions labeled).

---

### Task 8: Model Versioning Support

**Files:**
- Modify: `classify.py`
- Modify: `tests/test_classify.py`

**Step 1: Write the failing test**

Add to `tests/test_classify.py`:

```python
class TestModelVersioning:
    def test_train_outcome_model_v2(self):
        from classify import train_outcome_model
        engine = _setup_labeled_data()
        result = train_outcome_model(engine, version=2)
        assert result["model_name"] == "outcome_logreg_v2"

    def test_compare_models(self):
        from classify import train_outcome_model, compare_models
        engine = _setup_labeled_data()
        train_outcome_model(engine, version=1)
        train_outcome_model(engine, version=2)
        comparison = compare_models(engine, "outcome")
        assert "v1" in comparison
        assert "v2" in comparison
        assert "accuracy" in comparison["v1"]
        assert "accuracy" in comparison["v2"]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classify.py::TestModelVersioning -v`
Expected: FAIL

**Step 3: Add version parameter to train functions**

In `classify.py`, update `train_outcome_model()` signature:

```python
def train_outcome_model(engine=None, version=1):
```

Update the model_name and path inside the function:

```python
    model_name = f"outcome_logreg_v{version}"
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, f"outcome_model_v{version}.pkl")
```

Also update the training query to exclude "settled" labels from training:

```python
    rows = session.execute(text(
        "SELECT o.id, o.plain_text, l.label_value "
        "FROM opinions o JOIN labels l ON o.id = l.opinion_id "
        "WHERE l.label_type = 'outcome' "
        "AND l.label_value IN ('plaintiff_win', 'defendant_win', 'mixed') "
        "AND o.plain_text IS NOT NULL AND o.plain_text != ''"
    )).fetchall()
```

Do the same for `train_claim_type_model()`:

```python
def train_claim_type_model(engine=None, version=1):
```

```python
    model_name = f"claim_type_ovr_v{version}"
    model_path = os.path.join(MODELS_DIR, f"claim_type_model_v{version}.pkl")
```

**Step 4: Add compare_models() function**

Add to `classify.py`:

```python
def compare_models(engine=None, label_type="outcome"):
    """Compare model versions by metrics stored in DB.

    Returns:
        dict with version keys containing accuracy and f1_score.
    """
    if engine is None:
        engine = get_local_engine()

    session = get_session(engine)
    rows = session.execute(text(
        "SELECT name, accuracy, f1_score, trained_at FROM models "
        "WHERE label_type = :lt ORDER BY name"
    ), {"lt": label_type}).fetchall()
    session.close()

    comparison = {}
    for name, acc, f1, trained in rows:
        # Extract version from name (e.g., "outcome_logreg_v1" -> "v1")
        version = name.split("_v")[-1] if "_v" in name else name
        comparison[f"v{version}"] = {
            "model_name": name,
            "accuracy": acc,
            "f1_score": f1,
            "trained_at": trained,
        }

    return comparison
```

**Step 5: Update predict functions to accept version**

Update `predict_outcomes()`:

```python
def predict_outcomes(engine=None, version=1):
```

```python
    model_path = os.path.join(MODELS_DIR, f"outcome_model_v{version}.pkl")
    model_name = f"outcome_logreg_v{version}"
```

Update `predict_claim_types()`:

```python
def predict_claim_types(engine=None, version=1):
```

```python
    model_path = os.path.join(MODELS_DIR, f"claim_type_model_v{version}.pkl")
    model_name = f"claim_type_ovr_v{version}"
```

Update `update_chunk_map_with_predictions()` to accept version:

```python
def update_chunk_map_with_predictions(engine=None, outcome_version=1, claim_version=1):
```

```python
    outcome_rows = session.execute(text(
        f"SELECT opinion_id, predicted_value FROM predictions "
        f"WHERE model_name = 'outcome_logreg_v{outcome_version}' AND label_type = 'outcome'"
    )).fetchall()
```

```python
    claim_rows = session.execute(text(
        f"SELECT opinion_id, predicted_value FROM predictions "
        f"WHERE model_name = 'claim_type_ovr_v{claim_version}' AND label_type = 'claim_type'"
    )).fetchall()
```

**Step 6: Update CLI**

Add to argparse:

```python
    parser.add_argument("--version", type=int, default=1, help="Model version (1 or 2)")
    parser.add_argument("--compare", action="store_true", help="Compare model versions")
```

Add handler:

```python
    if args.compare:
        print("\n=== Outcome Models ===")
        comp = compare_models(engine, "outcome")
        for v, metrics in sorted(comp.items()):
            print(f"  {v}: accuracy={metrics['accuracy']}, f1={metrics['f1_score']}, trained={metrics['trained_at']}")
        print("\n=== Claim Type Models ===")
        comp = compare_models(engine, "claim_type")
        for v, metrics in sorted(comp.items()):
            print(f"  {v}: f1={metrics['f1_score']}, trained={metrics['trained_at']}")
        return
```

Pass version to train/predict calls:

```python
    if not args.predict_only:
        outcome_result = train_outcome_model(engine, version=args.version)
        claim_result = train_claim_type_model(engine, version=args.version)

    if not args.train_only:
        predict_outcomes(engine, version=args.version)
        predict_claim_types(engine, version=args.version)
```

**Step 7: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add classify.py tests/test_classify.py
git commit -m "feat: add model versioning and comparison support"
```

---

### Task 9: Retrain Models as v2 and Compare

**Files:**
- No code changes — run training and comparison

**Step 1: Delete existing predictions (v1 predictions are stale with new labels)**

Run:
```python
python -c "
from db import get_local_engine, get_session
from sqlalchemy import text
engine = get_local_engine()
session = get_session(engine)
session.execute(text('DELETE FROM predictions'))
session.commit()
print('Cleared predictions table')
session.close()
"
```

**Step 2: Train v2 models**

Run: `cd /c/PythonProject/ml && python classify.py --version 2 --train-only`

Expected: Trains on expanded label set, prints accuracy/F1 for both models.

**Step 3: Compare v1 vs v2**

Run: `python classify.py --compare`

Expected: Shows side-by-side comparison of v1 and v2 metrics.

**Step 4: Predict with v2**

Run: `python classify.py --version 2 --predict-only`

Expected: Predicts all 30K opinions with v2 models.

**Step 5: Update FAISS metadata with v2 predictions**

Run: `python classify.py --update-index`

Note: update_chunk_map_with_predictions needs to be called with the right version. If the CLI doesn't pass version yet, run:

```python
python -c "
from classify import update_chunk_map_with_predictions
from db import get_local_engine
n = update_chunk_map_with_predictions(get_local_engine(), outcome_version=2, claim_version=2)
print(f'Updated {n} entries')
"
```

**Step 6: No commit** — validation step.

---

### Task 10: Final Validation and Commit

**Files:**
- No code changes — validation and push

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Check final label coverage**

Run: `python label.py --stats`
Expected: Shows improved coverage (target: 80%+)

**Step 3: Compare models**

Run: `python classify.py --compare`
Expected: v2 shows improvement over v1

**Step 4: Test Streamlit app**

Run: `python -m streamlit run app.py --server.headless true`
Verify: Analytics tab shows updated charts, search filters work with new predictions

**Step 5: Commit and push**

```bash
git add -A
git commit -m "feat: complete labeling improvements — expanded regex + LLM labeling + v2 models"
git push origin master
```

---

## Summary

| Task | Description | Phase |
|------|-------------|-------|
| 1 | Expand regex patterns (motions, appellate, settlements) | Phase 1 |
| 2 | Re-run regex labeling on real data | Phase 1 |
| 3 | Add Ollama config settings | Phase 2 |
| 4 | Implement LLM labeling function | Phase 2 |
| 5 | Batch LLM labeling pipeline | Phase 2 |
| 6 | Install and test Ollama | Phase 2 |
| 7 | Run full LLM labeling (~18-36 hours) | Phase 2 |
| 8 | Model versioning support | Phase 3 |
| 9 | Retrain as v2 and compare | Phase 3 |
| 10 | Final validation and push | Phase 3 |

## Commands Reference

```bash
# Regex labeling
python label.py                         # label new opinions (regex)
python label.py --relabel               # re-label all (regex)
python label.py --stats                 # show distribution

# LLM labeling
python label.py --llm                   # LLM-label all unlabeled
python label.py --llm --llm-limit 10    # test with 10 opinions

# Training and prediction
python classify.py --version 2 --train-only     # train v2 models
python classify.py --version 2 --predict-only   # predict with v2
python classify.py --compare                     # compare v1 vs v2
python classify.py --update-index                # update FAISS metadata

# Tests
python -m pytest tests/ -v              # all tests
```
