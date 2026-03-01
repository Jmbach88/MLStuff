# Project 2 Improvements: Labeling Coverage — Design Document

## Goal

Improve outcome labeling coverage from 38.5% to 80%+ through better regex patterns and LLM-assisted labeling (Ollama), then retrain both models and measure improvement.

## Current State

- 30,609 opinions with text
- 11,796 labeled by regex (38.5%): 3,843 plaintiff_win, 3,542 defendant_win, 4,411 mixed
- 18,813 unlabeled (61.5%)
- Outcome model: 79.8% accuracy, 0.80 macro F1
- Claim type model: 0.35 avg F1 across 40 sections

## Three-Phase Approach

### Phase 1: Regex Improvements

Add patterns the current regex misses. Analysis of unlabeled opinions shows:
- 55% contain "granted"/"denied" (generic motion language not captured)
- 33% contain "settled"/"settlement"
- 10% contain "affirmed" (appellate)
- 8% contain "order granting"/"order denying"
- 6% contain "default judgment"

**New patterns:**

Appellate outcomes (low weight=1):
- "affirmed" → defendant_win (lower court stands)
- "reversed" → plaintiff_win (overturned)
- "affirmed in part.*reversed in part" → mixed
- "vacated and remanded" → mixed

Motion/order language:
- "motion to dismiss.*granted" → defendant_win
- "motion to dismiss.*denied" → plaintiff_win
- "order granting.*motion to dismiss" → defendant_win
- "order denying.*motion to dismiss" → plaintiff_win
- "default judgment.*entered" → plaintiff_win

Settlement handling:
- "settled", "stipulated dismissal", "voluntary dismissal" → `settled` label
- Excluded from classifier training (not judicial rulings on merits)

Expected gain: ~20-30% of 18,813 unlabeled → ~4,000-5,600 new labels.

### Phase 2: Ollama LLM Labeling

**Setup:**
- Install Ollama for Windows
- Model: `mistral:7b-instruct-v0.3-q4_K_M` (4.1 GB, fits in 16 GB RAM)

**Prompt:**
```
You are a legal analyst. Read this federal court opinion and classify its outcome.

Respond with ONLY a JSON object, no other text:
{"outcome": "<LABEL>", "confidence": <0.0-1.0>}

Where LABEL is one of:
- "plaintiff_win" - plaintiff prevailed
- "defendant_win" - defendant prevailed
- "mixed" - split decision
- "settled" - case settled or voluntarily dismissed
- "unclear" - cannot determine from the text

OPINION TEXT:
{first 3000 characters}
```

**Batch processing:**
- Process only still-unlabeled opinions (after Phase 1)
- Ollama HTTP API (localhost:11434/api/generate)
- ~5-10 sec/opinion → ~13K opinions ≈ 18-36 hours
- Save after each opinion (crash-safe)
- Skip "unclear" or confidence < 0.5
- Store as `source="llm"` in labels table

Expected gain: ~50-70% of remaining unlabeled → ~6,500-9,000 new labels.

### Phase 3: Retrain & Compare

- Retrain outcome model with expanded labels → save as `outcome_logreg_v2`
- Same architecture (TF-IDF + LogReg) for fair comparison
- Retrain claim type model too → `claim_type_ovr_v2`
- Compare v1 vs v2 metrics (accuracy, F1, per-class)
- Re-predict all opinions and update FAISS metadata
- Keep v1 models for reference

Expected outcome: 83-88% accuracy (from 79.8%), F1 improvement proportional.

## New/Modified Files

| File | Change |
|------|--------|
| `label.py` | Add new regex patterns, add `label_with_llm()`, add `--llm` flag |
| `classify.py` | Add `--compare` flag for v1 vs v2 comparison |
| `config.py` | Add OLLAMA_MODEL, OLLAMA_URL, LLM_TEXT_LIMIT settings |
| `tests/test_label.py` | Add tests for new regex patterns |

## Dependencies

- Ollama (system install, not pip)
- `requests` (for Ollama HTTP API, already available via other deps)
