# Laptop Setup & LLM Processing Instructions

## Prerequisites

- Ollama installed and running (`ollama serve`)
- llama3.1:latest already pulled (`ollama list` to verify)
- Project files + data/ directory copied from desktop

## Step 1: Install Python Dependencies

```bash
cd C:\PythonProject\ml
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Step 2: Update config.py

Open `config.py` and make these changes:

**Change the source DB path** (line 6) to wherever the cld project is on this machine:
```python
SOURCE_DB = "C:/path/to/cld/fdcpa.db"
```
If you don't have the cld project on this machine, leave it as-is — you just won't be able to sync new opinions.

**Change the Ollama model** (line 25):
```python
OLLAMA_MODEL = "llama3.1:latest"   # was "gemma2:2b"
```

**Enable LLM labeling** (line 27):
```python
USE_LLM_LABELING = True            # was False
```

## Step 3: Verify Setup

```bash
# Check the database loads and see current label counts
python label.py --stats

# Check entity counts
python ner.py --info

# Run test suite (should see 162 passed)
python -m pytest tests/ -v
```

## Step 4: Test LLM Labeling (1 opinion)

```bash
python label.py --llm --llm-limit 1
```

You should see output like:
```
Labeling unlabeled opinions with LLM...
  [1/1] Opinion 12345: defendant_win (confidence: 0.85)
LLM labeling complete: 1 labeled
```

If you get connection errors, make sure Ollama is running (`ollama serve`).

## Step 5: Run Full LLM Labeling (~18,813 opinions)

```bash
python label.py --llm
```

This will label all unlabeled opinions. Progress is saved after each opinion, so it's safe to interrupt and resume. With llama3.1 on GPU, expect roughly 5-15 seconds per opinion (~25-75 hours total).

To do a partial run first:
```bash
python label.py --llm --llm-limit 500
```

Check progress at any time:
```bash
python label.py --stats
```

## Step 6: Retrain Models with New Labels

After LLM labeling is done (or after a significant batch):

```bash
python classify.py
```

This retrains the outcome and claim type classifiers with the expanded labeled dataset, then re-predicts all opinions.

## Step 7: Update FAISS Metadata

```bash
python classify.py --predict-only
```

This updates the chunk_map with new predictions so the search and predictor pages reflect the updated model.

## Step 8: Launch Dashboard

```bash
python -m streamlit run app.py --server.headless true
```

Open http://localhost:8501 in your browser.

## Optional: GPU Upgrades

### spaCy Transformer Model (better NER)

```bash
pip install spacy[cuda12x]
python -m spacy download en_core_web_trf
```

Then in `config.py`:
```python
SPACY_MODEL = "en_core_web_trf"    # was "en_core_web_sm"
```

Re-run NER extraction:
```bash
python ner.py --no-spacy
python ner.py                       # with spaCy transformer
```

### FAISS GPU (faster search)

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

No code changes needed — same API.
