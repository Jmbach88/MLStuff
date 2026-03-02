# Named Entity Recognition — Design Document

## Goal

Extract 8 entity types from 30K federal court opinions using hybrid rule-based regex + spaCy NER, store in a structured `entities` table, and visualize in an Entities dashboard with analytics overview and entity browser.

## Entity Types

| Type | Method | Source |
|------|--------|--------|
| JUDGE | Regex on first 1000 chars of plain_text | Header patterns |
| PLAINTIFF | Title parsing ("X v. Y") | Opinion title |
| DEFENDANT | Title parsing ("X v. Y") | Opinion title |
| ORIGINAL_CREDITOR | spaCy ORG + context patterns | plain_text |
| DEBT_TYPE | Keyword matching | First 3000 chars |
| DOLLAR_AMOUNT | Regex + context classification | plain_text |
| DAMAGES_AWARDED | Dollar regex near "damages"/"awarded" | plain_text |
| ATTORNEY_FEES | Dollar regex near "attorney fees" | plain_text |

Skipped: CASE_CITATION (done in Project 5), STATUTORY_SECTION (partially done in label.py), DATE (low value).

## Extraction Strategy

### Title Parsing (Plaintiff/Defendant)

Split opinion titles on ` v. ` or ` vs. `. Strip "et al", trim whitespace. Near-100% coverage.

### Judge Extraction

Regex on first 1000 chars of `plain_text`:
- `JUDGE\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)*)`
- `before\s+(?:the\s+Honorable\s+)?Judge\s+([A-Z].+?)[\s,.]`
- `([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s+(?:United States\s+)?(?:District|Circuit|Magistrate)\s+Judge`

### Dollar Amounts

Regex: `\$\s?[\d,]+(?:\.\d{2})?`

Context classification (±50 chars):
- Contains "attorney" + "fee" → ATTORNEY_FEES
- Contains "damages", "awarded", "judgment for" → DAMAGES_AWARDED
- Otherwise → DOLLAR_AMOUNT

### Debt Type

Keyword matching against lookup list: credit card, medical debt, student loan, auto loan, mortgage, utility, payday loan, etc. Search first 3000 chars.

### Original Creditor

spaCy ORG entities near context phrases: "original creditor", "assignee of", "on behalf of", "successor to".

### spaCy Role

Used to validate/enrich plaintiff and defendant names (PERSON/ORG) and find original creditors. Not the primary extraction method — regex handles structured patterns.

Model configurable via `config.py`: start with `en_core_web_sm`, upgrade to `en_core_web_trf` when GPU available.

## Database Schema

```
Entity:
    id (PK, autoincrement)
    opinion_id (FK opinions.id)
    entity_type (Text)
    entity_value (Text)
    context_snippet (Text)
    start_char (Integer, nullable)
    end_char (Integer, nullable)
```

Index on `(opinion_id, entity_type)`.

## Entities Dashboard UI

Page: `pages/5_Entities.py`

Sidebar filters: Statute, Circuit, Court Type.

**Analytics Overview:**
1. Summary metrics — total entities, opinions with judges, opinions with damages, opinions with attorney fees
2. Damages by Circuit — bar chart of avg/median damages per circuit
3. Attorney Fees by Circuit — bar chart of avg attorney fees per circuit
4. Top Defendants — most-sued by count, with avg damages
5. Top Judges — most opinions, with avg damages and avg attorney fees
6. Debt Type Distribution — bar/pie chart

**Entity Browser:**
- Filter by entity type
- Searchable table: opinion title, entity type, entity value, date
- Per-opinion view showing all extracted entities

## New/Modified Files

| File | Change |
|------|--------|
| `ner.py` | New: extraction functions, CLI |
| `tests/test_ner.py` | New: unit tests for all extraction functions |
| `pages/5_Entities.py` | New: dashboard |
| `db.py` | Add Entity ORM class |
| `config.py` | Add SPACY_MODEL setting |
| `pipeline.py` | Add `--ner` flag |
| `requirements.txt` | Add `spacy>=3.7.0` |

## CLI

```
python ner.py                          # extract all entities
python ner.py --types judge,plaintiff  # extract specific types only
python ner.py --info                   # print entity summary
```

## Dependencies

- `spacy>=3.7.0`
- spaCy model: `python -m spacy download en_core_web_sm`

## Expected Runtime

- Title parsing + regex: ~5 minutes (30K opinions)
- spaCy processing: ~5 minutes with en_core_web_sm on CPU
- Total: ~10 minutes
