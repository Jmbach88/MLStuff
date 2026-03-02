import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Entity


class TestTitleParsing:
    def test_simple_v_title(self):
        from ner import extract_parties_from_title
        p, d = extract_parties_from_title("Smith v. Jones")
        assert p == "Smith"
        assert d == "Jones"

    def test_vs_title(self):
        from ner import extract_parties_from_title
        p, d = extract_parties_from_title("Smith vs. Jones")
        assert p == "Smith"
        assert d == "Jones"

    def test_et_al(self):
        from ner import extract_parties_from_title
        p, d = extract_parties_from_title("Hawkins v. I.C. System, Inc. et al")
        assert p == "Hawkins"
        assert d == "I.C. System, Inc."

    def test_multi_word_names(self):
        from ner import extract_parties_from_title
        p, d = extract_parties_from_title("JOHN DOE v. ABC COLLECTIONS, LLC")
        assert p == "JOHN DOE"
        assert d == "ABC COLLECTIONS, LLC"

    def test_no_v_returns_none(self):
        from ner import extract_parties_from_title
        p, d = extract_parties_from_title("In re Some Matter")
        assert p is None
        assert d is None

    def test_empty_title(self):
        from ner import extract_parties_from_title
        p, d = extract_parties_from_title("")
        assert p is None
        assert d is None

    def test_strips_whitespace(self):
        from ner import extract_parties_from_title
        p, d = extract_parties_from_title("  WEST  v.  GLOBAL LENDING SERVICES, LLC  et al  ")
        assert p == "WEST"
        assert d == "GLOBAL LENDING SERVICES, LLC"


class TestJudgeExtraction:
    def test_judge_uppercase(self):
        from ner import extract_judges
        text = "JUDGE PALERMO'S ORDER AND REPORT"
        judges = extract_judges(text)
        assert len(judges) >= 1
        assert any("PALERMO" in j["entity_value"] for j in judges)

    def test_before_judge(self):
        from ner import extract_judges
        text = "This matter came before Judge Sarah Hughes for hearing."
        judges = extract_judges(text)
        assert len(judges) >= 1
        assert any("Sarah Hughes" in j["entity_value"] for j in judges)

    def test_honorable_judge(self):
        from ner import extract_judges
        text = "before the Honorable Judge Robert Smith on this date."
        judges = extract_judges(text)
        assert len(judges) >= 1
        assert any("Robert Smith" in j["entity_value"] for j in judges)

    def test_district_judge_pattern(self):
        from ner import extract_judges
        text = "John Roberts, United States District Judge, presiding."
        judges = extract_judges(text)
        assert len(judges) >= 1
        assert any("John Roberts" in j["entity_value"] for j in judges)

    def test_magistrate_judge(self):
        from ner import extract_judges
        text = "Alice Walker, Magistrate Judge."
        judges = extract_judges(text)
        assert len(judges) >= 1
        assert any("Alice Walker" in j["entity_value"] for j in judges)

    def test_no_judge(self):
        from ner import extract_judges
        text = "The plaintiff filed a motion for summary judgment."
        judges = extract_judges(text)
        assert len(judges) == 0


class TestDollarExtraction:
    def test_simple_amount(self):
        from ner import extract_dollar_amounts
        text = "The total was $1,000."
        results = extract_dollar_amounts(text)
        assert len(results) == 1
        assert results[0]["entity_value"] == "$1,000"
        assert results[0]["entity_type"] == "DOLLAR_AMOUNT"

    def test_amount_with_cents(self):
        from ner import extract_dollar_amounts
        text = "Plaintiff seeks $1,234.56 in relief."
        results = extract_dollar_amounts(text)
        assert len(results) == 1
        assert results[0]["entity_value"] == "$1,234.56"

    def test_damages_classification(self):
        from ner import extract_dollar_amounts
        text = "The court awarded $5,000 in damages to the plaintiff."
        results = extract_dollar_amounts(text)
        assert len(results) == 1
        assert results[0]["entity_type"] == "DAMAGES_AWARDED"

    def test_attorney_fees_classification(self):
        from ner import extract_dollar_amounts
        text = "Plaintiff is entitled to $15,000 in attorney's fees."
        results = extract_dollar_amounts(text)
        assert len(results) == 1
        assert results[0]["entity_type"] == "ATTORNEY_FEES"

    def test_attorney_fee_variant(self):
        from ner import extract_dollar_amounts
        text = "The court awards attorney fees of $8,500."
        results = extract_dollar_amounts(text)
        assert len(results) == 1
        assert results[0]["entity_type"] == "ATTORNEY_FEES"

    def test_judgment_for_classification(self):
        from ner import extract_dollar_amounts
        text = "Judgment for plaintiff in the amount of $25,000."
        results = extract_dollar_amounts(text)
        assert len(results) == 1
        assert results[0]["entity_type"] == "DAMAGES_AWARDED"

    def test_multiple_amounts(self):
        from ner import extract_dollar_amounts
        text = "The court awarded $5,000 in damages and $2,000 in attorney's fees."
        results = extract_dollar_amounts(text)
        assert len(results) == 2
        types = {r["entity_type"] for r in results}
        assert "DAMAGES_AWARDED" in types
        assert "ATTORNEY_FEES" in types

    def test_no_amounts(self):
        from ner import extract_dollar_amounts
        text = "The motion is granted."
        results = extract_dollar_amounts(text)
        assert len(results) == 0

    def test_large_amount(self):
        from ner import extract_dollar_amounts
        text = "Damages of $1,000,000 are appropriate."
        results = extract_dollar_amounts(text)
        assert len(results) == 1
        assert results[0]["entity_value"] == "$1,000,000"


class TestDebtTypeExtraction:
    def test_credit_card_debt(self):
        from ner import extract_debt_types
        text = "The underlying obligation was credit card debt owed to Chase Bank."
        results = extract_debt_types(text)
        assert len(results) >= 1
        assert any("credit card" in r["entity_value"].lower() for r in results)

    def test_medical_debt(self):
        from ner import extract_debt_types
        text = "Defendant attempted to collect a medical debt."
        results = extract_debt_types(text)
        assert len(results) >= 1
        assert any("medical" in r["entity_value"].lower() for r in results)

    def test_student_loan(self):
        from ner import extract_debt_types
        text = "The student loan was in default for over a year."
        results = extract_debt_types(text)
        assert len(results) >= 1
        assert any("student loan" in r["entity_value"].lower() for r in results)

    def test_no_debt_type(self):
        from ner import extract_debt_types
        text = "The court finds for the plaintiff."
        results = extract_debt_types(text)
        assert len(results) == 0

    def test_auto_loan(self):
        from ner import extract_debt_types
        text = "The auto loan balance was $15,000."
        results = extract_debt_types(text)
        assert len(results) >= 1
        assert any("auto loan" in r["entity_value"].lower() for r in results)


class TestEntityStorage:
    def test_store_entities(self):
        from ner import store_entities
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM entities"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()
        session.add(Opinion(id=1, package_id="pkg_1", title="Case 1"))
        session.commit()
        session.close()

        entities = [
            {"entity_type": "JUDGE", "entity_value": "Smith",
             "context_snippet": "Judge Smith presiding", "start_char": 0, "end_char": 11},
            {"entity_type": "DOLLAR_AMOUNT", "entity_value": "$1,000",
             "context_snippet": "awarded $1,000", "start_char": 50, "end_char": 56},
        ]
        store_entities(engine, 1, entities)

        session = get_session(engine)
        stored = session.query(Entity).filter_by(opinion_id=1).all()
        assert len(stored) == 2
        types = {e.entity_type for e in stored}
        assert types == {"JUDGE", "DOLLAR_AMOUNT"}
        session.close()

    def test_store_clears_existing(self):
        from ner import store_entities
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM entities"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()
        session.add(Opinion(id=1, package_id="pkg_1", title="Case 1"))
        session.commit()
        session.close()

        entities1 = [{"entity_type": "JUDGE", "entity_value": "Smith",
                       "context_snippet": "ctx", "start_char": 0, "end_char": 5}]
        store_entities(engine, 1, entities1)

        entities2 = [{"entity_type": "JUDGE", "entity_value": "Jones",
                       "context_snippet": "ctx", "start_char": 0, "end_char": 5}]
        store_entities(engine, 1, entities2)

        session = get_session(engine)
        stored = session.query(Entity).filter_by(opinion_id=1).all()
        assert len(stored) == 1
        assert stored[0].entity_value == "Jones"
        session.close()


class TestEndToEnd:
    def test_extracts_multiple_entity_types(self):
        from ner import extract_entities_from_opinion
        title = "Smith v. ABC Collections, LLC"
        text = (
            "JUDGE PALERMO'S ORDER\n\n"
            "Before the Court is Plaintiff Smith's motion. "
            "The underlying credit card debt of $5,000 was assigned to Defendant. "
            "The court awards $3,000 in damages and $1,500 in attorney's fees."
        )
        entities = extract_entities_from_opinion(title, text, use_spacy=False)
        types = {e["entity_type"] for e in entities}
        assert "PLAINTIFF" in types
        assert "DEFENDANT" in types
        assert "JUDGE" in types
        assert "DEBT_TYPE" in types
        assert any(e["entity_type"] == "DAMAGES_AWARDED" for e in entities)
        assert any(e["entity_type"] == "ATTORNEY_FEES" for e in entities)
