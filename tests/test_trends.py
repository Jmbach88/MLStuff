import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Entity
from sqlalchemy import text as sql_text


@pytest.fixture
def engine():
    """Create in-memory DB with test data for trend queries."""
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    # Clean slate
    for table in ["entities", "predictions", "labels", "opinion_statutes", "opinions", "statutes"]:
        session.execute(sql_text(f"DELETE FROM {table}"))
    session.commit()

    # Create statutes
    session.execute(sql_text("INSERT INTO statutes (id, key, name) VALUES (1, 'fdcpa', 'FDCPA')"))
    session.execute(sql_text("INSERT INTO statutes (id, key, name) VALUES (2, 'tcpa', 'TCPA')"))
    session.commit()

    # Create opinions with varied dates, circuits
    opinions = [
        (1, "pkg1", "Smith v. Jones", "district", "3rd", "2020-01-15"),
        (2, "pkg2", "Doe v. Acme", "district", "3rd", "2020-06-20"),
        (3, "pkg3", "Roe v. Bank", "district", "9th", "2021-03-10"),
        (4, "pkg4", "Lee v. Corp", "district", "9th", "2021-07-05"),
        (5, "pkg5", "Kim v. LLC", "district", "3rd", "2022-01-01"),
        (6, "pkg6", "Park v. Inc", "district", "9th", "2022-08-15"),
        (7, "pkg7", "Chen v. Co", "district", "5th", "2020-04-01"),
        (8, "pkg8", "Wang v. Ltd", "district", "5th", "2021-11-01"),
        (9, "pkg9", "Ali v. Firm", "district", "5th", "2022-05-01"),
        (10, "pkg10", "Cruz v. Grp", "district", "3rd", "2022-12-01"),
        (11, "pkg11", "Test v. Case", "district", "3rd", "2020-03-01"),
        (12, "pkg12", "More v. Data", "district", "9th", "2021-09-01"),
    ]
    for oid, pkg, title, ct, circ, date in opinions:
        session.add(Opinion(id=oid, package_id=pkg, title=title,
                            court_type=ct, circuit=circ, date_issued=date))
    session.commit()

    # Link opinions to statutes
    for oid in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        session.execute(sql_text(
            f"INSERT INTO opinion_statutes (opinion_id, statute_id) VALUES ({oid}, 1)"
        ))
    for oid in [3, 4, 6, 12]:
        session.execute(sql_text(
            f"INSERT INTO opinion_statutes (opinion_id, statute_id) VALUES ({oid}, 2)"
        ))
    session.commit()

    # Add outcome predictions
    outcomes = [
        (1, "plaintiff_win"), (2, "defendant_win"), (3, "plaintiff_win"),
        (4, "defendant_win"), (5, "plaintiff_win"), (6, "plaintiff_win"),
        (7, "defendant_win"), (8, "plaintiff_win"), (9, "defendant_win"),
        (10, "plaintiff_win"), (11, "defendant_win"), (12, "plaintiff_win"),
    ]
    for oid, outcome in outcomes:
        session.execute(sql_text(
            f"INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            f"VALUES ({oid}, 'outcome_logreg_v1', 'outcome', '{outcome}', 0.85)"
        ))
    session.commit()

    # Add entities: DAMAGES_AWARDED, ATTORNEY_FEES, JUDGE, DEFENSE_TYPE
    entities = [
        (1, "DAMAGES_AWARDED", "$5,000"),
        (1, "ATTORNEY_FEES", "$2,000"),
        (1, "JUDGE", "Smith"),
        (2, "DAMAGES_AWARDED", "$3,000"),
        (2, "JUDGE", "Smith"),
        (3, "DAMAGES_AWARDED", "$10,000"),
        (3, "ATTORNEY_FEES", "$4,000"),
        (3, "JUDGE", "Jones"),
        (4, "JUDGE", "Jones"),
        (5, "DAMAGES_AWARDED", "$8,000"),
        (5, "ATTORNEY_FEES", "$3,000"),
        (5, "JUDGE", "Smith"),
        (6, "DAMAGES_AWARDED", "$12,000"),
        (6, "ATTORNEY_FEES", "$5,000"),
        (6, "JUDGE", "Jones"),
        (7, "DAMAGES_AWARDED", "$2,000"),
        (7, "JUDGE", "Williams"),
        (8, "DAMAGES_AWARDED", "$7,000"),
        (8, "ATTORNEY_FEES", "$3,500"),
        (8, "JUDGE", "Williams"),
        (9, "JUDGE", "Williams"),
        (10, "DAMAGES_AWARDED", "$6,000"),
        (10, "ATTORNEY_FEES", "$2,500"),
        (10, "JUDGE", "Smith"),
        (11, "JUDGE", "Smith"),
        (12, "DAMAGES_AWARDED", "$9,000"),
        (12, "ATTORNEY_FEES", "$4,500"),
        (12, "JUDGE", "Jones"),
        # Defense types
        (1, "DEFENSE_TYPE", "bona fide error"),
        (2, "DEFENSE_TYPE", "statute of limitations"),
        (2, "DEFENSE_TYPE", "bona fide error"),
        (3, "DEFENSE_TYPE", "arbitration"),
        (4, "DEFENSE_TYPE", "bona fide error"),
        (5, "DEFENSE_TYPE", "statute of limitations"),
        (7, "DEFENSE_TYPE", "bona fide error"),
        (9, "DEFENSE_TYPE", "arbitration"),
        (9, "DEFENSE_TYPE", "bona fide error"),
    ]
    for oid, etype, evalue in entities:
        session.add(Entity(opinion_id=oid, entity_type=etype, entity_value=evalue,
                           context_snippet="test context"))
    session.commit()
    session.close()

    return engine


class TestOutcomeTrends:
    def test_outcome_trends_returns_dataframe(self, engine):
        from trends import outcome_trends_by_year
        df = outcome_trends_by_year(engine)
        assert not df.empty
        assert "year" in df.columns
        assert "plaintiff_win_rate" in df.columns
        assert "total" in df.columns

    def test_outcome_trends_filter_by_circuit(self, engine):
        from trends import outcome_trends_by_year
        df = outcome_trends_by_year(engine, circuit="3rd")
        assert not df.empty
        assert all(df["total"] > 0)

    def test_outcome_trends_filter_by_statute(self, engine):
        from trends import outcome_trends_by_year
        df = outcome_trends_by_year(engine, statute="FDCPA")
        assert not df.empty


class TestDamagesTrends:
    def test_damages_trends_returns_dataframe(self, engine):
        from trends import damages_trends_by_year
        df = damages_trends_by_year(engine)
        assert not df.empty
        assert "year" in df.columns
        assert "mean_damages" in df.columns
        assert "median_damages" in df.columns

    def test_damages_trends_filter_by_circuit(self, engine):
        from trends import damages_trends_by_year
        df = damages_trends_by_year(engine, circuit="9th")
        assert not df.empty


class TestJudgeStats:
    def test_judge_stats_returns_dataframe(self, engine):
        from trends import judge_stats
        df = judge_stats(engine, min_opinions=2)
        assert not df.empty
        assert "judge" in df.columns
        assert "opinion_count" in df.columns
        assert "plaintiff_win_rate" in df.columns

    def test_judge_stats_filters_minimum(self, engine):
        from trends import judge_stats
        df = judge_stats(engine, min_opinions=100)
        assert df.empty


class TestCircuitComparison:
    def test_circuit_comparison_returns_dataframe(self, engine):
        from trends import circuit_comparison
        df = circuit_comparison(engine)
        assert not df.empty
        assert "circuit" in df.columns
        assert "opinion_count" in df.columns
        assert "plaintiff_win_rate" in df.columns

    def test_circuit_outcome_heatmap_returns_dataframe(self, engine):
        from trends import circuit_outcome_heatmap
        df = circuit_outcome_heatmap(engine)
        assert not df.empty
        assert "circuit" in df.columns
        assert "statute" in df.columns
        assert "plaintiff_win_rate" in df.columns


class TestDefenseAnalysis:
    def test_defense_frequency(self, engine):
        from trends import defense_frequency
        df = defense_frequency(engine)
        assert not df.empty
        assert "defense_type" in df.columns
        assert "count" in df.columns

    def test_defense_effectiveness(self, engine):
        from trends import defense_effectiveness
        df = defense_effectiveness(engine, min_count=1)
        assert not df.empty
        assert "defense_type" in df.columns
        assert "times_raised" in df.columns
        assert "defendant_win_rate_when_raised" in df.columns

    def test_defense_by_circuit(self, engine):
        from trends import defense_by_circuit
        df = defense_by_circuit(engine)
        assert not df.empty
        assert "defense_type" in df.columns
        assert "circuit" in df.columns


class TestStatisticalFunctions:
    def test_compare_outcome_rates_significant(self):
        from trends import compare_outcome_rates
        result = compare_outcome_rates(0.8, 100, 0.3, 100)
        assert "p_value" in result
        assert "significant" in result
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_compare_outcome_rates_not_significant(self):
        from trends import compare_outcome_rates
        result = compare_outcome_rates(0.5, 10, 0.55, 10)
        assert result["significant"] is False

    def test_compare_outcome_rates_badges(self):
        from trends import compare_outcome_rates
        result = compare_outcome_rates(0.9, 500, 0.1, 500)
        assert "badge" in result
        assert result["badge"] in ("*", "**", "***")

    def test_compare_damages_returns_result(self):
        from trends import compare_damages
        group1 = [1000, 2000, 3000, 4000, 5000]
        group2 = [10000, 20000, 30000, 40000, 50000]
        result = compare_damages(group1, group2)
        assert "p_value" in result
        assert "significant" in result

    def test_compare_damages_empty_group(self):
        from trends import compare_damages
        result = compare_damages([], [1000, 2000])
        assert result["significant"] is False


class TestClaimFrequency:
    def test_claim_frequency_returns_dataframe(self, engine):
        from trends import claim_frequency_by_year
        # Add some claim type predictions
        session = get_session(engine)
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            "VALUES (1, 'claim_type_logreg_v1', 'claim_type', '1692e', 0.8)"
        ))
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            "VALUES (3, 'claim_type_logreg_v1', 'claim_type', '1692e', 0.7)"
        ))
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, predicted_value, confidence) "
            "VALUES (5, 'claim_type_logreg_v1', 'claim_type', '1692g', 0.9)"
        ))
        session.commit()
        session.close()

        df = claim_frequency_by_year(engine)
        assert not df.empty
        assert "year" in df.columns
        assert "claim_section" in df.columns
        assert "count" in df.columns
