import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Citation


class TestCitationExtraction:
    """Tests for extract_citations_from_text()."""

    def test_extracts_f2d_citation(self):
        from citations import extract_citations_from_text
        text = "The court relied on Smith v. Jones, 123 F.2d 456 (2d Cir. 1990)."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "123"
        assert cites[0]["reporter"] == "F.2d"
        assert cites[0]["page"] == "456"

    def test_extracts_f3d_citation(self):
        from citations import extract_citations_from_text
        text = "See also Doe v. Roe, 789 F.3d 101 (9th Cir. 2015)."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "789"
        assert cites[0]["reporter"] == "F.3d"
        assert cites[0]["page"] == "101"

    def test_extracts_f4th_citation(self):
        from citations import extract_citations_from_text
        text = "In Brown v. Green, 45 F.4th 678 (5th Cir. 2022), the court held..."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "45"
        assert cites[0]["reporter"] == "F.4th"
        assert cites[0]["page"] == "678"

    def test_extracts_f_first_series(self):
        from citations import extract_citations_from_text
        text = "The seminal case is 100 F. 200."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["reporter"] == "F."

    def test_extracts_f_supp_citations(self):
        from citations import extract_citations_from_text
        text = "See 100 F. Supp. 200 and 300 F. Supp. 2d 400 and 500 F. Supp. 3d 600."
        cites = extract_citations_from_text(text)
        assert len(cites) == 3
        reporters = {c["reporter"] for c in cites}
        assert reporters == {"F. Supp.", "F. Supp. 2d", "F. Supp. 3d"}

    def test_extracts_us_reports(self):
        from citations import extract_citations_from_text
        text = "The Supreme Court in 410 U.S. 113 held that..."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["reporter"] == "U.S."
        assert cites[0]["volume"] == "410"
        assert cites[0]["page"] == "113"

    def test_extracts_s_ct(self):
        from citations import extract_citations_from_text
        text = "See 123 S. Ct. 456 and 789 S.Ct. 101."
        cites = extract_citations_from_text(text)
        assert len(cites) == 2

    def test_extracts_westlaw_cite(self):
        from citations import extract_citations_from_text
        text = "See Smith v. Jones, 2024 WL 1234567 (S.D.N.Y. 2024)."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "2024"
        assert cites[0]["reporter"] == "WL"
        assert cites[0]["page"] == "1234567"

    def test_extracts_lexis_cites(self):
        from citations import extract_citations_from_text
        text = "See 2023 U.S. Dist. LEXIS 12345 and 2022 U.S. App. LEXIS 67890."
        cites = extract_citations_from_text(text)
        assert len(cites) == 2
        reporters = {c["reporter"] for c in cites}
        assert "U.S. Dist. LEXIS" in reporters
        assert "U.S. App. LEXIS" in reporters

    def test_captures_context_snippet(self):
        from citations import extract_citations_from_text
        text = "X" * 200 + "cited 123 F.2d 456 here" + "Y" * 200
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert len(cites[0]["context_snippet"]) <= 250
        assert "123 F.2d 456" in cites[0]["context_snippet"]

    def test_deduplicates_same_citation(self):
        from citations import extract_citations_from_text
        text = "See 123 F.2d 456. The court in 123 F.2d 456 also held..."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1

    def test_multiple_different_citations(self):
        from citations import extract_citations_from_text
        text = "See 123 F.2d 456 and 789 F.3d 101."
        cites = extract_citations_from_text(text)
        assert len(cites) == 2

    def test_empty_text_returns_empty(self):
        from citations import extract_citations_from_text
        assert extract_citations_from_text("") == []
        assert extract_citations_from_text(None) == []


class TestCitationDedup:
    def test_db_dedup_constraint(self):
        from citations import store_citations
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM citations"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()
        session.add(Opinion(id=1, package_id="pkg_1", title="Case 1"))
        session.commit()
        session.close()

        citations = [
            {"volume": "123", "reporter": "F.2d", "page": "456",
             "citation_string": "123 F.2d 456", "context_snippet": "context"},
        ]
        store_citations(engine, 1, citations)

        session = get_session(engine)
        count = session.query(Citation).filter_by(citing_opinion_id=1).count()
        session.close()
        assert count == 1


class TestCitationResolution:
    def test_build_citation_index_from_titles(self):
        from citations import build_citation_index
        opinions = [
            (1, "Smith v. Jones, 123 F.2d 456 (2d Cir. 1990)"),
            (2, "Doe v. Roe, 789 F.3d 101 (9th Cir. 2015)"),
            (3, "No citation in this title"),
        ]
        index = build_citation_index(opinions)
        assert index[("123", "F.2d", "456")] == 1
        assert index[("789", "F.3d", "101")] == 2
        assert len(index) == 2

    def test_resolve_citations_links_to_corpus(self):
        from citations import resolve_citations
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM citations"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()
        session.add(Opinion(id=1, package_id="pkg_1", title="Case A"))
        session.add(Opinion(id=2, package_id="pkg_2",
                            title="Case B, 500 F.3d 100 (3d Cir. 2007)"))
        session.commit()
        session.add(Citation(
            citing_opinion_id=1,
            volume="500", reporter="F.3d", page="100",
            citation_string="500 F.3d 100", context_snippet="context",
        ))
        session.commit()
        session.close()

        resolved = resolve_citations(engine)
        assert resolved > 0

        session = get_session(engine)
        cite = session.query(Citation).filter_by(citing_opinion_id=1).first()
        assert cite.cited_opinion_id == 2
        session.close()

    def test_unresolved_stays_null(self):
        from citations import resolve_citations
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM citations"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()
        session.add(Opinion(id=1, package_id="pkg_1", title="Case A"))
        session.commit()
        session.add(Citation(
            citing_opinion_id=1,
            volume="999", reporter="F.2d", page="888",
            citation_string="999 F.2d 888", context_snippet="context",
        ))
        session.commit()
        session.close()

        resolve_citations(engine)

        session = get_session(engine)
        cite = session.query(Citation).filter_by(citing_opinion_id=1).first()
        assert cite.cited_opinion_id is None
        session.close()


class TestGraphMetrics:
    def test_build_graph_from_citations(self):
        from citations import build_citation_graph
        edges = [(1, 2), (1, 3), (2, 3), (4, 1)]
        G = build_citation_graph(edges)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4
        assert G.is_directed()

    def test_compute_metrics(self):
        from citations import build_citation_graph, compute_graph_metrics
        edges = [(1, 2), (1, 3), (2, 3), (4, 1), (4, 2)]
        G = build_citation_graph(edges)
        metrics = compute_graph_metrics(G)
        assert metrics[2]["in_degree"] == 2
        assert metrics[3]["in_degree"] == 2
        assert metrics[1]["out_degree"] == 2
        for node_id in [1, 2, 3, 4]:
            assert "pagerank" in metrics[node_id]
            assert "hub_score" in metrics[node_id]
            assert "authority_score" in metrics[node_id]
            assert "community_id" in metrics[node_id]

    def test_store_and_load_metrics(self):
        from citations import store_metrics
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM opinion_metrics"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()
        for oid in [1, 2, 3]:
            session.add(Opinion(id=oid, package_id=f"pkg_{oid}", title=f"Case {oid}"))
        session.commit()
        session.close()

        metrics = {
            1: {"in_degree": 2, "out_degree": 1, "pagerank": 0.4,
                "hub_score": 0.1, "authority_score": 0.5, "community_id": 0},
            2: {"in_degree": 1, "out_degree": 2, "pagerank": 0.3,
                "hub_score": 0.6, "authority_score": 0.2, "community_id": 0},
            3: {"in_degree": 0, "out_degree": 0, "pagerank": 0.1,
                "hub_score": 0.0, "authority_score": 0.0, "community_id": 1},
        }
        store_metrics(engine, metrics)

        session = get_session(engine)
        from db import OpinionMetric
        m = session.query(OpinionMetric).filter_by(opinion_id=1).first()
        assert m.in_degree == 2
        assert abs(m.pagerank - 0.4) < 1e-6
        assert m.community_id == 0
        session.close()
