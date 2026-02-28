import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

os.environ["ML_LOCAL_DB"] = ":memory:"


@pytest.fixture
def source_engine(tmp_path):
    db_path = str(tmp_path / "source.db")
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE statutes (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE opinions (
                id INTEGER PRIMARY KEY,
                package_id TEXT UNIQUE NOT NULL,
                granule_id TEXT,
                title TEXT NOT NULL,
                court_name TEXT,
                court_type TEXT,
                circuit TEXT,
                date_issued TEXT,
                last_modified TEXT,
                pdf_url TEXT,
                plain_text TEXT,
                xml_url TEXT,
                govinfo_url TEXT,
                created_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE opinion_statutes (
                opinion_id INTEGER,
                statute_id INTEGER,
                alerted_at TEXT,
                citation_count INTEGER,
                relevance TEXT,
                review_status TEXT,
                reviewed_at TEXT,
                scanned_at TEXT,
                PRIMARY KEY (opinion_id, statute_id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE fdcpa_sections (
                id INTEGER PRIMARY KEY,
                opinion_id INTEGER NOT NULL,
                subsection TEXT NOT NULL,
                description TEXT,
                UNIQUE(opinion_id, subsection)
            )
        """))
        conn.execute(text("""
            INSERT INTO statutes VALUES (1, 'tcpa', 'Telephone Consumer Protection Act'),
                                        (2, 'fdcpa', 'Fair Debt Collection Practices Act'),
                                        (3, 'fcra', 'Fair Credit Reporting Act')
        """))
        conn.execute(text("""
            INSERT INTO opinions (id, package_id, title, court_name, court_type, circuit,
                                  date_issued, plain_text, created_at)
            VALUES (1, 'PKG-001', 'Smith v. Collections Inc', 'US District Court SD Texas',
                    'district', '5th', '2024-01-15', 'This is the opinion text about FDCPA violations.',
                    '2024-01-16T00:00:00Z'),
                   (2, 'PKG-002', 'Jones v. Bank Corp', 'US Court of Appeals 11th Circuit',
                    'circuit', '11th', '2024-03-20', 'This opinion discusses TCPA and FDCPA claims.',
                    '2024-03-21T00:00:00Z'),
                   (3, 'PKG-003', 'No Text Case', 'US District Court ND Ohio',
                    'district', '6th', '2024-05-01', NULL,
                    '2024-05-02T00:00:00Z')
        """))
        conn.execute(text("""
            INSERT INTO opinion_statutes (opinion_id, statute_id)
            VALUES (1, 2), (2, 1), (2, 2), (3, 2)
        """))
        conn.execute(text("""
            INSERT INTO fdcpa_sections (opinion_id, subsection, description)
            VALUES (1, '1692e', 'False representations')
        """))
    return engine


def test_sync_copies_opinions_with_text(source_engine):
    from db import init_local_db, get_session, Opinion
    from sync import sync_from_source

    local_engine = init_local_db()
    source_url = str(source_engine.url)
    count = sync_from_source(source_url, local_engine)

    session = get_session(local_engine)
    opinions = session.query(Opinion).all()
    assert len(opinions) == 2
    assert count == 2
    assert all(o.chunked == 0 for o in opinions)
    session.close()


def test_sync_copies_statutes_and_links(source_engine):
    from db import init_local_db, get_session, Opinion, Statute
    from sync import sync_from_source

    local_engine = init_local_db()
    source_url = str(source_engine.url)
    sync_from_source(source_url, local_engine)

    session = get_session(local_engine)
    statutes = session.query(Statute).all()
    assert len(statutes) == 3

    op2 = session.query(Opinion).filter_by(package_id="PKG-002").first()
    statute_keys = sorted([s.key for s in op2.statutes])
    assert statute_keys == ["fdcpa", "tcpa"]
    session.close()


def test_sync_incremental_skips_existing(source_engine):
    from db import init_local_db
    from sync import sync_from_source

    local_engine = init_local_db()
    source_url = str(source_engine.url)

    count1 = sync_from_source(source_url, local_engine)
    assert count1 == 2

    count2 = sync_from_source(source_url, local_engine)
    assert count2 == 0
