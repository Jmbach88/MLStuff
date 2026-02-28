import os
import pytest
from sqlalchemy import inspect

os.environ["ML_LOCAL_DB"] = ":memory:"


def test_init_db_creates_all_tables():
    from db import init_local_db, get_local_engine
    engine = get_local_engine()
    init_local_db(engine)
    table_names = inspect(engine).get_table_names()
    assert "opinions" in table_names
    assert "statutes" in table_names
    assert "opinion_statutes" in table_names
    assert "fdcpa_sections" in table_names


def test_opinion_columns_exist():
    from db import init_local_db, get_local_engine
    engine = get_local_engine()
    init_local_db(engine)
    cols = [c["name"] for c in inspect(engine).get_columns("opinions")]
    for expected in ["id", "package_id", "title", "court_name", "court_type",
                     "circuit", "date_issued", "plain_text", "pdf_url",
                     "synced_at", "chunked"]:
        assert expected in cols, f"Missing column: {expected}"
