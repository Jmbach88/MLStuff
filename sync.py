# sync.py
import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import config
from db import (
    Base, Opinion, Statute, OpinionStatute, FDCPASection,
    init_local_db, get_local_engine
)

logger = logging.getLogger(__name__)


def sync_from_source(source_url=None, local_engine=None):
    if source_url is None:
        source_url = f"sqlite:///{config.SOURCE_DB}"
    if local_engine is None:
        local_engine = get_local_engine()
        init_local_db(local_engine)

    source_engine = create_engine(source_url)
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0

    local_session = Session(local_engine)
    try:
        _sync_statutes(source_engine, local_session)

        existing_ids = set(
            row[0] for row in local_session.execute(
                text("SELECT package_id FROM opinions")
            ).fetchall()
        )

        with Session(source_engine) as source_session:
            rows = source_session.execute(text("""
                SELECT id, package_id, title, court_name, court_type, circuit,
                       date_issued, plain_text, pdf_url
                FROM opinions
                WHERE plain_text IS NOT NULL AND plain_text != ''
            """)).fetchall()

            for row in rows:
                src_id, package_id, title, court_name, court_type, circuit, \
                    date_issued, plain_text, pdf_url = row

                if package_id in existing_ids:
                    continue

                opinion = Opinion(
                    id=src_id,
                    package_id=package_id,
                    title=title,
                    court_name=court_name,
                    court_type=court_type,
                    circuit=circuit,
                    date_issued=date_issued,
                    plain_text=plain_text,
                    pdf_url=pdf_url,
                    synced_at=now,
                    chunked=0,
                )
                local_session.add(opinion)
                new_count += 1

                links = source_session.execute(text(
                    "SELECT statute_id FROM opinion_statutes WHERE opinion_id = :oid"
                ), {"oid": src_id}).fetchall()
                for (statute_id,) in links:
                    local_session.add(OpinionStatute(
                        opinion_id=src_id, statute_id=statute_id
                    ))

                sections = source_session.execute(text(
                    "SELECT subsection, description FROM fdcpa_sections WHERE opinion_id = :oid"
                ), {"oid": src_id}).fetchall()
                for subsection, description in sections:
                    local_session.add(FDCPASection(
                        opinion_id=src_id,
                        subsection=subsection,
                        description=description,
                    ))

        local_session.commit()
        logger.info(f"Synced {new_count} new opinions")

    except Exception:
        local_session.rollback()
        raise
    finally:
        local_session.close()

    return new_count


def _sync_statutes(source_engine, local_session):
    existing = set(
        row[0] for row in local_session.execute(
            text("SELECT key FROM statutes")
        ).fetchall()
    )
    if existing:
        return

    with Session(source_engine) as source_session:
        rows = source_session.execute(
            text("SELECT id, key, name FROM statutes")
        ).fetchall()
        for sid, key, name in rows:
            local_session.add(Statute(id=sid, key=key, name=name))
    local_session.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = get_local_engine()
    init_local_db(engine)
    count = sync_from_source(local_engine=engine)
    print(f"Synced {count} opinions")
