import os
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, Text, ForeignKey, UniqueConstraint,
    create_engine, inspect
)
from sqlalchemy.orm import declarative_base, relationship, Session

import config

logger = logging.getLogger(__name__)

Base = declarative_base()


class Statute(Base):
    __tablename__ = "statutes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)
    opinions = relationship("Opinion", secondary="opinion_statutes", back_populates="statutes")


class Opinion(Base):
    __tablename__ = "opinions"
    id = Column(Integer, primary_key=True)
    package_id = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    court_name = Column(Text)
    court_type = Column(Text)
    circuit = Column(Text)
    date_issued = Column(Text)
    plain_text = Column(Text)
    pdf_url = Column(Text)
    synced_at = Column(Text)
    chunked = Column(Integer, default=0)
    statutes = relationship("Statute", secondary="opinion_statutes", back_populates="opinions")
    sections = relationship("FDCPASection", back_populates="opinion")


class OpinionStatute(Base):
    __tablename__ = "opinion_statutes"
    opinion_id = Column(Integer, ForeignKey("opinions.id"), primary_key=True)
    statute_id = Column(Integer, ForeignKey("statutes.id"), primary_key=True)


class FDCPASection(Base):
    __tablename__ = "fdcpa_sections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    subsection = Column(Text, nullable=False)
    description = Column(Text)
    opinion = relationship("Opinion", back_populates="sections")
    __table_args__ = (
        UniqueConstraint("opinion_id", "subsection", name="uq_opinion_subsection"),
    )


class Label(Base):
    __tablename__ = "labels"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    label_type = Column(Text, nullable=False)
    label_value = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    confidence = Column(Float)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    model_name = Column(Text, nullable=False)
    label_type = Column(Text, nullable=False)
    predicted_value = Column(Text, nullable=False)
    confidence = Column(Float)
    created_at = Column(Text)


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    label_type = Column(Text, nullable=False)
    accuracy = Column(Float)
    f1_score = Column(Float)
    trained_at = Column(Text)
    params_json = Column(Text)


class Citation(Base):
    __tablename__ = "citations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    citing_opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    cited_opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=True)
    volume = Column(Text)
    reporter = Column(Text)
    page = Column(Text)
    citation_string = Column(Text, nullable=False)
    context_snippet = Column(Text)
    __table_args__ = (
        UniqueConstraint("citing_opinion_id", "volume", "reporter", "page",
                         name="uq_citation_dedup"),
    )


class OpinionMetric(Base):
    __tablename__ = "opinion_metrics"
    opinion_id = Column(Integer, ForeignKey("opinions.id"), primary_key=True)
    in_degree = Column(Integer, default=0)
    out_degree = Column(Integer, default=0)
    pagerank = Column(Float)
    hub_score = Column(Float)
    authority_score = Column(Float)
    community_id = Column(Integer)


class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    entity_type = Column(Text, nullable=False)
    entity_value = Column(Text, nullable=False)
    context_snippet = Column(Text)
    start_char = Column(Integer)
    end_char = Column(Integer)


def get_local_engine():
    db_path = os.environ.get("ML_LOCAL_DB", config.LOCAL_DB)
    if db_path == ":memory:":
        return create_engine("sqlite:///:memory:")
    return create_engine(f"sqlite:///{db_path}")


def init_local_db(engine=None):
    if engine is None:
        engine = get_local_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    if engine is None:
        engine = get_local_engine()
    return Session(engine)
