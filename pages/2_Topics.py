import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

import config
from db import get_local_engine, init_local_db, get_session

st.set_page_config(page_title="Topics", page_icon="⚖️", layout="wide")
st.title("Topic Modeling")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


@st.cache_data
def load_topic_assignments(_engine):
    """Load all topic predictions joined with opinion metadata."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT p.opinion_id, p.predicted_value, p.confidence, "
        "o.title, o.circuit, o.court_type, o.date_issued "
        "FROM predictions p "
        "JOIN opinions o ON p.opinion_id = o.id "
        "WHERE p.label_type = 'topic' AND p.model_name = 'bertopic_v1'"
    )).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "opinion_id", "topic", "confidence",
        "title", "circuit", "court_type", "date_issued",
    ])
    df["topic"] = df["topic"].astype(int)
    return df


@st.cache_data
def load_top_topics(_engine):
    """Load top_topics from the bertopic_v1 model record."""
    session = get_session(_engine)
    row = session.execute(text(
        "SELECT params_json FROM models WHERE name = 'bertopic_v1'"
    )).fetchone()
    session.close()
    if row is None or row[0] is None:
        return []
    params = json.loads(row[0])
    return params.get("top_topics", [])


@st.cache_data
def load_opinion_statutes(_engine):
    """Map opinion_id -> statute key for filtering."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT os.opinion_id, UPPER(s.key) "
        "FROM opinion_statutes os "
        "JOIN statutes s ON os.statute_id = s.id"
    )).fetchall()
    session.close()
    return pd.DataFrame(rows, columns=["opinion_id", "statute"])


@st.cache_data
def load_coords():
    """Load 2-D UMAP coordinates from npz file."""
    path = os.path.join(config.PROJECT_ROOT, "data", "models", "topic_coords_v1.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["opinion_ids"], data["coords"]


engine = get_db_engine()

# --- Load data ---
df_all = load_topic_assignments(engine)
if df_all.empty:
    st.warning("No topic assignments found. Run `python topics.py` to fit topics.")
    st.stop()

top_topics = load_top_topics(engine)
df_statutes = load_opinion_statutes(engine)

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    statute_options = ["All", "FDCPA", "TCPA", "FCRA"]
    selected_statute = st.selectbox("Statute", statute_options)

    circuits = sorted(
        df_all.loc[df_all["circuit"].notna() & (df_all["circuit"] != ""), "circuit"].unique()
    )
    selected_circuit = st.selectbox("Circuit", ["All"] + list(circuits))

    court_types = ["All", "district", "circuit"]
    selected_court_type = st.selectbox("Court Type", court_types)

# --- Apply filters ---
df = df_all.copy()

if selected_statute != "All":
    valid_ids = df_statutes.loc[
        df_statutes["statute"] == selected_statute.upper(), "opinion_id"
    ]
    df = df[df["opinion_id"].isin(valid_ids)]

if selected_circuit != "All":
    df = df[df["circuit"] == selected_circuit]

if selected_court_type != "All":
    df = df[df["court_type"] == selected_court_type]

if df.empty:
    st.info("No topics match the current filters.")
    st.stop()

# --- Summary Metrics ---
total_assigned = len(df[df["topic"] != -1])
total_outliers = len(df[df["topic"] == -1])
n_topics = df.loc[df["topic"] != -1, "topic"].nunique()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Topics Found", f"{n_topics:,}")
with col2:
    st.metric("Opinions Assigned", f"{total_assigned:,}")
with col3:
    st.metric("Outliers", f"{total_outliers:,}")

# --- Build topic name lookup from top_topics ---
topic_name_map = {}
for entry in top_topics:
    tid = entry.get("Topic")
    name = entry.get("Name", "")
    if tid is not None:
        topic_name_map[int(tid)] = name

# --- Section 1: Topic Overview Table ---
st.subheader("Topic Overview")

df_non_outlier = df[df["topic"] != -1]
topic_counts = df_non_outlier["topic"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]
total_non_outlier = topic_counts["Count"].sum()
topic_counts["Percentage"] = (topic_counts["Count"] / total_non_outlier * 100).round(1)

# Extract top words from topic name (BERTopic names are like "0_word1_word2_word3")
def extract_top_words(topic_id):
    name = topic_name_map.get(topic_id, "")
    if not name:
        return ""
    # BERTopic format: "0_word1_word2_word3_..."
    parts = name.split("_")
    if len(parts) > 1:
        return ", ".join(parts[1:])
    return name

topic_counts["Top Words"] = topic_counts["Topic"].apply(extract_top_words)
topic_counts = topic_counts.sort_values("Count", ascending=False).reset_index(drop=True)
topic_counts["Percentage"] = topic_counts["Percentage"].apply(lambda x: f"{x}%")

st.dataframe(topic_counts, use_container_width=True, hide_index=True)

# --- Section 2: Topic Size Bar Chart ---
st.subheader("Top 20 Topics by Opinion Count")

top20 = topic_counts.head(20).copy()
top20["Percentage"] = top20["Percentage"].str.rstrip("%").astype(float)
top20["Label"] = top20["Topic"].apply(
    lambda t: f"Topic {t}: {extract_top_words(t)[:40]}"
)
top20 = top20.sort_values("Count", ascending=True)

fig_bar = px.bar(
    top20, x="Count", y="Label", orientation="h",
    labels={"Count": "Number of Opinions", "Label": ""},
    color="Count", color_continuous_scale="Blues",
)
fig_bar.update_layout(
    height=max(400, len(top20) * 28),
    showlegend=False,
    coloraxis_showscale=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Section 3: Interactive Scatter Plot ---
st.subheader("Topic Scatter Plot (UMAP 2-D)")

opinion_ids_coords, coords = load_coords()

if opinion_ids_coords is not None and coords is not None:
    df_coords = pd.DataFrame({
        "opinion_id": opinion_ids_coords,
        "x": coords[:, 0],
        "y": coords[:, 1],
    })
    # Merge with topic assignments (filtered)
    df_scatter = df_coords.merge(
        df[["opinion_id", "topic", "title"]],
        on="opinion_id", how="inner",
    )
    if not df_scatter.empty:
        df_scatter["topic_label"] = df_scatter["topic"].apply(
            lambda t: "Outlier" if t == -1 else f"Topic {t}"
        )
        # Sort so outliers render behind real topics
        df_scatter = df_scatter.sort_values("topic", ascending=False)

        fig_scatter = px.scatter(
            df_scatter, x="x", y="y",
            color="topic_label",
            hover_data={"title": True, "topic_label": True, "x": False, "y": False},
            labels={"topic_label": "Topic"},
            opacity=0.6,
        )
        fig_scatter.update_layout(
            height=600,
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No scatter data available for the current filters.")
else:
    st.info(
        "UMAP coordinates not found. Run `python topics.py` to generate "
        "`data/models/topic_coords_v1.npz`."
    )

# --- Section 4: Topic Evolution Over Time ---
st.subheader("Topic Evolution Over Time")

df_time = df[df["topic"] != -1].copy()
df_time["year"] = pd.to_datetime(df_time["date_issued"], errors="coerce").dt.year
df_time = df_time.dropna(subset=["year"])
df_time["year"] = df_time["year"].astype(int)

# Identify top 10 topics by total count
top10_topics = (
    df_time["topic"].value_counts().head(10).index.tolist()
)
df_time = df_time[df_time["topic"].isin(top10_topics)]

evolution = df_time.groupby(["year", "topic"]).size().reset_index(name="count")
evolution["topic_label"] = evolution["topic"].apply(lambda t: f"Topic {t}")

fig_line = px.line(
    evolution, x="year", y="count", color="topic_label",
    labels={"count": "Opinions", "year": "Year", "topic_label": "Topic"},
    markers=True,
)
fig_line.update_layout(height=500)
st.plotly_chart(fig_line, use_container_width=True)
