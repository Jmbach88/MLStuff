import streamlit as st

st.set_page_config(
    page_title="Federal Opinion Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Federal Opinion Intelligence Platform")
st.markdown(
    "Search and analyze ~30,000 federal court opinions across FDCPA, TCPA, and FCRA."
)
st.markdown("Use the sidebar to navigate between tools.")
