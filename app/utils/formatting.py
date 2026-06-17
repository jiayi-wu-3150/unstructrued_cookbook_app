import streamlit as st


AUDIO_TYPE_COLORS = {
    "speech":       "blue",
    "music":        "violet",
    "sound_effect": "orange",
    "ambient":      "green",
    "mixed":        "red",
    "unknown":      "gray",
}


def confidence_color(score: float) -> str:
    if score is None:
        return "gray"
    if score >= 0.9:
        return "green"
    if score >= 0.8:
        return "orange"
    return "red"


def recipe_header(icon: str, title: str, subtitle: str, badge_label: str, badge_value: str):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"{icon} {title}")
        st.caption(subtitle)
    with col2:
        st.metric(badge_label, badge_value)


def code_tab_content(snippet: str, language: str = "sql"):
    st.code(snippet, language=language)


def requirements_tab_content(permissions: list[str], resources: list[str], dependencies: list[str]):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Permissions**")
        for p in permissions:
            st.markdown(f"- {p}")
    with c2:
        st.markdown("**Databricks Resources**")
        for r in resources:
            st.markdown(f"- {r}")
    with c3:
        st.markdown("**Dependencies**")
        for d in dependencies:
            st.markdown(f"- {d}")


def show_endpoint_health(endpoint_names: list[str]):
    """Display live health status for a list of serving endpoints."""
    from utils.db import get_endpoint_statuses
    with st.spinner("Checking endpoint health..."):
        statuses = get_endpoint_statuses(tuple(endpoint_names))
    for name, (ready, state) in statuses.items():
        icon = "🟢" if ready else "🔴"
        st.markdown(f"{icon} `{name}` — **{state}**")
