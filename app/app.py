import importlib
import streamlit as st

st.set_page_config(
    page_title="Unstructured Cookbook",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RECIPES = {
    "🏠 Home":                      "views.home",
    "📄 PDF — W2 Forms":            "views.recipe_01_w2",
    "📑 PDF — Research Papers":     "views.recipe_02_research",
    "📊 PowerPoint (PPTX)":         "views.recipe_03_pptx",
    "🎤 Audio — Speech":            "views.recipe_04_audio",
    "🔊 Audio — Non-Speech":        "views.recipe_05_nonspeech",
    "🎬 Video (MP4 + VS Search)":   "views.recipe_06_video",
}

with st.sidebar:
    st.markdown("## 🗂️ Unstructured Cookbook")
    st.caption("Any file type → Databricks Vector Search")
    st.divider()

    selection = st.radio(
        "Choose a recipe:",
        list(RECIPES.keys()),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Universal output schema**")
    st.code(
        "chunk_id\nchunk_to_embed\nchunk_to_retrieve\nsource_path\nchunk_index\nmetadata",
        language="text",
    )

module = importlib.import_module(RECIPES[selection])
module.render()
