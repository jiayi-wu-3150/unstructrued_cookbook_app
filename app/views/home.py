import streamlit as st
from utils.db import query, CATALOG, SCHEMA


# --- Coverage Map Definition ---
# Each cell: {"label": str, "active": bool, "recipe": str (sidebar key) or None}
# None = empty cell
COLUMN_HEADERS = [
    "txt / md", "HTML", "JSON /\nJSONL", "PDF\n(text-heavy)",
    "PDF\n(with images)", "PDF\n(tables)", "PPTX",
    "Audios\n(speech)", "Audios\n(nonspeech)", "Videos\n(short)", "Videos\n(long)",
]

GRID_ROWS = [
    # Row 1: Specific variations
    [
        None, None, None, None,
        {"label": "Research\nPapers", "active": True, "recipe": "📑 PDF — Research Papers"},
        {"label": "W2", "active": True, "recipe": "📄 PDF — W2 Forms"},
        {"label": "General\nPPTX", "active": True, "recipe": "📊 PowerPoint (PPTX)"},
        {"label": "Meeting\nRecordings", "active": True, "recipe": "🎤 Audio — Speech"},
        {"label": "Sound\nEffect", "active": True, "recipe": "🔊 Audio — Non-Speech"},
        {"label": "Movie\nClips", "active": True, "recipe": "🎬 Video (MP4 + VS Search)"},
        {"label": "Meetings", "active": False},
    ],
    # Row 2: Deeper variations
    [
        None, None, None, None,
        {"label": "Loan\nMortgage", "active": False},
        None, None,
        {"label": "Long Audios\n(>8 min)", "active": False},
        None, None, None,
    ],
    # Row 3: Even deeper
    [
        None, None, None, None,
        {"label": "CAD\nDrawings", "active": False},
        None, None,
        {"label": "Audios w/\ndiarization", "active": False},
        None, None, None,
    ],
]

# Custom CSS for uniform grid buttons
GRID_CSS = """
<style>
[data-testid="stHorizontalBlock"] [data-testid="stButton"] {
    height: 72px !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button {
    height: 72px !important;
    max-height: 72px !important;
    min-height: 72px !important;
    font-size: 0.7rem !important;
    white-space: normal !important;
    line-height: 1.2 !important;
    padding: 6px 4px !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button[kind="primary"] {
    background-color: #ef5350 !important;
    border-color: #ef5350 !important;
    color: white !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button[kind="secondary"] {
    background-color: #e3f2fd !important;
    border-color: #1976d2 !important;
    color: #1565c0 !important;
    font-weight: 600 !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button:disabled {
    background-color: #fafafa !important;
    border-color: #e0e0e0 !important;
    color: #bdbdbd !important;
    opacity: 1 !important;
}
</style>
"""


def render():
    st.title("🗂️ Unstructured Cookbook")
    st.markdown(
        "Code examples for ingesting **unstructured files** into Databricks — "
        "per file type per variation, ready for agents, analytics, or any downstream workflow."
    )

    st.divider()

    # --- Coverage Map ---
    st.subheader("Coverage Map")
    st.caption("Click any highlighted block to jump to that recipe. Gray blocks are coming soon.")
    st.markdown(GRID_CSS, unsafe_allow_html=True)

    grid_container = st.container()
    grid_container.markdown('<div class="grid-map">', unsafe_allow_html=True)

    with grid_container:
        # Axis label
        st.markdown(
            '<p style="font-size:0.72rem; color:#666; text-align:right; margin-bottom:2px;">'
            'Broader (Text to Multimodal) →</p>',
            unsafe_allow_html=True,
        )

        # Row 0: Column headers (base categories) — always red
        header_cols = st.columns(len(COLUMN_HEADERS))
        for i, hdr in enumerate(COLUMN_HEADERS):
            with header_cols[i]:
                label = hdr.replace("\n", " ")
                st.button(label, key=f"hdr_{i}", type="primary", use_container_width=True)

        # Y-axis label
        st.markdown(
            '<p style="font-size:0.72rem; color:#666; margin:6px 0 2px;">'
            '↓ Deeper (More customized and industrialized)</p>',
            unsafe_allow_html=True,
        )

        # Variation rows
        for row_idx, row in enumerate(GRID_ROWS):
            row_cols = st.columns(len(COLUMN_HEADERS))
            for col_idx, cell in enumerate(row):
                with row_cols[col_idx]:
                    if cell is None:
                        # Invisible spacer to maintain grid
                        st.button("\u00A0", key=f"empty_{row_idx}_{col_idx}", disabled=True, use_container_width=True)
                    elif cell["active"]:
                        label = cell["label"].replace("\n", " ")
                        if st.button(label, key=f"cell_{row_idx}_{col_idx}", type="secondary", use_container_width=True):
                            st.session_state["navigate_to"] = cell["recipe"]
                            st.rerun()
                    else:
                        label = cell["label"].replace("\n", " ")
                        st.button(label, key=f"cell_{row_idx}_{col_idx}", disabled=True, use_container_width=True)

    grid_container.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- Active Recipes Summary ---
    st.subheader("Active Recipes")

    RECIPES = [
        {
            "icon": "📄", "title": "PDF — W2 Forms", "table": "w2_parsed",
            "parser": "ai_parse_document",
            "description": "Structured field/value extraction from tax forms",
        },
        {
            "icon": "📑", "title": "PDF — Research Papers", "table": "research_parsed",
            "parser": "ai_parse_document + figure descriptions",
            "description": "Text + AI-written figure descriptions",
        },
        {
            "icon": "📊", "title": "PowerPoint (PPTX)", "table": "pptx_chunks",
            "parser": "ai_parse_document",
            "description": "One chunk per slide — titles, bullets, and image captions",
        },
        {
            "icon": "🎤", "title": "Audio — Speech", "table": "voice_celebrities_chunks",
            "parser": "Whisper Large v3",
            "description": "Word-window chunked transcripts with speaker context",
        },
        {
            "icon": "🔊", "title": "Audio — Non-Speech", "table": "sound_chunks",
            "parser": "Gemini 2.5 Flash",
            "description": "Classify & describe sound effects, music, ambient audio",
        },
        {
            "icon": "🎬", "title": "Video (MP4)", "table": "video_clips_gold",
            "parser": "CLIP + Gemini 2.5 Flash",
            "description": "Frame embeddings (768-dim) + Gemini descriptions",
        },
    ]

    # Live row counts
    union_sql = "\nUNION ALL\n".join(
        f"SELECT '{r['table']}' AS tbl, COUNT(*) AS n FROM {CATALOG}.{SCHEMA}.{r['table']}"
        for r in RECIPES
    )
    with st.spinner("Loading live row counts..."):
        try:
            counts_df = query(union_sql)
            counts = dict(zip(counts_df["tbl"], counts_df["n"]))
        except Exception:
            counts = {}

    # Recipe cards in 3-column grid
    cols = st.columns(3)
    for i, recipe in enumerate(RECIPES):
        with cols[i % 3]:
            n = counts.get(recipe["table"], "—")
            st.markdown(
                f"""
<div style="border:1px solid #e0e0e0; border-radius:8px; padding:16px; margin-bottom:12px;">
<span style="font-size:2rem">{recipe['icon']}</span>
<h4 style="margin:8px 0 4px">{recipe['title']}</h4>
<p style="color:#666; font-size:0.85rem; margin:0">{recipe['description']}</p>
<hr style="margin:8px 0"/>
<code style="font-size:0.8rem">{recipe['table']}</code>
<span style="float:right; font-weight:bold; color:#e97f04">{f'{n:,}' if isinstance(n, int) else n} rows</span><br/>
<small style="color:#888">Parser: {recipe['parser']}</small>
</div>
""",
                unsafe_allow_html=True,
            )

    st.divider()

    # Pipeline overview
    st.subheader("How It Works")
    st.code(
        """\
Raw file (UC Volume)
  └─► ai_parse_document / Whisper / CLIP / Gemini
        └─► parsed_raw table  (VARIANT / raw output)
              └─► parsed table  (exploded elements)
                    └─► chunks table  (canonical schema)
                          └─► Vector Search index  (embed + sync)""",
        language="text",
    )
