import streamlit as st
from utils.db import query, CATALOG, SCHEMA


RECIPES = [
    {
        "icon": "📄",
        "title": "PDF — W2 Forms",
        "table": "w2_parsed",
        "parser": "ai_parse_document",
        "description": "Structured field/value extraction from tax forms",
    },
    {
        "icon": "📑",
        "title": "PDF — Research Papers",
        "table": "research_parsed",
        "parser": "ai_parse_document + figure descriptions",
        "description": "Text + AI-written descriptions of figures and charts",
    },
    {
        "icon": "📊",
        "title": "PowerPoint (PPTX)",
        "table": "pptx_chunks",
        "parser": "ai_parse_document",
        "description": "One chunk per slide — titles, bullets, and image captions",
    },
    {
        "icon": "🎤",
        "title": "Audio — Speech",
        "table": "voice_celebrities_chunks",
        "parser": "Whisper Large v3",
        "description": "Word-window chunked transcripts with speaker context",
    },
    {
        "icon": "🔊",
        "title": "Audio — Non-Speech",
        "table": "sound_chunks",
        "parser": "Gemini 2.5 Flash",
        "description": "Classify & describe sound effects, music, ambient audio",
    },
    {
        "icon": "🎬",
        "title": "Video (MP4)",
        "table": "video_clips_gold",
        "parser": "CLIP + Gemini 2.5 Flash",
        "description": "Frame embeddings (768-dim) + Gemini descriptions → Vector Search",
    },
]


def render():
    st.title("🗂️ Unstructured Cookbook")
    st.markdown(
        "A recipe book for ingesting **any file type** into Databricks for RAG and semantic search. "
        "One self-contained pipeline per file type — all producing the same output schema."
    )
    st.divider()

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

    # Output schema section
    st.subheader("Universal Output Schema")
    st.markdown("Every recipe produces this canonical schema — drop any table into a Vector Search index.")

    schema_cols = st.columns(2)
    with schema_cols[0]:
        st.code(
            """\
chunk_id          STRING   -- hash(source_path + chunk_index)
chunk_to_embed    STRING   -- text sent to the embedding model
chunk_to_retrieve STRING   -- richer text shown in retrieval results
source_path       STRING   -- original file path / URL
chunk_index       INT      -- position within source document
metadata          MAP<STRING,STRING>  -- file-type-specific fields""",
            language="sql",
        )
    with schema_cols[1]:
        st.markdown(
            """
**chunk_to_embed** is optimized for the embedding model — concise and focused.

**chunk_to_retrieve** is richer — includes surrounding context, speaker labels, slide titles,
or figure descriptions that are useful to show the user but would dilute the embedding.

**metadata** captures file-type-specific fields:
- PDF: `{element_type, confidence, page_id}`
- Audio: `{speaker, word_start, word_end}`
- Video: `{frame_num, video_id, embedding_dim}`
"""
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
