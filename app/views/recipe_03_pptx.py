import streamlit as st
from utils.db import query, download_volume_file, CATALOG, SCHEMA
from utils.formatting import recipe_header, code_tab_content, requirements_tab_content

CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.pptx_chunks"
PARSED_TABLE = f"{CATALOG}.{SCHEMA}.pptx_parsed"

CODE_SNIPPET = """\
-- Step 3: Aggregate elements per slide into one chunk
CREATE OR REPLACE TABLE pptx_chunks AS
SELECT
  p.slide_id,
  MAX(CASE WHEN p.element_type = 'title' THEN p.content END) AS slide_title,
  -- All text + figure descriptions joined into one chunk
  ARRAY_JOIN(
    ARRAY_SORT(
      COLLECT_LIST(
        COALESCE(p.description, p.content)
      )
    ), ' '
  )                                AS slide_text,
  COUNT(*)                         AS element_count,
  ROUND(AVG(p.confidence), 3)      AS avg_confidence
FROM pptx_parsed AS p
GROUP BY p.slide_id
ORDER BY p.slide_id;"""


def render():
    recipe_header("📊", "PowerPoint (PPTX)", "One chunk per slide — titles, bullets, and AI-described figure captions", "Table", "pptx_chunks")

    tab_try, tab_code, tab_req = st.tabs(["**Try it**", "**Code snippet**", "**Requirements**"])

    with tab_try:
        with st.spinner("Loading slide overview..."):
            overview_df = query(f"""
                SELECT slide_id, slide_title, element_count, avg_confidence
                FROM {CHUNKS_TABLE}
                ORDER BY slide_id
            """)

        if overview_df.empty:
            st.warning("No data found in pptx_chunks.")
            return

        total_slides = len(overview_df)
        avg_elements = overview_df["element_count"].mean()
        avg_conf = overview_df["avg_confidence"].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Slides", total_slides)
        m2.metric("Avg elements / slide", f"{avg_elements:.1f}")
        m3.metric("Avg confidence", f"{avg_conf:.3f}")

        st.divider()

        slide_ids = overview_df["slide_id"].tolist()
        slide_titles = overview_df["slide_title"].fillna("(no title)").tolist()
        slide_labels = [f"Slide {sid+1}: {title[:40]}" for sid, title in zip(slide_ids, slide_titles)]

        col_nav, col_view = st.columns([3, 1])
        with col_nav:
            selected_idx = st.select_slider("Slide", options=range(len(slide_labels)), format_func=lambda i: slide_labels[i])
        with col_view:
            view = st.radio("View", ["Chunk text", "Raw elements"], horizontal=True, label_visibility="collapsed")
        selected_slide = slide_ids[selected_idx]

        col_img, col_results = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("**Slide image**")
            with st.spinner("Loading slide image..."):
                try:
                    img_df = query(f"""
                        SELECT p.value:id::INT AS page_id,
                               p.value:image_uri::STRING AS image_uri
                        FROM {CATALOG}.{SCHEMA}.pptx_parsed_raw,
                        LATERAL variant_explode(parsed:document.pages) AS p
                        WHERE p.value:id::INT = {selected_slide}
                        LIMIT 1
                    """)
                    if not img_df.empty:
                        img_bytes = download_volume_file(img_df.iloc[0]["image_uri"])
                        st.image(img_bytes, use_container_width=True)
                    else:
                        st.info("No image for this slide.")
                except Exception as e:
                    st.warning(f"Image not available: {e}")

        with col_results:
            if view == "Chunk text":
                with st.spinner("Loading chunk..."):
                    chunk_df = query(f"""
                        SELECT slide_id, slide_title, slide_text, element_count, avg_confidence
                        FROM {CHUNKS_TABLE}
                        WHERE slide_id = {selected_slide}
                    """)
                if not chunk_df.empty:
                    row = chunk_df.iloc[0]
                    st.markdown(f"**Slide {row['slide_id']+1}** — {row['slide_title'] or '(no title)'}")
                    st.caption(f"Elements: {row['element_count']}  ·  Avg confidence: {row['avg_confidence']}")
                    st.text_area("Chunk text (what gets embedded)", value=row["slide_text"] or "", height=420, disabled=True)

            else:  # Raw elements
                with st.spinner("Loading elements..."):
                    el_df = query(f"""
                        SELECT element_type, content, description, confidence
                        FROM {PARSED_TABLE}
                        WHERE slide_id = {selected_slide}
                        ORDER BY element_id
                    """)
                st.markdown(f"**{len(el_df)} raw elements** on slide {selected_slide+1}")
                st.dataframe(el_df, use_container_width=True, hide_index=True, height=440)

    with tab_code:
        code_tab_content(CODE_SNIPPET, language="sql")
        st.markdown("""
**Key points:**
- `GROUP BY slide_id` — one row per slide is the natural chunk unit for presentations
- `COALESCE(description, content)` — figure descriptions are included, making image slides retrievable
- `ARRAY_JOIN(ARRAY_SORT(COLLECT_LIST(...)))` — deterministic text assembly from unordered elements
- The chunk includes the slide title as context prefix for retrieval
""")

    with tab_req:
        requirements_tab_content(
            permissions=["SELECT on table", "READ FILES on volume", "USE SCHEMA"],
            resources=["SQL warehouse", "UC Volume (pptx)", "Unity Catalog tables"],
            dependencies=["ai_parse_document (built-in)", "No extra pip installs required"],
        )
