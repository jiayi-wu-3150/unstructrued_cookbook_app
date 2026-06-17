import streamlit as st
import pandas as pd
import plotly.express as px
from utils.db import query, download_volume_file, CATALOG, SCHEMA
from utils.formatting import recipe_header, code_tab_content, requirements_tab_content

IMAGES_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/research_papers_images"

TABLE = f"{CATALOG}.{SCHEMA}.research_parsed"

CODE_SNIPPET = """\
-- Parse research PDFs with figure descriptions (runs once)
CREATE TABLE IF NOT EXISTS research_parsed_raw AS
SELECT
  path AS file_path,
  ai_parse_document(
    content,
    map(
      'version',                '2.0',
      'dpi',                    '300',
      'descriptionElementTypes','figure',
      'imageOutputPath',        '/Volumes/.../research_papers_images'
    )
  ) AS parsed,
  current_timestamp() AS parsed_at
FROM READ_FILES('/Volumes/.../research_papers', format => 'binaryFile');

-- Explode + use COALESCE so figure descriptions are searchable
CREATE OR REPLACE TABLE research_parsed AS
SELECT
  r.file_path,
  el.value:id::INT                        AS element_id,
  el.value:type::STRING                   AS element_type,
  el.value:content::STRING                AS content,
  el.value:description::STRING            AS description,
  ROUND(el.value:confidence::DOUBLE, 3)   AS confidence,
  -- chunk_to_embed: use description for figures, content for text
  COALESCE(el.value:description::STRING,
           el.value:content::STRING)      AS chunk_to_embed
FROM research_parsed_raw AS r,
LATERAL variant_explode(r.parsed:document.elements) AS el;"""


def render():
    recipe_header("📑", "PDF — Research Papers", "Text + AI-written figure descriptions via `ai_parse_document`", "Table", "research_parsed")

    tab_try, tab_code, tab_req = st.tabs(["**Try it**", "**Code snippet**", "**Requirements**"])

    with tab_try:
        with st.spinner("Loading papers..."):
            papers_df = query(f"SELECT DISTINCT regexp_extract(file_path, '[^/]+$', 0) AS paper FROM {TABLE} ORDER BY paper")
        papers = papers_df["paper"].tolist() if not papers_df.empty else []

        col_sel, col_view = st.columns([2, 1])
        with col_sel:
            selected_paper = st.selectbox("Select a paper", papers)
        with col_view:
            view = st.radio("View", ["Page images", "Text elements", "Figure descriptions", "Type distribution"], label_visibility="collapsed")

        if not selected_paper:
            return

        col_preview, col_results = st.columns([1, 1], gap="large")

        with col_preview:
            st.markdown("**Paper pages**")
            with st.spinner("Loading page images..."):
                try:
                    # Get page image URIs from the raw table
                    paper_stem = selected_paper.rsplit(".", 1)[0]
                    images_df = query(f"""
                        SELECT p.value:id::INT AS page_id,
                               p.value:image_uri::STRING AS image_uri
                        FROM {CATALOG}.{SCHEMA}.research_parsed_raw,
                        LATERAL variant_explode(parsed:document.pages) AS p
                        WHERE file_path LIKE '%{selected_paper}%'
                        ORDER BY page_id
                        LIMIT 30
                    """)
                    if not images_df.empty:
                        page_nums = images_df["page_id"].fillna(pd.Series(range(len(images_df)))).astype(int).tolist()
                        sel_page = st.selectbox("Page", page_nums, format_func=lambda p: f"Page {p+1}")
                        img_row = images_df[images_df["page_id"] == sel_page]
                        if not img_row.empty:
                            img_uri = img_row.iloc[0]["image_uri"]
                            img_bytes = download_volume_file(img_uri)
                            st.image(img_bytes, use_container_width=True)
                    else:
                        st.info("No page images available.")
                except Exception as e:
                    st.warning(f"Page images not available: {e}")

        with col_results:
            if selected_paper:
                if view == "Text elements":
                    with st.spinner("Loading text..."):
                        df = query(f"""
                            SELECT element_id, element_type, content, confidence
                            FROM {TABLE}
                            WHERE file_path LIKE '%{selected_paper}%'
                              AND element_type != 'figure'
                              AND content IS NOT NULL
                            ORDER BY element_id
                            LIMIT 30
                        """)
                    st.markdown(f"**{len(df)} text elements** from `{selected_paper}`")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                elif view == "Figure descriptions":
                    with st.spinner("Loading figures..."):
                        df = query(f"""
                            SELECT element_id, description, confidence
                            FROM {TABLE}
                            WHERE file_path LIKE '%{selected_paper}%'
                              AND element_type = 'figure'
                            ORDER BY element_id
                        """)
                    if df.empty:
                        st.info("No figure elements found for this paper.")
                    else:
                        st.markdown(f"**{len(df)} figures** — AI-generated descriptions make images searchable")
                        for _, row in df.iterrows():
                            conf = row.get("confidence")
                            conf_label = f"{conf:.3f}" if conf is not None else "—"
                            with st.expander(f"Figure {row['element_id']}  ·  confidence {conf_label}"):
                                st.markdown(row["description"] or "_No description_")

                else:  # Type distribution
                    with st.spinner("Loading distribution..."):
                        dist_df = query(f"""
                            SELECT element_type,
                                   COUNT(*) AS n,
                                   ROUND(AVG(confidence), 3) AS avg_conf,
                                   COUNT(CASE WHEN description IS NOT NULL THEN 1 END) AS with_description
                            FROM {TABLE}
                            WHERE file_path LIKE '%{selected_paper}%'
                            GROUP BY element_type
                            ORDER BY n DESC
                        """)
                    fig = px.bar(
                        dist_df,
                        x="n",
                        y="element_type",
                        orientation="h",
                        color="avg_conf",
                        color_continuous_scale="RdYlGn",
                        range_color=[0.7, 1.0],
                        title=f"Element types — {selected_paper}",
                        labels={"n": "Count", "element_type": "Type", "avg_conf": "Avg confidence"},
                    )
                    fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    with tab_code:
        code_tab_content(CODE_SNIPPET, language="sql")
        st.markdown("""
**Key differences from W2:**
- `dpi=300` — higher resolution improves parsing of dense layouts and small figures
- `descriptionElementTypes=figure` — triggers AI description generation for each figure element
- `imageOutputPath` — saves rendered page images to a volume; `pages[].image_uri` is populated
- `COALESCE(description, content)` — figures have `description` instead of `content`, so both are searchable
""")

    with tab_req:
        requirements_tab_content(
            permissions=["SELECT on table", "READ FILES on volume", "USE SCHEMA"],
            resources=["SQL warehouse", "UC Volume (research_papers)", "Unity Catalog table"],
            dependencies=["ai_parse_document (built-in)", "No extra pip installs required"],
        )
