import json
import streamlit as st
import plotly.express as px
from utils.db import query, download_volume_file, CATALOG, SCHEMA
from utils.formatting import recipe_header, code_tab_content, requirements_tab_content, draw_bboxes_on_image, render_pdf_page

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/w2_sample"

TABLE = f"{CATALOG}.{SCHEMA}.w2_parsed"

CODE_SNIPPET = """\
-- Step 1: Parse all W2 PDFs with ai_parse_document (runs once)
CREATE TABLE IF NOT EXISTS w2_parsed_raw AS
SELECT
  path                                               AS file_path,
  ai_parse_document(content, map('version', '2.0')) AS parsed,
  current_timestamp()                                AS parsed_at
FROM READ_FILES('/Volumes/.../w2_sample', format => 'binaryFile');

-- Step 2: Explode elements (no model call — reads from raw table)
CREATE OR REPLACE TABLE w2_parsed AS
SELECT
  r.file_path,
  el.value:id::INT                      AS element_id,
  el.value:type::STRING                 AS element_type,
  el.value:content::STRING              AS content,
  ROUND(el.value:confidence::DOUBLE, 3) AS confidence
FROM w2_parsed_raw AS r,
LATERAL variant_explode(r.parsed:document.elements) AS el;"""


def render():
    recipe_header("📄", "PDF — W2 Forms", "Structured field/value extraction from tax forms using `ai_parse_document`", "Table", "w2_parsed")

    tab_try, tab_code, tab_req = st.tabs(["**Try it**", "**Code snippet**", "**Requirements**"])

    with tab_try:
        with st.spinner("Loading files..."):
            files_df = query(f"SELECT DISTINCT regexp_extract(file_path, '[^/]+$', 0) AS file_name FROM {TABLE} ORDER BY file_name")
        file_names = files_df["file_name"].tolist() if not files_df.empty else []

        col_sel, col_view = st.columns([2, 1])
        with col_sel:
            selected_file = st.selectbox("Select a W2 file", file_names)
        with col_view:
            view = st.radio("View", ["Parsed elements", "Type distribution"], horizontal=True, label_visibility="collapsed")

        if not selected_file:
            return

        col_pdf, col_data = st.columns([1, 1], gap="large")

        with col_pdf:
            st.markdown("**Page with bounding boxes**")
            show_bboxes = st.checkbox("Show bounding boxes", value=True, key="w2_bbox")
            with st.spinner("Rendering page..."):
                try:
                    pdf_bytes = download_volume_file(f"{VOLUME_PATH}/{selected_file}")
                    page_img = render_pdf_page(pdf_bytes, page_num=0, dpi=200)

                    if show_bboxes:
                        bbox_df = query(f"""
                            SELECT element_id, element_type, bbox::STRING AS bbox_str
                            FROM {TABLE}
                            WHERE file_path LIKE '%{selected_file}%'
                              AND bbox IS NOT NULL
                        """)
                        bboxes = []
                        for _, row in bbox_df.iterrows():
                            bbox_list = json.loads(row["bbox_str"])
                            for b in bbox_list:
                                b["element_type"] = row["element_type"]
                                b["label"] = f"{row['element_type']} ({row['element_id']})"
                                bboxes.append(b)
                        page_img = draw_bboxes_on_image(page_img, bboxes)

                    st.image(page_img, use_column_width=True)
                except Exception as e:
                    st.error(f"Could not render page: {e}")

        with col_data:
            if view == "Parsed elements":
                with st.spinner("Loading elements..."):
                    df = query(f"""
                        SELECT element_id, element_type, content, confidence
                        FROM {TABLE}
                        WHERE file_path LIKE '%{selected_file}%'
                        ORDER BY element_id
                    """)
                st.markdown(f"**{len(df)} elements** extracted")

                def color_conf(val):
                    if val is None:
                        return ""
                    color = "#2ecc71" if val >= 0.9 else "#f39c12" if val >= 0.8 else "#e74c3c"
                    return f"background-color: {color}22"

                styled = df.style.map(color_conf, subset=["confidence"])
                st.dataframe(styled, use_container_width=True, hide_index=True, height=560)

            else:  # Type distribution
                with st.spinner("Loading distribution..."):
                    dist_df = query(f"""
                        SELECT element_type,
                               COUNT(*) AS total_elements,
                               ROUND(AVG(confidence), 3) AS avg_confidence
                        FROM {TABLE}
                        GROUP BY element_type
                        ORDER BY total_elements DESC
                    """)
                fig = px.bar(
                    dist_df,
                    x="total_elements",
                    y="element_type",
                    orientation="h",
                    color="avg_confidence",
                    color_continuous_scale="RdYlGn",
                    range_color=[0.7, 1.0],
                    labels={"total_elements": "Element count", "element_type": "Type", "avg_confidence": "Avg confidence"},
                    title="Element types across all W2s",
                )
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(dist_df, use_container_width=True, hide_index=True)

    with tab_code:
        code_tab_content(CODE_SNIPPET, language="sql")
        st.markdown("""
**Key points:**
- `ai_parse_document(content, map('version', '2.0'))` returns a `VARIANT` with `document.elements[]`
- `LATERAL variant_explode(...)` expands each element into its own row
- `CREATE TABLE IF NOT EXISTS` means the model call only runs **once** — re-running the notebook is safe
- Each element has `id`, `type`, `content`, `confidence`, and `bbox`
""")

    with tab_req:
        requirements_tab_content(
            permissions=["SELECT on table", "READ FILES on volume", "USE SCHEMA"],
            resources=["SQL warehouse", "UC Volume (w2_sample)", "Unity Catalog table"],
            dependencies=["ai_parse_document (built-in)", "No extra pip installs required"],
        )
