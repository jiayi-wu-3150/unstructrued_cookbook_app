import base64
import streamlit as st
from utils.db import query, download_volume_file, CATALOG, SCHEMA, get_workspace_client
from utils.formatting import recipe_header, code_tab_content, requirements_tab_content, show_endpoint_health

GOLD_TABLE  = f"{CATALOG}.{SCHEMA}.video_clips_gold"
VS_ENDPOINT = "video-search-endpoint"
VS_INDEX    = f"{CATALOG}.{SCHEMA}.video_clips_index"

DEMO_QUERIES = [
    "person cooking in a kitchen",
    "outdoor scene with people",
    "close up of a face",
    "people sitting at a table",
    "kitchen appliances",
    "empty room interior",
    "someone eating food",
    "daytime outdoor activity",
]

CODE_SNIPPET_EMBED = """\
-- Step 3: Embed every frame with CLIP via ai_query (768-dim)
CREATE OR REPLACE TABLE video_clips_embeddings AS
SELECT
  frame_id,
  video_id,
  ai_query(
    'clip_embedding_endpoint',
    request   => named_struct('model_input', model_input),
    returnType => 'ARRAY<STRUCT<model_input: ARRAY<DOUBLE>>>'
  )[0].model_input AS image_embeddings
FROM video_clips;

-- Step 4: Frame descriptions with Gemini 2.5 Flash (multimodal)
CREATE OR REPLACE TABLE video_descriptions_raw AS
SELECT
  frame_id, video_id,
  ai_query(
    'databricks-gemini-2-5-flash',
    'Describe this video frame in 2–3 sentences for search indexing.',
    files => unbase64(model_input)   -- BINARY, not ARRAY<BINARY>
  ) AS frame_description
FROM video_clips;

-- Step 5: Gold table (CDF enabled for Delta Sync)
CREATE OR REPLACE TABLE video_clips_gold
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS
SELECT e.frame_id, e.video_id, e.image_embeddings,
       d.frame_description, c.frame_num, c.frame_path
FROM video_clips_embeddings e
JOIN video_descriptions_raw d USING (frame_id)
JOIN video_clips c USING (frame_id);"""

CODE_SNIPPET_SEARCH = """\
from transformers import CLIPProcessor, CLIPModel
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

# Encode text query with CLIP text encoder (same 768-dim space as image embeddings)
def get_text_embedding(text: str) -> list:
    model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs    = processor(text=text, return_tensors="pt", padding=True)
    features  = model.get_text_features(**inputs)
    return features.detach().numpy().tolist()[0]

query_text   = "person running outdoors"
query_vector = get_text_embedding(query_text)

# HYBRID search: vector similarity + BM25 on frame_description + reranker
index = VectorSearchClient(disable_notice=True).get_index(
    "video-search-endpoint",
    "serverless_stable_r4umw1_catalog.unstructured_data.video_clips_index",
)
results = index.similarity_search(
    query_text=query_text,
    query_vector=query_vector,
    columns=["frame_id", "video_id", "frame_num", "frame_description", "frame_path"],
    query_type="HYBRID",
    num_results=5,
    reranker=DatabricksReranker(columns_to_rerank=["frame_description"]),
)"""


def _vs_search(query_text: str, n: int = 5) -> list[dict]:
    """Text search on frame_description using SQL LIKE matching."""
    keywords = [w for w in query_text.lower().split() if len(w) > 3]
    if not keywords:
        keywords = query_text.lower().split()
    conditions = " OR ".join(f"LOWER(frame_description) LIKE '%{kw}%'" for kw in keywords)
    df = query(f"""
        SELECT frame_id, video_id, frame_num, frame_description, frame_path
        FROM {GOLD_TABLE}
        WHERE {conditions}
        ORDER BY frame_num
        LIMIT {n}
    """)
    return df.to_dict("records")


def render():
    recipe_header(
        "🎬", "Video (MP4 + Vector Search)",
        "Frame extraction → CLIP embeddings (768-dim) → Gemini descriptions → hybrid VS index",
        "Table", "video_clips_gold",
    )

    tab_try, tab_code, tab_req = st.tabs(["**Try it**", "**Code snippet**", "**Requirements**"])

    with tab_try:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames indexed", "82")
        m2.metric("Embedding dimensions", "768")
        m3.metric("Image model", "CLIP ViT-L/14")
        m4.metric("Description model", "Gemini 2.5 Flash")

        st.divider()

        col_vid, col_table = st.columns([1, 1], gap="large")

        with col_vid:
            st.markdown("**Original video**")
            with st.spinner("Loading video..."):
                try:
                    video_bytes = download_volume_file(
                        f"/Volumes/{CATALOG}/{SCHEMA}/video_clips_sample/Breakfas1939_512kb.mp4"
                    )
                    b64 = base64.b64encode(video_bytes).decode()
                    st.markdown(
                        f'<video controls width="100%" style="border-radius:4px">'
                        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                        f'</video>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.warning(f"Could not load video: {e}")

        with col_table:
            st.markdown("**Gold table — `video_clips_gold`**")
            with st.spinner("Loading..."):
                gold_df = query(f"""
                    SELECT frame_num,
                           SIZE(image_embeddings) AS embedding_dim,
                           LEFT(frame_description, 100) AS description_preview
                    FROM {GOLD_TABLE}
                    ORDER BY frame_num
                    LIMIT 10
                """)
            st.dataframe(gold_df, use_container_width=True, hide_index=True, height=300)

        st.divider()

        # Hybrid VS search
        st.subheader("Hybrid search (CLIP vector + BM25 + reranker)")
        st.caption("Keyword search on `frame_description` — production pipeline uses CLIP vector + HYBRID VS search")

        selected_query = st.selectbox("Demo query", DEMO_QUERIES, label_visibility="collapsed")

        if st.button("🔍 Search frames", type="primary"):
            with st.spinner("Searching frames..."):
                try:
                    results = _vs_search(selected_query)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    return

            if results:
                st.success(f"**{len(results)} results**")
                cols = st.columns(min(len(results), 3))
                for i, r in enumerate(results):
                    with cols[i % 3]:
                        frame_path = r.get("frame_path") or ""
                        if frame_path:
                            try:
                                img_bytes = download_volume_file(frame_path)
                                st.image(img_bytes, use_column_width=True)
                            except Exception as e:
                                st.caption(f"_(image unavailable: {e})_")
                        else:
                            st.caption("_(no frame path)_")
                        desc = r.get("frame_description") or ""
                        st.caption(f"Frame {r.get('frame_num', '?')} · {desc[:80]}{'...' if len(desc) > 80 else ''}")
            else:
                st.info("No matching frames found. Try a different query.")

    with tab_code:
        st.subheader("Embedding + description pipeline")
        code_tab_content(CODE_SNIPPET_EMBED, language="sql")
        st.subheader("Query-time: CLIP text encoding + HYBRID VS search")
        code_tab_content(CODE_SNIPPET_SEARCH, language="python")
        st.markdown("""
**Key design decisions:**
- **`files => unbase64(model_input)`** — Gemini's `files =>` expects `BINARY`, not `ARRAY<BINARY>`
- **`ai_query` returnType** — CLIP returns `ARRAY<STRUCT<model_input: ARRAY<DOUBLE>>>`, index with `[0].model_input`
- **CLIP shared embedding space** — text and image embeddings are directly comparable with no per-frame labels
- **`recursiveFileLookup=true`** — frames saved in `{video_id}/` subdirs, not the root volume path
- **CDF enabled** on `video_clips_gold` — required for Delta Sync to the VS index
- **`clip_embedding_endpoint` handles images only** — the CLIP text encoder runs locally at query time
""")

    with tab_req:
        st.subheader("Endpoint health")
        show_endpoint_health(["clip_embedding_endpoint", "databricks-gemini-2-5-flash"])

        st.subheader("VS Index status")
        with st.spinner("Checking VS index..."):
            try:
                ws = get_workspace_client()
                idx = ws.vector_search.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
                ready = idx.status.ready_for_search if idx.status else False
                icon = "🟢" if ready else "🔴"
                st.markdown(f"{icon} `{VS_INDEX}` — **{'ONLINE' if ready else 'NOT READY'}**")
            except Exception as e:
                st.error(f"Could not check VS index: {e}")

        st.divider()
        requirements_tab_content(
            permissions=["SELECT on table", "CAN QUERY vector search endpoint", "USE SCHEMA"],
            resources=[
                "SQL warehouse",
                "clip_embedding_endpoint (GPU_SMALL, already deployed)",
                "databricks-gemini-2-5-flash (FMAPI)",
                "video-search-endpoint + video_clips_index (ONLINE)",
            ],
            dependencies=["databricks-vectorsearch==0.63 (pipeline only)", "No extra pip installs for the app"],
        )
