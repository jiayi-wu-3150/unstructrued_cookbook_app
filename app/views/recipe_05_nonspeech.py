import streamlit as st
import plotly.express as px
from utils.db import query, CATALOG, SCHEMA
from utils.formatting import recipe_header, code_tab_content, requirements_tab_content, show_endpoint_health, AUDIO_TYPE_COLORS

RAW_TABLE    = f"{CATALOG}.{SCHEMA}.sound_descriptions_raw"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.sound_chunks"

CODE_SNIPPET = """\
CLASSIFY_PROMPT = \"\"\"Listen to this audio and respond with valid JSON only:
{
  "audio_type": "<speech | music | sound_effect | ambient | mixed>",
  "description": "<one sentence for search indexing>"
}\"\"\"

@pandas_udf(returnType=StructType([
    StructField("audio_type",   StringType()),
    StructField("description",  StringType()),
]))
def classify_describe_udf(paths: pd.Series, contents: pd.Series) -> pd.DataFrame:
    results = []
    for path, content in zip(paths, contents):
        mime = MIME_MAP.get(path.rsplit(".", 1)[-1].lower(), "audio/wav")
        b64  = base64.b64encode(content).decode()
        payload = {
            "messages": [{"role":"user","content":[
                {"type":"text","text": CLASSIFY_PROMPT},
                {"type":"image_url","image_url":{"url": f"data:{mime};base64,{b64}"}}
            ]}]
        }
        resp = client.predict(endpoint="databricks-gemini-2-5-flash", inputs=payload)
        parsed = json.loads(resp["choices"][0]["message"]["content"])
        results.append((parsed["audio_type"], parsed["description"]))
    return pd.DataFrame(results, columns=["audio_type", "description"])"""


def render():
    recipe_header("🔊", "Audio — Non-Speech", "Classify & describe sound effects, music, and ambient audio with Gemini 2.5 Flash", "Table", "sound_chunks")

    tab_try, tab_code, tab_req = st.tabs(["**Try it**", "**Code snippet**", "**Requirements**"])

    with tab_try:
        with st.spinner("Loading classifications..."):
            try:
                all_df = query(f"""
                    SELECT clip_name, audio_type,
                           description
                    FROM {RAW_TABLE}
                    ORDER BY audio_type, clip_name
                """)
            except Exception as e:
                st.error(f"Could not load data: {e}")
                return

        if all_df.empty:
            st.warning("No data found in sound_descriptions_raw.")
            return

        # Summary metrics
        type_counts = all_df["audio_type"].value_counts()
        cols = st.columns(len(type_counts) + 1)
        cols[0].metric("Total clips", len(all_df))
        for i, (atype, cnt) in enumerate(type_counts.items()):
            cols[i + 1].metric(atype.replace("_", " ").title(), cnt)

        st.divider()

        col_filter, col_cards = st.columns([1, 3], gap="large")

        with col_filter:
            st.markdown("#### Filter by type")
            audio_types = ["All"] + sorted(all_df["audio_type"].dropna().unique().tolist())
            selected_type = st.radio("Type", audio_types, label_visibility="collapsed")

            st.divider()

            # Distribution chart
            dist_df = all_df.groupby("audio_type").size().reset_index(name="count")
            fig = px.pie(
                dist_df,
                names="audio_type",
                values="count",
                color="audio_type",
                color_discrete_map={t: f"#{c}" for t, c in {"speech": "3498db", "music": "9b59b6", "sound_effect": "e67e22", "ambient": "27ae60", "mixed": "e74c3c"}.items()},
                hole=0.4,
            )
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_cards:
            filtered = all_df if selected_type == "All" else all_df[all_df["audio_type"] == selected_type]
            st.markdown(f"**{len(filtered)} clips**")

            for _, row in filtered.iterrows():
                atype = row.get("audio_type") or "unknown"
                color = AUDIO_TYPE_COLORS.get(atype, "gray")
                clip = row.get("clip_name") or "unknown"
                desc = row.get("description") or "_No description_"

                st.markdown(
                    f"**{clip}** &nbsp; :{color}[{atype}]",
                    unsafe_allow_html=False,
                )
                st.caption(desc)
                st.markdown("---")

    with tab_code:
        code_tab_content(CODE_SNIPPET, language="python")
        st.markdown("""
**Key points:**
- Single Gemini call returns both `audio_type` and `description` as structured JSON
- Handles WAV, MP3, FLAC, OGG, M4A via `MIME_MAP` — Gemini accepts any audio format
- `CREATE TABLE IF NOT EXISTS` — model only called once per file
- Chunk format: `[Sound: Forest Rain (ambient)] A steady rainfall with occasional thunder...`
- Non-speech clips from notebook 04 are automatically routed here via a UNION
""")

    with tab_req:
        st.subheader("Endpoint health")
        show_endpoint_health(["databricks-gemini-2-5-flash"])
        st.divider()
        requirements_tab_content(
            permissions=["SELECT on table", "READ FILES on volume", "USE SCHEMA"],
            resources=["SQL warehouse", "UC Volume (sound_effects)", "databricks-gemini-2-5-flash endpoint"],
            dependencies=["mlflow", "pandas", "base64 (stdlib)"],
        )
