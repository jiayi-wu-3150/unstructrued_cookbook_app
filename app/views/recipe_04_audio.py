import streamlit as st
from utils.db import query, download_volume_file, CATALOG, SCHEMA
from utils.formatting import recipe_header, code_tab_content, requirements_tab_content, show_endpoint_health

RAW_TABLE    = f"{CATALOG}.{SCHEMA}.voice_celebrities_raw"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.voice_celebrities_chunks"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/voice_celebrities"

# Map from speaker display name back to filename stem
SPEAKER_FILE_MAP = {
    "Andrew Tate":   "andrew-tate",
    "Barack Obama":  "barack-obama",
    "Bill Gates":    "bill_gates",
    "Donald Trump":  "donald-trump",
    "Elon Musk":     "elon-musk",
    "Joe Rogan":     "joe-rogan",
    "Jordan Peterson": "jordan-peterson",
    "Keanu Reeves":  "keanu-reeves",
    "Mark Zuckerberg": "mark-zuckerberg",
    "Morgan Freeman": "morgan-freeman",
    "Sam Altman":    "sam-altman",
    "Steve Jobs":    "steve-jobs",
    "Taylor Swift":  "taylor-swift",
}

CODE_SNIPPET = """\
-- Downsample to 16 kHz mono (reduces ~15 MB WAV → ~2 MB)
def _compress(content_bytes: bytes) -> bytes:
    y, sr = sf.read(io.BytesIO(content_bytes))
    if y.ndim > 1:
        y = y.mean(axis=1)          # stereo → mono
    if sr != 16000:
        n = int(len(y) * 16000 / sr)
        y = np.interp(np.linspace(0, len(y)-1, n), np.arange(len(y)), y)
    out = io.BytesIO()
    sf.write(out, y.astype(np.float32), 16000, format="WAV", subtype="PCM_16")
    return out.getvalue()

-- Transcribe with Whisper (CREATE TABLE IF NOT EXISTS = runs once)
SELECT
  file_path,
  speaker,
  CAST(ai_query('whisper_large_v3', content) AS STRING) AS full_text
FROM audio_to_transcribe;

-- Word-window chunking (150 words, 20-word overlap)
chunk_text = f"[Speaker: {speaker}] " + " ".join(words[start:end])"""


def render():
    recipe_header("🎤", "Audio — Speech", "Transcription with Whisper Large v3 + word-window chunking", "Table", "voice_celebrities_chunks")

    tab_try, tab_code, tab_req = st.tabs(["**Try it**", "**Code snippet**", "**Requirements**"])

    with tab_try:
        with st.spinner("Loading speakers..."):
            speakers_df = query(f"SELECT speaker FROM {RAW_TABLE} ORDER BY speaker")
        speakers = speakers_df["speaker"].tolist() if not speakers_df.empty else []

        col_sel, col_view = st.columns([2, 1])
        with col_sel:
            selected_speaker = st.selectbox("Select a speaker", speakers)
        with col_view:
            view = st.radio("View", ["Full transcript", "Chunks"], horizontal=True, label_visibility="collapsed")

        if not selected_speaker:
            return

        # Audio player
        st.markdown("**Listen to the recording**")
        file_stem = SPEAKER_FILE_MAP.get(selected_speaker)
        if file_stem:
            for ext in ["wav", "mp3", "flac"]:
                try:
                    audio_bytes = download_volume_file(f"{VOLUME_PATH}/{file_stem}.{ext}")
                    st.audio(audio_bytes, format=f"audio/{ext}")
                    break
                except Exception:
                    continue
        else:
            st.caption("Audio file not mapped for this speaker.")

        st.divider()

        col_results, _ = st.columns([2, 1])
        with col_results:
            if view == "Full transcript":
                with st.spinner("Loading transcript..."):
                    df = query(f"""
                        SELECT speaker, full_text
                        FROM {RAW_TABLE}
                        WHERE speaker = '{selected_speaker}'
                    """)
                if not df.empty:
                    row = df.iloc[0]
                    words = len(row["full_text"].split()) if row["full_text"] else 0
                    st.markdown(f"**{row['speaker']}** — ~{words:,} words")
                    st.text_area("Transcript", value=row["full_text"] or "", height=320, disabled=True)

            else:  # Chunks
                with st.spinner("Loading chunks..."):
                    chunks_df = query(f"""
                        SELECT chunk_index, chunk_text
                        FROM {CHUNKS_TABLE}
                        WHERE speaker = '{selected_speaker}'
                        ORDER BY chunk_index
                    """)
                total_chunks = len(chunks_df)

                with st.spinner("Loading word counts..."):
                    raw_df = query(f"SELECT full_text FROM {RAW_TABLE} WHERE speaker = '{selected_speaker}'")

                total_words = 0
                if not raw_df.empty and raw_df.iloc[0]["full_text"]:
                    total_words = len(raw_df.iloc[0]["full_text"].split())

                m1, m2, m3 = st.columns(3)
                m1.metric("Total words", f"{total_words:,}")
                m2.metric("Chunks (150w, 20w overlap)", total_chunks)
                m3.metric("Avg words/chunk", f"{total_words // total_chunks if total_chunks else 0}")

                st.markdown("---")
                for _, chunk_row in chunks_df.iterrows():
                    with st.expander(f"Chunk {chunk_row['chunk_index']}"):
                        st.markdown(chunk_row["chunk_text"])

    with tab_code:
        code_tab_content(CODE_SNIPPET, language="python")
        st.markdown("""
**Key points:**
- `_compress()` downsamples WAV to 16 kHz mono — reduces file size ~8–10× (Whisper's native format)
- `ai_query('whisper_large_v3', content)` — passes raw BINARY audio, returns transcript STRING
- `ai_query` has a **16 MB request limit** — compression is required for longer recordings
- Chunk format `[Speaker: Morgan Freeman] text...` — speaker context baked into every chunk for retrieval
- 150-word window with 20-word overlap ensures no context is lost at chunk boundaries
""")

    with tab_req:
        st.subheader("Endpoint health")
        show_endpoint_health(["whisper_large_v3"])
        st.divider()
        requirements_tab_content(
            permissions=["SELECT on table", "READ FILES on volume", "USE SCHEMA"],
            resources=["SQL warehouse", "UC Volume (voice_celebrities)", "whisper_large_v3 endpoint"],
            dependencies=["soundfile", "numpy", "ai_query (built-in)"],
        )
