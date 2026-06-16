# Databricks notebook source
# MAGIC %md
# MAGIC # Audio Parsing — Transcription with `faster-whisper`
# MAGIC
# MAGIC Transcribes celebrity voice clips from a UC Volume and prepares them for RAG.
# MAGIC
# MAGIC **Source volume:** `serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities`
# MAGIC
# MAGIC **Two audio types handled differently:**
# MAGIC - **Speech** → `faster-whisper` transcription → sentence-grouped text chunks
# MAGIC - **Non-speech** (sound effects, music) → Whisper `no_speech_prob` flags these;
# MAGIC   use a multimodal LLM description instead (see note at bottom)
# MAGIC
# MAGIC **Pipeline:**
# MAGIC ```
# MAGIC WAV files → Pandas UDF (faster-whisper) → voice_celebrities_raw
# MAGIC           → explode segments            → voice_celebrities_segments
# MAGIC           → group into 30s windows      → voice_celebrities_chunks
# MAGIC ```

# COMMAND ----------

CATALOG     = "serverless_stable_r4umw1_catalog"
SCHEMA      = "unstructured_data"
VOLUME      = "voice_celebrities"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_TABLE   = f"{CATALOG}.{SCHEMA}.voice_celebrities_raw"
SEG_TABLE   = f"{CATALOG}.{SCHEMA}.voice_celebrities_segments"
OUT_TABLE   = f"{CATALOG}.{SCHEMA}.voice_celebrities_chunks"

WINDOW_SEC  = 30   # target chunk length in seconds
OVERLAP_SEC = 5    # overlap between consecutive chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — List files in volume

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/voice_celebrities'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Transcribe with `faster-whisper` and save to `voice_celebrities_raw`
# MAGIC
# MAGIC Uses a Pandas UDF so Spark parallelises across files.
# MAGIC `no_speech_prob` per segment flags non-speech content (sound effects, music, silence).
# MAGIC `CREATE TABLE IF NOT EXISTS` — transcription only runs once per file.

# COMMAND ----------

# MAGIC %pip install faster-whisper --quiet

# COMMAND ----------

import json
import re
import pandas as pd
from faster_whisper import WhisperModel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType

# Load model once per worker (broadcast via closure)
_model = None
def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model

def _speaker_from_path(path: str) -> str:
    name = re.sub(r"\.wav$", "", path.split("/")[-1], flags=re.I)
    return name.replace("-", " ").replace("_", " ").title()

@F.pandas_udf(StructType([
    StructField("full_text",    StringType()),
    StructField("segments_json",StringType()),
    StructField("language",     StringType()),
    StructField("avg_no_speech_prob", DoubleType()),
]))
def transcribe_udf(paths: pd.Series, contents: pd.Series) -> pd.DataFrame:
    import io
    results = []
    model = _get_model()
    for path, content in zip(paths, contents):
        try:
            audio_io = io.BytesIO(bytes(content))
            segs, info = model.transcribe(audio_io, beam_size=5, word_timestamps=False)
            seg_list = [
                {"id": i, "start": round(s.start, 2), "end": round(s.end, 2),
                 "text": s.text.strip(), "no_speech_prob": round(s.no_speech_prob, 4)}
                for i, s in enumerate(segs)
            ]
            full_text = " ".join(s["text"] for s in seg_list)
            avg_nsp   = sum(s["no_speech_prob"] for s in seg_list) / max(len(seg_list), 1)
            results.append((full_text, json.dumps(seg_list), info.language, avg_nsp))
        except Exception as e:
            results.append((f"ERROR: {e}", "[]", "unknown", 1.0))
    return pd.DataFrame(results, columns=["full_text", "segments_json", "language", "avg_no_speech_prob"])

# COMMAND ----------

from pyspark.sql import functions as F

if not spark.catalog.tableExists(RAW_TABLE):
    raw_df = spark.read.format("binaryFile").load(VOLUME_PATH)

    result = raw_df.select(
        F.col("path").alias("file_path"),
        F.regexp_replace(
            F.regexp_replace(F.regexp_extract(F.col("path"), r"([^/]+)\.wav$", 1), "-", " "),
            "_", " "
        ).alias("speaker_raw"),
        transcribe_udf(F.col("path"), F.col("content")).alias("t"),
        F.current_timestamp().alias("transcribed_at"),
    ).select(
        "file_path",
        F.initcap(F.col("speaker_raw")).alias("speaker"),
        F.col("t.full_text"),
        F.col("t.segments_json"),
        F.col("t.language"),
        F.col("t.avg_no_speech_prob"),
        "transcribed_at",
    )

    result.write.format("delta").saveAsTable(RAW_TABLE)
    print(f"Saved {result.count()} rows to {RAW_TABLE}")
else:
    print(f"{RAW_TABLE} already exists — skipping transcription.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Explode segments to `voice_celebrities_segments`

# COMMAND ----------

from pyspark.sql.types import IntegerType

raw = spark.table(RAW_TABLE)

segments_df = raw.select(
    "file_path", "speaker", "language", "avg_no_speech_prob",
    F.from_json(
        F.col("segments_json"),
        ArrayType(StructType([
            StructField("id",             IntegerType()),
            StructField("start",          DoubleType()),
            StructField("end",            DoubleType()),
            StructField("text",           StringType()),
            StructField("no_speech_prob", DoubleType()),
        ]))
    ).alias("segments")
).withColumn("seg", F.explode("segments")).select(
    "file_path", "speaker", "language",
    F.col("seg.id").alias("segment_id"),
    F.col("seg.start").alias("start_sec"),
    F.col("seg.end").alias("end_sec"),
    F.col("seg.text").alias("text"),
    F.col("seg.no_speech_prob").alias("no_speech_prob"),
)

segments_df.write.format("delta").mode("overwrite").saveAsTable(SEG_TABLE)
print(f"Saved {segments_df.count()} segments to {SEG_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Group segments into ~30s chunks and save to `voice_celebrities_chunks`
# MAGIC
# MAGIC Adjacent segments are grouped into windows of ~`WINDOW_SEC` seconds.
# MAGIC Speaker name is prepended as context prefix for retrieval.

# COMMAND ----------

from pyspark.sql import Window

segs = spark.table(SEG_TABLE).filter(F.col("no_speech_prob") < 0.8)  # drop silent/noise segments

# Assign window_id: increments when cumulative duration exceeds WINDOW_SEC
segs_with_duration = segs.withColumn("duration", F.col("end_sec") - F.col("start_sec"))

# Use Python to group (simpler than Spark window for variable-length grouping)
rows = segs_with_duration.orderBy("file_path", "start_sec").collect()

chunks = []
chunk_id = 0
i = 0
while i < len(rows):
    row = rows[i]
    file_path = row["file_path"]
    speaker   = row["speaker"]
    language  = row["language"]
    chunk_start = row["start_sec"]
    chunk_texts = []
    chunk_end   = row["start_sec"]

    while i < len(rows) and rows[i]["file_path"] == file_path:
        r = rows[i]
        chunk_texts.append(r["text"])
        chunk_end = r["end_sec"]
        i += 1
        if (chunk_end - chunk_start) >= WINDOW_SEC:
            break

    chunk_text = f"[Speaker: {speaker}] " + " ".join(chunk_texts).strip()
    chunks.append((
        f"{file_path}::{chunk_id}",
        file_path,
        speaker,
        language,
        chunk_id,
        round(chunk_start, 2),
        round(chunk_end, 2),
        chunk_text,
    ))
    chunk_id += 1
    # overlap: step back OVERLAP_SEC worth of segments
    while i > 0 and rows[i-1]["file_path"] == file_path and rows[i-1]["end_sec"] > chunk_end - OVERLAP_SEC:
        i -= 1

chunks_df = spark.createDataFrame(chunks, schema=[
    "chunk_id", "file_path", "speaker", "language",
    "chunk_index", "start_sec", "end_sec", "chunk_text",
])

chunks_df.write.format("delta").mode("overwrite").saveAsTable(OUT_TABLE)
print(f"Saved {chunks_df.count()} chunks to {OUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Inspect results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transcription summary per speaker

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   speaker,
# MAGIC   language,
# MAGIC   ROUND(avg_no_speech_prob, 3)                        AS avg_no_speech_prob,
# MAGIC   -- flag potential non-speech files
# MAGIC   CASE WHEN avg_no_speech_prob > 0.5 THEN 'WARNING: likely non-speech' ELSE 'OK' END AS speech_quality,
# MAGIC   LENGTH(full_text)                                   AS transcript_chars
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw
# MAGIC ORDER BY avg_no_speech_prob DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample chunks

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   speaker,
# MAGIC   chunk_index,
# MAGIC   ROUND(start_sec, 1) AS start_sec,
# MAGIC   ROUND(end_sec,   1) AS end_sec,
# MAGIC   chunk_text
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_chunks
# MAGIC ORDER BY speaker, chunk_index
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunk distribution per speaker

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   speaker,
# MAGIC   COUNT(*)                              AS num_chunks,
# MAGIC   ROUND(MAX(end_sec) - MIN(start_sec))  AS total_duration_sec,
# MAGIC   ROUND(AVG(end_sec - start_sec), 1)    AS avg_chunk_sec
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_chunks
# MAGIC GROUP BY speaker
# MAGIC ORDER BY total_duration_sec DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Note: Non-speech audio (sound effects, music)
# MAGIC
# MAGIC Whisper halluccinates on non-speech. If `avg_no_speech_prob > 0.5`, use a
# MAGIC multimodal LLM to **describe** the audio instead:
# MAGIC
# MAGIC ```python
# MAGIC # Gemini 2.5 Flash supports audio via ai_query (when audio input is enabled)
# MAGIC # Fallback: describe using base64 + multimodal prompt
# MAGIC spark.sql("""
# MAGIC   SELECT path,
# MAGIC     ai_query(
# MAGIC       'databricks-gemini-2-5-flash',
# MAGIC       CONCAT('Describe this audio clip in one sentence for search indexing: ', base64(content))
# MAGIC     ) AS description
# MAGIC   FROM READ_FILES('/Volumes/.../sound_effects', format => 'binaryFile')
# MAGIC """)
# MAGIC ```
# MAGIC
# MAGIC The description becomes `chunk_text` — same schema as speech transcripts.
