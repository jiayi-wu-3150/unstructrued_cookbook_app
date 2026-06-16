# Databricks notebook source
# MAGIC %md
# MAGIC # Audio Parsing — Transcription with `ai_query` (Whisper Large v3)
# MAGIC
# MAGIC Transcribes celebrity voice clips from a UC Volume using the
# MAGIC `whisper_large_v3` Databricks Model Serving endpoint.
# MAGIC
# MAGIC **Source volume:** `serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities`
# MAGIC
# MAGIC **Non-speech detection:** clips where the transcript is very short or matches
# MAGIC known Whisper hallucination patterns (`[ Music ]`, `[ Silence ]`, etc.) are
# MAGIC flagged. Route those to notebook `05_audio_nonspeech.py` for Gemini description.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC ```
# MAGIC WAV files → ai_query(whisper_large_v3) → voice_celebrities_raw
# MAGIC           → sentence-split chunks         → voice_celebrities_chunks
# MAGIC ```

# COMMAND ----------

CATALOG     = "serverless_stable_r4umw1_catalog"
SCHEMA      = "unstructured_data"
VOLUME      = "voice_celebrities"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_TABLE   = f"{CATALOG}.{SCHEMA}.voice_celebrities_raw"
OUT_TABLE   = f"{CATALOG}.{SCHEMA}.voice_celebrities_chunks"

WHISPER_ENDPOINT = "whisper_large_v3"

CHUNK_WORDS   = 150   # target words per chunk
OVERLAP_WORDS = 20    # word overlap between consecutive chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — List files in volume

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/voice_celebrities'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Transcribe with `ai_query` and save to `voice_celebrities_raw`
# MAGIC
# MAGIC `ai_query` calls the Whisper Large v3 endpoint on each audio file's binary content.
# MAGIC `CREATE TABLE IF NOT EXISTS` — transcription only runs once.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw
# MAGIC AS
# MAGIC SELECT
# MAGIC   path                                                                         AS file_path,
# MAGIC   INITCAP(REGEXP_REPLACE(REGEXP_REPLACE(
# MAGIC     REGEXP_EXTRACT(path, r'([^/]+)\.[^.]+$', 1), '-', ' '), '_', ' '))        AS speaker,
# MAGIC   CAST(ai_query('whisper_large_v3', content) AS STRING)                       AS full_text,
# MAGIC   current_timestamp()                                                          AS transcribed_at
# MAGIC FROM READ_FILES(
# MAGIC   '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/voice_celebrities',
# MAGIC   format => 'binaryFile'
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Flag non-speech clips
# MAGIC
# MAGIC Whisper hallucinations on non-speech audio produce short outputs or
# MAGIC bracketed tokens like `[ Music ]`, `[ Silence ]`, `(music)`.
# MAGIC These are flagged so they can be re-routed to notebook 05.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   speaker,
# MAGIC   full_text,
# MAGIC   LENGTH(full_text) AS transcript_len,
# MAGIC   CASE
# MAGIC     WHEN LENGTH(full_text) < 20                    THEN 'likely_nonspeech: too short'
# MAGIC     WHEN full_text RLIKE '(?i)\\[\\s*(music|silence|applause|noise|laughter)\\s*\\]'
# MAGIC                                                    THEN 'likely_nonspeech: whisper tag'
# MAGIC     WHEN full_text RLIKE '(?i)\\(\\s*(music|silence|applause)\\s*\\)'
# MAGIC                                                    THEN 'likely_nonspeech: whisper tag'
# MAGIC     ELSE 'speech'
# MAGIC   END AS speech_type
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw
# MAGIC ORDER BY transcript_len

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Chunk transcripts into overlapping windows
# MAGIC
# MAGIC Text-based chunking: split into ~`CHUNK_WORDS`-word windows with `OVERLAP_WORDS` overlap.
# MAGIC Speaker name is prepended as context prefix.
# MAGIC Non-speech clips (very short transcripts) are excluded — use notebook 05 for those.

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def _chunk_text(speaker: str, text: str, chunk_words: int, overlap_words: int) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(words):
        window = words[i : i + chunk_words]
        chunk_text = f"[Speaker: {speaker}] " + " ".join(window)
        chunks.append((chunk_idx, chunk_text, i, i + len(window) - 1))
        chunk_idx += 1
        i += max(1, chunk_words - overlap_words)
    return chunks

@F.pandas_udf(StructType([
    StructField("chunk_index", IntegerType()),
    StructField("chunk_text",  StringType()),
    StructField("word_start",  IntegerType()),
    StructField("word_end",    IntegerType()),
]))
def chunk_transcript_udf(speakers: pd.Series, texts: pd.Series) -> pd.DataFrame:
    rows = []
    for speaker, text in zip(speakers, texts):
        rows.extend(_chunk_text(speaker, text or "", CHUNK_WORDS, OVERLAP_WORDS))
    return pd.DataFrame(rows, columns=["chunk_index", "chunk_text", "word_start", "word_end"])

# COMMAND ----------

from pyspark.sql import functions as F

raw = spark.table(RAW_TABLE).filter(F.length(F.col("full_text")) >= 20)

chunks_df = raw.select(
    F.col("file_path"),
    F.col("speaker"),
    chunk_transcript_udf(F.col("speaker"), F.col("full_text")).alias("c"),
).select(
    F.concat(F.col("file_path"), F.lit("::"), F.col("c.chunk_index").cast("string")).alias("chunk_id"),
    F.col("file_path"),
    F.col("speaker"),
    F.col("c.chunk_index"),
    F.col("c.chunk_text"),
    F.col("c.word_start"),
    F.col("c.word_end"),
)

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
# MAGIC   LENGTH(full_text)                                           AS transcript_chars,
# MAGIC   CASE
# MAGIC     WHEN LENGTH(full_text) < 20 THEN 'non-speech → use notebook 05'
# MAGIC     ELSE 'speech'
# MAGIC   END AS speech_type
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw
# MAGIC ORDER BY transcript_chars

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample chunks

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   speaker,
# MAGIC   chunk_index,
# MAGIC   word_start,
# MAGIC   word_end,
# MAGIC   chunk_text
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_chunks
# MAGIC ORDER BY speaker, chunk_index
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunk count per speaker

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   speaker,
# MAGIC   COUNT(*)           AS num_chunks,
# MAGIC   MAX(word_end)      AS total_words
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_chunks
# MAGIC GROUP BY speaker
# MAGIC ORDER BY total_words DESC
