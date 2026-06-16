# Databricks notebook source
# MAGIC %md
# MAGIC # Audio Parsing — Transcription with `ai_query` (Whisper Large v3)
# MAGIC
# MAGIC Transcribes celebrity voice clips from a UC Volume using the
# MAGIC `whisper_large_v3` Databricks Model Serving endpoint.
# MAGIC
# MAGIC **Source volume:** `serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities`
# MAGIC
# MAGIC **Note on file size:** `ai_query` has a 16 MB request limit. Uncompressed WAV
# MAGIC files are often larger than that. Step 2 downsamples audio to 16 kHz mono
# MAGIC (Whisper's native format) before transcription, reducing size ~8–10×.
# MAGIC
# MAGIC **Non-speech detection:** clips where the transcript is very short or matches
# MAGIC known Whisper hallucination patterns (`[ Music ]`, `[ Silence ]`, etc.) are
# MAGIC flagged. Route those to notebook `05_audio_nonspeech.py` for Gemini description.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC ```
# MAGIC WAV files → downsample 16kHz mono (UDF) → ai_query(whisper_large_v3)
# MAGIC           → voice_celebrities_raw
# MAGIC           → word-window chunks → voice_celebrities_chunks
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
# MAGIC ## Step 2 — Downsample audio and transcribe with `ai_query`
# MAGIC
# MAGIC WAV files are downsampled to 16 kHz mono before being sent to Whisper.
# MAGIC This keeps each request well under the 16 MB limit while matching the
# MAGIC sample rate Whisper expects.
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` — transcription only runs once.

# COMMAND ----------

# MAGIC %pip install soundfile --quiet

# COMMAND ----------

import io
import re as _re
import numpy as np
import soundfile as sf
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BinaryType

def _compress(content_bytes: bytes) -> bytes:
    """Resample to 16 kHz mono WAV (Whisper's native format; reduces ~15 MB WAV to ~2 MB)."""
    y, sr = sf.read(io.BytesIO(content_bytes))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        n = int(len(y) * 16000 / sr)
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
    out = io.BytesIO()
    sf.write(out, y.astype(np.float32), 16000, format="WAV", subtype="PCM_16")
    return out.getvalue()

# COMMAND ----------

if not spark.catalog.tableExists(RAW_TABLE):
    # Collect files to driver (13 files, fine for a small dataset)
    rows = spark.read.format("binaryFile").load(VOLUME_PATH).collect()

    data = []
    for row in rows:
        raw_name = _re.sub(r"\.[^.]+$", "", row["path"].split("/")[-1])
        speaker = raw_name.replace("-", " ").replace("_", " ").title()
        compressed = _compress(bytes(row["content"]))
        data.append((row["path"], speaker, compressed))

    compressed_df = spark.createDataFrame(data, schema=["file_path", "speaker", "content"])
    compressed_df.createOrReplaceTempView("audio_to_transcribe")

    result = spark.sql(f"""
        SELECT
          file_path,
          speaker,
          CAST(ai_query('{WHISPER_ENDPOINT}', content) AS STRING) AS full_text,
          current_timestamp()                                      AS transcribed_at
        FROM audio_to_transcribe
    """)

    result.write.format("delta").saveAsTable(RAW_TABLE)
    print(f"Saved {result.count()} rows to {RAW_TABLE}")
else:
    print(f"{RAW_TABLE} already exists — skipping transcription.")

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
# MAGIC     WHEN LENGTH(full_text) < 20
# MAGIC       THEN 'likely_nonspeech: too short'
# MAGIC     WHEN full_text RLIKE '(?i)\\[\\s*(music|silence|applause|noise|laughter)\\s*\\]'
# MAGIC       THEN 'likely_nonspeech: whisper tag'
# MAGIC     WHEN full_text RLIKE '(?i)\\(\\s*(music|silence|applause)\\s*\\)'
# MAGIC       THEN 'likely_nonspeech: whisper tag'
# MAGIC     ELSE 'speech'
# MAGIC   END AS speech_type
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw
# MAGIC ORDER BY transcript_len

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Chunk transcripts and save to `voice_celebrities_chunks`
# MAGIC
# MAGIC Text-based chunking: ~`CHUNK_WORDS`-word windows with `OVERLAP_WORDS` overlap.
# MAGIC Non-speech clips (short transcripts) are excluded.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

_CHUNK_SCHEMA = ArrayType(StructType([
    StructField("chunk_index", IntegerType()),
    StructField("chunk_text",  StringType()),
    StructField("word_start",  IntegerType()),
    StructField("word_end",    IntegerType()),
]))

@F.udf(_CHUNK_SCHEMA)
def chunk_transcript_udf(speaker, text):
    """Returns an array of chunks — use explode() to expand to one row per chunk."""
    words = (text or "").split()
    if not words:
        return []
    result, i, idx = [], 0, 0
    while i < len(words):
        window = words[i : i + CHUNK_WORDS]
        result.append((idx, f"[Speaker: {speaker}] " + " ".join(window), i, i + len(window) - 1))
        idx += 1
        i += max(1, CHUNK_WORDS - OVERLAP_WORDS)
    return result

# COMMAND ----------

raw = spark.table(RAW_TABLE).filter(F.length(F.col("full_text")) >= 20)

chunks_df = raw.select(
    "file_path", "speaker",
    F.explode(chunk_transcript_udf(F.col("speaker"), F.col("full_text"))).alias("c"),
).select(
    F.concat(F.col("file_path"), F.lit("::"), F.col("c.chunk_index").cast("string")).alias("chunk_id"),
    "file_path", "speaker",
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

# MAGIC %sql
# MAGIC SELECT
# MAGIC   speaker,
# MAGIC   LENGTH(full_text) AS transcript_chars,
# MAGIC   CASE WHEN LENGTH(full_text) < 20 THEN 'non-speech → use notebook 05' ELSE 'speech' END AS speech_type
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw
# MAGIC ORDER BY transcript_chars

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT speaker, chunk_index, word_start, word_end, chunk_text
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_chunks
# MAGIC ORDER BY speaker, chunk_index
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT speaker, COUNT(*) AS num_chunks, MAX(word_end) AS total_words
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_chunks
# MAGIC GROUP BY speaker
# MAGIC ORDER BY total_words DESC
