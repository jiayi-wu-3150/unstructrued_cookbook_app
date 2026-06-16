# Databricks notebook source
# MAGIC %md
# MAGIC # Non-Speech Audio — Description with Gemini 2.5 Flash
# MAGIC
# MAGIC Generates searchable text descriptions for non-speech audio (sound effects,
# MAGIC music, ambient, nature sounds) using a multimodal LLM.
# MAGIC
# MAGIC Unlike speech, these clips cannot be transcribed — the model listens and
# MAGIC describes what it hears in natural language, which becomes the `chunk_text`
# MAGIC for embedding and retrieval.
# MAGIC
# MAGIC **Source volume:** `serverless_stable_r4umw1_catalog.unstructured_data.sound_effects`
# MAGIC
# MAGIC **Also handles:** clips flagged as non-speech by `faster-whisper` in notebook 04
# MAGIC (`avg_no_speech_prob > 0.5` from `voice_celebrities_raw`)
# MAGIC
# MAGIC **Pipeline:**
# MAGIC ```
# MAGIC Audio files → Pandas UDF (Gemini 2.5 Flash multimodal) → sound_descriptions_raw
# MAGIC             → one chunk per clip (description = chunk_text) → sound_chunks
# MAGIC ```

# COMMAND ----------

CATALOG     = "serverless_stable_r4umw1_catalog"
SCHEMA      = "unstructured_data"
VOLUME      = "sound_effects"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_TABLE   = f"{CATALOG}.{SCHEMA}.sound_descriptions_raw"
OUT_TABLE   = f"{CATALOG}.{SCHEMA}.sound_chunks"

GEMINI_ENDPOINT = "databricks-gemini-2-5-flash"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — List files in volume

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/sound_effects'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Describe audio clips with Gemini 2.5 Flash
# MAGIC
# MAGIC Each audio file is base64-encoded and sent to Gemini 2.5 Flash via the
# MAGIC Databricks Foundation Model endpoint. The model returns a one-to-two sentence
# MAGIC description of what it hears.
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` — model is only called once per file.

# COMMAND ----------

# MAGIC %pip install mlflow --quiet

# COMMAND ----------

import base64
import re
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

MIME_MAP = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "flac": "audio/flac",
    "ogg":  "audio/ogg",
    "m4a":  "audio/mp4",
    "aac":  "audio/aac",
}

def _mime_from_path(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    return MIME_MAP.get(ext, "audio/wav")

@F.pandas_udf(StructType([
    StructField("description", StringType()),
    StructField("mime_type",   StringType()),
]))
def describe_audio_udf(paths: pd.Series, contents: pd.Series) -> pd.DataFrame:
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    results = []
    for path, content in zip(paths, contents):
        mime = _mime_from_path(path)
        try:
            audio_b64 = base64.b64encode(bytes(content)).decode("utf-8")
            response = client.predict(
                endpoint=GEMINI_ENDPOINT,
                inputs={
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Listen to this audio clip and describe it in one to two sentences "
                                    "for search indexing. Include: type of sound (music, nature, machinery, "
                                    "crowd, animal, etc.), mood or energy level, any notable instruments or "
                                    "sound sources, and approximate duration feel. Be concise and specific."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{audio_b64}"
                                },
                            },
                        ],
                    }]
                },
            )
            description = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            description = f"ERROR: {e}"
        results.append((description, mime))
    return pd.DataFrame(results, columns=["description", "mime_type"])

# COMMAND ----------

if not spark.catalog.tableExists(RAW_TABLE):
    raw_df = spark.read.format("binaryFile").load(VOLUME_PATH)

    result = raw_df.select(
        F.col("path").alias("file_path"),
        F.regexp_replace(
            F.regexp_replace(F.regexp_extract(F.col("path"), r"([^/]+)\.[^.]+$", 1), "-", " "),
            "_", " "
        ).alias("clip_name_raw"),
        describe_audio_udf(F.col("path"), F.col("content")).alias("d"),
        F.col("length").alias("file_size_bytes"),
        F.current_timestamp().alias("described_at"),
    ).select(
        "file_path",
        F.initcap(F.col("clip_name_raw")).alias("clip_name"),
        F.col("d.description"),
        F.col("d.mime_type"),
        "file_size_bytes",
        "described_at",
    )

    result.write.format("delta").saveAsTable(RAW_TABLE)
    print(f"Saved {result.count()} rows to {RAW_TABLE}")
else:
    print(f"{RAW_TABLE} already exists — skipping description.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build `sound_chunks` (one chunk per clip)
# MAGIC
# MAGIC For non-speech audio there is no segmentation — the entire clip is one chunk.
# MAGIC The description becomes `chunk_text` for embedding, prefixed with clip name.

# COMMAND ----------

descriptions = spark.table(RAW_TABLE)

chunks_df = descriptions.select(
    F.concat(F.col("file_path"), F.lit("::0")).alias("chunk_id"),
    F.col("file_path"),
    F.col("clip_name"),
    F.col("mime_type"),
    F.lit(0).alias("chunk_index"),
    F.concat(
        F.lit("[Sound: "), F.col("clip_name"), F.lit("] "),
        F.col("description")
    ).alias("chunk_text"),
    F.col("file_size_bytes"),
    F.col("described_at"),
)

chunks_df.write.format("delta").mode("overwrite").saveAsTable(OUT_TABLE)
print(f"Saved {chunks_df.count()} chunks to {OUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Inspect results

# COMMAND ----------

# MAGIC %md
# MAGIC ### All descriptions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   clip_name,
# MAGIC   mime_type,
# MAGIC   ROUND(file_size_bytes / 1024, 1) AS size_kb,
# MAGIC   description
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.sound_descriptions_raw
# MAGIC ORDER BY clip_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunks ready for embedding

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   clip_name,
# MAGIC   chunk_text
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.sound_chunks
# MAGIC ORDER BY clip_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Handle non-speech clips from voice_celebrities (optional)
# MAGIC
# MAGIC If notebook 04 flagged any clips as non-speech (`avg_no_speech_prob > 0.5`),
# MAGIC re-describe them here and merge into `sound_chunks`.

# COMMAND ----------

from pyspark.sql.types import BinaryType

# Only run if the voice_celebrities_raw table exists
if spark.catalog.tableExists("serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw"):
    nonspeech = spark.table("serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw") \
        .filter(F.col("avg_no_speech_prob") > 0.5) \
        .select("file_path", "speaker")

    if nonspeech.count() == 0:
        print("No non-speech clips found in voice_celebrities_raw.")
    else:
        print(f"Found {nonspeech.count()} non-speech clips — re-describing with Gemini.")

        # Read binary content for flagged files
        nonspeech_files = [r["file_path"] for r in nonspeech.collect()]
        ns_binary = spark.read.format("binaryFile").load(nonspeech_files)

        ns_result = ns_binary.select(
            F.col("path").alias("file_path"),
            describe_audio_udf(F.col("path"), F.col("content")).alias("d"),
            F.col("length").alias("file_size_bytes"),
            F.current_timestamp().alias("described_at"),
        ).join(nonspeech, "file_path") \
         .select(
            F.concat(F.col("file_path"), F.lit("::0")).alias("chunk_id"),
            F.col("file_path"),
            F.col("speaker").alias("clip_name"),
            F.col("d.mime_type"),
            F.lit(0).alias("chunk_index"),
            F.concat(
                F.lit("[Sound: "), F.col("speaker"), F.lit("] "),
                F.col("d.description")
            ).alias("chunk_text"),
            F.col("file_size_bytes"),
            F.col("described_at"),
        )

        ns_result.write.format("delta").mode("append").saveAsTable(OUT_TABLE)
        print(f"Appended {ns_result.count()} non-speech chunks to {OUT_TABLE}")
else:
    print("voice_celebrities_raw not found — skipping non-speech detection.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Note: Multimodal audio support in Gemini
# MAGIC
# MAGIC The UDF above uses the OpenAI-compatible `image_url` field with a `data:audio/...;base64,`
# MAGIC URI. This works when the Databricks Gemini endpoint has audio input enabled.
# MAGIC
# MAGIC If the endpoint returns an error for audio content, use the **spectrogram fallback**:
# MAGIC
# MAGIC ```python
# MAGIC import librosa
# MAGIC import librosa.display
# MAGIC import matplotlib.pyplot as plt
# MAGIC import io, base64
# MAGIC
# MAGIC def audio_to_spectrogram_b64(audio_bytes: bytes) -> str:
# MAGIC     """Convert audio bytes to a mel-spectrogram PNG, return base64."""
# MAGIC     import soundfile as sf
# MAGIC     import numpy as np
# MAGIC     audio_io = io.BytesIO(audio_bytes)
# MAGIC     y, sr = sf.read(audio_io)
# MAGIC     if y.ndim > 1:
# MAGIC         y = y.mean(axis=1)  # stereo → mono
# MAGIC     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# MAGIC     S_dB = librosa.power_to_db(S, ref=np.max)
# MAGIC     fig, ax = plt.subplots(figsize=(6, 3))
# MAGIC     librosa.display.specshow(S_dB, sr=sr, ax=ax)
# MAGIC     fig.tight_layout()
# MAGIC     buf = io.BytesIO()
# MAGIC     fig.savefig(buf, format="png")
# MAGIC     plt.close(fig)
# MAGIC     return base64.b64encode(buf.getvalue()).decode()
# MAGIC
# MAGIC # Then in the UDF, send the spectrogram image to Gemini instead of raw audio.
# MAGIC # Gemini will describe the visual frequency patterns.
# MAGIC ```
# MAGIC
# MAGIC Install: `%pip install librosa soundfile matplotlib`
