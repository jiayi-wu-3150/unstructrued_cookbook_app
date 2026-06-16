# Databricks notebook source
# MAGIC %md
# MAGIC # Non-Speech Audio — Classify & Describe with Gemini 2.5 Flash
# MAGIC
# MAGIC Uses Gemini 2.5 Flash multimodal to:
# MAGIC 1. **Classify** the audio: speech, music, sound effect, ambient, or mixed
# MAGIC 2. **Describe** non-speech clips in one sentence for search indexing
# MAGIC
# MAGIC This handles two sources:
# MAGIC - A dedicated non-speech volume (e.g. `sound_effects`)
# MAGIC - Clips flagged by notebook 04 as non-speech (short transcript in `voice_celebrities_raw`)
# MAGIC
# MAGIC **Pipeline:**
# MAGIC ```
# MAGIC Audio files → Pandas UDF (Gemini classify + describe) → sound_descriptions_raw
# MAGIC             → one chunk per clip                       → sound_chunks
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
# MAGIC ## Step 2 — Classify and describe with Gemini 2.5 Flash
# MAGIC
# MAGIC Single Gemini call per clip returns:
# MAGIC - `audio_type`: speech | music | sound_effect | ambient | mixed
# MAGIC - `description`: one sentence for search indexing
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` — model only called once per file.

# COMMAND ----------

# MAGIC %pip install mlflow --quiet

# COMMAND ----------

import base64
import json
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

CLASSIFY_PROMPT = """Listen to this audio clip and respond with valid JSON only, no markdown:
{
  "audio_type": "<one of: speech, music, sound_effect, ambient, mixed>",
  "description": "<one sentence describing what you hear, suitable for search indexing>"
}
Focus on: type of sound, mood/energy, notable instruments or sources."""

def _mime(path: str) -> str:
    return MIME_MAP.get(path.rsplit(".", 1)[-1].lower(), "audio/wav")

@F.pandas_udf(StructType([
    StructField("audio_type",  StringType()),
    StructField("description", StringType()),
    StructField("mime_type",   StringType()),
]))
def classify_describe_udf(paths: pd.Series, contents: pd.Series) -> pd.DataFrame:
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    results = []
    for path, content in zip(paths, contents):
        mime = _mime(path)
        try:
            audio_b64 = base64.b64encode(bytes(content)).decode("utf-8")
            response = client.predict(
                endpoint=GEMINI_ENDPOINT,
                inputs={
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CLASSIFY_PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": f"data:{mime};base64,{audio_b64}"
                            }},
                        ],
                    }]
                },
            )
            raw = response["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            parsed = json.loads(raw)
            audio_type  = parsed.get("audio_type", "unknown")
            description = parsed.get("description", raw)
        except Exception as e:
            audio_type  = "error"
            description = f"ERROR: {e}"
        results.append((audio_type, description, mime))
    return pd.DataFrame(results, columns=["audio_type", "description", "mime_type"])

# COMMAND ----------

if not spark.catalog.tableExists(RAW_TABLE):
    raw_df = spark.read.format("binaryFile").load(VOLUME_PATH)

    result = raw_df.select(
        F.col("path").alias("file_path"),
        F.initcap(F.regexp_replace(
            F.regexp_replace(F.regexp_extract(F.col("path"), r"([^/]+)\.[^.]+$", 1), "-", " "),
            "_", " "
        )).alias("clip_name"),
        classify_describe_udf(F.col("path"), F.col("content")).alias("d"),
        F.col("length").alias("file_size_bytes"),
        F.current_timestamp().alias("described_at"),
    ).select(
        "file_path",
        "clip_name",
        F.col("d.audio_type"),
        F.col("d.description"),
        F.col("d.mime_type"),
        "file_size_bytes",
        "described_at",
    )

    result.write.format("delta").saveAsTable(RAW_TABLE)
    print(f"Saved {result.count()} rows to {RAW_TABLE}")
else:
    print(f"{RAW_TABLE} already exists — skipping.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build `sound_chunks` (one chunk per clip)

# COMMAND ----------

descriptions = spark.table(RAW_TABLE)

chunks_df = descriptions.select(
    F.concat(F.col("file_path"), F.lit("::0")).alias("chunk_id"),
    F.col("file_path"),
    F.col("clip_name"),
    F.col("audio_type"),
    F.col("mime_type"),
    F.lit(0).alias("chunk_index"),
    F.concat(
        F.lit("[Sound: "), F.col("clip_name"),
        F.lit(" ("), F.col("audio_type"), F.lit(")] "),
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
# MAGIC ### All classifications and descriptions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   clip_name,
# MAGIC   audio_type,
# MAGIC   ROUND(file_size_bytes / 1024, 1) AS size_kb,
# MAGIC   description
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.sound_descriptions_raw
# MAGIC ORDER BY audio_type, clip_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunks ready for embedding

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT audio_type, COUNT(*) AS clips FROM serverless_stable_r4umw1_catalog.unstructured_data.sound_chunks GROUP BY audio_type

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT clip_name, audio_type, chunk_text
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.sound_chunks
# MAGIC ORDER BY audio_type, clip_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Handle non-speech clips flagged by notebook 04 (optional)
# MAGIC
# MAGIC Clips in `voice_celebrities_raw` with very short transcripts are likely non-speech.
# MAGIC Re-describe them with Gemini and append to `sound_chunks`.

# COMMAND ----------

if spark.catalog.tableExists("serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw"):
    nonspeech = spark.table("serverless_stable_r4umw1_catalog.unstructured_data.voice_celebrities_raw") \
        .filter(F.length(F.col("full_text")) < 20) \
        .select("file_path", "speaker")

    count = nonspeech.count()
    if count == 0:
        print("No non-speech clips found in voice_celebrities_raw.")
    else:
        print(f"Found {count} non-speech clips — re-describing with Gemini.")
        ns_binary = spark.read.format("binaryFile").load(
            [r["file_path"] for r in nonspeech.collect()]
        )
        ns_result = ns_binary.select(
            F.col("path").alias("file_path"),
            classify_describe_udf(F.col("path"), F.col("content")).alias("d"),
            F.col("length").alias("file_size_bytes"),
            F.current_timestamp().alias("described_at"),
        ).join(nonspeech, "file_path").select(
            F.concat(F.col("file_path"), F.lit("::0")).alias("chunk_id"),
            F.col("file_path"),
            F.col("speaker").alias("clip_name"),
            F.col("d.audio_type"),
            F.col("d.mime_type"),
            F.lit(0).alias("chunk_index"),
            F.concat(
                F.lit("[Sound: "), F.col("speaker"),
                F.lit(" ("), F.col("d.audio_type"), F.lit(")] "),
                F.col("d.description")
            ).alias("chunk_text"),
            F.col("file_size_bytes"),
            F.col("described_at"),
        )
        ns_result.write.format("delta").mode("append").saveAsTable(OUT_TABLE)
        print(f"Appended {ns_result.count()} non-speech chunks to {OUT_TABLE}")
else:
    print("voice_celebrities_raw not found — skipping.")
