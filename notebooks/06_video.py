# Databricks notebook source
# MAGIC %md
# MAGIC # Video Processing — Frame Extraction, CLIP Embeddings & LLM Descriptions
# MAGIC
# MAGIC Processes MP4 video files from a UC Volume into a searchable Vector Search index.
# MAGIC
# MAGIC **Prerequisite**: `model_setup/clip_model.py` must have run successfully and
# MAGIC `clip_embedding_endpoint` must be in READY state.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC ```
# MAGIC MP4 files (UC Volume)
# MAGIC   → OpenCV frame extraction (every Nth frame, driver-side)
# MAGIC   → video_clips  (frame_id, video_id, frame_num, model_input: base64 JPEG)
# MAGIC   → CLIP embeddings via ai_query(clip_embedding_endpoint)
# MAGIC   → video_clips_embeddings  (frame_id, image_embeddings: ARRAY<DOUBLE>)
# MAGIC   → Gemini 2.5 Flash multimodal descriptions via ai_query
# MAGIC   → video_descriptions_raw  (frame_id, frame_description: STRING)
# MAGIC   → JOIN → video_clips_gold  (embeddings + descriptions + metadata)
# MAGIC   → Vector Search index  (768-dim CLIP space, hybrid text+vector)
# MAGIC ```
# MAGIC
# MAGIC **Search at query time**: encode a text query with the CLIP text encoder
# MAGIC locally, then run a vector similarity search against the frame embeddings.
# MAGIC Because CLIP image and text embeddings share the same 768-dim space, the
# MAGIC search is cross-modal — "cat jumping" finds frames of cats jumping even
# MAGIC without per-frame text labels.

# COMMAND ----------

# MAGIC %pip install opencv-python pillow databricks-vectorsearch --quiet
# MAGIC %restart_python

# COMMAND ----------

# Constants — defined AFTER %restart_python so they survive the kernel restart
CATALOG       = "serverless_stable_r4umw1_catalog"
SCHEMA        = "unstructured_data"
VOLUME        = "video_clips_sample"
VOLUME_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

CLIPS_TABLE   = f"{CATALOG}.{SCHEMA}.video_clips"
EMBED_TABLE   = f"{CATALOG}.{SCHEMA}.video_clips_embeddings"
DESC_TABLE    = f"{CATALOG}.{SCHEMA}.video_descriptions_raw"
GOLD_TABLE    = f"{CATALOG}.{SCHEMA}.video_clips_gold"

CLIP_ENDPOINT   = "clip_embedding_endpoint"
GEMINI_ENDPOINT = "databricks-gemini-2-5-flash"

FRAME_INTERVAL = 30   # extract every 30th frame (≈1 frame/sec at 30 fps)
VS_ENDPOINT    = "video-search-endpoint"
VS_INDEX       = f"{CATALOG}.{SCHEMA}.video_clips_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — List video files in volume

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/video_clips_sample'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Extract keyframes from video files
# MAGIC
# MAGIC OpenCV reads each MP4 on the driver and samples every `FRAME_INTERVAL`-th frame.
# MAGIC Frames are encoded as JPEG bytes then base64-encoded to STRING for storage in
# MAGIC Delta — the same `model_input` format expected by the CLIP endpoint.
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` — extraction runs only once.

# COMMAND ----------

import os
import cv2
import base64
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def _extract_frames(video_path: str, frame_interval: int) -> list:
    """Return list of (video_id, frame_num, base64_jpeg) for every Nth frame."""
    cap = cv2.VideoCapture(video_path)
    video_id = os.path.basename(video_path).rsplit(".", 1)[0]
    rows, count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            rows.append((video_id, count, b64))
        count += 1
    cap.release()
    return rows

# COMMAND ----------

if not spark.catalog.tableExists(CLIPS_TABLE):
    mp4_files = [
        os.path.join(VOLUME_PATH, f)
        for f in os.listdir(VOLUME_PATH)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    print(f"Found {len(mp4_files)} video file(s): {[os.path.basename(f) for f in mp4_files]}")

    all_frames = []
    for path in mp4_files:
        frames = _extract_frames(path, FRAME_INTERVAL)
        print(f"  {os.path.basename(path)}: {len(frames)} frames extracted")
        all_frames.extend(frames)

    schema = StructType([
        StructField("video_id",    StringType()),
        StructField("frame_num",   IntegerType()),
        StructField("model_input", StringType()),  # base64 JPEG
    ])
    clips_df = spark.createDataFrame(all_frames, schema=schema) \
        .withColumn("frame_id", F.concat(F.col("video_id"), F.lit("::"), F.col("frame_num").cast("string")))

    clips_df.write.format("delta").saveAsTable(CLIPS_TABLE)
    print(f"\nSaved {clips_df.count()} frames to {CLIPS_TABLE}")
else:
    print(f"{CLIPS_TABLE} already exists — skipping extraction.")
    clips_df = spark.table(CLIPS_TABLE)

clips_df.select("frame_id", "video_id", "frame_num").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Embed frames with CLIP via `ai_query()`
# MAGIC
# MAGIC Calls the `clip_embedding_endpoint` (deployed in `model_setup/clip_model.py`) for each
# MAGIC frame. Returns a 768-dim ARRAY<DOUBLE> per frame in the same embedding space
# MAGIC used for text queries at search time.
# MAGIC
# MAGIC **Note**: the endpoint must be in READY state. Check its status at:
# MAGIC `<workspace>/ml/endpoints/clip_embedding_endpoint`

# COMMAND ----------

if not spark.catalog.tableExists(EMBED_TABLE):
    spark.sql(f"""
        CREATE TABLE {EMBED_TABLE} AS
        SELECT
            frame_id,
            video_id,
            frame_num,
            ai_query(
                '{CLIP_ENDPOINT}',
                request    => named_struct('model_input', model_input),
                returnType => 'ARRAY<STRUCT<model_input: ARRAY<DOUBLE>>>'
            )[0].model_input AS image_embeddings
        FROM {CLIPS_TABLE}
    """)
    count = spark.table(EMBED_TABLE).count()
    print(f"Saved {count} embeddings to {EMBED_TABLE}")
else:
    print(f"{EMBED_TABLE} already exists — skipping embedding.")

spark.table(EMBED_TABLE).select("frame_id", "video_id", "frame_num").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Describe frames with Gemini 2.5 Flash (multimodal)
# MAGIC
# MAGIC Generates a short natural-language description of each keyframe using
# MAGIC Gemini 2.5 Flash multimodal via `ai_query()`.
# MAGIC
# MAGIC The `files` parameter requires BINARY; `unbase64(model_input)` converts
# MAGIC the base64 STRING stored in `video_clips` to BINARY inline.
# MAGIC
# MAGIC Descriptions enrich the VS index with searchable text alongside the CLIP vectors.

# COMMAND ----------

if not spark.catalog.tableExists(DESC_TABLE):
    spark.sql(f"""
        CREATE TABLE {DESC_TABLE} AS
        SELECT
            frame_id,
            video_id,
            frame_num,
            ai_query(
                '{GEMINI_ENDPOINT}',
                'Describe this video frame in 2-3 sentences for search indexing. '
                'Include: main subjects, actions, setting, and notable objects. '
                'Be specific and factual.',
                files => array(unbase64(model_input))
            ) AS frame_description
        FROM {CLIPS_TABLE}
    """)
    count = spark.table(DESC_TABLE).count()
    print(f"Saved {count} descriptions to {DESC_TABLE}")
else:
    print(f"{DESC_TABLE} already exists — skipping description.")

spark.table(DESC_TABLE).show(5, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Build `video_clips_gold`: join embeddings + descriptions
# MAGIC
# MAGIC Merges CLIP embeddings and Gemini descriptions into a single table ready
# MAGIC for Vector Search indexing. Change Data Feed is enabled so the VS Delta Sync
# MAGIC index can track incremental updates.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {GOLD_TABLE} AS
    SELECT
        e.frame_id,
        e.video_id,
        e.frame_num,
        e.image_embeddings,
        d.frame_description
    FROM {EMBED_TABLE} e
    JOIN {DESC_TABLE} d USING (frame_id)
""")

spark.sql(f"ALTER TABLE {GOLD_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
count = spark.table(GOLD_TABLE).count()
print(f"Saved {count} rows to {GOLD_TABLE}")
spark.table(GOLD_TABLE).select("frame_id", "video_id", "frame_num", "frame_description").show(5, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Create Vector Search endpoint and index
# MAGIC
# MAGIC Uses a **Delta Sync** index with pre-computed embeddings.
# MAGIC The `image_embeddings` column (768-dim) was pre-computed by the CLIP endpoint.
# MAGIC `frame_description` is also synced for hybrid text+vector search.
# MAGIC
# MAGIC Endpoint creation takes ~5–10 min; index sync takes ~1–3 min for small datasets.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs = VectorSearchClient(disable_notice=True)

# Create endpoint if it doesn't exist
existing_endpoints = {e["name"] for e in vs.list_endpoints().get("endpoints", [])}
if VS_ENDPOINT not in existing_endpoints:
    print(f"Creating VS endpoint '{VS_ENDPOINT}'...")
    vs.create_endpoint(name=VS_ENDPOINT, endpoint_type="STORAGE_OPTIMIZED")
    print(f"Endpoint '{VS_ENDPOINT}' created.")
else:
    print(f"Endpoint '{VS_ENDPOINT}' already exists.")

# COMMAND ----------

# Create or sync the index
existing_indexes = {idx["name"] for idx in vs.list_indexes(VS_ENDPOINT).get("vector_indexes", [])}

if VS_INDEX not in existing_indexes:
    print(f"Creating index '{VS_INDEX}'...")
    vs.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=GOLD_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="frame_id",
        embedding_dimension=768,
        embedding_vector_column="image_embeddings",
        columns_to_sync=["video_id", "frame_num", "frame_description"],
    )
    print(f"Index '{VS_INDEX}' created — initial sync in progress.")
else:
    print(f"Index '{VS_INDEX}' already exists — triggering sync...")
    vs.get_index(VS_ENDPOINT, VS_INDEX).sync()
    print("Sync triggered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Smoke test: inspect sample embeddings

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   frame_id,
# MAGIC   video_id,
# MAGIC   frame_num,
# MAGIC   SIZE(image_embeddings) AS embedding_dim,
# MAGIC   ROUND(image_embeddings[0], 6) AS emb_0,
# MAGIC   ROUND(image_embeddings[1], 6) AS emb_1,
# MAGIC   LEFT(frame_description, 120) AS description_preview
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.video_clips_gold
# MAGIC ORDER BY frame_num
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Aggregate descriptions per video (optional summary)
# MAGIC
# MAGIC Concatenates all frame descriptions for a video and uses an LLM to generate
# MAGIC a single summary. Useful for building a video-level search layer on top of
# MAGIC the frame-level index.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   video_id,
# MAGIC   COUNT(*) AS num_frames,
# MAGIC   ai_query(
# MAGIC     'databricks-claude-sonnet-4',
# MAGIC     CONCAT(
# MAGIC       'Summarise this video in 3-4 sentences based on its keyframe descriptions. ',
# MAGIC       'Focus on: what is happening, who/what is present, and the overall setting.\n\n',
# MAGIC       'Frame descriptions:\n',
# MAGIC       CONCAT_WS('\n', COLLECT_LIST(CONCAT('Frame ', CAST(frame_num AS STRING), ': ', frame_description)))
# MAGIC     )
# MAGIC   ) AS video_summary
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.video_clips_gold
# MAGIC GROUP BY video_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Summary
# MAGIC
# MAGIC | Layer | Technology | Details |
# MAGIC |-------|-----------|---------|
# MAGIC | Frame extraction | OpenCV (`cv2`) | Every 30th frame, driver-side, base64 JPEG |
# MAGIC | Visual embedding | CLIP (`clip_embedding_endpoint`) | 768-dim, same space for image + text |
# MAGIC | Frame description | Gemini 2.5 Flash multimodal | `ai_query` with `files => array(unbase64(...))` |
# MAGIC | Vector index | Databricks Vector Search | Delta Sync, pre-computed embeddings, 768-dim |
# MAGIC | Query-time encoding | CLIP text encoder (local) | `model.get_text_features(**inputs)` — see `model_setup/clip_model.py` note |
# MAGIC | Search | Cross-modal cosine similarity | Text query → CLIP embedding → VS similarity search |
# MAGIC
# MAGIC ## Design decisions
# MAGIC
# MAGIC **Why driver-side frame extraction?**
# MAGIC For small video datasets (a few files), reading MP4s with OpenCV on the
# MAGIC driver and creating a DataFrame from the extracted frames avoids the overhead
# MAGIC of a distributed UDF. For large datasets (100+ videos), use a Pandas UDF
# MAGIC with `mapInPandas` to distribute extraction across workers.
# MAGIC
# MAGIC **Why base64 STRING for `model_input`?**
# MAGIC The CLIP pyfunc endpoint expects a STRING column (base64). Storing as STRING
# MAGIC avoids the `base64()` → `unbase64()` round-trip at embedding time. Gemini's
# MAGIC `files` param needs BINARY, so `unbase64(model_input)` converts inline in SQL.
# MAGIC
# MAGIC **Why CLIP for embeddings instead of `ai_parse_document`?**
# MAGIC `ai_parse_document` is optimised for documents (PDFs, DOCX) and returns text
# MAGIC descriptions of image elements — not dense embeddings. CLIP provides a
# MAGIC continuous 768-dim space shared by both images and text, enabling cross-modal
# MAGIC search. Use `ai_parse_document` for OCR-style extraction from document images;
# MAGIC use CLIP for semantic visual search.
# MAGIC
# MAGIC **Why Gemini for descriptions instead of `ai_parse_document`?**
# MAGIC `ai_parse_document` supports images (JPG/PNG) and produces element-level
# MAGIC descriptions, but requires saving frames as files to a volume first. Gemini
# MAGIC multimodal via `ai_query` accepts inline BINARY, making the pipeline fully
# MAGIC table-driven with no intermediate file writes.
