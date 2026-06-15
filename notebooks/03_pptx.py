# Databricks notebook source
# MAGIC %md
# MAGIC # PPTX Parsing with `ai_parse_document`
# MAGIC
# MAGIC Parses a PowerPoint file from a UC Volume using `ai_parse_document` and chunks
# MAGIC by slide — one chunk per slide is the natural atomic unit for presentations.
# MAGIC
# MAGIC **Source volume:** `serverless_stable_r4umw1_catalog.unstructured_data.pptx`
# MAGIC
# MAGIC **Chunking strategy:** all elements on a slide are aggregated into one row.
# MAGIC Slide title/header becomes the context prefix; figure descriptions are included
# MAGIC so image-heavy slides are still retrievable.

# COMMAND ----------

CATALOG     = "serverless_stable_r4umw1_catalog"
SCHEMA      = "unstructured_data"
VOLUME      = "pptx"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
IMAGE_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/pptx_images"
RAW_TABLE   = f"{CATALOG}.{SCHEMA}.pptx_parsed_raw"
EL_TABLE    = f"{CATALOG}.{SCHEMA}.pptx_parsed"
OUT_TABLE   = f"{CATALOG}.{SCHEMA}.pptx_chunks"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Preview: list files in volume

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/pptx'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Parse PPTX and save raw VARIANT to `pptx_parsed_raw`
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` — safe to re-run, model only called once.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS serverless_stable_r4umw1_catalog.unstructured_data.pptx_images

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed_raw
# MAGIC AS
# MAGIC SELECT
# MAGIC   path                AS file_path,
# MAGIC   ai_parse_document(
# MAGIC     content,
# MAGIC     map(
# MAGIC       'version',                 '2.0',
# MAGIC       'descriptionElementTypes', 'figure',
# MAGIC       'imageOutputPath',         '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/pptx_images'
# MAGIC     )
# MAGIC   )                   AS parsed,
# MAGIC   current_timestamp() AS parsed_at
# MAGIC FROM READ_FILES(
# MAGIC   '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/pptx',
# MAGIC   format => 'binaryFile'
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Explode elements to `pptx_parsed`
# MAGIC
# MAGIC `slide_id` is extracted from `bbox[0].page_id` — the slide number (0-indexed).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed
# MAGIC AS
# MAGIC SELECT
# MAGIC   r.file_path,
# MAGIC   el.value:id::INT                                                   AS element_id,
# MAGIC   el.value:bbox[0]:page_id::INT                                      AS slide_id,
# MAGIC   el.value:type::STRING                                              AS element_type,
# MAGIC   el.value:content::STRING                                           AS content,
# MAGIC   el.value:description::STRING                                       AS description,
# MAGIC   COALESCE(el.value:description::STRING, el.value:content::STRING)   AS text_to_embed,
# MAGIC   ROUND(el.value:confidence::DOUBLE, 3)                              AS confidence,
# MAGIC   el.value:bbox                                                      AS bbox,
# MAGIC   r.parsed_at
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed_raw AS r,
# MAGIC LATERAL variant_explode(r.parsed:document.elements) AS el

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Aggregate to slide chunks in `pptx_chunks`
# MAGIC
# MAGIC One row per slide. Title/header is prepended as context so each chunk is
# MAGIC self-contained for retrieval.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE serverless_stable_r4umw1_catalog.unstructured_data.pptx_chunks
# MAGIC AS
# MAGIC SELECT
# MAGIC   file_path,
# MAGIC   slide_id,
# MAGIC   -- First title or section_header on the slide
# MAGIC   MAX(CASE WHEN element_type IN ('title', 'section_header') THEN content END) AS slide_title,
# MAGIC   -- All text on the slide joined in element order
# MAGIC   ARRAY_JOIN(
# MAGIC     ARRAY_SORT(
# MAGIC       COLLECT_LIST(
# MAGIC         CASE WHEN text_to_embed IS NOT NULL
# MAGIC              THEN STRUCT(element_id AS id, text_to_embed AS txt) END
# MAGIC       ),
# MAGIC       (l, r) -> CASE WHEN l.id < r.id THEN -1 WHEN l.id > r.id THEN 1 ELSE 0 END
# MAGIC     ).txt,
# MAGIC     '\n'
# MAGIC   )                                                                  AS slide_text,
# MAGIC   ROUND(AVG(confidence), 3)                                          AS avg_confidence,
# MAGIC   COUNT(*)                                                           AS element_count
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed
# MAGIC GROUP BY file_path, slide_id
# MAGIC ORDER BY file_path, slide_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Inspect results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Slide count and element type distribution

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   element_type,
# MAGIC   COUNT(*)                  AS total_elements,
# MAGIC   COUNT(DISTINCT slide_id)  AS slides_with_type,
# MAGIC   ROUND(AVG(confidence), 3) AS avg_confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed
# MAGIC GROUP BY element_type
# MAGIC ORDER BY total_elements DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample slide chunks

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   slide_id,
# MAGIC   slide_title,
# MAGIC   slide_text,
# MAGIC   element_count,
# MAGIC   avg_confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.pptx_chunks
# MAGIC ORDER BY slide_id
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Slides with figures (image-heavy slides)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   c.slide_id,
# MAGIC   c.slide_title,
# MAGIC   p.description             AS figure_description,
# MAGIC   p.confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.pptx_chunks AS c
# MAGIC JOIN serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed AS p
# MAGIC   ON c.file_path = p.file_path AND c.slide_id = p.slide_id
# MAGIC WHERE p.element_type = 'figure'
# MAGIC ORDER BY c.slide_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Visualize one slide with bounding boxes

# COMMAND ----------

# MAGIC %pip install pillow --quiet

# COMMAND ----------

import json
from PIL import Image, ImageDraw

SLIDE_NUM = 0  # change to inspect a different slide

PAGE_URI = spark.sql(f"""
    SELECT p.value:image_uri::STRING AS image_uri
    FROM serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed_raw AS r,
    LATERAL variant_explode(r.parsed:document.pages) AS p
    WHERE p.value:id::INT = {SLIDE_NUM}
    LIMIT 1
""").collect()[0]["image_uri"]

local_path = PAGE_URI if PAGE_URI.startswith("/Volumes/") else PAGE_URI.replace("dbfs:", "/dbfs")
img  = Image.open(local_path)
draw = ImageDraw.Draw(img, "RGBA")

rows = spark.sql(f"""
    SELECT element_type, bbox::STRING AS bbox
    FROM serverless_stable_r4umw1_catalog.unstructured_data.pptx_parsed
    WHERE slide_id = {SLIDE_NUM} AND bbox IS NOT NULL
""").collect()

COLORS = {
    "title":          (255, 165,   0, 80),
    "section_header": (255, 165,   0, 80),
    "text":           ( 80,  80, 255, 50),
    "figure":         ( 80, 200,  80, 90),
    "table":          (255,  80,  80, 70),
}

for row in rows:
    color = COLORS.get(row["element_type"], (180, 180, 180, 50))
    for bb in json.loads(row["bbox"]):
        if bb["page_id"] != SLIDE_NUM:
            continue
        x1, y1, x2, y2 = bb["coord"]
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (220,), width=2)

display(img)
