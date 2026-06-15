# Databricks notebook source
# MAGIC %md
# MAGIC # Research Paper PDF Parsing with `ai_parse_document`
# MAGIC
# MAGIC Parses ML research papers (YOLO, Attention Is All You Need) from a UC Volume.
# MAGIC Key difference from W2: research papers have **figures** that need AI-generated
# MAGIC descriptions to be useful for RAG.
# MAGIC
# MAGIC **Source volume:** `serverless_stable_r4umw1_catalog.unstructured_data.research_papers`
# MAGIC
# MAGIC **Extra params vs W2:**
# MAGIC - `dpi=300` — higher resolution for dense multi-column text and small figures
# MAGIC - `descriptionElementTypes=figure` — generates AI text description for each figure element
# MAGIC - `imageOutputPath` — saves rendered page images to a UC Volume; populates `pages[].image_uri`
# MAGIC
# MAGIC **Design:** parse once into `research_parsed_raw`, derive `research_parsed` from it.

# COMMAND ----------

CATALOG      = "serverless_stable_r4umw1_catalog"
SCHEMA       = "unstructured_data"
VOLUME       = "research_papers"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
IMAGE_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/research_papers_images"
RAW_TABLE    = f"{CATALOG}.{SCHEMA}.research_parsed_raw"
OUT_TABLE    = f"{CATALOG}.{SCHEMA}.research_parsed"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Preview: list files in volume

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/research_papers'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Parse all papers and save raw VARIANT to `research_parsed_raw`
# MAGIC
# MAGIC - `dpi=300`: better quality for dense layouts and small figures (default is 200)
# MAGIC - `descriptionElementTypes=figure`: triggers AI description generation for figures
# MAGIC - `imageOutputPath`: saves rendered page images; `pages[].image_uri` is populated
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` — only runs on first execution, safe to re-run.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- imageOutputPath requires the volume to exist first
# MAGIC CREATE VOLUME IF NOT EXISTS serverless_stable_r4umw1_catalog.unstructured_data.research_papers_images

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS serverless_stable_r4umw1_catalog.unstructured_data.research_parsed_raw
# MAGIC AS
# MAGIC SELECT
# MAGIC   path                AS file_path,
# MAGIC   ai_parse_document(
# MAGIC     content,
# MAGIC     map(
# MAGIC       'version',                '2.0',
# MAGIC       'dpi',                    '300',
# MAGIC       'descriptionElementTypes','figure',
# MAGIC       'imageOutputPath',        '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/research_papers_images'
# MAGIC     )
# MAGIC   )                   AS parsed,
# MAGIC   current_timestamp() AS parsed_at
# MAGIC FROM READ_FILES(
# MAGIC   '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/research_papers',
# MAGIC   format => 'binaryFile'
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Explode elements and save to `research_parsed`
# MAGIC
# MAGIC Figure elements: `content` is null, `description` holds the AI-generated text.
# MAGIC For RAG, use `description` as the embeddable text for figures.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE serverless_stable_r4umw1_catalog.unstructured_data.research_parsed
# MAGIC AS
# MAGIC SELECT
# MAGIC   r.file_path,
# MAGIC   el.value:id::INT                                                   AS element_id,
# MAGIC   el.value:type::STRING                                              AS element_type,
# MAGIC   el.value:content::STRING                                           AS content,
# MAGIC   el.value:description::STRING                                       AS description,
# MAGIC   -- For RAG: figures use description; all other elements use content
# MAGIC   COALESCE(el.value:description::STRING, el.value:content::STRING)   AS text_to_embed,
# MAGIC   ROUND(el.value:confidence::DOUBLE, 3)                              AS confidence,
# MAGIC   el.value:bbox                                                      AS bbox,
# MAGIC   r.parsed:metadata                                                  AS doc_metadata,
# MAGIC   r.parsed_at
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.research_parsed_raw AS r,
# MAGIC LATERAL variant_explode(r.parsed:document.elements) AS el

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Inspect results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Element type distribution across all papers

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   regexp_extract(file_path, '[^/]+$', 0) AS paper,
# MAGIC   element_type,
# MAGIC   COUNT(*)                               AS total_elements,
# MAGIC   ROUND(AVG(confidence), 3)              AS avg_confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.research_parsed
# MAGIC GROUP BY paper, element_type
# MAGIC ORDER BY paper, total_elements DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure elements — AI-generated descriptions
# MAGIC
# MAGIC These descriptions become the embeddable text for figures in RAG.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   regexp_extract(file_path, '[^/]+$', 0) AS paper,
# MAGIC   element_id,
# MAGIC   description,
# MAGIC   confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.research_parsed
# MAGIC WHERE element_type = 'figure'
# MAGIC ORDER BY file_path, element_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Page images saved to volume
# MAGIC
# MAGIC `imageOutputPath` saves one PNG per page. These can be used to crop figures by bbox.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   r.file_path,
# MAGIC   p.value:id::INT       AS page_id,
# MAGIC   p.value:image_uri::STRING AS image_uri
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.research_parsed_raw AS r,
# MAGIC LATERAL variant_explode(r.parsed:document.pages) AS p
# MAGIC ORDER BY file_path, page_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Low-confidence elements

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   regexp_extract(file_path, '[^/]+$', 0) AS paper,
# MAGIC   element_id,
# MAGIC   element_type,
# MAGIC   content,
# MAGIC   confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.research_parsed
# MAGIC WHERE confidence < 0.8
# MAGIC ORDER BY confidence
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Visualize bounding boxes on a paper page
# MAGIC
# MAGIC Page images are already saved to the volume via `imageOutputPath`.
# MAGIC Load one, overlay element bboxes coloured by type.

# COMMAND ----------

# MAGIC %pip install pillow --quiet

# COMMAND ----------

import json
from PIL import Image, ImageDraw

PAGE_URI = spark.sql("""
    SELECT p.value:image_uri::STRING AS image_uri
    FROM serverless_stable_r4umw1_catalog.unstructured_data.research_parsed_raw AS r,
    LATERAL variant_explode(r.parsed:document.pages) AS p
    WHERE r.file_path LIKE '%1706%'       -- Attention Is All You Need
      AND p.value:id::INT = 0             -- first page
""").collect()[0]["image_uri"]

# Load saved page image from volume (already at the parse DPI)
# UC Volumes are accessible at /Volumes/... directly on serverless; /dbfs/Volumes/... on classic compute
local_path = PAGE_URI if PAGE_URI.startswith("/Volumes/") else PAGE_URI.replace("dbfs:", "/dbfs")
img  = Image.open(local_path)
draw = ImageDraw.Draw(img, "RGBA")

rows = spark.sql("""
    SELECT element_type, bbox::STRING AS bbox
    FROM serverless_stable_r4umw1_catalog.unstructured_data.research_parsed
    WHERE file_path LIKE '%1706%'
      AND bbox IS NOT NULL
""").collect()

COLORS = {
    "table":          (255,  80,  80,  70),
    "text":           ( 80,  80, 255,  50),
    "figure":         ( 80, 200,  80,  90),
    "section_header": (255, 165,   0,  80),
}

for row in rows:
    color = COLORS.get(row["element_type"], (180, 180, 180, 50))
    for bb in json.loads(row["bbox"]):
        if bb["page_id"] != 0:
            continue
        x1, y1, x2, y2 = bb["coord"]
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (220,), width=2)

display(img)
