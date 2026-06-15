# Databricks notebook source
# MAGIC %md
# MAGIC # W2 PDF Parsing with `ai_parse_document`
# MAGIC
# MAGIC Parses 10 W2 tax form PDFs from a UC Volume using `ai_parse_document`,
# MAGIC inspects the structured element output, and saves results to Delta tables.
# MAGIC
# MAGIC **Source volume:** `serverless_stable_r4umw1_catalog.unstructured_data.w2_sample`
# MAGIC
# MAGIC **Design — two tables, one model call:**
# MAGIC - `w2_parsed_raw` — raw VARIANT output, created once (`IF NOT EXISTS` skips re-parsing)
# MAGIC - `w2_parsed`     — exploded elements, derived from the raw table (no model call)

# COMMAND ----------

CATALOG     = "serverless_stable_r4umw1_catalog"
SCHEMA      = "unstructured_data"
VOLUME      = "w2_sample"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_TABLE   = f"{CATALOG}.{SCHEMA}.w2_parsed_raw"
OUT_TABLE   = f"{CATALOG}.{SCHEMA}.w2_parsed"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Preview: list files in volume

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/w2_sample'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Parse all W2s and save raw VARIANT to `w2_parsed_raw`
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` means `ai_parse_document` only runs on the first execution.
# MAGIC Re-running the notebook skips this step entirely if the table already exists.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed_raw
# MAGIC AS
# MAGIC SELECT
# MAGIC   path                                               AS file_path,
# MAGIC   ai_parse_document(content, map('version', '2.0')) AS parsed,
# MAGIC   current_timestamp()                                AS parsed_at
# MAGIC FROM READ_FILES(
# MAGIC   '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/w2_sample',
# MAGIC   format => 'binaryFile'
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Explode elements and save to `w2_parsed`
# MAGIC
# MAGIC Reads from `w2_parsed_raw` — no additional model calls.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
# MAGIC AS
# MAGIC SELECT
# MAGIC   r.file_path,
# MAGIC   el.value:id::INT                      AS element_id,
# MAGIC   el.value:type::STRING                 AS element_type,
# MAGIC   el.value:content::STRING              AS content,
# MAGIC   ROUND(el.value:confidence::DOUBLE, 3) AS confidence,
# MAGIC   el.value:bbox                         AS bbox,
# MAGIC   r.parsed:metadata                     AS doc_metadata,
# MAGIC   r.parsed_at
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed_raw AS r,
# MAGIC LATERAL variant_explode(r.parsed:document.elements) AS el

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Inspect results
# MAGIC
# MAGIC All queries below read from `w2_parsed` — no model calls.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 5 elements from one W2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   regexp_extract(file_path, '[^/]+$', 0) AS file_name,
# MAGIC   element_id,
# MAGIC   element_type,
# MAGIC   content,
# MAGIC   confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
# MAGIC WHERE file_path = (SELECT MIN(file_path) FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed)
# MAGIC ORDER BY element_id
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Element type distribution across all W2s

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   element_type,
# MAGIC   COUNT(*)                       AS total_elements,
# MAGIC   ROUND(AVG(confidence), 3)      AS avg_confidence,
# MAGIC   ROUND(MIN(confidence), 3)      AS min_confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
# MAGIC GROUP BY element_type
# MAGIC ORDER BY total_elements DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Low-confidence elements (potential parse quality issues)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   regexp_extract(file_path, '[^/]+$', 0) AS file_name,
# MAGIC   element_type,
# MAGIC   content,
# MAGIC   confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
# MAGIC WHERE confidence < 0.8
# MAGIC ORDER BY confidence
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full parse output for one W2 (W2_XL_input_clean_2990.pdf)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   element_id,
# MAGIC   element_type,
# MAGIC   content,
# MAGIC   confidence
# MAGIC FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
# MAGIC WHERE file_path LIKE '%2990%'
# MAGIC ORDER BY element_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Visualize bounding boxes on the PDF
# MAGIC
# MAGIC Renders the PDF page at 200 DPI (the same resolution `ai_parse_document` uses for its
# MAGIC coordinate space) and overlays each element's bounding box, coloured by type.

# COMMAND ----------

# MAGIC %pip install pymupdf --quiet

# COMMAND ----------

import json
import fitz                          # pymupdf
from PIL import Image, ImageDraw

FILE     = "W2_XL_input_clean_2990.pdf"
PDF_PATH = f"/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/w2_sample/{FILE}"

# Read elements + bboxes from the saved Delta table (no model call)
rows = spark.sql("""
    SELECT element_id, element_type, confidence, bbox::STRING AS bbox
    FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
    WHERE file_path LIKE '%2990%'
    ORDER BY element_id
""").collect()

# Render page at 200 DPI to match ai_parse_document coordinate space
doc  = fitz.open(PDF_PATH)
page = doc[0]
pix  = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
draw = ImageDraw.Draw(img, "RGBA")

# Colour per element type: (R, G, B, fill_alpha) + solid border
COLORS = {
    "table":  (255,  80,  80,  70),
    "text":   ( 80,  80, 255,  50),
    "figure": ( 80, 200,  80,  70),
}

for row in rows:
    if not row["bbox"]:
        continue
    color  = COLORS.get(row["element_type"], (180, 180, 180, 50))
    border = color[:3] + (220,)
    for bb in json.loads(row["bbox"]):
        x1, y1, x2, y2 = bb["coord"]
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=border, width=2)

display(img)
