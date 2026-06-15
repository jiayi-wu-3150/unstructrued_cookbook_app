# Declarative Vectorization — Project Plan

> **Goal:** Show how to ingest any file type into Databricks for RAG/search — one self-contained recipe per file type. Each recipe: download sample data → load into UC Volume → parse → chunk → output to `chunk_id / chunk_to_embed / chunk_to_retrieve` schema.

---

## 1. Scope

This is **not** a unified pipeline. It is a **recipe book** — one notebook per file type, each demonstrating the idiomatic Databricks ingestion path for that format. Recipes are composable building blocks; users pick the ones relevant to their data.

---

## 2. Output Schema (all recipes converge here)

```
chunk_id          STRING   -- unique hash of (source_path + chunk_index)
chunk_to_embed    STRING   -- text sent to embedding model
chunk_to_retrieve STRING   -- text shown to user in retrieval results (may be richer)
source_path       STRING   -- original file path / URL
chunk_index       INT      -- position within source document
metadata          MAP<STRING, STRING>  -- file-type-specific fields
```

---

## 3. File Types & Recipes

| # | File Type | Parser | Chunking Strategy | Small Dataset | Large Dataset | License |
|---|-----------|--------|------------------|---------------|---------------|---------|
| 1 | PDF (text-heavy) | `ai_parse_document` | Section/heading → semantic | arXiv single paper | arXiv bulk (S3) | CC-BY 4.0 |
| 2 | PDF (tables / financial) | `ai_parse_document` | 1 table = 1 chunk, never split | SEC EDGAR single 10-K | SEC EDGAR bulk | Public domain |
| 3 | PDF (scanned / OCR) | `ai_parse_document` (OCR path) | 1 page = 1 chunk | RVL-CDIP 10 samples | RVL-CDIP full (400K) | Apache 2.0 |
| 4 | DOCX | `ai_parse_document` | Heading-based → 512 tok + 10% overlap | USPTO sample patents | superdoc-dev/docx-corpus | ODC-BY |
| 5 | PPTX | `ai_parse_document` | 1 slide = 1 chunk | OpenReview NeurIPS slides (manual) | Same, full archive | CC-BY |
| 6 | Image (invoices / forms) | `ai_parse_document` | Whole image = 1 chunk | SROIE dataset (50 samples) | RVL-CDIP (400K) | Apache 2.0 |
| 7 | Image (charts / figures) | `ai_parse_document` | Image = 1 chunk + caption as text | cmarkea/aftdb (50 samples) | Same, full (178K) | Apache 2.0 |
| 8 | HTML (text + images) | `playwright` → PDF → `ai_parse_document` | `ai_prep_search` (post-convert) | WebSight 50 samples | HuggingFaceM4/WebSight (1.9M) | CC-BY 4.0 |
| 9 | HTML (text only) | BeautifulSoup | Paragraph / heading | Wikipedia API (10 articles) | HuggingFace wikipedia 20220301.en | CC-BY-SA |
| 10 | Markdown | Pass-through | `##` heading boundaries | GitHub README via GH API | github-code corpus (filtered) | MIT / Apache 2.0 |
| 11 | CSV / TSV | Spark `read.csv` | N-row batch → Markdown table string | NYC Taxi trips (1 month) | NYC Taxi multi-year (~50 GB) | Public domain |
| 12 | Excel (XLSX) | `openpyxl` Spark UDF | Per-sheet Markdown table | World Bank open data XLSX | Kaggle financial datasets | CC-BY 4.0 |
| 13 | Audio (short clips) | Whisper → sentence split | ~30s segments, 2–5s overlap | Mozilla Common Voice (10 clips) | Common Voice English (~70 GB) | CC-0 |
| 14 | Audio (long form) | Whisper → sentence split | ~30s segments, 5s overlap | LibriSpeech dev-clean (5 files) | LibriSpeech train-960 (60 GB) | CC-BY 4.0 |
| 15 | Video (MP4) | OpenCV frames + Whisper | 30–60s segments: visual chunk + audio chunk | UCF-101 (5 clips) | UCF-101 full (7 GB) | Apache 2.0 |

> **License note:** All datasets above are commercial-friendly (CC0, CC-BY, Apache 2.0, Public Domain, ODC-BY). Wikipedia is CC-BY-SA — attribution required; derivative code is fine.

> **HTML with images note:** `ai_parse_document` does NOT support HTML natively. Use `playwright` (headless Chromium) to render HTML → PDF preserving layout and images, then run the PDF through `ai_parse_document` → `ai_prep_search`. Use `weasyprint` as a lighter fallback for simple HTML with no JS.

---

## 4. Chunking Strategy Summary

### Recommended: ai_prep_search (Beta, v2 — June 2026)
For all `ai_parse_document`-supported formats, **use `ai_prep_search` as the chunking step** instead of custom chunking:
- Takes `ai_parse_document` VARIANT output → produces `chunk_to_embed` + `chunk_to_retrieve`
- v2: LLM-powered contextual summary, per-table summaries, richer embedding metadata, soft table boundaries
- Requires: DBR 18.2+ or serverless env ≥3, workspace preview flag **"AI Prep Search (Beta)"** enabled
- Known limitation: no configurable overlap or chunk size

```sql
-- Composable IDP golden pattern
WITH parsed AS (
  SELECT path, ai_parse_document(content, map('version', '2.0')) AS parsed
  FROM READ_FILES('/Volumes/catalog/schema/docs', format => 'binaryFile')
),
prepped AS (SELECT path, ai_prep_search(parsed) AS result FROM parsed)
SELECT
  chunk.value:chunk_id::STRING          AS chunk_id,
  chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
  chunk.value:chunk_to_embed::STRING    AS chunk_to_embed,
  path                                  AS source_uri   -- NOTE: must pass path explicitly (docs bug)
FROM prepped, LATERAL variant_explode(prepped.result:document.contents) AS chunk
```

### Custom chunking (fallback for non-APD formats)
- **Section/heading-based** split at natural boundaries (`##`, Word styles, `<h2>`)
- **512 tokens** per chunk, **10% overlap**
- **Contextual retrieval prefix**: prepend a 1–2 sentence document summary to each chunk

### Special Cases
| Situation | Strategy |
|-----------|----------|
| Tables (PDF, Excel) | Keep whole table intact — never split mid-table |
| PPTX slides | 1 slide = 1 chunk (slide is atomic unit) |
| Scanned PDF pages | 1 page = 1 chunk |
| Images (invoice, chart) | Whole image = 1 chunk; caption/extracted text is `chunk_to_embed` |
| CSV/TSV rows | Batch N rows → Markdown table string; embed the string |
| Audio/Video | 30–60s segments; two chunks per segment for video (visual + audio) |

---

## 5. Key Technology Context

### ai_parse_document (GA)
- **Supported formats**: PDF, DOC/DOCX, PPT/PPTX, JPG/JPEG, PNG, TIF/TIFF
- **NOT supported**: audio, video, HTML, plain text, CSV, Excel
- **Limits**: 500-page / 100 MB hard limit → use `pageRange` for large PDFs (e.g. `'1-500'`)
- **Requires**: DBR 17.3+ or serverless env ≥3
- **Output**: VARIANT with `document.elements[]` — each element has `id, type, content, confidence, bbox`
- **Known issues**: non-deterministic element count across runs, table OCR errors on text-based PDFs, non-English quality gaps

### ai_prep_search (Beta)
- **Requires**: DBR 18.2+ or serverless env ≥3 + workspace preview flag **"AI Prep Search (Beta)"**
- **Output fields**: `chunk_id`, `chunk_position`, `chunk_to_retrieve`, `chunk_to_embed`
- **Known limitation**: no configurable overlap; no chunk size control; table HTML may split across chunks

### Format Dispatch
```python
APD_FORMATS     = {"pdf", "docx", "doc", "pptx", "ppt", "png", "jpg", "jpeg", "tif", "tiff"}
HTML_FORMATS    = {"html", "xml", "htm"}   # with images → playwright→PDF; text-only → BeautifulSoup
TEXT_FORMATS    = {"txt", "md"}
TABULAR_FORMATS = {"csv", "tsv", "xlsx", "xls"}
AUDIO_FORMATS   = {"mp3", "wav", "m4a", "ogg", "flac"}
VIDEO_FORMATS   = {"mp4", "mov", "avi", "mkv"}
```

---

## 6. Notebook Structure (per recipe)

Each notebook follows this template:

```
0. Setup & imports
1. Download sample data → UC Volume  (or point at existing volume)
2. Load / read file (READ_FILES binaryFile for APD; format-specific for others)
3. Parse  (file-type-specific)
4. Chunk  (ai_prep_search for APD formats; custom for others)
5. Normalize to output schema
6. Write to Delta table
7. Smoke test: spot-check chunk quality (element type distribution, confidence scores)
```

---

## 7. Project Layout

```
declarative_vectorization/
├── databricks.yml                   # DAB config (existing)
├── PLAN.md                          # This file
├── notebooks/
│   ├── 01_pdf_w2_parse.py           # CURRENT: W2 tax forms — parsing step only
│   ├── 02_pdf_text.py
│   ├── 03_pdf_tables.py
│   ├── 04_pdf_scanned.py
│   ├── 05_docx.py
│   ├── 06_pptx.py
│   ├── 07_image_invoice.py
│   ├── 08_image_charts.py
│   ├── 09_html_rich.py              # playwright → PDF → ai_parse_document
│   ├── 10_html_text.py
│   ├── 11_markdown.py
│   ├── 12_csv_tsv.py
│   ├── 13_excel.py
│   ├── 14_audio_short.py
│   ├── 15_audio_long.py
│   └── 16_video.py
├── shared/
│   ├── schema.py                    # Output schema definition
│   ├── chunking_utils.py            # Shared chunking helpers
│   └── download_utils.py            # Dataset download helpers
└── tests/
    └── test_output_schema.py        # Validate each recipe's output shape
```

---

## 8. Build Order (suggested)

Start with the easiest parsers and validate the output schema end-to-end before moving to harder types:

1. **Phase 1 — APD native (PDF/DOCX/images)**: W2 PDFs → PDF text → PDF tables → DOCX → PPTX → images
2. **Phase 2 — HTML**: HTML with images (playwright path) → HTML text-only (BeautifulSoup)
3. **Phase 3 — Text-native**: Markdown, CSV/TSV
4. **Phase 4 — Structured data**: Excel
5. **Phase 5 — Media**: Audio short → Audio long → Video

---

## 9. Current Step: W2 PDF Parsing

**Data already in workspace**:
- Volume: `/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/w2_sample`
- Files: `W2_XL_input_clean_2990.pdf` → `2999.pdf` (10 files, ~140 KB each)
- Workspace: `fevm-serverless-stable-r4umw1.cloud.databricks.com`

**Notebook**: `notebooks/01_pdf_w2_parse.py`

Parse SQL:
```sql
CREATE OR REPLACE TABLE serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed AS
WITH raw AS (
  SELECT path, content
  FROM READ_FILES(
    '/Volumes/serverless_stable_r4umw1_catalog/unstructured_data/w2_sample',
    format => 'binaryFile'
  )
),
parsed AS (
  SELECT path, ai_parse_document(content, map('version', '2.0')) AS parsed
  FROM raw
)
SELECT
  path                                    AS file_path,
  parsed:document.metadata               AS doc_metadata,
  el.value:id::INT                        AS element_id,
  el.value:type::STRING                   AS element_type,
  el.value:content::STRING               AS content,
  el.value:confidence::DOUBLE            AS confidence
FROM parsed,
LATERAL variant_explode(parsed:document.elements) AS el
```

Inspection queries:
```sql
-- Element type distribution across all 10 W2s
SELECT element_type, COUNT(*) AS cnt, ROUND(AVG(confidence), 3) AS avg_confidence
FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
GROUP BY element_type ORDER BY cnt DESC;

-- Low confidence elements
SELECT file_path, element_type, content, confidence
FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
WHERE confidence < 0.7 ORDER BY confidence;

-- One W2 full parse output
SELECT element_type, content, confidence
FROM serverless_stable_r4umw1_catalog.unstructured_data.w2_parsed
WHERE file_path LIKE '%2990%' ORDER BY element_id;
```

---

## 10. Open Questions / Decisions

- [x] Which UC catalog/schema/volume? → `serverless_stable_r4umw1_catalog.unstructured_data`
- [ ] Target embedding model (determines max token size per chunk)?
- [ ] Should recipes also write to Vector Search index, or just Delta?
- [ ] Contextual retrieval prefix: use `claude-3-haiku` or Databricks FMAPI?
- [ ] PPTX sample data: manual download needed — confirm acceptable?
- [ ] Video: UCF-101 requires registration — preferred alternative (Something-Something v2 on HuggingFace)?
- [ ] HTML with images: confirm `playwright` is available/installable on cluster?
- [ ] Is `ai_prep_search` preview flag enabled on `fevm-serverless-stable-r4umw1`?
