# Databricks notebook source
# MAGIC %md
# MAGIC # CLIP Model — Register to Unity Catalog & Deploy as Serving Endpoint
# MAGIC
# MAGIC Downloads `openai/clip-vit-large-patch14` from HuggingFace, wraps it as an
# MAGIC MLflow pyfunc model, registers it to Unity Catalog, and deploys it as a
# MAGIC GPU model serving endpoint.
# MAGIC
# MAGIC **Input**: base64-encoded image string (one row per image)
# MAGIC **Output**: 768-dimensional image embedding (ARRAY<DOUBLE>)
# MAGIC
# MAGIC The endpoint is used by notebook `07_video.py` to embed video keyframes for
# MAGIC visual similarity search.
# MAGIC
# MAGIC **Reference**: Databricks HuggingFace model registration pattern
# MAGIC https://docs.databricks.com/aws/en/notebooks/source/machine-learning/train-register-hugging-face-model-serving.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install CPU dependencies for local model loading
# MAGIC
# MAGIC We install CPU-only torch here to load the model on the driver and infer
# MAGIC the MLflow signature. The serving endpoint uses GPU torch (in pip_requirements below).

# COMMAND ----------

# MAGIC %pip install torch torchvision transformers pillow mlflow databricks-sdk --quiet
# MAGIC %restart_python

# COMMAND ----------

# Constants — defined AFTER %restart_python so they survive the kernel restart
CATALOG       = "serverless_stable_r4umw1_catalog"
SCHEMA        = "unstructured_data"
MODEL_NAME    = f"{CATALOG}.{SCHEMA}.clip_vit_large_patch14"
ENDPOINT_NAME = "clip_embedding_endpoint"
HF_MODEL      = "openai/clip-vit-large-patch14"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Define the MLflow pyfunc model class
# MAGIC
# MAGIC Input: pandas DataFrame with column `model_input` (base64-encoded image string)
# MAGIC Output: pandas Series of 768-dim embedding lists

# COMMAND ----------

import mlflow
import torch
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPImageEmbedding(mlflow.pyfunc.PythonModel):
    """CLIP image embedding model. Accepts base64-encoded images, returns 768-dim embeddings."""

    def load_context(self, context):
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(HF_MODEL)
        self.processor = CLIPProcessor.from_pretrained(HF_MODEL)
        self.model.eval()

    def _embed(self, b64_str: str) -> list:
        img_bytes = base64.b64decode(b64_str)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            # Pass pixel_values explicitly to get a plain tensor back
            features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        return features[0].tolist()  # shape (1, 768) → list of 768 floats

    def predict(self, context, df: pd.DataFrame) -> pd.Series:
        return df["model_input"].apply(self._embed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Load model locally and create a synthetic test signature
# MAGIC
# MAGIC We use a 1×1 white PNG as a synthetic test image — no dependency on
# MAGIC the video tables at this stage.

# COMMAND ----------

import io, base64
from mlflow.models.signature import infer_signature

# Synthetic 1x1 white PNG → base64
buf = io.BytesIO()
Image.new("RGB", (224, 224), color=(255, 255, 255)).save(buf, format="PNG")
test_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
test_df = pd.DataFrame({"model_input": [test_b64]})

# Load model on driver (CPU) to infer output shape
clip = CLIPImageEmbedding()
clip.load_context(context=None)
test_output = clip.predict(context=None, df=test_df)
print(f"Embedding dim: {len(test_output[0])}")  # should be 768

signature = infer_signature(test_df, [test_output[0]])
print(f"Signature:\n  inputs:  {signature.inputs}\n  outputs: {signature.outputs}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Log and register model to Unity Catalog
# MAGIC
# MAGIC The `pip_requirements` specify GPU torch builds used by the serving endpoint.
# MAGIC Local driver uses CPU torch; endpoint workers use CUDA 12.1 builds.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# GPU pip requirements for the serving endpoint
pip_requirements = [
    "--extra-index-url https://download.pytorch.org/whl/cu121",
    "mlflow>=2.14.0",
    "torch==2.3.1+cu121",
    "torchvision==0.18.1+cu121",
    "transformers==4.41.2",
    "accelerate==0.31.0",
    "pillow",
    "pandas",
    "setuptools<70.0.0",
]

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

with mlflow.start_run(run_name="clip_vit_large_patch14_registration"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="clip_image_embedding",
        python_model=CLIPImageEmbedding(),
        registered_model_name=MODEL_NAME,
        signature=signature,
        pip_requirements=pip_requirements,
        input_example=test_df,
    )

print(f"Registered: {MODEL_NAME}")
print(f"Run ID: {model_info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Get the latest model version

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)
print(f"Latest version: {latest_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Deploy as GPU model serving endpoint
# MAGIC
# MAGIC Uses `GPU_SMALL` (1× T4 GPU). Scale-to-zero is enabled to save cost.
# MAGIC The endpoint may take 10–20 minutes to reach READY state.

# COMMAND ----------

import requests, time

HOST  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
BASE_URL = f"https://{HOST}/api/2.0/serving-endpoints"

served_entity_payload = {
    "entity_name":          MODEL_NAME,
    "entity_version":       str(latest_version),
    "workload_size":        "Small",
    "workload_type":        "GPU_SMALL",
    "scale_to_zero_enabled": True,
}

# Check if endpoint already exists
existing = {e["name"] for e in requests.get(BASE_URL, headers=HEADERS).json().get("endpoints", [])}

if ENDPOINT_NAME in existing:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists — updating to version {latest_version}.")
    resp = requests.put(
        f"{BASE_URL}/{ENDPOINT_NAME}/config",
        headers=HEADERS,
        json={"served_entities": [served_entity_payload]},
    )
else:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    resp = requests.post(
        BASE_URL,
        headers=HEADERS,
        json={
            "name": ENDPOINT_NAME,
            "config": {"served_entities": [served_entity_payload]},
        },
    )

resp.raise_for_status()
print(f"Request accepted (HTTP {resp.status_code}). Endpoint is being provisioned.")
print(f"Monitor at: https://{HOST}/ml/endpoints/{ENDPOINT_NAME}")
print("Note: GPU endpoints typically take 10–20 minutes to reach READY state.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Smoke test the endpoint

# COMMAND ----------

import requests

HOST  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Check endpoint state first
state_resp = requests.get(f"https://{HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=HEADERS)
state = state_resp.json().get("state", {}).get("ready", "UNKNOWN")
print(f"Endpoint state: {state}")

if state != "READY":
    print("Endpoint not yet READY — re-run this cell once it finishes provisioning (10–20 min).")
else:
    resp = requests.post(
        f"https://{HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers=HEADERS,
        json={"dataframe_records": [{"model_input": test_b64}]},
    )
    resp.raise_for_status()
    embedding = resp.json()["predictions"][0]
    print(f"Embedding type  : {type(embedding)}")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values  : {embedding[:5]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Inspect the registered model in UC

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE MODEL serverless_stable_r4umw1_catalog.unstructured_data.clip_vit_large_patch14

# COMMAND ----------

# MAGIC %md
# MAGIC ## Note: Text embedding (for query-time search)
# MAGIC
# MAGIC The CLIP endpoint handles **image** embedding only (used at index time for
# MAGIC video keyframes). At **query time**, text queries are encoded using the CLIP
# MAGIC text encoder locally in the search notebook:
# MAGIC
# MAGIC ```python
# MAGIC from transformers import CLIPProcessor, CLIPModel
# MAGIC import torch
# MAGIC
# MAGIC model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# MAGIC processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# MAGIC
# MAGIC def get_text_embedding(text: str) -> list:
# MAGIC     inputs = processor(text=text, return_tensors="pt", padding=True)
# MAGIC     with torch.no_grad():
# MAGIC         features = model.get_text_features(**inputs)
# MAGIC     return features.squeeze().tolist()
# MAGIC ```
# MAGIC
# MAGIC Both image and text embeddings live in the same 768-dim CLIP space, so
# MAGIC cosine similarity between a text query and an image frame is meaningful.
