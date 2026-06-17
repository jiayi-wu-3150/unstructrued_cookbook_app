# Run this script once on the Databricks workspace to pre-compute CLIP text vectors
# for the video search demo page. Output is a Python dict literal to paste into
# pages/recipe_06_video.py under DEMO_QUERY_VECTORS.
#
# Usage (in a Databricks notebook or serverless run):
#   %pip install transformers torch --quiet
#   exec(open("generate_video_vectors.py").read())

import json
import torch
from transformers import CLIPProcessor, CLIPModel

DEMO_QUERIES = [
    "person cooking in a kitchen",
    "outdoor scene with people",
    "close up of a face",
    "people sitting at a table",
    "kitchen appliances",
    "empty room interior",
    "someone eating food",
    "daytime outdoor activity",
]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

vectors = {}
with torch.no_grad():
    for q in DEMO_QUERIES:
        inputs = processor(text=[q], return_tensors="pt", padding=True)
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        vectors[q] = features[0].tolist()

# Print as Python dict literal to paste into recipe_06_video.py
print("DEMO_QUERY_VECTORS = {")
for k, v in vectors.items():
    print(f'    "{k}": {json.dumps(v)},')
print("}")
