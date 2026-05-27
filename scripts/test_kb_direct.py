"""Test KB retrieval with direct credentials (no role assumption)."""
import time
import boto3
from botocore.config import Config

print("Testing KB with direct credentials (no role assumption)...")
t0 = time.time()
client = boto3.client(
    "bedrock-agent-runtime",
    region_name="ap-southeast-1",
    config=Config(read_timeout=120, connect_timeout=10),
)

t1 = time.time()
r = client.retrieve(
    knowledgeBaseId="O3OSLRFXOD",
    retrievalQuery={"text": "sales by product category"},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
)
elapsed = time.time() - t1
print(f"KB retrieval: {elapsed:.1f}s, {len(r['retrievalResults'])} results")
print(f"Total: {time.time()-t0:.1f}s")
