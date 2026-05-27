"""Test KB retrieval — check if SQL query is included in response metadata."""
import json
import boto3
from botocore.config import Config

client = boto3.client(
    "bedrock-agent-runtime",
    region_name="ap-southeast-1",
    config=Config(read_timeout=120),
)

# Method 1: Try Retrieve and check for metadata/SQL
print("=== Method 1: Retrieve ===")
r = client.retrieve(
    knowledgeBaseId="O3OSLRFXOD",
    retrievalQuery={"text": "total sales by category"},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}},
)

# Check full response structure for SQL
for key in r.keys():
    if key != "retrievalResults":
        print(f"  Top-level key: {key} = {r[key]}")

if r["retrievalResults"]:
    first = r["retrievalResults"][0]
    print(f"  Result keys: {list(first.keys())}")
    if "metadata" in first:
        print(f"  Metadata: {json.dumps(first['metadata'], indent=2, default=str)}")
    if "location" in first:
        print(f"  Location: {json.dumps(first['location'], indent=2, default=str)}")

# Method 2: Try GenerateQuery API
print("\n=== Method 2: GenerateQuery ===")
try:
    r2 = client.generate_query(
        knowledgeBaseId="O3OSLRFXOD",
        queryGenerationInput={"text": "total sales by product category"},
    )
    print(f"  Response keys: {list(r2.keys())}")
    if "queries" in r2:
        for q in r2["queries"]:
            print(f"  Generated SQL: {q}")
except Exception as e:
    print(f"  GenerateQuery failed: {type(e).__name__}: {e}")
