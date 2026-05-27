"""Quick test: verify KB retrieval with structured row parsing."""
import boto3
from botocore.config import Config

client = boto3.client(
    "bedrock-agent-runtime",
    region_name="ap-southeast-1",
    config=Config(read_timeout=120),
)

r = client.retrieve(
    knowledgeBaseId="O3OSLRFXOD",
    retrievalQuery={"text": "top products by revenue"},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
)

results = r["retrievalResults"]
print(f"{len(results)} results retrieved:\n")

for i, res in enumerate(results, 1):
    content = res["content"]
    if "row" in content:
        row_str = ", ".join(
            f"{col['columnName']}: {col['columnValue']}"
            for col in content["row"]
        )
        print(f"  [{i}] {row_str}")
    elif "text" in content:
        print(f"  [{i}] {content['text'][:150]}")
    else:
        print(f"  [{i}] Unknown format: {list(content.keys())}")
