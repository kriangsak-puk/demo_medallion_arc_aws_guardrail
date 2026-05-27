"""Test GenerateQuery API to get the SQL that KB generates."""
import json
import boto3
from botocore.config import Config

client = boto3.client(
    "bedrock-agent-runtime",
    region_name="ap-southeast-1",
    config=Config(read_timeout=60),
)

print("Calling GenerateQuery...")
try:
    r = client.generate_query(
        queryGenerationInput={
            "text": "show total sales by product category",
            "type": "TEXT",
        },
        transformationConfiguration={
            "mode": "TEXT_TO_SQL",
            "textToSqlConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseArn": "arn:aws:bedrock:ap-southeast-1:095833340753:knowledge-base/O3OSLRFXOD",
                }
            },
        },
    )
    print(f"Response keys: {list(r.keys())}")
    if "queries" in r:
        for i, q in enumerate(r["queries"], 1):
            print(f"\n  Query {i}:")
            print(f"    {json.dumps(q, indent=4, default=str)}")
    else:
        print(json.dumps(r, indent=2, default=str))
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
