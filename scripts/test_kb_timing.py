"""Test KB retrieval timing with assumed role."""
import time
import boto3
from botocore.config import Config

print("Step 1: Assuming role...")
t0 = time.time()
sts = boto3.client("sts", region_name="ap-southeast-1")
creds = sts.assume_role(
    RoleArn="arn:aws:iam::095833340753:role/service-role/AmazonBedrockExecutionRoleForKnowledgeBase_demo_silver",
    RoleSessionName="kb-timing-test",
)["Credentials"]
print(f"  Role assumed in {time.time()-t0:.1f}s")

print("Step 2: Creating KB client...")
client = boto3.client(
    "bedrock-agent-runtime",
    region_name="ap-southeast-1",
    aws_access_key_id=creds["AccessKeyId"],
    aws_secret_access_key=creds["SecretAccessKey"],
    aws_session_token=creds["SessionToken"],
    config=Config(read_timeout=120, connect_timeout=10),
)

print("Step 3: Calling KB retrieve...")
t1 = time.time()
r = client.retrieve(
    knowledgeBaseId="O3OSLRFXOD",
    retrievalQuery={"text": "sales by product category"},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
)
elapsed = time.time() - t1
print(f"  KB retrieval: {elapsed:.1f}s, {len(r['retrievalResults'])} results")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("\nFirst 3 results:")
for i, res in enumerate(r["retrievalResults"][:3], 1):
    content = res["content"]
    if "row" in content:
        row = ", ".join(f"{c['columnName']}={c['columnValue']}" for c in content["row"])
        print(f"  [{i}] {row}")
