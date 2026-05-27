"""Quick test: invoke Haiku 4.5 via inference profile with assumed role."""
import boto3
from botocore.config import Config

# Assume the permissive role first
sts = boto3.client("sts", region_name="ap-southeast-1")
creds = sts.assume_role(
    RoleArn="arn:aws:iam::095833340753:role/service-role/AmazonBedrockExecutionRoleForKnowledgeBase_demo_silver",
    RoleSessionName="test-model",
)["Credentials"]

# Create bedrock-runtime client with assumed role
client = boto3.client(
    "bedrock-runtime",
    region_name="ap-southeast-1",
    aws_access_key_id=creds["AccessKeyId"],
    aws_secret_access_key=creds["SecretAccessKey"],
    aws_session_token=creds["SessionToken"],
    config=Config(read_timeout=60),
)

# Try the inference profile ARN
model_id = "apac.anthropic.claude-haiku-4-5-20251001-v1:0"

print(f"Testing model: {model_id}")
try:
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "Say hello in 5 words"}]}],
        inferenceConfig={"maxTokens": 50},
    )
    text = response["output"]["message"]["content"][0]["text"]
    print(f"✅ Response: {text}")
except Exception as e:
    print(f"❌ Failed with {model_id}: {e}")
    
    # Try global profile
    model_id2 = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    print(f"\nTrying: {model_id2}")
    try:
        response = client.converse(
            modelId=model_id2,
            messages=[{"role": "user", "content": [{"text": "Say hello in 5 words"}]}],
            inferenceConfig={"maxTokens": 50},
        )
        text = response["output"]["message"]["content"][0]["text"]
        print(f"✅ Response: {text}")
    except Exception as e2:
        print(f"❌ Failed with {model_id2}: {e2}")
        
        # Try full ARN
        model_id3 = f"arn:aws:bedrock:ap-southeast-1:095833340753:inference-profile/apac.anthropic.claude-haiku-4-5-20251001-v1:0"
        print(f"\nTrying full ARN: {model_id3}")
        try:
            response = client.converse(
                modelId=model_id3,
                messages=[{"role": "user", "content": [{"text": "Say hello in 5 words"}]}],
                inferenceConfig={"maxTokens": 50},
            )
            text = response["output"]["message"]["content"][0]["text"]
            print(f"✅ Response: {text}")
        except Exception as e3:
            print(f"❌ Failed: {e3}")
