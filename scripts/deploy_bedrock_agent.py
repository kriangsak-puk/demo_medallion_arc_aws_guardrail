"""Deploy Strands Agent to AWS Bedrock Agent (Agents for Amazon Bedrock).

This script creates or updates a Bedrock Agent with:
- A Knowledge Base association (for RAG retrieval)
- Guardrails configuration
- An agent execution IAM role (created if not exists)
- Agent alias for invocation

Usage:
    # First time — creates everything:
    python scripts/deploy_bedrock_agent.py

    # Update existing agent (reads AGENT_ID from .env):
    python scripts/deploy_bedrock_agent.py --update

    # Delete agent and role:
    python scripts/deploy_bedrock_agent.py --delete

Prerequisites:
    - AWS credentials configured (aws configure or env vars)
    - .env file with BEDROCK_MODEL_ID, KNOWLEDGE_BASE_ID, GUARDRAILS_ID, etc.
    - uv sync (to have boto3 available)

Run with:
    uv run python scripts/deploy_bedrock_agent.py
"""

import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# Configuration from environment
# =============================================================================

REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "")
GUARDRAILS_ID = os.getenv("GUARDRAILS_ID", "")
GUARDRAILS_VERSION = os.getenv("GUARDRAILS_VERSION", "1")
GLUE_DATABASE_NAME = os.getenv("GLUE_DATABASE_NAME", "")
GLUE_TABLE_NAME = os.getenv("GLUE_TABLE_NAME", "")

# Agent-specific config
AGENT_NAME = "safe-haven-demo-agent"
AGENT_DESCRIPTION = (
    "Safe Haven Demo Booth agent for AWS Summit Bangkok. "
    "Queries the Gold Table via Glue Catalog and uses Knowledge Base for RAG."
)
AGENT_INSTRUCTION = """You are a data analytics assistant for the Safe Haven Demo Booth.
You help visitors query structured business data from the Gold Table in AWS Glue Catalog.
When asked about revenue, sales, products, or customers, retrieve relevant data and provide clear answers.
Always cite the data source. Never fabricate data that isn't in the knowledge base or Gold Table.
If you cannot find the requested information, say so clearly."""

IDLE_SESSION_TTL = 600  # 10 minutes

# IAM role for the agent
AGENT_ROLE_NAME = "BedrockAgentRole-SafeHavenDemo"

# =============================================================================
# Clients
# =============================================================================

sts = boto3.client("sts", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)
bedrock_agent = boto3.client("bedrock-agent", region_name=REGION)

ACCOUNT_ID = sts.get_caller_identity()["Account"]


# =============================================================================
# IAM Role Management
# =============================================================================

AGENT_TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
            },
        }
    ],
})

AGENT_PERMISSIONS_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockModelInvocation",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            "Resource": f"arn:aws:bedrock:{REGION}::foundation-model/{BEDROCK_MODEL_ID}",
        },
        {
            "Sid": "BedrockKnowledgeBase",
            "Effect": "Allow",
            "Action": [
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate",
            ],
            "Resource": f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:knowledge-base/{KNOWLEDGE_BASE_ID}",
        },
        {
            "Sid": "BedrockGuardrails",
            "Effect": "Allow",
            "Action": [
                "bedrock:ApplyGuardrail",
            ],
            "Resource": f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:guardrail/{GUARDRAILS_ID}",
        },
        {
            "Sid": "GlueCatalogAccess",
            "Effect": "Allow",
            "Action": [
                "glue:GetTable",
                "glue:GetTables",
                "glue:GetDatabase",
                "glue:GetPartitions",
            ],
            "Resource": [
                f"arn:aws:glue:{REGION}:{ACCOUNT_ID}:catalog",
                f"arn:aws:glue:{REGION}:{ACCOUNT_ID}:database/{GLUE_DATABASE_NAME}",
                f"arn:aws:glue:{REGION}:{ACCOUNT_ID}:table/{GLUE_DATABASE_NAME}/{GLUE_TABLE_NAME}",
            ],
        },
        {
            "Sid": "S3DataAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket",
            ],
            "Resource": ["*"],  # Narrow this to your specific bucket in production
        },
    ],
})


def create_or_get_agent_role() -> str:
    """Create the agent execution IAM role or return existing ARN."""
    role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{AGENT_ROLE_NAME}"

    try:
        iam.get_role(RoleName=AGENT_ROLE_NAME)
        print(f"✓ IAM role already exists: {AGENT_ROLE_NAME}")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    print(f"Creating IAM role: {AGENT_ROLE_NAME}...")
    iam.create_role(
        RoleName=AGENT_ROLE_NAME,
        AssumeRolePolicyDocument=AGENT_TRUST_POLICY,
        Description="Execution role for Safe Haven Demo Bedrock Agent",
    )

    iam.put_role_policy(
        RoleName=AGENT_ROLE_NAME,
        PolicyName="BedrockAgentPermissions",
        PolicyDocument=AGENT_PERMISSIONS_POLICY,
    )

    # Wait for IAM propagation
    print("  Waiting for IAM role propagation (10s)...")
    time.sleep(10)

    print(f"✓ IAM role created: {role_arn}")
    return role_arn


def delete_agent_role():
    """Delete the agent execution IAM role."""
    try:
        # Remove inline policies first
        policies = iam.list_role_policies(RoleName=AGENT_ROLE_NAME)
        for policy_name in policies.get("PolicyNames", []):
            iam.delete_role_policy(RoleName=AGENT_ROLE_NAME, PolicyName=policy_name)

        iam.delete_role(RoleName=AGENT_ROLE_NAME)
        print(f"✓ IAM role deleted: {AGENT_ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"  Role {AGENT_ROLE_NAME} does not exist, skipping.")
        else:
            raise


# =============================================================================
# Bedrock Agent Management
# =============================================================================


def create_agent(role_arn: str) -> str:
    """Create a new Bedrock Agent and return its ID."""
    print(f"Creating Bedrock Agent: {AGENT_NAME}...")

    create_params = {
        "agentName": AGENT_NAME,
        "description": AGENT_DESCRIPTION,
        "instruction": AGENT_INSTRUCTION,
        "foundationModel": BEDROCK_MODEL_ID,
        "agentResourceRoleArn": role_arn,
        "idleSessionTTLInSeconds": IDLE_SESSION_TTL,
    }

    # Add guardrails if configured
    if GUARDRAILS_ID:
        create_params["guardrailConfiguration"] = {
            "guardrailIdentifier": GUARDRAILS_ID,
            "guardrailVersion": GUARDRAILS_VERSION,
        }

    response = bedrock_agent.create_agent(**create_params)
    agent_id = response["agent"]["agentId"]
    print(f"✓ Agent created: {agent_id}")

    # Wait for agent to be ready
    _wait_for_agent_status(agent_id, "NOT_PREPARED")

    return agent_id


def associate_knowledge_base(agent_id: str):
    """Associate the Knowledge Base with the agent."""
    if not KNOWLEDGE_BASE_ID:
        print("  Skipping KB association (KNOWLEDGE_BASE_ID not set)")
        return

    print(f"Associating Knowledge Base {KNOWLEDGE_BASE_ID} with agent...")

    try:
        bedrock_agent.associate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion="DRAFT",
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            description="Gold Table knowledge base for RAG retrieval",
            knowledgeBaseState="ENABLED",
        )
        print(f"✓ Knowledge Base associated: {KNOWLEDGE_BASE_ID}")
    except ClientError as e:
        if "already associated" in str(e).lower() or "ConflictException" in str(e):
            print(f"  Knowledge Base already associated, skipping.")
        else:
            raise


def prepare_agent(agent_id: str):
    """Prepare the agent for invocation (compiles the agent)."""
    print("Preparing agent...")
    bedrock_agent.prepare_agent(agentId=agent_id)
    _wait_for_agent_status(agent_id, "PREPARED")
    print("✓ Agent prepared and ready")


def create_agent_alias(agent_id: str) -> str:
    """Create a LIVE alias for the agent."""
    alias_name = "live"
    print(f"Creating agent alias: {alias_name}...")

    try:
        response = bedrock_agent.create_agent_alias(
            agentId=agent_id,
            agentAliasName=alias_name,
            description="Live alias for Safe Haven Demo Booth",
        )
        alias_id = response["agentAlias"]["agentAliasId"]
        print(f"✓ Agent alias created: {alias_id}")
        return alias_id
    except ClientError as e:
        if "ConflictException" in str(e):
            # Alias already exists, list and return it
            aliases = bedrock_agent.list_agent_aliases(agentId=agent_id)
            for alias in aliases.get("agentAliasSummaries", []):
                if alias["agentAliasName"] == alias_name:
                    alias_id = alias["agentAliasId"]
                    print(f"  Alias already exists: {alias_id}")
                    return alias_id
        raise


def update_agent(agent_id: str, role_arn: str):
    """Update an existing Bedrock Agent."""
    print(f"Updating Bedrock Agent: {agent_id}...")

    update_params = {
        "agentId": agent_id,
        "agentName": AGENT_NAME,
        "description": AGENT_DESCRIPTION,
        "instruction": AGENT_INSTRUCTION,
        "foundationModel": BEDROCK_MODEL_ID,
        "agentResourceRoleArn": role_arn,
        "idleSessionTTLInSeconds": IDLE_SESSION_TTL,
    }

    if GUARDRAILS_ID:
        update_params["guardrailConfiguration"] = {
            "guardrailIdentifier": GUARDRAILS_ID,
            "guardrailVersion": GUARDRAILS_VERSION,
        }

    bedrock_agent.update_agent(**update_params)
    print("✓ Agent updated")

    # Re-prepare after update
    prepare_agent(agent_id)


def delete_agent(agent_id: str):
    """Delete a Bedrock Agent."""
    print(f"Deleting Bedrock Agent: {agent_id}...")

    # Delete aliases first
    try:
        aliases = bedrock_agent.list_agent_aliases(agentId=agent_id)
        for alias in aliases.get("agentAliasSummaries", []):
            bedrock_agent.delete_agent_alias(
                agentId=agent_id,
                agentAliasId=alias["agentAliasId"],
            )
            print(f"  Deleted alias: {alias['agentAliasName']}")
    except ClientError:
        pass

    bedrock_agent.delete_agent(agentId=agent_id, skipResourceInUseCheck=True)
    print(f"✓ Agent deleted: {agent_id}")


def _wait_for_agent_status(agent_id: str, target_status: str, timeout: int = 60):
    """Wait for agent to reach a target status."""
    start = time.time()
    while time.time() - start < timeout:
        response = bedrock_agent.get_agent(agentId=agent_id)
        status = response["agent"]["agentStatus"]
        if status == target_status:
            return
        if status == "FAILED":
            reasons = response["agent"].get("failureReasons", [])
            raise RuntimeError(f"Agent failed: {reasons}")
        time.sleep(3)
    raise TimeoutError(f"Agent did not reach '{target_status}' within {timeout}s (current: {status})")


# =============================================================================
# Main
# =============================================================================


def deploy():
    """Full deployment: create role, agent, KB association, prepare, alias."""
    print("=" * 60)
    print("  Deploying Bedrock Agent — Safe Haven Demo")
    print("=" * 60)
    print(f"  Region:     {REGION}")
    print(f"  Model:      {BEDROCK_MODEL_ID}")
    print(f"  KB ID:      {KNOWLEDGE_BASE_ID or '(not set)'}")
    print(f"  Guardrails: {GUARDRAILS_ID or '(not set)'}")
    print(f"  Glue DB:    {GLUE_DATABASE_NAME}/{GLUE_TABLE_NAME}")
    print("=" * 60)
    print()

    # Step 1: Create IAM role
    role_arn = create_or_get_agent_role()

    # Step 2: Create agent
    agent_id = create_agent(role_arn)

    # Step 3: Associate Knowledge Base
    associate_knowledge_base(agent_id)

    # Step 4: Prepare agent
    prepare_agent(agent_id)

    # Step 5: Create alias
    alias_id = create_agent_alias(agent_id)

    # Output results
    print()
    print("=" * 60)
    print("  ✅ Deployment Complete!")
    print("=" * 60)
    print(f"  Agent ID:    {agent_id}")
    print(f"  Alias ID:    {alias_id}")
    print(f"  Role ARN:    {role_arn}")
    print()
    print("  Add these to your .env file:")
    print(f"    BEDROCK_AGENT_ID={agent_id}")
    print(f"    BEDROCK_AGENT_ALIAS_ID={alias_id}")
    print("=" * 60)

    return agent_id, alias_id


def update():
    """Update an existing agent (reads BEDROCK_AGENT_ID from .env)."""
    agent_id = os.getenv("BEDROCK_AGENT_ID", "")
    if not agent_id:
        print("ERROR: BEDROCK_AGENT_ID not set in .env. Run deploy first.")
        sys.exit(1)

    role_arn = create_or_get_agent_role()
    update_agent(agent_id, role_arn)
    associate_knowledge_base(agent_id)

    print()
    print("✅ Agent updated and re-prepared.")


def delete():
    """Delete agent and IAM role (reads BEDROCK_AGENT_ID from .env)."""
    agent_id = os.getenv("BEDROCK_AGENT_ID", "")
    if not agent_id:
        print("ERROR: BEDROCK_AGENT_ID not set in .env.")
        sys.exit(1)

    delete_agent(agent_id)
    delete_agent_role()

    print()
    print("✅ Agent and role deleted. Remove BEDROCK_AGENT_ID from .env.")


if __name__ == "__main__":
    if "--update" in sys.argv:
        update()
    elif "--delete" in sys.argv:
        delete()
    else:
        deploy()
