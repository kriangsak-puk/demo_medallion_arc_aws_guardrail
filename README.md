# 🏰 Secure your AI pipeline: Medallion Architecture with AWS Guardrails

An interactive Chainlit web application for the **AWS Summit Bangkok** expo floor that demonstrates the security contrast between an ungoverned AI pipeline ("Data Swamp") and a fully secured Zero-Trust pipeline ("Safe Haven").

Visitors submit attack prompts and see side-by-side how each engine handles them — one leaks sensitive data, the other blocks it through a 3-Gate defense architecture.

## What It Does

| Engine A — 🧟 Data Swamp | Engine B — 🛡️ Safe Haven |
|---|---|
| Uses Permissive IAM Role | Uses Restricted IAM Role |
| Sees ALL columns (including PII) | PII columns blocked by Lake Formation |
| No Guardrails applied | Bedrock Guardrails enforce input/output filtering |
| Leaks sensitive data freely | 3-Gate defense pipeline protects data |

### The 3-Gate Defense Pipeline (Engine B)

1. **Gate 1 — Amazon Macie**: Displays pre-computed PII scan results from the source data
2. **Gate 2 — AWS Lake Formation**: Shows column-level access control (allowed vs denied columns)
3. **Gate 3 — Amazon Bedrock Guardrails**: Evaluates prompts for jailbreak attempts and filters outputs

### Features

- Dual-engine concurrent processing with streaming responses
- Pre-built attack presets (PII extraction, prompt injection, role escalation, etc.)
- Analytics query presets with interactive Plotly chart visualization
- Automatic offline Mock Mode for exhibition floor resilience
- Health endpoint for container orchestration

## Architecture

The system has two deployment components:

```
┌─────────────────────────────────────────────────────────┐
│  1. BACKEND — Bedrock Agent (Strands Agent on AWS)      │
│     Deployed via: scripts/deploy_bedrock_agent.py       │
│     - Bedrock Agent with Knowledge Base (RAG)           │
│     - Guardrails (jailbreak + output filtering)         │
│     - IAM roles for column-level access control         │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  2. FRONTEND — Chainlit Web App                         │
│     Deployed via: Docker → Fargate                      │
│     - Dual-engine UI (Data Swamp vs Safe Haven)         │
│     - Connects to Bedrock Agent via Strands SDK         │
│     - Streaming responses, charts, preset buttons       │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
demo_medallion_arc_aws_guardrail/
├── app.py                  # Chainlit entry point (UI, routing, streaming)
├── engine_a.py             # Ungoverned "Data Swamp" pipeline
├── engine_b.py             # Governed "Safe Haven" pipeline with 3-Gate defense
├── agent_wrapper.py        # Strands Agent SDK wrapper with STS role assumption
├── gates/
│   ├── base.py             # Abstract Gate class and GatePipeline
│   ├── gate1_macie.py      # Amazon Macie scan results display
│   ├── gate2_lakeformation.py  # Lake Formation column access check
│   └── gate3_guardrails.py # Bedrock Guardrails evaluation
├── config.py               # Environment variable loading and AppMode enum
├── mode_detector.py        # Live/Mock mode detection
├── mock_engine.py          # Pre-programmed responses for offline operation
├── validators.py           # Input validation and sanitization
├── presets.py              # Attack and analytics prompt presets
├── chart_renderer.py       # Plotly chart generation for analytics queries
├── health.py               # /health endpoint (HTTP 200/503)
├── scripts/
│   └── deploy_bedrock_agent.py  # Deploy Strands Agent to Bedrock
├── Dockerfile              # Container image (Python 3.11 slim)
├── docker-compose.yml      # Local development with placeholder env vars
├── pyproject.toml          # Project config and dependencies (uv)
├── uv.lock                 # Locked dependency versions for reproducibility
├── .env.example            # Template for environment variables
└── tests/                  # 425 tests (property, unit, integration)
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- AWS CLI configured (`aws configure`)
- Docker (for containerized deployment)

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Running Locally

Running locally requires two steps: deploy the agent backend, then start the Chainlit frontend.

### Step 0: Install Dependencies

```bash
cd demo_medallion_arc_aws_guardrail
uv sync
cp .env.example .env
# Edit .env with your AWS resource IDs
```

### Step 1: Deploy the Bedrock Agent

The Strands Agent must be deployed to AWS Bedrock before the frontend can connect to it in Live Mode.

```bash
# Deploy agent (creates IAM role, agent, KB association, alias)
uv run python scripts/deploy_bedrock_agent.py
```

This outputs `BEDROCK_AGENT_ID` and `BEDROCK_AGENT_ALIAS_ID` — add them to your `.env` file.

To update or delete:
```bash
uv run python scripts/deploy_bedrock_agent.py --update
uv run python scripts/deploy_bedrock_agent.py --delete
```

### Step 2: Start the Chainlit Frontend

```bash
# Live Mode (connects to deployed Bedrock Agent)
uv run chainlit run app.py --port 8000
```

Open http://localhost:8000. The app connects to your deployed Bedrock Agent and live AWS services.

### Quick Start (Mock Mode — No AWS Needed)

If you just want to see the UI without deploying anything:

```bash
cd demo_medallion_arc_aws_guardrail
uv sync
uv run chainlit run app.py --port 8000
```

No `.env` needed — the app detects missing credentials and runs in Mock Mode with simulated responses. You'll see the "⚡ Offline Mock Mode" indicator.

### Docker Compose (Mock Mode)

```bash
docker compose up --build
```

Open http://localhost:8000. Health endpoint at http://localhost:8000/health.

---

## Deploying to AWS (Production)

Production deployment also has two steps: deploy the agent, then deploy the frontend container to Fargate.

### Prerequisites (AWS Infrastructure)

These must be pre-provisioned before deployment:
- Amazon Bedrock Knowledge Base (with your data ingested)
- Amazon Bedrock Guardrails (jailbreak detection + output filtering)
- Amazon Macie (PII scan results pre-computed on source data)
- AWS Glue Catalog with Gold Table (PII + non-PII columns)
- AWS Lake Formation (column-level access control configured)
- Two IAM Roles:
  - **Permissive_Role** — Lake Formation grants to ALL columns
  - **Restricted_Role** — Lake Formation grants to non-PII columns only

### Step 1: Deploy the Bedrock Agent

```bash
# Ensure .env has all required values
uv run python scripts/deploy_bedrock_agent.py
```

The script:
1. Creates an IAM execution role for the agent
2. Creates the Bedrock Agent with model, instructions, and guardrails
3. Associates the Knowledge Base for RAG
4. Prepares the agent (compiles it)
5. Creates a `live` alias for stable invocation

### Step 2: Deploy the Chainlit Frontend to Fargate

**Build and push the container image:**

```bash
# Build
docker build -t demo-medallion-arc-aws-guardrail .

# Tag and push to ECR
aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com

docker tag demo-medallion-arc-aws-guardrail:latest \
  <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/demo-medallion-arc-aws-guardrail:latest

docker push <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/demo-medallion-arc-aws-guardrail:latest
```

**Create a Fargate task definition** with:
- Container image from ECR
- Port mapping: 8000
- IAM Task Role with `sts:AssumeRole` on both Permissive_Role and Restricted_Role
- All environment variables from `.env`
- Health check: `curl -f http://localhost:8000/health`
- Start period: 30 seconds

**Create a Fargate service** behind an Application Load Balancer:
- Target group health check path: `/health`
- Desired count: 1 (single booth instance)

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AWS_DEFAULT_REGION` | AWS region (default: ap-southeast-1) | No |
| `BEDROCK_MODEL_ID` | Bedrock model ID for inference | Yes |
| `KNOWLEDGE_BASE_ID` | Bedrock Knowledge Base ID | Yes |
| `GUARDRAILS_ID` | Bedrock Guardrails ID | Yes |
| `GUARDRAILS_VERSION` | Guardrails version string | Yes |
| `GLUE_DATABASE_NAME` | Glue Catalog database name | Yes |
| `GLUE_TABLE_NAME` | Gold table name in Glue | Yes |
| `PERMISSIVE_ROLE_ARN` | IAM role ARN for Engine A (all columns) | Yes |
| `RESTRICTED_ROLE_ARN` | IAM role ARN for Engine B (non-PII only) | Yes |
| `BEDROCK_AGENT_ID` | Agent ID (from deploy script output) | For Live Mode |
| `BEDROCK_AGENT_ALIAS_ID` | Agent alias ID (from deploy script output) | For Live Mode |
| `MOCK_MODE` | Force mock mode if set to "true" | No |

If any required variable is missing, the app logs the missing names and switches to Mock Mode automatically.

---

## Running Tests

```bash
cd demo_medallion_arc_aws_guardrail

# Full test suite (425 tests)
uv run pytest -v

# Property-based tests only
uv run pytest tests/test_properties/ -v

# Unit tests only
uv run pytest tests/test_unit/ -v

# Integration tests only
uv run pytest tests/test_integration/ -v
```

---

## How It Works for Booth Visitors

1. Visitor opens the app and sees preset buttons (red = attacks, blue = analytics)
2. Visitor clicks a preset or types their own prompt
3. Both engines process the prompt concurrently
4. **Engine A** (Data Swamp) leaks PII and sensitive data — annotated with "⚠️ Ungoverned — Data Leaked"
5. **Engine B** (Safe Haven) blocks jailbreaks and protects data — annotated with "✅ Safe Haven — Protected"
6. For analytics queries, Engine B renders interactive Plotly charts while Engine A shows raw data with PII visible

The contrast demonstrates why Zero-Trust data governance matters.
