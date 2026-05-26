# 🏰 Safe Haven Demo Booth

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

```
aws_demo_booth/
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
├── Dockerfile              # Container image (Python 3.11 slim)
├── docker-compose.yml      # Local development with placeholder env vars
├── requirements.txt        # Pinned Python dependencies
└── tests/                  # 425 tests (property, unit, integration)
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager (recommended) or pip
- Docker (for containerized deployment)

## Running Locally

### Option 1: Direct (Mock Mode)

No AWS credentials needed — the app runs with simulated responses.

```bash
cd aws_demo_booth

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Run the Chainlit app (launches in Mock Mode automatically)
chainlit run app.py --port 8000
```

Open http://localhost:8000 in your browser. You'll see the "⚡ Offline Mock Mode" indicator.

### Option 2: Docker Compose (Mock Mode)

```bash
cd aws_demo_booth

docker compose up --build
```

Open http://localhost:8000. The health endpoint is at http://localhost:8000/health.

### Option 3: Direct (Live Mode)

Set the required environment variables to connect to your pre-deployed AWS infrastructure:

```bash
export KNOWLEDGE_BASE_ID=your-kb-id
export GUARDRAILS_ID=your-guardrails-id
export GUARDRAILS_VERSION=1
export GLUE_DATABASE_NAME=your-database
export GLUE_TABLE_NAME=your-gold-table
export BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
export PERMISSIVE_ROLE_ARN=arn:aws:iam::123456789012:role/PermissiveRole
export RESTRICTED_ROLE_ARN=arn:aws:iam::123456789012:role/RestrictedRole
export AWS_DEFAULT_REGION=ap-southeast-1

chainlit run app.py --port 8000
```

The app will detect valid credentials and connect to live AWS services. If any service is unreachable, it falls back to Mock Mode for that engine automatically.

## Running on AWS (Fargate)

### Prerequisites

All backend infrastructure must be pre-deployed:
- Amazon Bedrock Knowledge Base
- Amazon Bedrock Guardrails (jailbreak detection + output filtering)
- Amazon Macie (PII scan results pre-computed)
- AWS Glue Catalog with Gold Table (PII + non-PII columns)
- AWS Lake Formation (column-level access via Permissive_Role and Restricted_Role)
- Two IAM Roles: Permissive_Role (all columns) and Restricted_Role (non-PII only)

### Deploy to Fargate

1. **Build and push the container image:**

```bash
cd aws_demo_booth

# Build
docker build -t safe-haven-demo-booth .

# Tag and push to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com
docker tag safe-haven-demo-booth:latest <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/safe-haven-demo-booth:latest
docker push <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/safe-haven-demo-booth:latest
```

2. **Create a Fargate task definition** with:
   - Container image from ECR
   - Port mapping: 8000
   - IAM Task Role with permissions to `sts:AssumeRole` on both Permissive_Role and Restricted_Role
   - Environment variables (all required vars listed above)
   - Health check: `curl -f http://localhost:8000/health`
   - Start period: 30 seconds

3. **Create a Fargate service** behind an Application Load Balancer:
   - Target group health check path: `/health`
   - Desired count: 1 (single booth instance)

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `KNOWLEDGE_BASE_ID` | Bedrock Knowledge Base ID | Yes |
| `GUARDRAILS_ID` | Bedrock Guardrails ID | Yes |
| `GUARDRAILS_VERSION` | Guardrails version string | Yes |
| `GLUE_DATABASE_NAME` | Glue Catalog database name | Yes |
| `GLUE_TABLE_NAME` | Gold table name in Glue | Yes |
| `BEDROCK_MODEL_ID` | Bedrock model ID for inference | Yes |
| `PERMISSIVE_ROLE_ARN` | IAM role ARN for Engine A (all columns) | Yes |
| `RESTRICTED_ROLE_ARN` | IAM role ARN for Engine B (non-PII only) | Yes |
| `AWS_DEFAULT_REGION` | AWS region (default: ap-southeast-1) | No |
| `MOCK_MODE` | Force mock mode if set to "true" | No |

If any required variable is missing, the app logs the missing variable names and switches to Mock Mode automatically.

## Running Tests

```bash
cd aws_demo_booth

# Run the full test suite (425 tests)
python -m pytest tests/ -v

# Run only property-based tests
python -m pytest tests/test_properties/ -v

# Run only unit tests
python -m pytest tests/test_unit/ -v

# Run only integration tests
python -m pytest tests/test_integration/ -v
```

## How It Works for Booth Visitors

1. Visitor opens the app and sees preset buttons (red = attacks, blue = analytics)
2. Visitor clicks a preset or types their own prompt
3. Both engines process the prompt concurrently
4. **Engine A** (Data Swamp) leaks PII and sensitive data — annotated with "⚠️ Ungoverned — Data Leaked"
5. **Engine B** (Safe Haven) blocks jailbreaks and protects data — annotated with "✅ Safe Haven — Protected"
6. For analytics queries, Engine B renders interactive Plotly charts while Engine A shows raw data with PII visible

The contrast demonstrates why Zero-Trust data governance matters.
