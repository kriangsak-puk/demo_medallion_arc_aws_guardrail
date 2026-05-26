From Bronze to AI: Safeguarding the Medallion Architecture for LLMs

Welcome to the official repository blueprint for the AWS Summit Bangkok Demo Booth project. This project showcases how to secure a Generative AI pipeline using a Zero-Trust Medallion Architecture built on AWS.

The interactive booth features the "Try to Jailbreak Me" Challenge, a dual-engine comparison web application powered by Chainlit that demonstrates the difference between an ungoverned "Data Swamp" and a fully secured "Zero-Trust Safe Haven."

🛠️ Architecture Overview

The system processes raw S3 data through a 3-Gate defense architecture before serving it as an AI Knowledge Base in Amazon Bedrock:

                                  [ USER INPUT ]
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       🧟 ENGINE A: DATA SWAMP                        🛡️ ENGINE B: SAFE HAVEN
┌───────────────────────────────┐              ┌───────────────────────────────┐
│ • Raw Bronze Bucket Context   │              │ • Sanitized Gold Data (Macie) │
│ • No Input Guardrails         │              │ • Column-Level Access Control │
│ • No Output Filtering         │              │ • Bedrock Guardrails Enabled  │
│                               │              │ • Contextual Grounding Filter │
└───────────────────────────────┘              └───────────────────────────────┘
                 │                                             │
                 ▼                                             ▼
     [ Uncensored Data Leaked! ]                  [ Blocked / Safe Output ]


The Three Security Gates

Gate 1 (Sanitization): Amazon Macie automatically detects PII (Personally Identifiable Information) in the Bronze bucket. AWS Glue tokenizes and redacts this data before promoting it to Silver/Gold.

Gate 2 (Authorization): AWS Lake Formation restricts database access, ensuring the AI service role only queries columns relevant to the prompt (Least Privilege).

Gate 3 (Interaction): Amazon Bedrock Guardrails inspects the prompt for jailbreaks (input protection) and filters the output for hallucinations and residual PII (output protection).

🎨 Interactive Booth Experience

To capture the interest of passersby in the Expo Hall, the Chainlit app provides an interactive showdown:

Dual Mode (Showdown): The visitor inputs a single "Attack Prompt" (e.g., trying to access proprietary company codes or customer phone numbers). The app evaluates the input against both engines simultaneously and streams their responses side-by-side or sequentially.

Under-the-Hood Step Tracking: Inside the UI, Chainlit dynamically expands collapsed steps showing real-time logs like [Checking Amazon Macie...] and [Evaluating Bedrock Guardrails...] to highlight your underlying data engineering pipelines.

Offline Resiliency: An automatic fallback to simulated AWS behaviors ensures your demo remains functional even during severe network drops on the exhibition floor.

📁 Repository Structure

├── README.md               # Project configuration and architectural guide
├── app.py                  # Chainlit dual-engine interactive application
├── requirements.txt        # Python dependency manifest
└── cloudformation/
    └── infrastructure.yaml # CloudFormation template for the backend resources


⚙️ Local Setup and Installation

1. Prerequisites

Ensure you have Python 3.9+ installed. Install the required dependencies:

pip install chainlit boto3 python-dotenv

framework for LLM is AWS strand and willing to deploy on Fargate

2. Configure AWS Credentials (Optional for Live Mode)

To connect the application to real AWS services, set up your AWS credentials file (~/.aws/credentials) or export your environment variables:

export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="ap-southeast-1" # Bangkok Summit region preference


If these keys are not present, the application will automatically launch in Offline Mock Mode, enabling full demo functionality with pre-programmed interactive jailbreak responses.

3. Running the Application

Launch your interactive booth screen using the Chainlit CLI:

chainlit run app.py -w


The -w flag enables hot-reloading so you can adjust your styling and prompt scripts on the fly.

