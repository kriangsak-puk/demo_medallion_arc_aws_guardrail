# Synthetic Customer Transactions & PII Dataset Generator

This contains a Python utility designed to generate realistic, relational synthetic data for testing Data Privacy, Data Masking, Tokenization, and Data Loss Prevention (DLP) pipelines. 

When developing PII (Personally Identifiable Information) protection systems, using real customer data introduces severe compliance and security risks. This script solves that problem by generating a mock relational dataset containing realistic direct and indirect identifiers integrated with e-commerce transaction logs.

## Features
- **Relational Integrity**: Simulates a `left join` between a unique Customer Profile pool and an Orders ledger.
- **Multi-Layered PII**: Generates structurally valid PII configurations (Social Security Numbers, Phone Patterns, Emails, Credit Cards, and IP Addresses) ideal for testing Regex engines and masking algorithms.
- **Dual Output Formats**: Automatically writes the synchronized data to both flat-text **CSV** and compressed columnar **Parquet** formats to test different pipeline performance metrics.

---

## Dataset Schema

The generated dataset combines transaction metrics with sensitive customer fields. Below are the generated columns categorized by security sensitivity:

| Column Name | Data Type | Classification | Description / Pattern Example |
| :--- | :--- | :--- | :--- |
| `order_id` | String | Public / Operational | Unique order identifier (`ORD-100001`) |
| `transaction_id` | String | Public / Operational | Unique bank gateway transaction token (`TXN-200001`) |
| `customer_id` | String | Public / Join Key | Relational link mapping orders back to customer accounts |
| `order_date` | Timestamp | Public | Timestamp of purchase (`YYYY-MM-DD HH:MM:SS`) |
| `product_sku` | String | Public | Formatted warehouse SKU key |
| `product_name` | String | Public | Name of purchased retail item |
| `category` | String | Public | Retail category categorization |
| `quantity` | Integer | Public | Quantity purchased |
| `price_per_unit` | Float | Public | Price per single unit |
| `total_amount` | Float | Public | Computed total order pricing |
| `payment_method` | String | Public | Payment provider channel (`Visa`, `MasterCard`, etc.) |
| **`first_name`** | String | **Direct PII** | Customer's given name |
| **`last_name`** | String | **Direct PII** | Customer's family name |
| **`email`** | String | **Direct PII** | Synthesized contact email |
| **`phone_number`** | String | **Direct PII** | Multi-format cellular number layout |
| **`national_id`** | String | **Highly Sensitive PII** | Government ID / SSN hyphenated mask (`###-##-####`) |
| **`date_of_birth`** | String (Date) | **Indirect / Quasi PII** | Date of birth (`YYYY-MM-DD`) |
| **`shipping_address`**| String | **Direct PII** | Full street, city, state, and zip routing block |
| **`billing_address`** | String | **Direct PII** | Financial statement address billing block |
| **`credit_card_number`**| String | **Highly Sensitive PII** | Full structured credit card string |
| **`device_ip_address`** | String | **Technical / Network PII** | Client IPv4 endpoint address |
| `order_status` | String | Public | Transaction state (`Completed`, `Shipped`, `Refunded`, `Failed`) |

---

## Getting Started with `uv`

This project uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver written in Rust.

### Prerequisites
Make sure `uv` is installed on your machine. If you don't have it yet, install it via curl or powershell:
```bash
# macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows (PowerShell)
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"


## How to run
```bash
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install pandas pyarrow
uv run generate_mock_data.py