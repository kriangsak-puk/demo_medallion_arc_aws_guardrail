"""Mock engine for offline operation (Mock_Mode).

Provides pre-programmed responses simulating AWS service behavior when
network connectivity is unavailable. Ensures the demo remains fully
functional on the exhibition floor without live AWS access.
"""

import asyncio
import random
from typing import Any, Dict, List


class MockEngine:
    """Simulates AWS service responses for offline demo operation.

    Provides mock implementations of both engines and all three gates,
    using keyword-based logic and pre-programmed data to replicate the
    behavior of the live AWS pipeline.
    """

    JAILBREAK_KEYWORDS: List[str] = [
        "ignore",
        "bypass",
        "override",
        "jailbreak",
        "pretend",
        "roleplay",
        "DAN",
    ]

    PII_KEYWORDS: List[str] = [
        "phone",
        "email",
        "national id",
        "ssn",
        "address",
        "credit card",
    ]

    async def mock_engine_a(self, prompt: str) -> str:
        """Return pre-programmed 'leaked data' response with mock PII.

        Simulates Engine_A (Data Swamp) behavior — unrestricted column access
        via Permissive_Role, returning sensitive data without guardrail filtering.

        Args:
            prompt: The visitor's attack prompt.

        Returns:
            A response string containing mock PII data simulating
            unrestricted column access.
        """
        return (
            "⚠️ Data Swamp Response (Ungoverned):\n\n"
            "Here is the requested data from the Gold Table:\n\n"
            "| customer_id | first_name | last_name | email | phone_number | national_id | category | product_name | total_amount |\n"
            "|-------------|-----------|-----------|-------|--------------|-------------|----------|--------------|-------------|\n"
            "| CUST-10042 | Somchai | Garcia | somchai.garcia55@gmail.com | +1-481-234-5678 | 312-45-6789 | Electronics | Laptop Pro 15 | 1,249.99 |\n"
            "| CUST-10087 | Ananya | Johnson | ananya.johnson12@yahoo.com | +1-692-876-5432 | 567-89-0123 | Apparel | Running Shoes | 189.50 |\n"
            "| CUST-10153 | Hans | Williams | hans.williams77@outlook.com | +1-370-123-4567 | 890-12-3456 | Home & Kitchen | Air Fryer 5L | 299.00 |\n\n"
            "Credit Card: 4532-1234-5678-9012\n"
            "Shipping Address: 4521 Oak St, Austin, TX 78701\n"
            "Access Level: FULL (Permissive_Role — all columns visible including PII)"
        )

    async def mock_engine_b(self, prompt: str) -> str:
        """Return 'blocked' or 'sanitized' response based on keyword matching.

        Simulates Engine_B (Safe Haven) behavior — restricted column access
        via Restricted_Role with guardrail enforcement.

        If the prompt contains any jailbreak or PII-extraction keyword,
        returns a "blocked" response. Otherwise returns a "sanitized"
        response showing clean data without PII.

        Args:
            prompt: The visitor's attack prompt.

        Returns:
            A "blocked" response if keywords matched, otherwise a
            "sanitized" response with clean analytics data.
        """
        prompt_lower = prompt.lower()

        # Check for jailbreak keywords
        for keyword in self.JAILBREAK_KEYWORDS:
            if keyword.lower() in prompt_lower:
                return (
                    "🛡️ Blocked by Bedrock Guardrails\n\n"
                    "Your request was identified as a potential jailbreak attempt "
                    "and has been blocked by the Safe Haven security pipeline.\n\n"
                    f"Detected keyword: '{keyword}'\n"
                    "Action: INPUT BLOCKED"
                )

        # Check for PII-extraction keywords
        for keyword in self.PII_KEYWORDS:
            if keyword.lower() in prompt_lower:
                return (
                    "🛡️ Blocked by Bedrock Guardrails\n\n"
                    "Your request attempted to access personally identifiable "
                    "information (PII) which is restricted by Lake Formation "
                    "column-level security.\n\n"
                    f"Detected PII keyword: '{keyword}'\n"
                    "Action: INPUT BLOCKED"
                )

        # No keywords matched — return sanitized response
        return (
            "✅ Safe Haven Response (Governed):\n\n"
            "Here is the requested data from the Gold Table:\n\n"
            "| customer_id | category | product_name | total_amount | order_date | order_status |\n"
            "|-------------|----------|--------------|-------------|------------|-------------|\n"
            "| CUST-10042 | Electronics | Laptop Pro 15 | 1,249.99 | 2026-01-15 | Completed |\n"
            "| CUST-10087 | Apparel | Running Shoes | 189.50 | 2026-02-03 | Shipped |\n"
            "| CUST-10153 | Home & Kitchen | Air Fryer 5L | 299.00 | 2026-03-22 | Completed |\n\n"
            "Access Level: RESTRICTED (Restricted_Role — PII columns excluded)\n"
            "Guardrail Status: ✅ PASS"
        )

    async def mock_gate1(self) -> Dict[str, Any]:
        """Return pre-programmed Macie results after simulated delay.

        Simulates Gate_1 (Sanitization) — displays pre-computed Amazon Macie
        PII detection findings with a realistic processing delay.

        Returns:
            A dictionary containing Macie scan findings with PII category
            counts.
        """
        delay = random.uniform(1.0, 2.0)
        await asyncio.sleep(delay)

        return {
            "findings_count": 7,
            "categories": {
                "EMAIL_ADDRESS": 1,
                "PHONE_NUMBER": 1,
                "NATIONAL_ID": 1,
                "CREDIT_CARD_NUMBER": 1,
                "DATE_OF_BIRTH": 1,
                "ADDRESS": 2,
            },
        }

    async def mock_gate2(self) -> Dict[str, Any]:
        """Return pre-programmed column access list after simulated delay.

        Simulates Gate_2 (Authorization) — shows the difference between
        Permissive_Role and Restricted_Role column-level access as enforced
        by AWS Lake Formation.

        Returns:
            A dictionary containing allowed and denied column lists for
            the Restricted_Role.
        """
        delay = random.uniform(1.0, 2.0)
        await asyncio.sleep(delay)

        return {
            "allowed": [
                "customer_id", "order_id", "order_date", "product_name",
                "category", "quantity", "price_per_unit", "total_amount",
                "payment_method", "order_status", "product_sku",
            ],
            "denied": [
                "first_name", "last_name", "email", "phone_number",
                "national_id", "date_of_birth", "credit_card_number",
                "shipping_address", "billing_address", "device_ip_address",
            ],
        }

    async def mock_gate3(self, prompt: str) -> Dict[str, Any]:
        """Return keyword-based guardrail action after simulated delay.

        Simulates Gate_3 (Interaction) — evaluates the prompt against
        jailbreak keywords and returns the appropriate guardrail action.

        If a jailbreak keyword is found in the prompt, the input action
        is BLOCKED. Otherwise both input and output actions are PASS.

        Args:
            prompt: The visitor's attack prompt to evaluate.

        Returns:
            A dictionary with input_action and output_action fields
            indicating the guardrail decision.
        """
        delay = random.uniform(1.0, 2.0)
        await asyncio.sleep(delay)

        prompt_lower = prompt.lower()

        for keyword in self.JAILBREAK_KEYWORDS:
            if keyword.lower() in prompt_lower:
                return {
                    "input_action": "BLOCKED",
                    "output_action": "PASS",
                }

        return {
            "input_action": "PASS",
            "output_action": "PASS",
        }
