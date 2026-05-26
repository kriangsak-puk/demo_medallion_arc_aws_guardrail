# Gate Pipeline - 3-Gate sequential defense architecture
"""Gate 1: Macie Scan Results, Gate 2: Lake Formation, Gate 3: Bedrock Guardrails."""

from gates.base import Gate, GateContext, GatePipeline, GateResult
from gates.gate1_macie import Gate1Macie
from gates.gate2_lakeformation import Gate2LakeFormation
from gates.gate3_guardrails import Gate3Guardrails

__all__ = [
    "Gate",
    "Gate1Macie",
    "Gate2LakeFormation",
    "Gate3Guardrails",
    "GateContext",
    "GatePipeline",
    "GateResult",
]
