"""Gate pipeline base classes for the 3-Gate sequential defense architecture.

Provides the abstract Gate class, GateContext/GateResult dataclasses,
and the GatePipeline orchestrator that executes gates in strict sequential order.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aws_demo_booth.config import AppMode


@dataclass
class GateContext:
    """Context passed to each gate during execution.

    Attributes:
        prompt: The attack prompt submitted by the visitor.
        session_mode: Current application mode (LIVE or MOCK).
        parent_message: The parent Chainlit message (typed as Any to avoid
            hard dependency on chainlit in the base module).
    """

    prompt: str
    session_mode: AppMode
    parent_message: Any  # cl.Message — use Any to avoid hard chainlit dependency


@dataclass
class GateResult:
    """Result returned by a gate after execution.

    Attributes:
        passed: Whether the gate completed successfully.
        blocked: Whether the gate blocked further pipeline progression.
        details: Arbitrary metadata about the gate execution.
        error: Error message if the gate encountered a failure.
    """

    passed: bool
    blocked: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Gate(ABC):
    """Abstract base class for all security gates.

    Each gate has a name and a timeout, and must implement the
    async execute() method that performs its logic and returns a GateResult.
    """

    def __init__(self, name: str, timeout_seconds: float) -> None:
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def execute(self, context: GateContext) -> GateResult:
        """Execute gate logic and return the result.

        Args:
            context: The gate context containing prompt, session mode,
                and parent message.

        Returns:
            A GateResult indicating whether the gate passed, blocked,
            or encountered an error.
        """
        ...


class GatePipeline:
    """Executes a list of gates in strict sequential order.

    Gates are executed one at a time. If a gate returns blocked=True or
    an error, the pipeline stops immediately and returns the results
    collected so far (including the blocking/error result).
    """

    def __init__(self, gates: List[Gate]) -> None:
        self.gates = gates

    async def execute(self, context: GateContext) -> List[GateResult]:
        """Run all gates sequentially, stopping on block or error.

        Args:
            context: The gate context to pass to each gate.

        Returns:
            A list of GateResult objects from all executed gates.
            If a gate blocks or errors, the list includes that result
            and no subsequent gates are executed.
        """
        results: List[GateResult] = []

        for gate in self.gates:
            result = await gate.execute(context)
            results.append(result)

            if result.blocked:
                break

            if result.error is not None:
                break

        return results
