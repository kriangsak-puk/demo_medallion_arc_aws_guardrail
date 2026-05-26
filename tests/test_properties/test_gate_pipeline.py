"""Property-based tests for gate pipeline sequential ordering.

Feature: safe-haven-demo-booth
Property 4: Gate sequential ordering
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from config import AppMode
from gates.base import Gate, GateContext, GatePipeline, GateResult


# --- Timestamp-recording test gate ---


@dataclass
class TimestampRecord:
    """Records start and end timestamps for a gate execution."""

    gate_name: str
    start_time: float = 0.0
    end_time: float = 0.0


class TimestampGate(Gate):
    """A test gate that records its start and end timestamps.

    Used to verify that the GatePipeline executes gates in strict
    sequential order with no overlap.
    """

    def __init__(self, name: str, delay_seconds: float, records: List[TimestampRecord]) -> None:
        super().__init__(name=name, timeout_seconds=10.0)
        self.delay_seconds = delay_seconds
        self.records = records

    async def execute(self, context: GateContext) -> GateResult:
        """Execute gate, recording start/end timestamps."""
        record = TimestampRecord(gate_name=self.name)
        record.start_time = time.monotonic()

        # Simulate some work with a small delay
        await asyncio.sleep(self.delay_seconds)

        record.end_time = time.monotonic()
        self.records.append(record)

        return GateResult(passed=True, blocked=False, details={"gate_name": self.name})


# --- Strategies ---


def _arbitrary_prompt() -> st.SearchStrategy[str]:
    """Generate arbitrary prompt strings for gate pipeline execution."""
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=200,
    )


def _gate_delay() -> st.SearchStrategy[float]:
    """Generate small random delays for test gates (0.001 to 0.05 seconds)."""
    return st.floats(min_value=0.001, max_value=0.05)


# =============================================================================
# Property 4: Gate pipeline executes in strict sequential order
# =============================================================================


class TestProperty4GateSequentialOrdering:
    """Property 4: Gate pipeline executes in strict sequential order.

    For any Engine_B processing of an Attack_Prompt, the gate execution
    timestamps SHALL satisfy: Gate_1 completion time < Gate_2 start time,
    and Gate_2 completion time < Gate_3 start time. No gate SHALL begin
    execution before the previous gate has completed.

    **Validates: Requirements 3.2**
    """

    @settings(max_examples=100, deadline=None)
    @given(prompt=_arbitrary_prompt(), delay1=_gate_delay(), delay2=_gate_delay(), delay3=_gate_delay())
    @pytest.mark.asyncio
    async def test_gate_pipeline_strict_sequential_order(
        self, prompt: str, delay1: float, delay2: float, delay3: float
    ):
        """Gate pipeline executes gates in strict sequential order with no overlap.

        For any random prompt and any gate processing delays, the pipeline
        SHALL ensure Gate_1 completes before Gate_2 starts, and Gate_2
        completes before Gate_3 starts.

        Feature: safe-haven-demo-booth, Property 4: Gate sequential ordering
        **Validates: Requirements 3.2**
        """
        records: List[TimestampRecord] = []

        # Create three test gates with random delays
        gate1 = TimestampGate(name="Gate_1", delay_seconds=delay1, records=records)
        gate2 = TimestampGate(name="Gate_2", delay_seconds=delay2, records=records)
        gate3 = TimestampGate(name="Gate_3", delay_seconds=delay3, records=records)

        pipeline = GatePipeline(gates=[gate1, gate2, gate3])

        context = GateContext(
            prompt=prompt,
            session_mode=AppMode.MOCK,
            parent_message=None,
        )

        results = await pipeline.execute(context)

        # All three gates should have executed
        assert len(results) == 3, (
            f"Expected 3 gate results, got {len(results)}"
        )
        assert len(records) == 3, (
            f"Expected 3 timestamp records, got {len(records)}"
        )

        # Verify strict sequential ordering:
        # Gate_1 completion time < Gate_2 start time
        assert records[0].end_time <= records[1].start_time, (
            f"Gate_1 end ({records[0].end_time:.6f}) must be <= "
            f"Gate_2 start ({records[1].start_time:.6f}). "
            f"Gate_1 did not complete before Gate_2 started."
        )

        # Gate_2 completion time < Gate_3 start time
        assert records[1].end_time <= records[2].start_time, (
            f"Gate_2 end ({records[1].end_time:.6f}) must be <= "
            f"Gate_3 start ({records[2].start_time:.6f}). "
            f"Gate_2 did not complete before Gate_3 started."
        )

    @settings(max_examples=100, deadline=None)
    @given(prompt=_arbitrary_prompt(), delay1=_gate_delay(), delay2=_gate_delay(), delay3=_gate_delay())
    @pytest.mark.asyncio
    async def test_gate_pipeline_no_gate_starts_before_previous_completes(
        self, prompt: str, delay1: float, delay2: float, delay3: float
    ):
        """No gate SHALL begin execution before the previous gate has completed.

        This verifies the invariant from a different angle: for each pair
        of consecutive gates (i, i+1), gate i+1's start time must be
        strictly after gate i's end time.

        Feature: safe-haven-demo-booth, Property 4: Gate sequential ordering
        **Validates: Requirements 3.2**
        """
        records: List[TimestampRecord] = []

        gate1 = TimestampGate(name="Gate_1", delay_seconds=delay1, records=records)
        gate2 = TimestampGate(name="Gate_2", delay_seconds=delay2, records=records)
        gate3 = TimestampGate(name="Gate_3", delay_seconds=delay3, records=records)

        pipeline = GatePipeline(gates=[gate1, gate2, gate3])

        context = GateContext(
            prompt=prompt,
            session_mode=AppMode.MOCK,
            parent_message=None,
        )

        results = await pipeline.execute(context)

        assert len(records) == 3, (
            f"Expected 3 timestamp records, got {len(records)}"
        )

        # Verify no overlap: each gate's start must be >= previous gate's end
        for i in range(len(records) - 1):
            current_gate = records[i]
            next_gate = records[i + 1]

            assert current_gate.end_time <= next_gate.start_time, (
                f"{current_gate.gate_name} end ({current_gate.end_time:.6f}) "
                f"must be <= {next_gate.gate_name} start ({next_gate.start_time:.6f}). "
                f"Sequential ordering violated: {next_gate.gate_name} started "
                f"before {current_gate.gate_name} completed."
            )

    @settings(max_examples=100, deadline=None)
    @given(prompt=_arbitrary_prompt(), delay1=_gate_delay(), delay2=_gate_delay(), delay3=_gate_delay())
    @pytest.mark.asyncio
    async def test_gate_pipeline_execution_order_matches_gate_list_order(
        self, prompt: str, delay1: float, delay2: float, delay3: float
    ):
        """Gates execute in the order they are provided to the pipeline.

        The recorded gate names must appear in the same order as the
        gates were added to the pipeline, confirming sequential execution.

        Feature: safe-haven-demo-booth, Property 4: Gate sequential ordering
        **Validates: Requirements 3.2**
        """
        records: List[TimestampRecord] = []

        gate1 = TimestampGate(name="Gate_1", delay_seconds=delay1, records=records)
        gate2 = TimestampGate(name="Gate_2", delay_seconds=delay2, records=records)
        gate3 = TimestampGate(name="Gate_3", delay_seconds=delay3, records=records)

        pipeline = GatePipeline(gates=[gate1, gate2, gate3])

        context = GateContext(
            prompt=prompt,
            session_mode=AppMode.MOCK,
            parent_message=None,
        )

        await pipeline.execute(context)

        assert len(records) == 3
        assert records[0].gate_name == "Gate_1", (
            f"First executed gate should be Gate_1, got {records[0].gate_name}"
        )
        assert records[1].gate_name == "Gate_2", (
            f"Second executed gate should be Gate_2, got {records[1].gate_name}"
        )
        assert records[2].gate_name == "Gate_3", (
            f"Third executed gate should be Gate_3, got {records[2].gate_name}"
        )

        # Also verify monotonic time ordering (end of previous <= start of next)
        assert records[0].end_time <= records[1].start_time, (
            "Gate_1 must complete before Gate_2 starts"
        )
        assert records[1].end_time <= records[2].start_time, (
            "Gate_2 must complete before Gate_3 starts"
        )
