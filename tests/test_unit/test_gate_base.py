"""Unit tests for the gate base classes and GatePipeline."""

import pytest

from aws_demo_booth.config import AppMode
from aws_demo_booth.gates.base import Gate, GateContext, GatePipeline, GateResult


# --- Concrete gate implementations for testing ---


class PassingGate(Gate):
    """A gate that always passes."""

    async def execute(self, context: GateContext) -> GateResult:
        return GateResult(passed=True, blocked=False)


class BlockingGate(Gate):
    """A gate that always blocks."""

    async def execute(self, context: GateContext) -> GateResult:
        return GateResult(passed=False, blocked=True, details={"reason": "blocked"})


class ErrorGate(Gate):
    """A gate that always returns an error."""

    async def execute(self, context: GateContext) -> GateResult:
        return GateResult(passed=False, blocked=False, error="service unavailable")


class TrackingGate(Gate):
    """A gate that records when it was executed."""

    execution_order: list = []

    def __init__(self, name: str, timeout_seconds: float, tracker: list) -> None:
        super().__init__(name, timeout_seconds)
        self.tracker = tracker

    async def execute(self, context: GateContext) -> GateResult:
        self.tracker.append(self.name)
        return GateResult(passed=True, blocked=False)


# --- Tests ---


@pytest.fixture
def gate_context() -> GateContext:
    return GateContext(
        prompt="test prompt",
        session_mode=AppMode.MOCK,
        parent_message=None,
    )


class TestGateContext:
    def test_create_context(self):
        ctx = GateContext(
            prompt="hello",
            session_mode=AppMode.LIVE,
            parent_message="msg",
        )
        assert ctx.prompt == "hello"
        assert ctx.session_mode == AppMode.LIVE
        assert ctx.parent_message == "msg"

    def test_context_with_mock_mode(self):
        ctx = GateContext(
            prompt="attack",
            session_mode=AppMode.MOCK,
            parent_message=None,
        )
        assert ctx.session_mode == AppMode.MOCK


class TestGateResult:
    def test_passed_result(self):
        result = GateResult(passed=True, blocked=False)
        assert result.passed is True
        assert result.blocked is False
        assert result.details == {}
        assert result.error is None

    def test_blocked_result(self):
        result = GateResult(passed=False, blocked=True, details={"reason": "jailbreak"})
        assert result.blocked is True
        assert result.details == {"reason": "jailbreak"}

    def test_error_result(self):
        result = GateResult(passed=False, blocked=False, error="timeout")
        assert result.error == "timeout"


class TestGate:
    def test_gate_has_name_and_timeout(self):
        gate = PassingGate(name="test_gate", timeout_seconds=5.0)
        assert gate.name == "test_gate"
        assert gate.timeout_seconds == 5.0

    @pytest.mark.asyncio
    async def test_gate_execute(self, gate_context):
        gate = PassingGate(name="pass", timeout_seconds=10.0)
        result = await gate.execute(gate_context)
        assert result.passed is True
        assert result.blocked is False


class TestGatePipeline:
    @pytest.mark.asyncio
    async def test_all_gates_pass(self, gate_context):
        gates = [
            PassingGate(name="gate1", timeout_seconds=5.0),
            PassingGate(name="gate2", timeout_seconds=10.0),
            PassingGate(name="gate3", timeout_seconds=10.0),
        ]
        pipeline = GatePipeline(gates)
        results = await pipeline.execute(gate_context)

        assert len(results) == 3
        assert all(r.passed for r in results)
        assert not any(r.blocked for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_block(self, gate_context):
        gates = [
            PassingGate(name="gate1", timeout_seconds=5.0),
            BlockingGate(name="gate2", timeout_seconds=10.0),
            PassingGate(name="gate3", timeout_seconds=10.0),
        ]
        pipeline = GatePipeline(gates)
        results = await pipeline.execute(gate_context)

        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].blocked is True

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_error(self, gate_context):
        gates = [
            PassingGate(name="gate1", timeout_seconds=5.0),
            ErrorGate(name="gate2", timeout_seconds=10.0),
            PassingGate(name="gate3", timeout_seconds=10.0),
        ]
        pipeline = GatePipeline(gates)
        results = await pipeline.execute(gate_context)

        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].error == "service unavailable"

    @pytest.mark.asyncio
    async def test_pipeline_executes_in_order(self, gate_context):
        tracker: list = []
        gates = [
            TrackingGate(name="first", timeout_seconds=5.0, tracker=tracker),
            TrackingGate(name="second", timeout_seconds=10.0, tracker=tracker),
            TrackingGate(name="third", timeout_seconds=10.0, tracker=tracker),
        ]
        pipeline = GatePipeline(gates)
        await pipeline.execute(gate_context)

        assert tracker == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_empty_pipeline(self, gate_context):
        pipeline = GatePipeline(gates=[])
        results = await pipeline.execute(gate_context)
        assert results == []

    @pytest.mark.asyncio
    async def test_first_gate_blocks(self, gate_context):
        gates = [
            BlockingGate(name="gate1", timeout_seconds=5.0),
            PassingGate(name="gate2", timeout_seconds=10.0),
        ]
        pipeline = GatePipeline(gates)
        results = await pipeline.execute(gate_context)

        assert len(results) == 1
        assert results[0].blocked is True

    @pytest.mark.asyncio
    async def test_blocked_gate_not_passed(self, gate_context):
        """A blocking result should not execute subsequent gates."""
        tracker: list = []
        gates = [
            TrackingGate(name="first", timeout_seconds=5.0, tracker=tracker),
            BlockingGate(name="blocker", timeout_seconds=10.0),
            TrackingGate(name="third", timeout_seconds=10.0, tracker=tracker),
        ]
        pipeline = GatePipeline(gates)
        results = await pipeline.execute(gate_context)

        assert len(results) == 2
        assert tracker == ["first"]  # "third" never executed
