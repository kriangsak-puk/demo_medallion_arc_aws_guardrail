"""Unit tests for the Strands Agent wrapper module.

Tests cover:
- AgentConfig dataclass creation and defaults
- AgentResponse dataclass creation and defaults
- StrandsAgentWrapper initialization
- STS AssumeRole success and failure handling
- Knowledge Base retrieval success and failure handling
- Timeout configuration
- Error logging
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_wrapper import (
    KB_RETRIEVAL_TIMEOUT_SECONDS,
    MODEL_INFERENCE_TIMEOUT_SECONDS,
    AgentConfig,
    AgentResponse,
    KnowledgeBaseError,
    RoleAssumptionError,
    StrandsAgentWrapper,
)


# --- AgentConfig Tests ---


class TestAgentConfig:
    """Tests for the AgentConfig dataclass."""

    def test_create_with_all_fields(self):
        """AgentConfig stores all service identifiers correctly."""
        config = AgentConfig(
            bedrock_model_id="anthropic.claude-3-sonnet",
            knowledge_base_id="kb-12345",
            guardrails_id="gr-67890",
            guardrails_version="1",
            glue_database_name="demo_db",
            glue_table_name="gold_table",
            permissive_role_arn="arn:aws:iam::123456789012:role/PermissiveRole",
            restricted_role_arn="arn:aws:iam::123456789012:role/RestrictedRole",
        )
        assert config.bedrock_model_id == "anthropic.claude-3-sonnet"
        assert config.knowledge_base_id == "kb-12345"
        assert config.guardrails_id == "gr-67890"
        assert config.guardrails_version == "1"
        assert config.glue_database_name == "demo_db"
        assert config.glue_table_name == "gold_table"
        assert config.permissive_role_arn == "arn:aws:iam::123456789012:role/PermissiveRole"
        assert config.restricted_role_arn == "arn:aws:iam::123456789012:role/RestrictedRole"

    def test_default_region(self):
        """AgentConfig defaults region to ap-southeast-1."""
        config = AgentConfig(
            bedrock_model_id="model",
            knowledge_base_id="kb",
            guardrails_id="gr",
            guardrails_version="1",
            glue_database_name="db",
            glue_table_name="table",
            permissive_role_arn="arn:permissive",
            restricted_role_arn="arn:restricted",
        )
        assert config.region == "ap-southeast-1"

    def test_custom_region(self):
        """AgentConfig accepts a custom region."""
        config = AgentConfig(
            bedrock_model_id="model",
            knowledge_base_id="kb",
            guardrails_id="gr",
            guardrails_version="1",
            glue_database_name="db",
            glue_table_name="table",
            permissive_role_arn="arn:permissive",
            restricted_role_arn="arn:restricted",
            region="us-east-1",
        )
        assert config.region == "us-east-1"


# --- AgentResponse Tests ---


class TestAgentResponse:
    """Tests for the AgentResponse dataclass."""

    def test_create_successful_response(self):
        """AgentResponse stores content and metadata for successful invocations."""
        response = AgentResponse(
            content="Hello, world!",
            metadata={"model_id": "claude-3", "role_arn": "arn:role"},
            guardrail_action="PASS",
            error=None,
        )
        assert response.content == "Hello, world!"
        assert response.metadata == {"model_id": "claude-3", "role_arn": "arn:role"}
        assert response.guardrail_action == "PASS"
        assert response.error is None

    def test_create_error_response(self):
        """AgentResponse stores error information when invocation fails."""
        response = AgentResponse(
            content="",
            metadata={},
            guardrail_action=None,
            error="Agent invocation failed: [TimeoutError] Request timed out",
        )
        assert response.content == ""
        assert response.error is not None
        assert "TimeoutError" in response.error

    def test_default_metadata_is_empty_dict(self):
        """AgentResponse defaults metadata to empty dict."""
        response = AgentResponse(content="test")
        assert response.metadata == {}
        assert response.guardrail_action is None
        assert response.error is None

    def test_guardrail_actions(self):
        """AgentResponse supports PASS, BLOCKED, and MODIFIED guardrail actions."""
        for action in ["PASS", "BLOCKED", "MODIFIED"]:
            response = AgentResponse(content="test", guardrail_action=action)
            assert response.guardrail_action == action


# --- Timeout Configuration Tests ---


class TestTimeoutConfiguration:
    """Tests for timeout constants."""

    def test_kb_retrieval_timeout_is_25_seconds(self):
        """Knowledge Base retrieval timeout is 25 seconds per requirement 9.13."""
        assert KB_RETRIEVAL_TIMEOUT_SECONDS == 25

    def test_model_inference_timeout_is_25_seconds(self):
        """Model inference timeout is 25 seconds per requirement 9.13."""
        assert MODEL_INFERENCE_TIMEOUT_SECONDS == 25


# --- RoleAssumptionError Tests ---


class TestRoleAssumptionError:
    """Tests for the RoleAssumptionError exception."""

    def test_stores_role_arn_and_error_type(self):
        """RoleAssumptionError stores role ARN and error type for logging."""
        error = RoleAssumptionError(
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            error_type="AccessDenied",
            message="Access denied for role",
        )
        assert error.role_arn == "arn:aws:iam::123456789012:role/TestRole"
        assert error.error_type == "AccessDenied"
        assert "Access denied" in str(error)


# --- KnowledgeBaseError Tests ---


class TestKnowledgeBaseError:
    """Tests for the KnowledgeBaseError exception."""

    def test_stores_kb_id_and_error_type(self):
        """KnowledgeBaseError stores KB ID and error type for logging."""
        error = KnowledgeBaseError(
            knowledge_base_id="kb-12345",
            error_type="ResourceNotFoundException",
            message="Knowledge Base not found",
        )
        assert error.knowledge_base_id == "kb-12345"
        assert error.error_type == "ResourceNotFoundException"
        assert "not found" in str(error)


# --- StrandsAgentWrapper Tests ---


def _make_config() -> AgentConfig:
    """Create a test AgentConfig."""
    return AgentConfig(
        bedrock_model_id="anthropic.claude-3-sonnet",
        knowledge_base_id="kb-test-123",
        guardrails_id="gr-test-456",
        guardrails_version="1",
        glue_database_name="test_db",
        glue_table_name="gold_table",
        permissive_role_arn="arn:aws:iam::123456789012:role/PermissiveRole",
        restricted_role_arn="arn:aws:iam::123456789012:role/RestrictedRole",
    )


class TestStrandsAgentWrapperInit:
    """Tests for StrandsAgentWrapper initialization."""

    @patch("agent_wrapper.boto3.client")
    def test_creates_sts_client_on_init(self, mock_boto_client):
        """Wrapper creates an STS client during initialization."""
        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        assert wrapper.config == config
        # STS client should be created
        mock_boto_client.assert_called_with(
            "sts",
            config=pytest.approx(mock_boto_client.call_args_list[0][1]["config"], abs=1),
        )


class TestAssumeRole:
    """Tests for the _assume_role method."""

    @patch("agent_wrapper.boto3.client")
    def test_successful_role_assumption(self, mock_boto_client):
        """_assume_role returns credentials on success."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIAIOSFODNN7EXAMPLE",
                "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "SessionToken": "FwoGZXIvYXdzEBYaDH...",
            }
        }
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        credentials = wrapper._assume_role(
            "arn:aws:iam::123456789012:role/TestRole", "test-session"
        )

        assert credentials["aws_access_key_id"] == "AKIAIOSFODNN7EXAMPLE"
        assert credentials["aws_secret_access_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert credentials["aws_session_token"] == "FwoGZXIvYXdzEBYaDH..."

    @patch("agent_wrapper.boto3.client")
    def test_role_assumption_access_denied(self, mock_boto_client, caplog):
        """_assume_role raises RoleAssumptionError on AccessDenied."""
        from botocore.exceptions import ClientError

        mock_sts = MagicMock()
        mock_sts.assume_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Not authorized"}},
            "AssumeRole",
        )
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with pytest.raises(RoleAssumptionError) as exc_info:
            wrapper._assume_role(
                "arn:aws:iam::123456789012:role/BadRole", "test-session"
            )

        assert exc_info.value.role_arn == "arn:aws:iam::123456789012:role/BadRole"
        assert exc_info.value.error_type == "AccessDenied"

    @patch("agent_wrapper.boto3.client")
    def test_role_assumption_logs_error(self, mock_boto_client, caplog):
        """_assume_role logs the role ARN and error type on failure."""
        from botocore.exceptions import ClientError

        mock_sts = MagicMock()
        mock_sts.assume_role.side_effect = ClientError(
            {"Error": {"Code": "MalformedPolicyDocument", "Message": "Bad policy"}},
            "AssumeRole",
        )
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RoleAssumptionError):
                wrapper._assume_role(
                    "arn:aws:iam::123456789012:role/BadRole", "test-session"
                )

        assert "arn:aws:iam::123456789012:role/BadRole" in caplog.text
        assert "MalformedPolicyDocument" in caplog.text

    @patch("agent_wrapper.boto3.client")
    def test_role_assumption_unexpected_error(self, mock_boto_client):
        """_assume_role handles unexpected exceptions gracefully."""
        mock_sts = MagicMock()
        mock_sts.assume_role.side_effect = ConnectionError("Network unreachable")
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with pytest.raises(RoleAssumptionError) as exc_info:
            wrapper._assume_role(
                "arn:aws:iam::123456789012:role/TestRole", "test-session"
            )

        assert exc_info.value.error_type == "ConnectionError"


class TestRetrieveContext:
    """Tests for the retrieve_context method."""

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_successful_retrieval(self, mock_boto_client):
        """retrieve_context returns concatenated passages from KB."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }

        mock_kb_client = MagicMock()
        mock_kb_client.retrieve.return_value = {
            "retrievalResults": [
                {"content": {"text": "First passage about sales data."}},
                {"content": {"text": "Second passage about revenue."}},
            ]
        }

        mock_boto_client.side_effect = [mock_sts, mock_kb_client]

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with patch.object(wrapper, "_create_kb_client", return_value=mock_kb_client):
            result = await wrapper.retrieve_context(
                "What are the sales figures?",
                config.restricted_role_arn,
            )

        assert "First passage about sales data." in result
        assert "Second passage about revenue." in result

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_retrieval_empty_results(self, mock_boto_client):
        """retrieve_context returns empty string when no results found."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }

        mock_kb_client = MagicMock()
        mock_kb_client.retrieve.return_value = {"retrievalResults": []}

        mock_boto_client.side_effect = [mock_sts, mock_kb_client]

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with patch.object(wrapper, "_create_kb_client", return_value=mock_kb_client):
            result = await wrapper.retrieve_context(
                "Unknown query",
                config.restricted_role_arn,
            )

        assert result == ""

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_retrieval_client_error_raises_kb_error(self, mock_boto_client, caplog):
        """retrieve_context raises KnowledgeBaseError on ClientError."""
        from botocore.exceptions import ClientError

        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }

        mock_kb_client = MagicMock()
        mock_kb_client.retrieve.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "KB not found"}},
            "Retrieve",
        )

        mock_boto_client.side_effect = [mock_sts, mock_kb_client]

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with patch.object(wrapper, "_create_kb_client", return_value=mock_kb_client):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(KnowledgeBaseError) as exc_info:
                    await wrapper.retrieve_context(
                        "test query",
                        config.restricted_role_arn,
                    )

        assert exc_info.value.knowledge_base_id == "kb-test-123"
        assert exc_info.value.error_type == "ResourceNotFoundException"
        assert "kb-test-123" in caplog.text

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_retrieval_role_assumption_failure(self, mock_boto_client):
        """retrieve_context raises RoleAssumptionError when STS fails."""
        from botocore.exceptions import ClientError

        mock_sts = MagicMock()
        mock_sts.assume_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Not authorized"}},
            "AssumeRole",
        )

        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with pytest.raises(RoleAssumptionError) as exc_info:
            await wrapper.retrieve_context(
                "test query",
                config.restricted_role_arn,
            )

        assert exc_info.value.role_arn == config.restricted_role_arn


class TestInvoke:
    """Tests for the invoke method."""

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_invoke_role_assumption_failure(self, mock_boto_client):
        """invoke raises RoleAssumptionError when STS fails."""
        from botocore.exceptions import ClientError

        mock_sts = MagicMock()
        mock_sts.assume_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Not authorized"}},
            "AssumeRole",
        )

        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        with pytest.raises(RoleAssumptionError):
            await wrapper.invoke(
                prompt="Hello",
                role_arn=config.permissive_role_arn,
                apply_guardrails=False,
            )

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_invoke_with_guardrails(self, mock_boto_client):
        """invoke passes guardrail config when apply_guardrails=True."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        # Mock the Strands SDK imports
        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = "Test response"

        import agent_wrapper as aw_module

        mock_model_cls = MagicMock()
        with patch.object(aw_module, "Agent", mock_agent_instance, create=True), \
             patch.object(aw_module, "BedrockModel", mock_model_cls, create=True), \
             patch.object(aw_module, "STRANDS_AVAILABLE", True):
            result = await wrapper.invoke(
                prompt="Tell me about sales",
                role_arn=config.restricted_role_arn,
                apply_guardrails=True,
            )

        # Verify BedrockModel was called with guardrail config
        call_kwargs = mock_model_cls.call_args[1]
        assert "guardrail_config" in call_kwargs
        assert call_kwargs["guardrail_config"]["guardrailIdentifier"] == "gr-test-456"
        assert call_kwargs["guardrail_config"]["guardrailVersion"] == "1"

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_invoke_without_guardrails(self, mock_boto_client):
        """invoke does NOT pass guardrail config when apply_guardrails=False."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = "Unrestricted response"

        import agent_wrapper as aw_module

        mock_model_cls = MagicMock()
        with patch.object(aw_module, "Agent", mock_agent_instance, create=True), \
             patch.object(aw_module, "BedrockModel", mock_model_cls, create=True), \
             patch.object(aw_module, "STRANDS_AVAILABLE", True):
            result = await wrapper.invoke(
                prompt="Show me all data",
                role_arn=config.permissive_role_arn,
                apply_guardrails=False,
            )

        # Verify BedrockModel was called WITHOUT guardrail config
        call_kwargs = mock_model_cls.call_args[1]
        assert "guardrail_config" not in call_kwargs

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_invoke_streams_tokens_via_callback(self, mock_boto_client):
        """invoke calls on_token callback for each token in the response."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        # Set up mock so Agent(model=model) returns a callable that returns
        # an object whose str() is "Hello world response"
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "Hello world response"
        mock_agent_obj = MagicMock(return_value=mock_response)
        mock_agent_cls = MagicMock(return_value=mock_agent_obj)

        tokens_received = []

        async def on_token(token: str):
            tokens_received.append(token)

        import agent_wrapper as aw_module

        with patch.object(aw_module, "Agent", mock_agent_cls, create=True), \
             patch.object(aw_module, "BedrockModel", MagicMock(), create=True), \
             patch.object(aw_module, "STRANDS_AVAILABLE", True):
            result = await wrapper.invoke(
                prompt="Hello",
                role_arn=config.permissive_role_arn,
                apply_guardrails=False,
                on_token=on_token,
            )

        assert len(tokens_received) > 0
        assert result.content == "Hello world response"
        assert result.error is None

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_invoke_returns_error_on_agent_failure(self, mock_boto_client):
        """invoke returns AgentResponse with error when agent invocation fails."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        import agent_wrapper as aw_module

        mock_agent_cls = MagicMock(side_effect=RuntimeError("Model unavailable"))
        with patch.object(aw_module, "Agent", mock_agent_cls, create=True), \
             patch.object(aw_module, "BedrockModel", MagicMock(), create=True), \
             patch.object(aw_module, "STRANDS_AVAILABLE", True):
            result = await wrapper.invoke(
                prompt="Hello",
                role_arn=config.permissive_role_arn,
                apply_guardrails=False,
            )

        assert result.error is not None
        assert "RuntimeError" in result.error
        assert result.content == ""

    @patch("agent_wrapper.boto3.client")
    @pytest.mark.asyncio
    async def test_invoke_metadata_includes_model_and_role(self, mock_boto_client):
        """invoke response metadata includes model_id, role_arn, and guardrails_applied."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA...",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }
        mock_boto_client.return_value = mock_sts

        config = _make_config()
        wrapper = StrandsAgentWrapper(config)
        wrapper._sts_client = mock_sts

        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = "Response"

        import agent_wrapper as aw_module

        with patch.object(aw_module, "Agent", mock_agent_instance, create=True), \
             patch.object(aw_module, "BedrockModel", MagicMock(), create=True), \
             patch.object(aw_module, "STRANDS_AVAILABLE", True):
            result = await wrapper.invoke(
                prompt="Hello",
                role_arn=config.permissive_role_arn,
                apply_guardrails=False,
            )

        assert result.metadata["model_id"] == "anthropic.claude-3-sonnet"
        assert result.metadata["role_arn"] == config.permissive_role_arn
        assert result.metadata["guardrails_applied"] is False
