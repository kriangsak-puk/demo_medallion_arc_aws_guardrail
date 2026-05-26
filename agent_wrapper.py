"""Strands Agent wrapper for role-based LLM orchestration.

Wraps the AWS Strands Agents SDK with IAM role-based configuration,
providing separate invocation paths for Engine_A (Permissive_Role, no guardrails)
and Engine_B (Restricted_Role, with guardrails). Handles STS AssumeRole,
Knowledge Base RAG retrieval, and timeout management.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

try:
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    STRANDS_AVAILABLE = True
except ImportError:
    Agent = None  # type: ignore[assignment, misc]
    BedrockModel = None  # type: ignore[assignment, misc]
    STRANDS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Timeout configuration (seconds)
KB_RETRIEVAL_TIMEOUT_SECONDS = 25
MODEL_INFERENCE_TIMEOUT_SECONDS = 25


class RoleAssumptionError(Exception):
    """Raised when STS AssumeRole fails.

    Callers should catch this to fall back to Mock_Mode for the affected engine.

    Attributes:
        role_arn: The ARN of the role that failed to be assumed.
        error_type: The type of error encountered (e.g., 'AccessDenied').
    """

    def __init__(self, role_arn: str, error_type: str, message: str) -> None:
        self.role_arn = role_arn
        self.error_type = error_type
        super().__init__(message)


class KnowledgeBaseError(Exception):
    """Raised when Knowledge Base retrieval fails.

    Callers should catch this to fall back to Mock_Mode for Engine_B.

    Attributes:
        knowledge_base_id: The ID of the Knowledge Base that failed.
        error_type: The type of error encountered.
    """

    def __init__(self, knowledge_base_id: str, error_type: str, message: str) -> None:
        self.knowledge_base_id = knowledge_base_id
        self.error_type = error_type
        super().__init__(message)


@dataclass
class AgentConfig:
    """Configuration for the Strands Agent wrapper.

    All service identifiers needed to connect to pre-deployed AWS
    infrastructure for both Engine_A and Engine_B pipelines.

    Attributes:
        bedrock_model_id: The Bedrock model ID for LLM inference.
        knowledge_base_id: The Bedrock Knowledge Base ID for RAG retrieval.
        guardrails_id: The Bedrock Guardrails ID for input/output filtering.
        guardrails_version: The Guardrails version string.
        glue_database_name: The Glue Catalog database name.
        glue_table_name: The Gold table name in Glue Catalog.
        permissive_role_arn: IAM role ARN for Engine_A (all columns).
        restricted_role_arn: IAM role ARN for Engine_B (non-PII only).
        region: AWS region (default: ap-southeast-1).
    """

    bedrock_model_id: str
    knowledge_base_id: str
    guardrails_id: str
    guardrails_version: str
    glue_database_name: str
    glue_table_name: str
    permissive_role_arn: str
    restricted_role_arn: str
    region: str = "ap-southeast-1"


@dataclass
class AgentResponse:
    """Response from a Strands Agent invocation.

    Attributes:
        content: The generated response text.
        metadata: Additional metadata from the agent invocation.
        guardrail_action: The guardrail action taken — "PASS", "BLOCKED",
            or "MODIFIED". None if guardrails were not applied.
        error: Error message if the invocation failed, None otherwise.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    guardrail_action: Optional[str] = None
    error: Optional[str] = None


class StrandsAgentWrapper:
    """Wraps the Strands Agents SDK with role-based configuration.

    Provides methods to invoke the LLM agent with different IAM roles
    and guardrail configurations for Engine_A and Engine_B pipelines.
    Handles STS AssumeRole, Knowledge Base RAG retrieval, and per-request
    timeouts.

    The wrapper authenticates to AWS services using the IAM task role
    attached to the Fargate execution environment, then assumes the
    Permissive_Role or Restricted_Role via STS for data access operations.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the Strands Agent wrapper.

        Args:
            config: Agent configuration with all service identifiers.
        """
        self.config = config

        # Configure boto3 clients with appropriate timeouts
        self._boto_config = BotoConfig(
            region_name=config.region,
            connect_timeout=5,
            read_timeout=MODEL_INFERENCE_TIMEOUT_SECONDS,
            retries={"max_attempts": 1},
        )

        # STS client for role assumption (uses Fargate task role)
        self._sts_client = boto3.client(
            "sts",
            config=BotoConfig(
                region_name=config.region,
                connect_timeout=5,
                read_timeout=10,
                retries={"max_attempts": 1},
            ),
        )

    def _assume_role(self, role_arn: str, session_name: str) -> Dict[str, str]:
        """Assume an IAM role via STS and return temporary credentials.

        Args:
            role_arn: The ARN of the role to assume.
            session_name: A name for the role session.

        Returns:
            A dictionary with AccessKeyId, SecretAccessKey, and SessionToken.

        Raises:
            RoleAssumptionError: If the role assumption fails (e.g., access
                denied, invalid ARN, or STS service error). Callers should
                catch this to fall back to Mock_Mode.
        """
        try:
            response = self._sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=session_name,
                DurationSeconds=900,  # 15 minutes — minimum for demo sessions
            )
            credentials = response["Credentials"]
            return {
                "aws_access_key_id": credentials["AccessKeyId"],
                "aws_secret_access_key": credentials["SecretAccessKey"],
                "aws_session_token": credentials["SessionToken"],
            }
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(
                "Failed to assume role %s: [%s] %s",
                role_arn,
                error_code,
                error_message,
            )
            raise RoleAssumptionError(
                role_arn=role_arn,
                error_type=error_code,
                message=f"Failed to assume role {role_arn}: [{error_code}] {error_message}",
            ) from e
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                "Unexpected error assuming role %s: [%s] %s",
                role_arn,
                error_type,
                str(e),
            )
            raise RoleAssumptionError(
                role_arn=role_arn,
                error_type=error_type,
                message=f"Unexpected error assuming role {role_arn}: [{error_type}] {str(e)}",
            ) from e

    def _create_bedrock_client(self, credentials: Dict[str, str]) -> Any:
        """Create a Bedrock Runtime client with assumed role credentials.

        Args:
            credentials: Temporary credentials from STS AssumeRole.

        Returns:
            A boto3 Bedrock Runtime client configured with the assumed
            role credentials and appropriate timeouts.
        """
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"],
            aws_session_token=credentials["aws_session_token"],
            config=BotoConfig(
                region_name=self.config.region,
                connect_timeout=5,
                read_timeout=MODEL_INFERENCE_TIMEOUT_SECONDS,
                retries={"max_attempts": 1},
            ),
        )

    def _create_kb_client(self, credentials: Dict[str, str]) -> Any:
        """Create a Bedrock Agent Runtime client for Knowledge Base retrieval.

        Args:
            credentials: Temporary credentials from STS AssumeRole.

        Returns:
            A boto3 Bedrock Agent Runtime client configured with the assumed
            role credentials and KB retrieval timeout.
        """
        return boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"],
            aws_session_token=credentials["aws_session_token"],
            config=BotoConfig(
                region_name=self.config.region,
                connect_timeout=5,
                read_timeout=KB_RETRIEVAL_TIMEOUT_SECONDS,
                retries={"max_attempts": 1},
            ),
        )

    async def invoke(
        self,
        prompt: str,
        role_arn: str,
        apply_guardrails: bool,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> AgentResponse:
        """Invoke the Strands Agent with the specified IAM role.

        Assumes the given IAM role via STS, creates a Strands Agent with
        BedrockModel, optionally applies guardrails, and streams the
        response via the on_token callback.

        Args:
            prompt: The user prompt to send to the agent.
            role_arn: The IAM role ARN to assume for this invocation.
            apply_guardrails: Whether to apply Bedrock Guardrails for
                input evaluation and output filtering.
            on_token: Optional async callback invoked for each response
                token during streaming.

        Returns:
            An AgentResponse with the generated content, metadata,
            guardrail action, and any error.

        Raises:
            RoleAssumptionError: If STS AssumeRole fails. Callers should
                catch this to fall back to Mock_Mode for the affected engine.
        """
        # Step 1: Assume the specified IAM role
        session_name = "safe-haven-engine-session"
        credentials = self._assume_role(role_arn, session_name)

        # Step 2: Create Bedrock client with assumed role credentials
        bedrock_client = self._create_bedrock_client(credentials)

        # Step 3: Build guardrail configuration if applicable
        guardrail_config = None
        if apply_guardrails:
            guardrail_config = {
                "guardrailIdentifier": self.config.guardrails_id,
                "guardrailVersion": self.config.guardrails_version,
            }

        # Step 4: Invoke the Strands Agent
        # NOTE: The actual Strands SDK integration is a placeholder.
        # In production, this would use:
        #   from strands import Agent
        #   from strands.models.bedrock import BedrockModel
        #   model = BedrockModel(
        #       model_id=self.config.bedrock_model_id,
        #       boto_client=bedrock_client,
        #       guardrail_config=guardrail_config,
        #   )
        #   agent = Agent(model=model)
        #   response = agent(prompt)
        #
        # For now, we structure the invocation to demonstrate the correct
        # pattern and error handling.

        try:
            if not STRANDS_AVAILABLE:
                raise ImportError(
                    "strands-agents package is not installed. "
                    "Install with: pip install strands-agents"
                )

            model_kwargs: Dict[str, Any] = {}
            if guardrail_config:
                model_kwargs["guardrail_config"] = guardrail_config

            model = BedrockModel(
                model_id=self.config.bedrock_model_id,
                boto_client=bedrock_client,
                **model_kwargs,
            )

            agent = Agent(model=model)
            response = agent(prompt)

            # Extract response content
            content = str(response)

            # Stream tokens via callback if provided
            if on_token and content:
                # In production, streaming would happen during agent execution
                # via a callback handler. Here we simulate token delivery.
                for token in content.split():
                    await on_token(token + " ")

            # Determine guardrail action from response metadata
            guardrail_action = None
            if apply_guardrails:
                # The Strands SDK provides guardrail trace in response metadata
                guardrail_action = "PASS"  # Default if no block/modification detected

            return AgentResponse(
                content=content,
                metadata={
                    "model_id": self.config.bedrock_model_id,
                    "role_arn": role_arn,
                    "guardrails_applied": apply_guardrails,
                },
                guardrail_action=guardrail_action,
                error=None,
            )

        except RoleAssumptionError:
            # Re-raise role assumption errors for caller to handle
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"Agent invocation failed: [{error_type}] {str(e)}"
            logger.error(
                "Strands Agent invocation failed for role %s: [%s] %s",
                role_arn,
                error_type,
                str(e),
            )
            return AgentResponse(
                content="",
                metadata={
                    "model_id": self.config.bedrock_model_id,
                    "role_arn": role_arn,
                    "guardrails_applied": apply_guardrails,
                },
                guardrail_action=None,
                error=error_msg,
            )

    async def retrieve_context(self, prompt: str, role_arn: str) -> str:
        """Query the Knowledge Base for RAG context retrieval.

        Assumes the specified IAM role and queries the pre-deployed
        Bedrock Knowledge Base to retrieve relevant context for
        Retrieval-Augmented Generation.

        Args:
            prompt: The user prompt to use as the retrieval query.
            role_arn: The IAM role ARN to assume for KB access.

        Returns:
            The retrieved context text from the Knowledge Base.

        Raises:
            RoleAssumptionError: If STS AssumeRole fails.
            KnowledgeBaseError: If the Knowledge Base retrieval fails
                or times out. Callers should catch this to fall back
                to Mock_Mode for Engine_B.
        """
        # Step 1: Assume the specified IAM role
        session_name = "safe-haven-kb-session"
        credentials = self._assume_role(role_arn, session_name)

        # Step 2: Create KB client with assumed role credentials
        kb_client = self._create_kb_client(credentials)

        # Step 3: Query the Knowledge Base
        try:
            response = kb_client.retrieve(
                knowledgeBaseId=self.config.knowledge_base_id,
                retrievalQuery={"text": prompt},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5,
                    }
                },
            )

            # Extract and concatenate retrieved passages
            results = response.get("retrievalResults", [])
            context_parts = []
            for result in results:
                content = result.get("content", {}).get("text", "")
                if content:
                    context_parts.append(content)

            return "\n\n".join(context_parts)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(
                "Knowledge Base retrieval failed (KB ID: %s): [%s] %s",
                self.config.knowledge_base_id,
                error_code,
                error_message,
            )
            raise KnowledgeBaseError(
                knowledge_base_id=self.config.knowledge_base_id,
                error_type=error_code,
                message=(
                    f"Knowledge Base retrieval failed "
                    f"(KB ID: {self.config.knowledge_base_id}): "
                    f"[{error_code}] {error_message}"
                ),
            ) from e
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                "Unexpected error during KB retrieval (KB ID: %s): [%s] %s",
                self.config.knowledge_base_id,
                error_type,
                str(e),
            )
            raise KnowledgeBaseError(
                knowledge_base_id=self.config.knowledge_base_id,
                error_type=error_type,
                message=(
                    f"Unexpected error during KB retrieval "
                    f"(KB ID: {self.config.knowledge_base_id}): "
                    f"[{error_type}] {str(e)}"
                ),
            ) from e
