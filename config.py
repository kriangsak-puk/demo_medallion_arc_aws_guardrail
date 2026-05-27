"""Application configuration module.

Loads AWS service identifiers from environment variables at startup.
Switches to Mock_Mode when required variables are missing.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Required environment variables that must be present for Live_Mode
REQUIRED_ENV_VARS: List[str] = [
    "KNOWLEDGE_BASE_ID",
    "GUARDRAILS_ID",
    "GUARDRAILS_VERSION",
    "GLUE_DATABASE_NAME",
    "GLUE_TABLE_NAME",
    "BEDROCK_MODEL_ID",
    "PERMISSIVE_ROLE_ARN",
    "RESTRICTED_ROLE_ARN",
]


class AppMode(Enum):
    """Application operating mode."""

    LIVE = "live"
    MOCK = "mock"


@dataclass
class ModeStatus:
    """Tracks the current operating mode and per-engine live status."""

    mode: AppMode
    engine_a_live: bool
    engine_b_live: bool
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class AppConfig:
    """Application configuration loaded from environment variables.

    All AWS service identifiers are read from env vars at startup.
    The application does NOT provision or modify any AWS infrastructure;
    it only connects to pre-existing resources identified by these variables.
    """

    knowledge_base_id: str
    knowledge_base_id_permissive: str  # Engine A (bad) — raw data with PII
    knowledge_base_id_restricted: str  # Engine B (good) — masked data
    guardrails_id: str
    guardrails_version: str
    glue_database_name: str
    glue_table_name: str
    bedrock_model_id: str
    permissive_role_arn: str
    restricted_role_arn: str
    aws_region: str = "ap-southeast-1"
    mock_mode_forced: bool = False

    @classmethod
    def from_environment(cls) -> Tuple["AppConfig", List[str]]:
        """Load configuration from environment variables.

        Returns:
            A tuple of (AppConfig, missing_vars) where missing_vars is a list
            of required environment variable names that were not found.
            When variables are missing, the config is populated with empty
            strings for those fields and an error is logged listing each
            missing variable by name.
        """
        missing_vars: List[str] = []

        for var_name in REQUIRED_ENV_VARS:
            if not os.environ.get(var_name):
                missing_vars.append(var_name)

        if missing_vars:
            logger.error(
                "Missing required environment variables: %s",
                ", ".join(missing_vars),
            )

        # Check if mock mode is explicitly forced
        mock_mode_forced = os.environ.get("MOCK_MODE", "").lower() == "true"

        config = cls(
            knowledge_base_id=os.environ.get("KNOWLEDGE_BASE_ID", ""),
            knowledge_base_id_permissive=os.environ.get("KNOWLEDGE_BASE_ID_PERMISSIVE", os.environ.get("KNOWLEDGE_BASE_ID", "")),
            knowledge_base_id_restricted=os.environ.get("KNOWLEDGE_BASE_ID_RESTRICTED", os.environ.get("KNOWLEDGE_BASE_ID", "")),
            guardrails_id=os.environ.get("GUARDRAILS_ID", ""),
            guardrails_version=os.environ.get("GUARDRAILS_VERSION", ""),
            glue_database_name=os.environ.get("GLUE_DATABASE_NAME", ""),
            glue_table_name=os.environ.get("GLUE_TABLE_NAME", ""),
            bedrock_model_id=os.environ.get("BEDROCK_MODEL_ID", ""),
            permissive_role_arn=os.environ.get("PERMISSIVE_ROLE_ARN", ""),
            restricted_role_arn=os.environ.get("RESTRICTED_ROLE_ARN", ""),
            aws_region=os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1"),
            mock_mode_forced=mock_mode_forced,
        )

        return config, missing_vars
