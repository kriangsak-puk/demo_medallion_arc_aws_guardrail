"""Unit tests for the configuration module."""

import os
from unittest.mock import patch

import pytest

from aws_demo_booth.config import (
    AppConfig,
    AppMode,
    ModeStatus,
    REQUIRED_ENV_VARS,
)


@pytest.fixture
def full_env_vars():
    """Complete set of required environment variables."""
    return {
        "KNOWLEDGE_BASE_ID": "kb-12345",
        "GUARDRAILS_ID": "gr-67890",
        "GUARDRAILS_VERSION": "1",
        "GLUE_DATABASE_NAME": "demo_db",
        "GLUE_TABLE_NAME": "gold_table",
        "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
        "PERMISSIVE_ROLE_ARN": "arn:aws:iam::123456789012:role/PermissiveRole",
        "RESTRICTED_ROLE_ARN": "arn:aws:iam::123456789012:role/RestrictedRole",
    }


class TestAppMode:
    """Tests for AppMode enum."""

    def test_live_mode_value(self):
        assert AppMode.LIVE.value == "live"

    def test_mock_mode_value(self):
        assert AppMode.MOCK.value == "mock"

    def test_enum_members(self):
        assert set(AppMode) == {AppMode.LIVE, AppMode.MOCK}


class TestModeStatus:
    """Tests for ModeStatus dataclass."""

    def test_create_live_status(self):
        status = ModeStatus(
            mode=AppMode.LIVE,
            engine_a_live=True,
            engine_b_live=True,
        )
        assert status.mode == AppMode.LIVE
        assert status.engine_a_live is True
        assert status.engine_b_live is True
        assert status.failure_reasons == []

    def test_create_mock_status_with_failures(self):
        reasons = ["STS AssumeRole failed", "Knowledge Base unreachable"]
        status = ModeStatus(
            mode=AppMode.MOCK,
            engine_a_live=False,
            engine_b_live=False,
            failure_reasons=reasons,
        )
        assert status.mode == AppMode.MOCK
        assert status.engine_a_live is False
        assert status.engine_b_live is False
        assert status.failure_reasons == reasons

    def test_default_failure_reasons_is_empty_list(self):
        status = ModeStatus(mode=AppMode.LIVE, engine_a_live=True, engine_b_live=True)
        assert status.failure_reasons == []

    def test_failure_reasons_not_shared_between_instances(self):
        status1 = ModeStatus(mode=AppMode.MOCK, engine_a_live=False, engine_b_live=False)
        status2 = ModeStatus(mode=AppMode.MOCK, engine_a_live=False, engine_b_live=False)
        status1.failure_reasons.append("error")
        assert status2.failure_reasons == []


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_create_config_with_all_fields(self):
        config = AppConfig(
            knowledge_base_id="kb-123",
            guardrails_id="gr-456",
            guardrails_version="1",
            glue_database_name="mydb",
            glue_table_name="mytable",
            bedrock_model_id="model-id",
            permissive_role_arn="arn:aws:iam::111:role/Perm",
            restricted_role_arn="arn:aws:iam::111:role/Restr",
        )
        assert config.knowledge_base_id == "kb-123"
        assert config.aws_region == "ap-southeast-1"
        assert config.mock_mode_forced is False

    def test_default_region(self):
        config = AppConfig(
            knowledge_base_id="",
            guardrails_id="",
            guardrails_version="",
            glue_database_name="",
            glue_table_name="",
            bedrock_model_id="",
            permissive_role_arn="",
            restricted_role_arn="",
        )
        assert config.aws_region == "ap-southeast-1"


class TestFromEnvironment:
    """Tests for AppConfig.from_environment() class method."""

    def test_all_vars_present_returns_empty_missing_list(self, full_env_vars):
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert missing == []
        assert config.knowledge_base_id == "kb-12345"
        assert config.guardrails_id == "gr-67890"
        assert config.guardrails_version == "1"
        assert config.glue_database_name == "demo_db"
        assert config.glue_table_name == "gold_table"
        assert config.bedrock_model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert config.permissive_role_arn == "arn:aws:iam::123456789012:role/PermissiveRole"
        assert config.restricted_role_arn == "arn:aws:iam::123456789012:role/RestrictedRole"

    def test_missing_single_var_returns_it_in_list(self, full_env_vars):
        del full_env_vars["KNOWLEDGE_BASE_ID"]
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert "KNOWLEDGE_BASE_ID" in missing
        assert len(missing) == 1
        assert config.knowledge_base_id == ""

    def test_missing_multiple_vars_returns_all_in_list(self):
        partial_env = {
            "KNOWLEDGE_BASE_ID": "kb-123",
            "GUARDRAILS_ID": "gr-456",
        }
        with patch.dict(os.environ, partial_env, clear=True):
            config, missing = AppConfig.from_environment()
        expected_missing = {
            "GUARDRAILS_VERSION",
            "GLUE_DATABASE_NAME",
            "GLUE_TABLE_NAME",
            "BEDROCK_MODEL_ID",
            "PERMISSIVE_ROLE_ARN",
            "RESTRICTED_ROLE_ARN",
        }
        assert set(missing) == expected_missing

    def test_no_vars_present_returns_all_required_as_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            config, missing = AppConfig.from_environment()
        assert set(missing) == set(REQUIRED_ENV_VARS)

    def test_custom_region_from_env(self, full_env_vars):
        full_env_vars["AWS_DEFAULT_REGION"] = "us-east-1"
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert config.aws_region == "us-east-1"
        assert missing == []

    def test_default_region_when_not_set(self, full_env_vars):
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert config.aws_region == "ap-southeast-1"

    def test_mock_mode_forced_true(self, full_env_vars):
        full_env_vars["MOCK_MODE"] = "true"
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert config.mock_mode_forced is True
        assert missing == []

    def test_mock_mode_forced_true_case_insensitive(self, full_env_vars):
        full_env_vars["MOCK_MODE"] = "True"
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert config.mock_mode_forced is True

    def test_mock_mode_not_forced_when_other_value(self, full_env_vars):
        full_env_vars["MOCK_MODE"] = "false"
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert config.mock_mode_forced is False

    def test_mock_mode_not_forced_when_absent(self, full_env_vars):
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert config.mock_mode_forced is False

    def test_empty_string_var_treated_as_missing(self, full_env_vars):
        full_env_vars["BEDROCK_MODEL_ID"] = ""
        with patch.dict(os.environ, full_env_vars, clear=True):
            config, missing = AppConfig.from_environment()
        assert "BEDROCK_MODEL_ID" in missing

    def test_logs_error_when_vars_missing(self, full_env_vars, caplog):
        del full_env_vars["GUARDRAILS_ID"]
        del full_env_vars["GLUE_TABLE_NAME"]
        with patch.dict(os.environ, full_env_vars, clear=True):
            import logging
            with caplog.at_level(logging.ERROR):
                config, missing = AppConfig.from_environment()
        assert "GUARDRAILS_ID" in caplog.text
        assert "GLUE_TABLE_NAME" in caplog.text

    def test_no_error_logged_when_all_present(self, full_env_vars, caplog):
        with patch.dict(os.environ, full_env_vars, clear=True):
            import logging
            with caplog.at_level(logging.ERROR):
                config, missing = AppConfig.from_environment()
        assert caplog.text == ""
