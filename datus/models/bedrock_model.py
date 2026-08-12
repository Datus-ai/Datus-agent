# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Amazon Bedrock model provider using the unified Converse API."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class BedrockModel(OpenAICompatibleModel):
    """Route Datus LLM calls through Amazon Bedrock Runtime Converse.

    Authentication is delegated to the standard AWS credential chain. Static
    access keys are intentionally not accepted as model configuration; local
    users can select an AWS profile, while deployed workloads should use an
    IAM role (for example EKS Pod Identity or IRSA).
    """

    _ALLOWED_PROVIDER_OPTIONS = frozenset(
        {
            "aws_region_name",
            "aws_profile_name",
            "aws_role_name",
            "aws_session_name",
            "aws_bedrock_runtime_endpoint",
        }
    )

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)
        logger.debug("Using Amazon Bedrock Converse model: %s", self.litellm_adapter.litellm_model_name)

    def _get_api_key(self) -> str:
        """Bedrock Runtime uses AWS credentials rather than an LLM API key."""
        return ""

    def _get_base_url(self) -> Optional[str]:
        """Let boto3 resolve the regional Bedrock Runtime endpoint."""
        return None

    def _default_sampling_params(self) -> Dict[str, float]:
        """Do not inject cross-vendor sampling defaults into Bedrock calls."""
        return {}

    def _json_response_format(self) -> Optional[Dict[str, str]]:
        """Use prompt-based JSON output across the heterogeneous Bedrock catalog.

        Bedrock Converse has no uniform JSON-mode control. In particular,
        forwarding OpenAI's ``response_format`` through LiteLLM causes Nova 2
        Lite to return an empty object. The shared parser still accepts raw or
        fenced JSON from the model response.
        """
        return None

    def _get_provider_request_options(self) -> Dict[str, Any]:
        raw_options = dict(self.model_config.provider_options or {})
        unknown = sorted(set(raw_options) - self._ALLOWED_PROVIDER_OPTIONS)
        if unknown:
            raise ValueError(
                "Unsupported Bedrock provider_options: "
                f"{', '.join(unknown)}. Allowed: {', '.join(sorted(self._ALLOWED_PROVIDER_OPTIONS))}"
            )

        options = {key: value for key, value in raw_options.items() if value not in (None, "")}
        if "aws_region_name" not in options:
            region = os.getenv("AWS_REGION_NAME") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            if not region:
                try:
                    import boto3

                    profile = options.get("aws_profile_name") or os.getenv("AWS_PROFILE")
                    region = boto3.Session(profile_name=str(profile) if profile else None).region_name
                except Exception as exc:
                    logger.debug("Unable to resolve AWS region from boto3 session: %s", exc)
            if region:
                options["aws_region_name"] = region
        return options
