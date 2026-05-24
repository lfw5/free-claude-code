"""Generic custom provider — works with any OpenAI-compatible or Anthropic-compatible endpoint."""

from __future__ import annotations

from typing import Any

import httpx

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)
from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.openai_compat import OpenAIChatTransport


def create_custom_provider(config: ProviderConfig, transport: str) -> BaseProvider:
    """Create a custom provider instance for the requested transport type."""
    if transport == "anthropic_messages":
        return _CustomAnthropicProvider(config)
    return _CustomOpenAIProvider(config)


class _CustomOpenAIProvider(OpenAIChatTransport):
    """Generic OpenAI-compatible chat completions provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CUSTOM",
            base_url=config.base_url or "",
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking = self._is_thinking_enabled(request, thinking_enabled)
        try:
            body = build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
                if effective_thinking
                else ReasoningReplayMode.DISABLED,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

        request_extra = getattr(request, "extra_body", None)
        if isinstance(request_extra, dict) and request_extra:
            merged = dict(request_extra)
            body["extra_body"] = merged

        return body


class _CustomAnthropicProvider(AnthropicMessagesTransport):
    """Generic Anthropic Messages-compatible provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CUSTOM",
            default_base_url="",
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking = self._is_thinking_enabled(request, thinking_enabled)
        return build_base_native_anthropic_request_body(
            request,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            thinking_enabled=effective_thinking,
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _send_model_list_request(self) -> httpx.Response:
        """Query the provider endpoint that advertises available model ids."""
        return await self._client.get(
            "/models",
            headers=self._model_list_headers(),
        )
