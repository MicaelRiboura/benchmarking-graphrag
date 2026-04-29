"""Centralized chat LLM factory for OpenAI or Azure OpenAI."""

from __future__ import annotations

from typing import Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from graphrag.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_ORGANIZATION,
    OPENAI_PROJECT,
    GRAPHRAG_LLM_PROVIDER,
)


def create_chat_llm(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ChatOpenAI | AzureChatOpenAI:
    """Create the project chat model using env-driven provider config."""
    selected_model = (model or LLM_MODEL).strip()
    provider = (GRAPHRAG_LLM_PROVIDER or "openai").strip().lower()

    if provider == "azure":
        azure_api_key = (AZURE_OPENAI_API_KEY or OPENAI_API_KEY or "").strip()
        if not AZURE_OPENAI_ENDPOINT.strip():
            raise RuntimeError(
                "GRAPHRAG_LLM_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT."
            )
        if not azure_api_key:
            raise RuntimeError(
                "GRAPHRAG_LLM_PROVIDER=azure requires AZURE_OPENAI_API_KEY or OPENAI_API_KEY."
            )
        return AzureChatOpenAI(
            model=selected_model,
            temperature=temperature,
            api_key=azure_api_key,
            azure_endpoint=AZURE_OPENAI_ENDPOINT.strip(),
            api_version=AZURE_OPENAI_API_VERSION.strip(),
            **kwargs,
        )

    openai_api_key = (OPENAI_API_KEY or "").strip()
    if not openai_api_key:
        raise RuntimeError(
            "GRAPHRAG_LLM_PROVIDER=openai requires OPENAI_API_KEY."
        )
    return ChatOpenAI(
        model=selected_model,
        temperature=temperature,
        api_key=openai_api_key,
        base_url=(OPENAI_BASE_URL or "").strip() or None,
        organization=(OPENAI_ORGANIZATION or "").strip() or None,
        project=(OPENAI_PROJECT or "").strip() or None,
        **kwargs,
    )
