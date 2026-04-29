"""Compatibilidade com APIs legadas de monitorização de custo/tokens.

Este módulo não faz contabilidade detalhada automaticamente.
Ele oferece a interface esperada por notebooks/scripts antigos:
- `tracked_chat_openai(...)`
- `TRACKER.reset(...)`
- `TRACKER.print_summary()`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from graphrag.llm_factory import create_chat_llm


@dataclass
class TokenCostTracker:
    """Tracker mínimo para manter compatibilidade com código legado."""

    run_name: str = "default"
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    _events: list[dict[str, Any]] = field(default_factory=list)

    def reset(self, run_name: str = "default") -> None:
        self.run_name = run_name
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._events.clear()

    def add_event(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.total_cost_usd += float(cost_usd or 0.0)
        self._events.append(
            {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "cost_usd": float(cost_usd or 0.0),
                "metadata": metadata or {},
            }
        )

    def print_summary(self) -> None:
        print(f"[TRACKER] run={self.run_name}")
        print(
            "[TRACKER] calls={calls} prompt_tokens={pt} completion_tokens={ct} "
            "total_tokens={tt} total_cost_usd={cost:.6f}".format(
                calls=self.calls,
                pt=self.prompt_tokens,
                ct=self.completion_tokens,
                tt=self.total_tokens,
                cost=self.total_cost_usd,
            )
        )


TRACKER = TokenCostTracker()


def tracked_chat_openai(
    *,
    model: str,
    temperature: float = 0.0,
    api_key: str | None = None,
    **kwargs: Any,
) -> ChatOpenAI | AzureChatOpenAI:
    """Cria `ChatOpenAI` com assinatura legada esperada pelo projeto."""
    if api_key and "api_key" not in kwargs:
        kwargs["api_key"] = api_key
    return create_chat_llm(model=model, temperature=temperature, **kwargs)
