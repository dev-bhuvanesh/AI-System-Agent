"""Local language-model provider boundary for System Agent."""

from config.config import AgentConfig
from llm.ollama_provider import OllamaProvider
from llm.provider import (
    ChatMessage,
    ChatProvider,
    DiagnosticDecision,
    LLMProvider,
    Plan,
    PlanAction,
    ProviderEvent,
)


def create_chat_provider(config: AgentConfig) -> ChatProvider:
    """Create the configured conversational backend."""
    if config.llm_backend.strip().lower() in {"ollama", "ollama-local"}:
        return OllamaProvider(config)
    raise ValueError(f"Unsupported chat backend: {config.llm_backend}")


__all__ = [
    "LLMProvider",
    "ChatMessage",
    "ChatProvider",
    "DiagnosticDecision",
    "OllamaProvider",
    "Plan",
    "PlanAction",
    "ProviderEvent",
    "create_chat_provider",
]
