from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Determine state, data and base directories
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Load provider parameters
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Sensible defaults for models
    default_models = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-sonnet-20240620",
        "ollama": "llama3",
        "openrouter": "meta-llama/llama-3-8b-instruct:free",
        "custom": "custom-model",
    }
    
    model_name = os.getenv("LLM_MODEL", default_models.get(provider, "gpt-4o-mini"))
    temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    # Retrieve api key based on provider
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif provider == "custom":
            api_key = os.getenv("CUSTOM_API_KEY")

    # Retrieve base URL
    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif provider == "custom":
            base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temp,
        api_key=api_key,
        base_url=base_url,
    )

    # Load judge parameters
    judge_provider = os.getenv("JUDGE_PROVIDER", provider).lower()
    judge_model_name = os.getenv("JUDGE_MODEL", default_models.get(judge_provider, model_name))
    judge_temp = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    
    judge_api_key = os.getenv("JUDGE_API_KEY")
    if not judge_api_key:
        if judge_provider == "openai":
            judge_api_key = os.getenv("OPENAI_API_KEY")
        elif judge_provider == "gemini":
            judge_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif judge_provider == "anthropic":
            judge_api_key = os.getenv("ANTHROPIC_API_KEY")
        elif judge_provider == "openrouter":
            judge_api_key = os.getenv("OPENROUTER_API_KEY")
        elif judge_provider == "custom":
            judge_api_key = os.getenv("CUSTOM_API_KEY")

    judge_base_url = os.getenv("JUDGE_BASE_URL")
    if not judge_base_url:
        if judge_provider == "ollama":
            judge_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif judge_provider == "custom":
            judge_base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        elif judge_provider == "openrouter":
            judge_base_url = "https://openrouter.ai/api/v1"

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temp,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    # Compact threshold configurations
    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000"))
    keep_msgs = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep_msgs,
        model=model_config,
        judge_model=judge_config,
    )
