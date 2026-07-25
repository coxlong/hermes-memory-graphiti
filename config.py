"""Configuration schema and helpers for the Graphiti memory provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 6379
_DEFAULT_USERNAME = ""
_DEFAULT_DATABASE = "default_db"
_DEFAULT_MEMORY_MODE = "hybrid"
_DEFAULT_MAX_INPUT_CHARS = 800

_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "ollama": "gemma3:12b",
    "openai_compatible": "local-model",
}


def _load_config() -> dict:
    """Load config from $HERMES_HOME/graphiti/config.json, fall back to env vars."""
    config_path = get_hermes_home() / "graphiti" / "config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "falkordb_host": os.environ.get("GRAPHITI_FALKORDB_HOST", _DEFAULT_HOST),
        "falkordb_port": int(os.environ.get("GRAPHITI_FALKORDB_PORT", str(_DEFAULT_PORT))),
        "falkordb_username": os.environ.get("GRAPHITI_FALKORDB_USERNAME", _DEFAULT_USERNAME),
        "falkordb_database": os.environ.get("GRAPHITI_FALKORDB_DATABASE", _DEFAULT_DATABASE),
        "openai_api_key": os.environ.get("GRAPHITI_OPENAI_API_KEY", ""),
        "llm_provider": os.environ.get("GRAPHITI_LLM_PROVIDER", "openai"),
        "llm_model": os.environ.get("GRAPHITI_LLM_MODEL", ""),
        "llm_base_url": os.environ.get("GRAPHITI_LLM_BASE_URL", ""),
        "embedding_model": os.environ.get("GRAPHITI_EMBEDDING_MODEL", ""),
        "memory_mode": os.environ.get("GRAPHITI_MEMORY_MODE", _DEFAULT_MEMORY_MODE),
        "auto_retain": os.environ.get("GRAPHITI_AUTO_RETAIN", "true").lower() != "false",
        "retain_every_n_turns": int(os.environ.get("GRAPHITI_RETAIN_EVERY_N_TURNS", "10")),
        "retain_min_interval_seconds": int(
            os.environ.get("GRAPHITI_RETAIN_MIN_INTERVAL_SECONDS", "300")
        ),
        "extraction_language_instruction": os.environ.get(
            "GRAPHITI_EXTRACTION_LANGUAGE_INSTRUCTION", ""
        ),
        "recall_max_tokens": int(os.environ.get("GRAPHITI_RECALL_MAX_TOKENS", "4096")),
    }


def get_config_schema() -> list[dict[str, Any]]:
    """Return config fields for hermes memory setup."""
    return [
        # --- Database ---
        {
            "key": "falkordb_host",
            "description": "FalkorDB host",
            "default": _DEFAULT_HOST,
        },
        {
            "key": "falkordb_port",
            "description": "FalkorDB port",
            "default": _DEFAULT_PORT,
        },
        {
            "key": "falkordb_username",
            "description": "FalkorDB username",
            "default": _DEFAULT_USERNAME,
        },
        {
            "key": "falkordb_password",
            "description": "FalkorDB password",
            "secret": True,
            "env_var": "GRAPHITI_FALKORDB_PASSWORD",
        },
        {
            "key": "falkordb_database",
            "description": "FalkorDB database name",
            "default": _DEFAULT_DATABASE,
        },
        # --- LLM (Graphiti uses this for entity/edge extraction) ---
        {
            "key": "openai_api_key",
            "description": "OpenAI API key for entity/edge extraction and embeddings",
            "secret": True,
            "env_var": "GRAPHITI_OPENAI_API_KEY",
            "url": "https://platform.openai.com/api-keys",
        },
        {
            "key": "llm_provider",
            "description": "LLM provider for entity extraction",
            "default": "openai",
            "choices": list(_PROVIDER_DEFAULT_MODELS.keys()),
        },
        {
            "key": "llm_model",
            "description": "LLM model for entity extraction (blank = provider default)",
            "default": "",
        },
        {
            "key": "llm_base_url",
            "description": "Custom LLM endpoint URL (for openai_compatible / openrouter)",
            "default": "",
        },
        {
            "key": "embedding_model",
            "description": "Embedding model name (blank = provider default)",
            "default": "",
        },
        # --- Memory behaviour ---
        {
            "key": "memory_mode",
            "description": "How Graphiti integrates with the agent",
            "default": _DEFAULT_MEMORY_MODE,
            "choices": ["context", "tools", "hybrid"],
        },
        {
            "key": "auto_retain",
            "description": "Automatically retain conversation turns as episodes",
            "default": True,
        },
        {
            "key": "retain_every_n_turns",
            "description": "Flush pending turns after this many (10) — whichever comes first with the debounce timer below",
            "default": 10,
        },
        {
            "key": "retain_min_interval_seconds",
            "description": "Flush pending turns after this many seconds of silence (default: 300 = 5 min)",
            "default": 300,
        },
        {
            "key": "extraction_language_instruction",
            "description": "Custom extraction language instruction appended to entity extraction prompts. Leave blank to use graphiti-core default.",
            "default": "",
        },
    ]


def save_config(values: dict[str, Any], hermes_home: str) -> None:
    """Write non-secret config to $HERMES_HOME/graphiti/config.json."""
    from utils import atomic_json_write

    config_dir = Path(hermes_home) / "graphiti"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"

    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(values)
    atomic_json_write(config_path, existing, mode=0o600)
