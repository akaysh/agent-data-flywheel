"""Builds chat models from environment variables.

Any OpenAI-compatible endpoint works: a LiteLLM proxy with a virtual key,
Groq, Gemini's OpenAI-compatible API, a local Ollama, etc.

  LLM_BASE_URL   endpoint, e.g. https://api.groq.com/openai/v1
  LLM_API_KEY    key for that endpoint
  AGENT_MODEL        small model the shipped support agent runs on
  FIXED_AGENT_MODEL  stronger model for the "swap the model" fix
  SIM_MODEL          user simulator (defaults to AGENT_MODEL)
  JUDGE_MODEL        stronger model for the judge
  LABELER_MODEL      strongest model, used to label traces for the kappa lab
  LLM_TEMPERATURE  default 0; "none" omits it (reasoning models reject temperature=0)
  OFFLINE        "1" replays cache/ only, never calls out
  CACHE_DIR      defaults to <repo>/cache
"""
import json
import os
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from llm.cache import DiskLLMCache

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"
MODELS_FILE = "_models.json"  # role -> model recorded with the cache; the model name is in every key
OFFLINE_PLACEHOLDER_KEY = "offline-no-key-needed"

Role = Literal["agent", "fixed_agent", "simulator", "judge", "labeler"]
ROLE_MODEL_ENV = {"agent": "AGENT_MODEL", "fixed_agent": "FIXED_AGENT_MODEL",
                  "simulator": "SIM_MODEL", "judge": "JUDGE_MODEL", "labeler": "LABELER_MODEL"}
ROLE_FALLBACK_ENV = {"simulator": "AGENT_MODEL"}


PROVIDER_BY_MODEL_KEYWORD = {"claude": "anthropic", "gemini": "google", "llama": "meta",
                             "gpt": "openai", "gemma": "google", "mistral": "mistral"}


def provider_for_model(model_name: str) -> str:
    lowered = model_name.lower()
    for keyword, provider in PROVIDER_BY_MODEL_KEYWORD.items():
        if keyword in lowered:
            return provider
    return "openai"


class GatewayChatModel(ChatOpenAI):
    """ChatOpenAI speaking to an OpenAI-compatible gateway.

    The protocol is OpenAI's but the model usually isn't, so traces report the model's
    real provider instead of always saying "openai". Does not affect the cache key.
    """

    def _get_ls_params(self, stop=None, **kwargs):
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = provider_for_model(params["ls_model_name"])
        return params


def is_offline() -> bool:
    return os.environ.get("OFFLINE", "0") == "1"


def build_chat_model(role: Role) -> BaseChatModel:
    cache_dir = Path(os.environ.get("CACHE_DIR", DEFAULT_CACHE_DIR))
    cache = DiskLLMCache(cache_dir, offline=is_offline())
    return GatewayChatModel(
        model=_model_name(role, cache_dir),
        base_url=os.environ.get("LLM_BASE_URL"),
        api_key=os.environ.get("LLM_API_KEY") or OFFLINE_PLACEHOLDER_KEY,
        temperature=_temperature(),
        cache=cache,
    )


def _temperature() -> float | None:
    raw = os.environ.get("LLM_TEMPERATURE", "0").strip().lower()
    return None if raw in ("none", "") else float(raw)


def _model_name(role: Role, cache_dir: Path) -> str:
    env_name = ROLE_MODEL_ENV[role]
    name = os.environ.get(env_name) or os.environ.get(ROLE_FALLBACK_ENV.get(role, ""))
    if name:
        if not is_offline():
            _record_role_model(cache_dir, role, name)
        return name
    if is_offline():
        return _recorded_models(cache_dir).get(role, f"offline-{role}")
    raise RuntimeError(f"Set {env_name} (and LLM_BASE_URL / LLM_API_KEY) or run with OFFLINE=1")


def _recorded_models(cache_dir: Path) -> dict[str, str]:
    path = cache_dir / MODELS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def _record_role_model(cache_dir: Path, role: Role, name: str) -> None:
    models = _recorded_models(cache_dir)
    if models.get(role) != name:
        (cache_dir / MODELS_FILE).write_text(json.dumps({**models, role: name}, indent=1))
