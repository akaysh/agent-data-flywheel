"""Disk cache for LLM responses, keyed by a hash of the request.

Record once with a real provider (OFFLINE=0), commit cache/, and anyone
without a key or wifi replays the same responses (OFFLINE=1). In offline
mode a request that was never recorded fails loudly instead of silently
calling out.
"""
import hashlib
import json
import os
from pathlib import Path

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation


ALLOWED_CLASSES = [
    Generation, ChatGeneration, ChatGenerationChunk, AIMessage, AIMessageChunk]


ENDPOINT_PARAMS = ("openai_api_base", "stream_usage")  # where the model runs, not what it says
HIT_LOG_ENV = "CACHE_HIT_LOG"  # set to a file path to record which entries a run used


class CacheMissError(RuntimeError):
    pass


class DiskLLMCache(BaseCache):
    def __init__(self, cache_dir: str | Path, offline: bool):
        self._dir = Path(cache_dir)
        self._offline = offline
        self._dir.mkdir(parents=True, exist_ok=True)

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        path = self._path(prompt, llm_string)
        if path.exists():
            self._record_hit(path)
            generations = json.loads(path.read_text())["generations"]
            return load(generations, allowed_objects=ALLOWED_CLASSES)
        if self._offline:
            raise CacheMissError(
                f"OFFLINE=1 but no recorded response for request {path.stem[:12]}. "
                "Re-record with OFFLINE=0 and a provider key.")
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        entry = {"generations": dumpd(list(return_val))}
        self._path(prompt, llm_string).write_text(json.dumps(entry, indent=1))

    def clear(self, **kwargs) -> None:
        for p in self._dir.glob("*.json"):
            p.unlink()

    @staticmethod
    def _record_hit(path: Path) -> None:
        log = os.environ.get(HIT_LOG_ENV)
        if log:
            with open(log, "a") as f:
                f.write(path.name + "\n")

    def _path(self, prompt: str, llm_string: str) -> Path:
        key = f"{normalize_llm_string(llm_string)}\x00{normalize_prompt(prompt)}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / f"{digest}.json"


def prune_unused(cache_dir: str | Path, hit_log: str | Path) -> int:
    """Deletes cache entries a recorded run never used. Returns how many were removed."""
    used = set(Path(hit_log).read_text().split())
    stale = [p for p in Path(cache_dir).glob("*.json")
             if p.name not in used and not p.name.startswith("_")]
    for p in stale:
        p.unlink()
    return len(stale)


# Our ChatOpenAI subclass must key like the plain class so existing recordings still replay.
SERIALIZED_CLASS_ALIASES = {"GatewayChatModel": "ChatOpenAI"}


def normalize_llm_string(llm_string: str) -> str:
    """Drops endpoint settings so a response recorded against a gateway replays without one."""
    head, sep, tail = llm_string.partition("---")
    try:
        config = json.loads(head)
    except json.JSONDecodeError:
        return llm_string
    for param in ENDPOINT_PARAMS:
        config.get("kwargs", {}).pop(param, None)
    config["id"] = [SERIALIZED_CLASS_ALIASES.get(part, part) for part in config.get("id", [])]
    config["name"] = SERIALIZED_CLASS_ALIASES.get(config.get("name"), config.get("name"))
    return json.dumps(config, sort_keys=True) + sep + tail


MESSAGE_FIELDS = ("content", "tool_call_id", "name")


def normalize_prompt(prompt: str) -> str:
    """Keys on what a message says, not on metadata that differs between a fresh and a cached copy."""
    try:
        messages = json.loads(prompt)
        canonical = [_canonical_message(m) for m in messages]
    except (json.JSONDecodeError, TypeError, KeyError):
        return prompt
    return json.dumps(canonical, sort_keys=True)


def _canonical_message(message: dict) -> dict:
    kwargs = message["kwargs"]
    canonical = {"type": message["id"][-1],
                 **{f: kwargs[f] for f in MESSAGE_FIELDS if kwargs.get(f)}}
    if kwargs.get("tool_calls"):
        canonical["tool_calls"] = [
            {"name": c["name"], "args": c["args"], "id": c.get("id")} for c in kwargs["tool_calls"]]
    return canonical
