"""Entry point that runs inside the task container: python -m harbor_adapter.container_main

Env: RIDECO_APP_DIR (default /app), INSTRUCTION_PATH, PROMPT_VERSION, MAX_TURNS,
plus the LLM_* / OFFLINE / CACHE_DIR variables from llm.factory.
"""
import asyncio
import os
from pathlib import Path

from agent.prompts import DEFAULT_PROMPT_VERSION
from llm.factory import build_chat_model
from runner.rollout import run_rollout
from runner.task_io import parse_opening_message
from sim.user_sim import DEFAULT_MAX_TURNS
from tools.mock_tools import APP_DIR_ENV, DEFAULT_APP_DIR


AGENT_ROLES = ("agent", "fixed_agent")


def agent_role_from_env() -> str:
    role = os.environ.get("AGENT_ROLE", "agent")
    if role not in AGENT_ROLES:
        raise ValueError(f"AGENT_ROLE must be one of {AGENT_ROLES}, got {role!r}")
    return role


def main() -> None:
    instruction = Path(os.environ["INSTRUCTION_PATH"]).read_text()
    transcript = asyncio.run(run_rollout(
        app_dir=os.environ.get(APP_DIR_ENV, DEFAULT_APP_DIR),
        opening_message=parse_opening_message(instruction),
        agent_llm=build_chat_model(agent_role_from_env()),
        prompt_version=os.environ.get("PROMPT_VERSION", DEFAULT_PROMPT_VERSION),
        max_turns=int(os.environ.get("MAX_TURNS", DEFAULT_MAX_TURNS))))
    print(f"conversation finished: {len(transcript)} messages")


if __name__ == "__main__":
    main()
