"""One conversation: agent + simulated customer + mock tools, inside app_dir.

Shared by the Colab runner (temp dir) and the Harbor adapter (/app).
Leaves tool_calls.jsonl and transcript.jsonl in app_dir for the verifier.
"""
import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import convert_to_openai_messages

from agent.graph import SupportAgent
from agent.mcp_tools import open_tools
from agent.prompts import DEFAULT_PROMPT_VERSION
from llm.factory import build_chat_model
from runner.persona import load_persona
from sim.conversation import run_conversation
from sim.user_sim import DEFAULT_MAX_TURNS, UserSimulator
from tools.mock_tools import TOOL_CALLS_FILE

TRANSCRIPT_FILE = "transcript.jsonl"
TRAJECTORY_FILE = "trajectory.jsonl"  # full OpenAI-style messages incl. tool calls and results


async def run_rollout(app_dir: str | Path, opening_message: str,
                      agent_llm: BaseChatModel | None = None,
                      sim_llm: BaseChatModel | None = None,
                      prompt_version: str = DEFAULT_PROMPT_VERSION,
                      max_turns: int = DEFAULT_MAX_TURNS) -> list[dict]:
    app_dir = Path(app_dir)
    agent_llm = agent_llm or build_chat_model("agent")
    sim_llm = sim_llm or build_chat_model("simulator")
    (app_dir / TOOL_CALLS_FILE).touch()  # an agent that calls nothing still leaves a log

    simulator = UserSimulator(sim_llm, load_persona(app_dir), opening_message, max_turns)
    async with open_tools(app_dir) as tools:
        agent = SupportAgent(agent_llm, tools, prompt_version)
        transcript = await run_conversation(agent, simulator, app_dir / TRANSCRIPT_FILE)
    _write_trajectory(agent, app_dir / TRAJECTORY_FILE)
    return transcript


def _write_trajectory(agent: SupportAgent, path: Path) -> None:
    messages = convert_to_openai_messages(agent.messages)
    path.write_text("".join(json.dumps(m) + "\n" for m in messages))
