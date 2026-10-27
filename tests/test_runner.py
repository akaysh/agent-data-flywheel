import json
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage

from runner.run_tasks import run_task, summarize
from runner.task_io import parse_opening_message
from sim.user_sim import END_TOKEN
from tests.conftest import ScriptedChatModel

REPO = Path(__file__).resolve().parent.parent
TASKS = REPO / "tasks"


def scripted(*messages):
    return ScriptedChatModel(messages=iter(messages))


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_parse_opening_message_reads_the_blockquote():
    text = (TASKS / "refund-under-limit" / "instruction.md").read_text()
    assert parse_opening_message(text) == (
        "I got charged an airport fee but I wasn't going to the airport.")


async def test_good_agent_passes_refund_under_limit(tmp_path):
    agent_llm = scripted(
        AIMessage(content='{"agents": ["charges"]}'),
        tool_call("get_charges", {"ride_id": "R200"}, "c1"),
        tool_call("issue_refund", {"ride_id": "R200", "charge_id": "C201", "amount": 12.0}, "c2"),
        AIMessage(content="Refunded $12. It shows up in 3-5 days."))
    sim_llm = scripted(AIMessage(content=END_TOKEN))

    result = await run_task(TASKS / "refund-under-limit", tmp_path, agent_llm, sim_llm)

    assert result.reward == 1
    assert result.transcript[-1]["content"].startswith("Refunded $12")
    assert [c["tool"] for c in result.tool_calls] == ["get_charges", "issue_refund"]


async def test_bad_agent_fails_refund_over_limit(tmp_path):
    agent_llm = scripted(
        AIMessage(content='{"agents": ["charges"]}'),
        tool_call("issue_refund", {"ride_id": "R300", "charge_id": "C301", "amount": 68.0}, "c1"),
        AIMessage(content="Refunded $68!"))
    sim_llm = scripted(AIMessage(content=END_TOKEN))

    result = await run_task(TASKS / "refund-over-limit", tmp_path, agent_llm, sim_llm)

    assert result.reward == 0
    assert result.checks["must_not_call:issue_refund"] is False


async def test_run_task_does_not_touch_the_task_folder(tmp_path):
    before = sorted(p.name for p in (TASKS / "refund-under-limit" / "environment").iterdir())
    await run_task(TASKS / "refund-under-limit", tmp_path,
                   scripted(AIMessage(content='{"agents": ["charges"]}'), AIMessage(content="hi")),
                   scripted(AIMessage(content=END_TOKEN)))
    after = sorted(p.name for p in (TASKS / "refund-under-limit" / "environment").iterdir())
    assert before == after


def test_summarize_reports_pass_rate_and_failures():
    from runner.run_tasks import TaskResult
    results = [TaskResult("a", 1, {}, [], []),
               TaskResult("b", 0, {"must_call:x": False, "ok": True}, [], [])]
    text = summarize(results)
    assert "1/2" in text
    assert "b" in text and "must_call:x" in text


async def test_run_task_captures_full_trajectory_with_tool_calls_and_results(tmp_path):
    agent_llm = scripted(
        AIMessage(content='{"agents": ["charges"]}'),
        tool_call("get_charges", {"ride_id": "R200"}, "c1"),
        AIMessage(content="Your airport fee was $12."))
    result = await run_task(TASKS / "fare-explain-basic", tmp_path, agent_llm,
                            scripted(AIMessage(content=END_TOKEN)))
    roles = [m["role"] for m in result.trajectory]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert result.trajectory[1]["tool_calls"][0]["function"]["name"] == "get_charges"
