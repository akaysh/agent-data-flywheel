"""Run every task, optionally judge it, and return traces. Used by the CLI and the notebook."""
import asyncio
from dataclasses import replace
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from agent.prompts import DEFAULT_PROMPT_VERSION
from flywheel.traces import Trace
from judge.run_judge import JudgeTrace, judge_good, load_rubric
from llm.factory import build_chat_model
from runner.run_tasks import TaskResult, run_task

DEFAULT_CONCURRENCY = 4


async def evaluate(tasks_dir: str | Path, run_label: str, work_root: str | Path,
                   prompt_version: str = DEFAULT_PROMPT_VERSION,
                   agent_llm: BaseChatModel | None = None, sim_llm: BaseChatModel | None = None,
                   judge_llm: BaseChatModel | None = None,
                   judge_rubric_path: str | Path | None = None,
                   concurrency: int = DEFAULT_CONCURRENCY) -> list[Trace]:
    agent_llm = agent_llm or build_chat_model("agent")
    sim_llm = sim_llm or build_chat_model("simulator")
    gate = asyncio.Semaphore(concurrency)

    async def one(task_dir: Path) -> Trace:
        async with gate:
            result = await run_task(task_dir, work_root, agent_llm, sim_llm, prompt_version)
            verdicts = await _judge(result, run_label, judge_llm, judge_rubric_path)
            return Trace.from_result(result, run_label, verdicts)

    task_dirs = sorted(p for p in Path(tasks_dir).iterdir() if (p / "task.toml").exists())
    return list(await asyncio.gather(*[one(d) for d in task_dirs]))


async def judge_traces(traces: list[Trace], judge_llm: BaseChatModel,
                       rubric_path: str | Path, concurrency: int = DEFAULT_CONCURRENCY) -> list[Trace]:
    """Returns copies of the traces with judge verdicts attached."""
    rubric = load_rubric(rubric_path)
    gate = asyncio.Semaphore(concurrency)

    async def one(trace: Trace) -> Trace:
        async with gate:
            verdicts = await judge_good(
                judge_llm, rubric, JudgeTrace(trace.trace_id, trace.transcript, trace.tool_calls))
        return replace(trace, verdicts=verdicts)

    return list(await asyncio.gather(*[one(t) for t in traces]))


async def _judge(result: TaskResult, run_label: str, judge_llm: BaseChatModel | None,
                 rubric_path: str | Path | None) -> dict[str, dict]:
    if rubric_path is None:
        return {}
    llm = judge_llm or build_chat_model("judge")
    trace = JudgeTrace(f"{run_label}/{result.task_id}", result.transcript, result.tool_calls)
    return await judge_good(llm, load_rubric(rubric_path), trace)
