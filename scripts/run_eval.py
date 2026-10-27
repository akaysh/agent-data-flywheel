"""Run all tasks with the RideCo agent, optionally judge them, save traces.

Usage:
  python -m scripts.run_eval --label before-fix [--prompt-version v1] [--judge good] [--tasks tasks]

Record the offline cache with a real provider (OFFLINE=0), then commit cache/.
Needs LLM_BASE_URL, LLM_API_KEY, AGENT_MODEL (and JUDGE_MODEL with --judge good).
"""
import argparse
import asyncio
import tempfile
from pathlib import Path

from agent.prompts import PROMPT_VERSIONS
from flywheel.traces import save_traces
from llm.factory import build_chat_model
from runner.evaluate import evaluate
from runner.run_tasks import TaskResult
from scripts.paths import JUDGE_DIR, REPO_ROOT, TRACES_DIR

JUDGE_RUBRICS = {"good": JUDGE_DIR / "rubric_good.yaml", "none": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-version", default="v1", choices=list(PROMPT_VERSIONS))
    ap.add_argument("--agent-role", default="agent", choices=["agent", "fixed_agent"],
                    help="fixed_agent = the stronger FIXED_AGENT_MODEL (swap-the-model fix)")
    ap.add_argument("--judge", default="none", choices=list(JUDGE_RUBRICS))
    ap.add_argument("--tasks", type=Path, default=REPO_ROOT / "tasks")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as work_root:
        traces = asyncio.run(evaluate(
            args.tasks, args.label, work_root, prompt_version=args.prompt_version,
            agent_llm=build_chat_model(args.agent_role),
            judge_rubric_path=JUDGE_RUBRICS[args.judge], concurrency=args.concurrency))

    TRACES_DIR.mkdir(exist_ok=True)
    out = TRACES_DIR / f"{args.label}.jsonl"
    save_traces(traces, out)
    print(_summary(traces))
    print(f"traces -> {out}")


def _summary(traces) -> str:
    from runner.run_tasks import summarize
    return summarize([TaskResult(t.task_id, t.reward, t.checks, t.transcript, t.tool_calls)
                      for t in traces])


if __name__ == "__main__":
    main()
