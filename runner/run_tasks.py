"""Docker-free runner: reads the same Harbor task folders and runs the same verifier.

Same tasks, same tests, same scores as `harbor run`. Used in Colab.
"""
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from agent.prompts import DEFAULT_PROMPT_VERSION
from runner.rollout import TRAJECTORY_FILE, TRANSCRIPT_FILE, run_rollout
from runner.task_io import parse_opening_message
from tools.mock_tools import FIXTURES_FILE, TOOL_CALLS_FILE

PERSONA_FILE = "persona.json"
VERIFIER_LOG_DIR = "verifier"


@dataclass
class TaskResult:
    task_id: str
    reward: int
    checks: dict[str, bool]
    transcript: list[dict]
    tool_calls: list[dict]
    trajectory: list[dict] = field(default_factory=list)


async def run_task(task_dir: str | Path, work_root: str | Path,
                   agent_llm: BaseChatModel | None = None, sim_llm: BaseChatModel | None = None,
                   prompt_version: str = DEFAULT_PROMPT_VERSION) -> TaskResult:
    task_dir = Path(task_dir)
    app_dir = _prepare_app_dir(task_dir, Path(work_root) / task_dir.name)
    opening_message = parse_opening_message((task_dir / "instruction.md").read_text())

    await run_rollout(app_dir, opening_message, agent_llm, sim_llm, prompt_version)
    return _verify(task_dir, app_dir)


async def run_all(tasks_dir: str | Path, work_root: str | Path, **kwargs) -> list[TaskResult]:
    task_dirs = sorted(p for p in Path(tasks_dir).iterdir() if (p / "task.toml").exists())
    return [await run_task(d, work_root, **kwargs) for d in task_dirs]


def summarize(results: list[TaskResult]) -> str:
    passed = sum(r.reward for r in results)
    lines = [f"{passed}/{len(results)} tasks passed"]
    for r in results:
        failed = [name for name, ok in r.checks.items() if not ok]
        lines.append(f"  {'PASS' if r.reward else 'FAIL'}  {r.task_id}"
                     + (f"   failed: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)


def _prepare_app_dir(task_dir: Path, app_dir: Path) -> Path:
    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)
    for name in (FIXTURES_FILE, PERSONA_FILE):
        shutil.copy(task_dir / "environment" / name, app_dir / name)
    return app_dir


def _verify(task_dir: Path, app_dir: Path) -> TaskResult:
    out_dir = app_dir / VERIFIER_LOG_DIR
    subprocess.run(
        [sys.executable, str(task_dir / "tests" / "check_state.py"),
         str(task_dir / "tests" / "checks.json"), str(app_dir), str(out_dir)],
        check=True, capture_output=True, text=True)
    return TaskResult(
        task_id=task_dir.name,
        reward=int((out_dir / "reward.txt").read_text()),
        checks=json.loads((out_dir / "checks.json").read_text()),
        transcript=_load_jsonl(app_dir / TRANSCRIPT_FILE),
        tool_calls=_load_jsonl(app_dir / TOOL_CALLS_FILE),
        trajectory=_load_jsonl(app_dir / TRAJECTORY_FILE))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
