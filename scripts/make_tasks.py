"""Render scenarios/scenarios.yaml into Harbor task folders under tasks/.

Usage: python scripts/make_tasks.py [--scenarios PATH ...] [--out tasks]

The same script is reused in workshop module 3: failed traces get turned
into new scenario entries, and this renders them as regression tasks.
"""
import argparse
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "scripts" / "check_state.py"

TASK_TOML = """schema_version = "1.4"

[task]
name = "rideco/{id}"
version = "1.0.0"

[metadata]
area = "{area}"
difficulty = "{difficulty}"
tags = {tags}
{extra_metadata}
[agent]
timeout_sec = 300

[verifier]
timeout_sec = 120

[environment]
cpus = 1
memory_mb = 1024
"""

INSTRUCTION = """You are the RideCo customer support agent. A customer just wrote in:

> {opening_message}

Help them using your tools and RideCo policy.
"""

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
# Fixture data the mock MCP tools serve. The persona is read by the
# user-simulator in the agent adapter, never exposed as a tool.
COPY fixtures.json persona.json /app/
RUN pip install --no-cache-dir pyyaml
"""

TEST_SH = """#!/bin/bash
# Deterministic checks only. Judge scoring runs separately (module 2).
mkdir -p /logs/verifier
python3 /tests/check_state.py /tests/checks.json /app /logs/verifier
"""

SOLVE_SH = """#!/bin/bash
# Oracle: writes the expected tool calls so the verifier should score 1.0.
# If the oracle fails, the task is broken, not the agent.
cp /solution/expected_calls.jsonl /app/tool_calls.jsonl
touch /app/transcript.jsonl
"""


def regression_metadata(s: dict) -> str:
    """Extra task.toml lines for tasks the flywheel generated from a failed trace."""
    lines = []
    if "origin" in s:
        lines.append(f"origin_trace_id = {json.dumps(s['origin']['trace_id'])}")
    if s.get("judge_criteria"):
        lines.append(f"judge_criteria = {json.dumps(s['judge_criteria'])}")
    return "".join(line + "\n" for line in lines)


def render(s: dict, out: Path) -> None:
    d = out / s["id"]
    if d.exists():
        shutil.rmtree(d)
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution").mkdir()

    tags = [s["area"]] + (["planted_bug"] if s.get("planted_bug") else [])
    (d / "task.toml").write_text(TASK_TOML.format(
        id=s["id"], area=s["area"], difficulty=s["difficulty"],
        tags=json.dumps(tags), extra_metadata=regression_metadata(s)))
    (d / "instruction.md").write_text(
        INSTRUCTION.format(opening_message=s["opening_message"]))

    (d / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (d / "environment" / "fixtures.json").write_text(
        json.dumps(s["fixtures"], indent=2))
    (d / "environment" / "persona.json").write_text(
        json.dumps(s["persona"], indent=2))

    (d / "tests" / "test.sh").write_text(TEST_SH)
    (d / "tests" / "checks.json").write_text(json.dumps(s["checks"], indent=2))
    shutil.copy(CHECKER, d / "tests" / "check_state.py")

    (d / "solution" / "solve.sh").write_text(SOLVE_SH)
    (d / "solution" / "expected_calls.jsonl").write_text(
        "\n".join(json.dumps(c) for c in s["expected_calls"]) + "\n")

    for f in ["tests/test.sh", "solution/solve.sh"]:
        (d / f).chmod(0o755)


def remove_stale_tasks(out: Path, keep: set[str]) -> None:
    """Deletes task folders whose scenario is gone (e.g. regressions.yaml was reset)."""
    for d in out.iterdir():
        if d.is_dir() and (d / "task.toml").exists() and d.name not in keep:
            shutil.rmtree(d)


def default_scenario_files() -> list[Path]:
    files = [ROOT / "scenarios" / "scenarios.yaml", ROOT / "scenarios" / "regressions.yaml"]
    return [f for f in files if f.exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", type=Path, default=default_scenario_files())
    ap.add_argument("--out", default=ROOT / "tasks")
    args = ap.parse_args()

    scenarios = [s for f in args.scenarios for s in yaml.safe_load(f.read_text())["scenarios"]]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    remove_stale_tasks(out, keep={s["id"] for s in scenarios})
    for s in scenarios:
        render(s, out)
    print(f"Rendered {len(scenarios)} tasks into {out}/")


if __name__ == "__main__":
    main()
