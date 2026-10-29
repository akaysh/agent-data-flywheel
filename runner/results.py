"""One row per task for the pass/fail table: code checks, plus the judge's verdicts if there are any.

`pass` is the code checks alone (what Harbor reports). `overall pass` also needs the judge to
find no failed criterion. A regression task made from a judge failure only counts the
criteria listed in its task.toml.
"""
import tomllib
from pathlib import Path

from flywheel.traces import Trace

FAILED_VERDICT = "fail"


def results_rows(traces: list[Trace], tasks_dir: str | Path) -> list[dict]:
    return [_row(trace, _listed_judge_criteria(Path(tasks_dir) / trace.task_id)) for trace in traces]


def _row(trace: Trace, listed_criteria: list[str] | None) -> dict:
    judge_failed = [cid for cid, v in trace.verdicts.items()
                    if v["verdict"] == FAILED_VERDICT and (listed_criteria is None or cid in listed_criteria)]
    return {"task": trace.task_id, "pass": bool(trace.reward),
            "failed checks": ", ".join(name for name, ok in trace.checks.items() if not ok),
            "judge failed": ", ".join(judge_failed),
            "overall pass": bool(trace.reward) and not judge_failed}


def _listed_judge_criteria(task_dir: Path) -> list[str] | None:
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return None
    return tomllib.loads(task_toml.read_text()).get("metadata", {}).get("judge_criteria")
