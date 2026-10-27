"""Failed traces become new eval tasks.

A regression scenario is the original scenario (same fixtures, persona, checks and
oracle) tagged with where it came from. scripts/make_tasks.py renders it like any other.
"""
import copy
from pathlib import Path

import yaml

from flywheel.traces import Trace

REGRESSION_ID_PREFIX = "regression-"
FAILED_VERDICT = "fail"


def failed_criteria(trace: Trace) -> tuple[list[str], list[str]]:
    """(failed code checks, failed judge criteria)."""
    code = [name for name, ok in trace.checks.items() if not ok]
    judge = [cid for cid, v in trace.verdicts.items() if v["verdict"] == FAILED_VERDICT]
    return code, judge


def scenario_from_failure(scenario: dict, trace: Trace) -> dict | None:
    failed_checks, failed_judge = failed_criteria(trace)
    if not failed_checks and not failed_judge:
        return None
    regression = copy.deepcopy(scenario)
    regression["id"] = f"{REGRESSION_ID_PREFIX}{scenario['id']}"
    regression["planted_bug"] = None
    regression["origin"] = {"trace_id": trace.trace_id, "failed_checks": failed_checks,
                            "failed_judge_criteria": failed_judge}
    if failed_judge:
        regression["judge_criteria"] = failed_judge
    return regression


def add_regressions(scenarios_path: str | Path, regressions_path: str | Path,
                    traces: list[Trace]) -> list[str]:
    """Appends regression scenarios for failing traces. Returns the ids that were new."""
    originals = {s["id"]: s for s in _load(scenarios_path)}
    existing = _load(regressions_path) if Path(regressions_path).exists() else []
    known_ids = {s["id"] for s in existing}

    added = []
    for trace in traces:
        original = originals.get(trace.task_id)
        regression = scenario_from_failure(original, trace) if original else None
        if regression and regression["id"] not in known_ids:
            existing.append(regression)
            known_ids.add(regression["id"])
            added.append(regression["id"])

    Path(regressions_path).write_text(yaml.safe_dump({"scenarios": existing}, sort_keys=False))
    return added


def _load(path: str | Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text())["scenarios"]
