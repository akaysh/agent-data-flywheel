import json
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "make_tasks.py"


def render(tmp_path, *scenario_files):
    out = tmp_path / "tasks"
    cmd = [sys.executable, str(SCRIPT), "--out", str(out), "--scenarios", *map(str, scenario_files)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_renders_every_scenario_from_every_file(tmp_path):
    regressions = tmp_path / "regressions.yaml"
    reg = yaml.safe_load((REPO / "scenarios" / "scenarios.yaml").read_text())["scenarios"][2]
    reg = {**reg, "id": "regression-refund-over-limit",
           "origin": {"trace_id": "run/refund-over-limit", "failed_checks": ["x"],
                      "failed_judge_criteria": ["honest_about_actions"]},
           "judge_criteria": ["honest_about_actions"]}
    regressions.write_text(yaml.safe_dump({"scenarios": [reg]}))

    out = render(tmp_path, REPO / "scenarios" / "scenarios.yaml", regressions)

    ids = {p.name for p in out.iterdir()}
    assert "refund-over-limit" in ids and "regression-refund-over-limit" in ids
    meta = tomllib.loads((out / "regression-refund-over-limit" / "task.toml").read_text())
    assert meta["metadata"]["origin_trace_id"] == "run/refund-over-limit"
    assert meta["metadata"]["judge_criteria"] == ["honest_about_actions"]


def test_regular_task_toml_has_no_regression_metadata(tmp_path):
    out = render(tmp_path, REPO / "scenarios" / "scenarios.yaml")
    meta = tomllib.loads((out / "refund-over-limit" / "task.toml").read_text())
    assert "origin_trace_id" not in meta["metadata"]
    assert meta["schema_version"] == "1.4"


def test_stale_task_folders_are_removed_but_unrelated_folders_are_kept(tmp_path):
    out = tmp_path / "tasks"
    stale = out / "regression-old"
    stale.mkdir(parents=True)
    (stale / "task.toml").write_text("")
    unrelated = out / "notes"
    unrelated.mkdir()

    render(tmp_path, REPO / "scenarios" / "scenarios.yaml")

    assert not stale.exists()
    assert unrelated.exists()
    assert (out / "refund-over-limit" / "task.toml").exists()
