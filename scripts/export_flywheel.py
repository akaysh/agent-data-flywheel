"""Turn judged traces into the flywheel's three outputs.

Usage: python -m scripts.export_flywheel --traces data/traces/before-fix.jsonl [more.jsonl ...]

  scenarios/regressions.yaml        failed traces -> new eval tasks (then run make_tasks.py)
  data/export/preference_pairs.jsonl  TRL DPO format {prompt, chosen, rejected}
  data/export/rewards.jsonl           one scalar reward per trace
"""
import argparse
from pathlib import Path

from flywheel.preference_pairs import build_preference_pairs, build_reward_records, write_jsonl
from flywheel.regressions import add_regressions
from flywheel.traces import load_traces
from scripts.paths import EXPORT_DIR, REPO_ROOT

SCENARIOS = REPO_ROOT / "scenarios" / "scenarios.yaml"
REGRESSIONS = REPO_ROOT / "scenarios" / "regressions.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", type=Path, required=True)
    args = ap.parse_args()
    traces = [t for path in args.traces for t in load_traces(path)]

    added = add_regressions(SCENARIOS, REGRESSIONS, traces)
    pairs = build_preference_pairs(traces)
    rewards = build_reward_records(traces)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(pairs, EXPORT_DIR / "preference_pairs.jsonl")
    write_jsonl(rewards, EXPORT_DIR / "rewards.jsonl")
    print(f"{len(added)} new regression scenarios: {added}")
    print(f"{len(pairs)} preference pairs, {len(rewards)} reward records -> {EXPORT_DIR}")


if __name__ == "__main__":
    main()
