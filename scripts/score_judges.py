"""Score the vague and good rubrics against expert labels (kappa). Workshop module 2.

Usage: python -m scripts.score_judges [--traces data/traces/labeled.jsonl]
                                      [--labels data/expert_labels.csv]
"""
import argparse
import asyncio
from pathlib import Path

from flywheel.traces import load_traces
from judge.compare import compare_rubrics, load_expert_labels
from judge.run_judge import load_rubric
from llm.factory import build_chat_model
from scripts.paths import JUDGE_DIR, REPO_ROOT

DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=DATA_DIR / "traces" / "labeled.jsonl")
    ap.add_argument("--labels", type=Path, default=DATA_DIR / "expert_labels.csv")
    args = ap.parse_args()

    llm = build_chat_model("judge")
    report = asyncio.run(compare_rubrics(
        llm, llm, load_rubric(JUDGE_DIR / "rubric_vague.yaml"),
        load_rubric(JUDGE_DIR / "rubric_good.yaml"),
        load_traces(args.traces), load_expert_labels(args.labels)))

    for rubric in ("vague", "good"):
        for criterion, r in report[rubric].items():
            print(f"{rubric:>5}  {criterion:<22} kappa={r['kappa']:.2f}  "
                  f"agree={r['percent_agreement']:.0%}  n={r['n']}")


if __name__ == "__main__":
    main()
