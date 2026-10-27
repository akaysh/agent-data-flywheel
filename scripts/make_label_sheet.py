"""Create the expert labeling sheet from saved traces.

Usage: python -m scripts.make_label_sheet --traces data/traces/*.jsonl

Writes data/label_sheet.csv (one row per trace x criterion, label column empty) and
data/label_sheet.md (readable traces to label from). Copy the filled-in sheet to
data/expert_labels.csv. Labels: pass | fail | na. The extra criterion `overall`
is the expert's overall good/bad call and is what the vague rubric is scored against.
"""
import argparse
import csv
import json
from pathlib import Path

from flywheel.traces import Trace, load_traces
from judge.run_judge import load_rubric
from scripts.paths import JUDGE_DIR, REPO_ROOT

OVERALL = "overall"
SHEET_CSV = REPO_ROOT / "data" / "label_sheet.csv"
SHEET_MD = REPO_ROOT / "data" / "label_sheet.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", type=Path, required=True)
    args = ap.parse_args()
    traces = [t for path in args.traces for t in load_traces(path)]
    criteria = [c["id"] for c in load_rubric(JUDGE_DIR / "rubric_good.yaml")["criteria"]]

    with SHEET_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trace_id", "criterion", "label"])
        for trace in traces:
            writer.writerows([trace.trace_id, c, ""] for c in [OVERALL, *criteria])
    SHEET_MD.write_text("\n\n---\n\n".join(_render(t) for t in traces))
    print(f"{len(traces)} traces -> {SHEET_CSV} and {SHEET_MD}")


def _render(trace: Trace) -> str:
    convo = "\n".join(f"**{'Customer' if m['role'] == 'user' else 'Agent'}:** {m['content']}"
                      for m in trace.transcript)
    calls = "\n".join(f"- `{json.dumps(c)}`" for c in trace.tool_calls) or "- (none)"
    return f"## {trace.trace_id}\n\n{convo}\n\nTool calls:\n{calls}"


if __name__ == "__main__":
    main()
