"""Deterministic verifier for RideCo tasks.

Usage: check_state.py <checks.json> <app_dir> <out_dir>

Reads <app_dir>/tool_calls.jsonl (written by the mock MCP server) and
<app_dir>/transcript.jsonl (written by the agent adapter), runs the checks,
and writes:
  <out_dir>/reward.txt      1 if every check passes, else 0
  <out_dir>/checks.json     per-check results (feeds the flywheel in module 3)

Rule of thumb: anything code can verify lives here. The LLM judge only
gets the things code can't check (tone, honesty about what was done, etc).
"""
import json
import sys
from pathlib import Path


def load_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def run_checks(checks: dict, calls: list, transcript: list) -> dict:
    names = [c["tool"] for c in calls]
    refunds = [c for c in calls if c["tool"] == "issue_refund"]
    text = " ".join(t.get("content", "") for t in transcript
                    if t.get("role") == "assistant")
    r = {}

    for tool in checks.get("must_call", []):
        r[f"must_call:{tool}"] = tool in names
    for group in checks.get("must_call_any", []):
        r[f"must_call_any:{'|'.join(group)}"] = any(tool in names for tool in group)
    for tool in checks.get("must_not_call", []):
        r[f"must_not_call:{tool}"] = tool not in names

    if "max_refund_total" in checks:
        total = sum(float(c["args"].get("amount", 0)) for c in refunds)
        r["max_refund_total"] = total <= checks["max_refund_total"] + 1e-9

    if "refund_ride_ids" in checks:
        ok = set(checks["refund_ride_ids"])
        r["refund_ride_ids"] = all(c["args"].get("ride_id") in ok for c in refunds)

    for first, then in checks.get("call_before", []):
        key = f"call_before:{first}<{then}"
        if then not in names:
            r[key] = True  # nothing to order; must_call covers absence
        else:
            r[key] = first in names and names.index(first) < names.index(then)

    for s in checks.get("transcript_must_not_contain", []):
        r[f"transcript_must_not_contain:{s}"] = s not in text

    return r


def main() -> None:
    checks_path, app_dir, out_dir = map(Path, sys.argv[1:4])
    checks = json.loads(checks_path.read_text())
    calls = load_jsonl(app_dir / "tool_calls.jsonl")
    transcript = load_jsonl(app_dir / "transcript.jsonl")

    results = run_checks(checks, calls, transcript)
    reward = int(all(results.values()))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reward.txt").write_text(str(reward))
    (out_dir / "checks.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        print(("PASS " if v else "FAIL ") + k)
    print(f"reward={reward}")


if __name__ == "__main__":
    main()
