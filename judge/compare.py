"""Vague vs good rubric, scored against expert labels. Workshop module 2's punchline."""
import asyncio
import csv
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from flywheel.traces import Trace
from judge.run_judge import JudgeTrace, agreement_report, judge_good, judge_vague

OVERALL = "overall"
SYSTEM_OVERALL = "system_overall"  # code checks AND judge criteria, scored against `overall`
IGNORED_VERDICTS = ("na",)


def load_expert_labels(path: str | Path) -> dict[tuple[str, str], str]:
    with Path(path).open(newline="") as f:
        return {(r["trace_id"], r["criterion"]): r["label"].strip()
                for r in csv.DictReader(f) if r["label"].strip()}


def overall_from_criteria(verdicts: dict[str, dict]) -> str:
    applicable = [v["verdict"] for v in verdicts.values() if v["verdict"] not in IGNORED_VERDICTS]
    return "pass" if all(v == "pass" for v in applicable) else "fail"


async def compare_rubrics(vague_llm: BaseChatModel, good_llm: BaseChatModel,
                          vague_rubric: dict, good_rubric: dict, traces: list[Trace],
                          labels: dict[tuple[str, str], str], concurrency: int = 4) -> dict:
    gate = asyncio.Semaphore(concurrency)

    async def judge_one(trace: Trace):
        jt = JudgeTrace(trace.trace_id, trace.transcript, trace.tool_calls)
        async with gate:
            return (await judge_vague(vague_llm, vague_rubric, jt),
                    await judge_good(good_llm, good_rubric, jt))

    judged = await asyncio.gather(*[judge_one(t) for t in traces])
    vague = {t.trace_id: v for t, (v, _) in zip(traces, judged)}
    good = {t.trace_id: g for t, (_, g) in zip(traces, judged)}

    criteria = [c["id"] for c in good_rubric["criteria"]]
    return {
        "vague": {OVERALL: _score(traces, labels, OVERALL,
                                  lambda t: vague[t.trace_id]["verdict"])},
        "good": {OVERALL: _score(traces, labels, OVERALL,
                                 lambda t: overall_from_criteria(good[t.trace_id])),
                 SYSTEM_OVERALL: _score(traces, labels, OVERALL,
                                        lambda t: _system_overall(t, good[t.trace_id])),
                 **{c: _score(traces, labels, c, lambda t, c=c: good[t.trace_id][c]["verdict"])
                    for c in criteria}},
        "verdicts": {"vague": vague, "good": good},
    }


def _system_overall(trace: Trace, verdicts: dict[str, dict]) -> str:
    code_ok = trace.reward == 1
    return "pass" if code_ok and overall_from_criteria(verdicts) == "pass" else "fail"


def _score(traces: list[Trace], labels: dict, criterion: str, predict) -> dict:
    labeled = [t for t in traces if (t.trace_id, criterion) in labels]
    return agreement_report([predict(t) for t in labeled],
                            [labels[(t.trace_id, criterion)] for t in labeled])
