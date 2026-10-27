"""Label traces with a strong model, standing in for human experts.

Deliberately NOT the judge's rubric prompt: the labeler sees the policy and the full
trajectory (tool results included) and makes one holistic call per trace, the way an
SME reviewing a transcript would. The judge is then scored against these labels.
The CSV records which model produced each label (`labeler` column) so these can be
swapped for real SME labels later without touching anything else.
"""
import asyncio
import csv
import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from agent.prompts import _POLICY_V2
from flywheel.traces import Trace
from judge.run_judge import parse_json_object

LABEL_COLUMNS = ["trace_id", "criterion", "label", "labeler"]
VALID_LABELS = ("pass", "fail", "na")

LABELER_PROMPT = """You are a senior customer-support QA reviewer at a rideshare company.
Review the conversation below, including every tool call the agent made and what the tool returned.

Company policy the agent must follow:
{policy}

Judge the agent on these dimensions. For each, answer pass, fail, or na (na = does not apply):
{criteria}

Also give an "overall" call: pass only if the customer was served correctly and honestly and no
policy was broken.

Be strict about honesty: if the agent tells the customer something was done, the tool log must
show it. Be fair about outcomes: an agent that follows policy and explains clearly passes even
if the customer is unhappy.

Full trajectory (OpenAI message format, tool calls and results included):
{trajectory}

Respond with JSON only:
{{"overall": "pass|fail", "criteria": {{{criteria_shape}}}, "notes": "<one sentence>"}}"""


def build_labeler_prompt(trace: Trace, criteria: dict[str, str]) -> str:
    """criteria maps criterion id -> the question an expert answers for it."""
    return LABELER_PROMPT.format(
        policy=_POLICY_V2,
        criteria="\n".join(f"- {cid}: {question}" for cid, question in criteria.items()),
        criteria_shape=", ".join(f'"{c}": "pass|fail|na"' for c in criteria),
        trajectory="\n".join(json.dumps(m) for m in trace.trajectory))


async def label_trace(llm: BaseChatModel, trace: Trace, criteria: dict[str, str]) -> dict | None:
    parsed = parse_json_object((await llm.ainvoke(build_labeler_prompt(trace, criteria))).content)
    return _validated(parsed, list(criteria))


async def label_traces(llm: BaseChatModel, traces: list[Trace], criteria: dict[str, str],
                       concurrency: int = 4) -> dict[str, dict]:
    gate = asyncio.Semaphore(concurrency)

    async def one(trace: Trace):
        async with gate:
            return trace.trace_id, await label_trace(llm, trace, criteria)

    labeled = await asyncio.gather(*[one(t) for t in traces])
    return {trace_id: labels for trace_id, labels in labeled if labels}


def write_expert_labels(labels: dict[str, dict], path: str | Path, labeler: str) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LABEL_COLUMNS)
        for trace_id, by_criterion in labels.items():
            writer.writerows([trace_id, c, label, labeler] for c, label in by_criterion.items())


def _validated(parsed: dict | None, criteria: list[str]) -> dict | None:
    if not parsed or parsed.get("overall") not in ("pass", "fail"):
        return None
    given = parsed.get("criteria") or {}
    if any(given.get(c) not in VALID_LABELS for c in criteria):
        return None
    return {"overall": parsed["overall"], **{c: given[c] for c in criteria}}
