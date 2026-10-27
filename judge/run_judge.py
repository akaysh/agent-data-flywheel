"""LLM judges for RideCo traces, plus agreement (Cohen's kappa) with expert labels.

judge_vague: the "before" rubric, one 1-10 score for the whole conversation.
judge_good:  the "after" rubric, one call per binary criterion with evidence.
"""
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.language_models import BaseChatModel
from sklearn.metrics import cohen_kappa_score

INVALID = "invalid"
KAPPA_TARGET = 0.7


@dataclass
class JudgeTrace:
    trace_id: str
    transcript: list[dict]
    tool_calls: list[dict]


def load_rubric(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


async def judge_vague(llm: BaseChatModel, rubric: dict, trace: JudgeTrace) -> dict:
    prompt = rubric["prompt"].format(transcript=_format_transcript(trace.transcript))
    parsed = parse_json_object((await llm.ainvoke(prompt)).content)
    if not parsed or "score" not in parsed:
        return {"verdict": INVALID, "score": None, "reason": "unparseable judge output"}
    verdict = "pass" if float(parsed["score"]) >= rubric["pass_threshold"] else "fail"
    return {"verdict": verdict, "score": parsed["score"], "reason": parsed.get("reason", "")}


async def judge_good(llm: BaseChatModel, rubric: dict, trace: JudgeTrace) -> dict[str, dict]:
    """One independent call per criterion, never bundled."""
    ids = [c["id"] for c in rubric["criteria"]]
    verdicts = await asyncio.gather(
        *[_judge_criterion(llm, rubric["prompt"], c, trace) for c in rubric["criteria"]])
    return dict(zip(ids, verdicts))


def parse_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def agreement_report(judge_labels: list[str], expert_labels: list[str]) -> dict:
    if len(judge_labels) != len(expert_labels):
        raise ValueError(f"{len(judge_labels)} judge labels vs {len(expert_labels)} expert labels")
    matches = sum(j == e for j, e in zip(judge_labels, expert_labels))
    return {"n": len(expert_labels),
            "percent_agreement": matches / len(expert_labels),
            "kappa": float(cohen_kappa_score(expert_labels, judge_labels)),
            "kappa_target": KAPPA_TARGET}


async def _judge_criterion(llm: BaseChatModel, prompt_template: str, criterion: dict,
                           trace: JudgeTrace) -> dict:
    prompt = prompt_template.format(
        question=criterion["question"], definition=criterion["definition"],
        pass_example=criterion["pass_example"], fail_example=criterion["fail_example"],
        tool_calls=_format_tool_calls(trace.tool_calls),
        transcript=_format_transcript(trace.transcript))
    parsed = parse_json_object((await llm.ainvoke(prompt)).content)
    if not parsed or parsed.get("verdict") not in ("pass", "fail", "na"):
        return {"verdict": INVALID, "evidence": [], "reason": "unparseable judge output"}
    return {"verdict": parsed["verdict"], "evidence": parsed.get("evidence", []),
            "reason": parsed.get("reason", "")}


def _format_transcript(transcript: list[dict]) -> str:
    speakers = {"user": "Customer", "assistant": "Agent"}
    return "\n".join(f"{speakers[t['role']]}: {t['content']}" for t in transcript)


def _format_tool_calls(tool_calls: list[dict]) -> str:
    return "\n".join(json.dumps(c) for c in tool_calls) or "(none)"
