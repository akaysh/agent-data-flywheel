from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from judge.run_judge import (
    JudgeTrace, agreement_report, judge_good, judge_vague, load_rubric, parse_json_object)
from tests.conftest import ScriptedChatModel

REPO = Path(__file__).resolve().parent.parent
TRACE = JudgeTrace(
    trace_id="t1",
    transcript=[{"role": "user", "content": "refund me"},
                {"role": "assistant", "content": "Done! Your refund is on its way."}],
    tool_calls=[{"tool": "escalate_to_human", "args": {"reason": "too big"}}])


def llm_replying(*contents):
    return ScriptedChatModel(messages=iter([AIMessage(content=c) for c in contents]))


def test_parse_json_object_handles_code_fences_and_chatter():
    assert parse_json_object('```json\n{"score": 4}\n```') == {"score": 4}
    assert parse_json_object('Sure! {"verdict": "fail"} hope that helps') == {"verdict": "fail"}
    assert parse_json_object("no json here") is None


async def test_good_rubric_makes_one_call_per_criterion():
    rubric = load_rubric(REPO / "judge" / "rubric_good.yaml")
    n = len(rubric["criteria"])
    replies = [f'{{"evidence": ["q"], "verdict": "fail", "reason": "r{i}"}}' for i in range(n)]
    verdicts = await judge_good(llm_replying(*replies), rubric, TRACE)
    assert set(verdicts) == {c["id"] for c in rubric["criteria"]}
    assert all(v["verdict"] == "fail" for v in verdicts.values())


async def test_good_rubric_prompt_shows_tool_calls_and_criterion():
    seen = []

    class Spy(ScriptedChatModel):
        async def _agenerate(self, messages, *a, **k):
            seen.append(messages[0].content)
            return await super()._agenerate(messages, *a, **k)

    rubric = load_rubric(REPO / "judge" / "rubric_good.yaml")
    replies = ['{"evidence": [], "verdict": "pass", "reason": "r"}'] * len(rubric["criteria"])
    await judge_good(Spy(messages=iter([AIMessage(content=r) for r in replies])), rubric, TRACE)
    assert "escalate_to_human" in seen[0]
    assert "Does every action the agent tells the customer" in seen[0]
    assert "Done! Your refund is on its way." in seen[0]


async def test_unparseable_judge_output_becomes_invalid_verdict():
    rubric = {"criteria": [{"id": "c", "question": "q", "definition": "d",
                            "pass_example": "p", "fail_example": "f"}],
              "prompt": "{question}{definition}{pass_example}{fail_example}{tool_calls}{transcript}"}
    verdicts = await judge_good(llm_replying("I think it is fine"), rubric, TRACE)
    assert verdicts["c"]["verdict"] == "invalid"


async def test_vague_rubric_thresholds_the_score():
    rubric = load_rubric(REPO / "judge" / "rubric_vague.yaml")
    high = await judge_vague(llm_replying('{"score": 8, "reason": "ok"}'), rubric, TRACE)
    low = await judge_vague(llm_replying('{"score": 3, "reason": "bad"}'), rubric, TRACE)
    assert high["verdict"] == "pass" and low["verdict"] == "fail"


def test_agreement_report_perfect_and_chance_level():
    expert = ["pass", "fail", "pass", "fail", "pass", "fail"]
    assert agreement_report(expert, expert)["kappa"] == 1.0
    always_pass = ["pass"] * 6
    assert agreement_report(always_pass, expert)["kappa"] == pytest.approx(0.0)


def test_agreement_report_counts_labels_and_agreement():
    report = agreement_report(["pass", "fail", "na"], ["pass", "pass", "na"])
    assert report["n"] == 3
    assert report["percent_agreement"] == pytest.approx(2 / 3)


def test_agreement_report_rejects_length_mismatch():
    with pytest.raises(ValueError):
        agreement_report(["pass"], ["pass", "fail"])
