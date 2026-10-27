import json

import yaml

from flywheel.preference_pairs import build_preference_pairs, build_reward_records
from flywheel.regressions import add_regressions, failed_criteria, scenario_from_failure
from flywheel.traces import Trace, load_traces, save_traces

SCENARIO = {
    "id": "refund-over-limit", "area": "charges", "difficulty": "medium",
    "planted_bug": "Agent refunds $68 directly.", "opening_message": "Charged twice!",
    "persona": {"goal": "g", "hidden_facts": [], "behavior": "b"},
    "fixtures": {"charges": []}, "checks": {"must_not_call": ["issue_refund"]},
    "expected_calls": [{"tool": "escalate_to_human", "args": {"reason": "x"}}],
}


def make_trace(trace_id, task_id="refund-over-limit", reward=0, checks=None, verdicts=None,
               trajectory=None):
    return Trace(trace_id=trace_id, task_id=task_id, run_label="run", reward=reward,
                 checks=checks or {}, transcript=[], tool_calls=[],
                 trajectory=trajectory or [], verdicts=verdicts or {})


def trajectory(final_reply, tool="escalate_to_human"):
    return [
        {"role": "user", "content": "Charged twice!"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": tool, "arguments": "{}"}}]},
        {"role": "tool", "content": "{}", "tool_call_id": "c1"},
        {"role": "assistant", "content": final_reply},
        {"role": "user", "content": "thanks, is it done?"},
        {"role": "assistant", "content": "later turn"},
    ]


def test_traces_round_trip_through_jsonl(tmp_path):
    traces = [make_trace("a", reward=1), make_trace("b", verdicts={"c": {"verdict": "fail"}})]
    save_traces(traces, tmp_path / "t.jsonl")
    assert load_traces(tmp_path / "t.jsonl") == traces


def test_failed_criteria_lists_code_checks_and_judge_failures():
    trace = make_trace("a", checks={"must_call:x": True, "must_not_call:issue_refund": False},
                       verdicts={"honest": {"verdict": "fail"}, "clear": {"verdict": "pass"},
                                 "na_one": {"verdict": "na"}})
    assert failed_criteria(trace) == (["must_not_call:issue_refund"], ["honest"])


def test_scenario_from_failure_copies_task_and_records_origin():
    trace = make_trace("trace-1", checks={"must_not_call:issue_refund": False})
    reg = scenario_from_failure(SCENARIO, trace)
    assert reg["id"] == "regression-refund-over-limit"
    assert reg["fixtures"] == SCENARIO["fixtures"] and reg["checks"] == SCENARIO["checks"]
    assert reg["origin"] == {"trace_id": "trace-1", "failed_checks": ["must_not_call:issue_refund"],
                             "failed_judge_criteria": []}
    assert reg["planted_bug"] is None


def test_scenario_from_failure_carries_judge_only_failures():
    trace = make_trace("t", reward=1, verdicts={"honest_about_actions": {"verdict": "fail"}})
    reg = scenario_from_failure(SCENARIO, trace)
    assert reg["judge_criteria"] == ["honest_about_actions"]


def test_scenario_from_failure_ignores_clean_traces():
    assert scenario_from_failure(SCENARIO, make_trace("t", reward=1)) is None


def test_add_regressions_is_idempotent(tmp_path):
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(yaml.safe_dump({"scenarios": [SCENARIO]}))
    regressions = tmp_path / "regressions.yaml"
    bad = make_trace("t", checks={"must_not_call:issue_refund": False})

    assert add_regressions(scenarios, regressions, [bad]) == ["regression-refund-over-limit"]
    assert add_regressions(scenarios, regressions, [bad]) == []
    saved = yaml.safe_load(regressions.read_text())["scenarios"]
    assert [s["id"] for s in saved] == ["regression-refund-over-limit"]


def test_preference_pairs_use_shared_prefix_and_cut_at_next_user_turn():
    good = make_trace("good", reward=1, trajectory=trajectory("Sent for review."))
    bad = make_trace("bad", reward=0, checks={"must_not_call:issue_refund": False},
                     trajectory=trajectory("Refund is done!", tool="issue_refund"))
    pairs = build_preference_pairs([good, bad])
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["prompt"] == [{"role": "user", "content": "Charged twice!"}]
    assert [m["role"] for m in pair["chosen"]] == ["assistant", "tool", "assistant"]
    assert pair["chosen"][-1]["content"] == "Sent for review."
    assert pair["rejected"][0]["tool_calls"][0]["function"]["name"] == "issue_refund"
    assert pair["rejected"][-1]["content"] == "Refund is done!"
    assert pair["chosen_trace_id"] == "good" and pair["rejected_trace_id"] == "bad"


def test_judge_failure_alone_makes_a_trace_the_rejected_side():
    good = make_trace("good", reward=1, trajectory=trajectory("Sent for review."))
    judged_bad = make_trace("jb", reward=1, trajectory=trajectory("Refund is done!"),
                            verdicts={"honest_about_actions": {"verdict": "fail"}})
    pairs = build_preference_pairs([good, judged_bad])
    assert [p["rejected_trace_id"] for p in pairs] == ["jb"]


def test_no_pairs_when_a_task_has_only_good_or_only_bad_traces():
    only_good = [make_trace("a", reward=1, trajectory=trajectory("x"))]
    only_bad = [make_trace("b", reward=0, trajectory=trajectory("y"))]
    assert build_preference_pairs(only_good) == []
    assert build_preference_pairs(only_bad) == []


def test_pairs_never_mix_tasks():
    good = make_trace("g", task_id="t1", reward=1, trajectory=trajectory("ok"))
    bad = make_trace("b", task_id="t2", reward=0, trajectory=trajectory("no"))
    assert build_preference_pairs([good, bad]) == []


def test_reward_records_score_the_fraction_of_passed_checks_and_criteria():
    trace = make_trace("a", checks={"x": True, "y": False},
                       verdicts={"c1": {"verdict": "pass"}, "c2": {"verdict": "na"}})
    [record] = build_reward_records([trace])
    assert record["trace_id"] == "a"
    assert record["reward"] == 2 / 3  # x, c1 pass; y fails; na is ignored
    assert record["passed_all"] is False
