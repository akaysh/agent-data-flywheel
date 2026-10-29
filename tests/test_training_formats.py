from flywheel.training_formats import build_sft_examples, build_verifiable_reward_records
from tests.test_flywheel import make_trace, trajectory

PASS = {"honest": {"verdict": "pass"}}
FAIL = {"honest": {"verdict": "fail"}}


def test_sft_keeps_only_traces_that_passed_every_check_and_judge_criterion():
    good = make_trace("good", reward=1, checks={"c": True}, verdicts=PASS, trajectory=trajectory("ok"))
    failed_check = make_trace("bad-check", reward=0, checks={"c": False}, trajectory=trajectory("no"))
    failed_judge = make_trace("bad-judge", reward=1, checks={"c": True}, verdicts=FAIL,
                              trajectory=trajectory("no"))
    examples = build_sft_examples([good, failed_check, failed_judge])
    assert [e["trace_id"] for e in examples] == ["good"]


def test_sft_example_is_the_whole_conversation_in_chat_format():
    good = make_trace("good", reward=1, trajectory=trajectory("ok"))
    [example] = build_sft_examples([good])
    assert example["messages"] == good.trajectory
    assert example["task_id"] == "refund-over-limit"


def test_sft_can_prepend_the_system_prompt_the_agent_ran_with():
    good = make_trace("good", reward=1, trajectory=trajectory("ok"))
    [example] = build_sft_examples([good], system_prompt="You are RideCo support.")
    assert example["messages"][0] == {"role": "system", "content": "You are RideCo support."}
    assert example["messages"][1:] == good.trajectory


def test_sft_skips_traces_with_no_conversation():
    assert build_sft_examples([make_trace("empty", reward=1)]) == []


def test_verifiable_reward_is_the_share_of_code_checks_passed():
    trace = make_trace("a", checks={"x": True, "y": False, "z": False, "w": True})
    [record] = build_verifiable_reward_records([trace])
    assert record["reward"] == 0.5


def test_verifiable_reward_ignores_the_judge():
    trace = make_trace("a", checks={"x": True}, verdicts=FAIL)
    [record] = build_verifiable_reward_records([trace])
    assert record["reward"] == 1.0


def test_verifiable_reward_falls_back_to_the_verifier_reward_without_checks():
    [record] = build_verifiable_reward_records([make_trace("a", reward=1)])
    assert record["reward"] == 1.0
