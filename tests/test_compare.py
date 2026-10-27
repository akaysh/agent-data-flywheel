import pytest
from langchain_core.messages import AIMessage

from flywheel.traces import Trace
from judge.compare import compare_rubrics, load_expert_labels, overall_from_criteria
from tests.conftest import ScriptedChatModel

CSV = """trace_id,criterion,label
t1,overall,fail
t1,honest_about_actions,fail
t2,overall,pass
t2,honest_about_actions,pass
t3,overall,fail
t3,honest_about_actions,fail
t4,overall,pass
t4,honest_about_actions,pass
"""


def trace(trace_id):
    return Trace(trace_id=trace_id, task_id="x", run_label="r", reward=1, checks={},
                 transcript=[{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "ok"}], tool_calls=[])


def test_load_expert_labels_keys_by_trace_and_criterion(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(CSV)
    labels = load_expert_labels(path)
    assert labels[("t1", "honest_about_actions")] == "fail"
    assert labels[("t2", "overall")] == "pass"


def test_unlabeled_rows_are_skipped(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("trace_id,criterion,label\nt1,overall,\nt1,x,pass\n")
    assert load_expert_labels(path) == {("t1", "x"): "pass"}


def test_overall_from_criteria_fails_if_any_criterion_fails_and_ignores_na():
    assert overall_from_criteria({"a": {"verdict": "pass"}, "b": {"verdict": "na"}}) == "pass"
    assert overall_from_criteria({"a": {"verdict": "pass"}, "b": {"verdict": "fail"}}) == "fail"


class RoutedJudge(ScriptedChatModel):
    """Answers vague-rubric prompts with a fixed score and good-rubric prompts by trace."""


async def test_compare_rubrics_scores_vague_and_good_against_experts(tmp_path):
    labels = {("t1", "overall"): "fail", ("t2", "overall"): "pass",
              ("t3", "overall"): "fail", ("t4", "overall"): "pass",
              ("t1", "honest_about_actions"): "fail", ("t2", "honest_about_actions"): "pass",
              ("t3", "honest_about_actions"): "fail", ("t4", "honest_about_actions"): "pass"}
    traces = [trace(f"t{i}") for i in range(1, 5)]
    vague_rubric = {"pass_threshold": 7, "prompt": "{transcript}"}
    good_rubric = {"criteria": [{"id": "honest_about_actions", "question": "q", "definition": "d",
                                 "pass_example": "p", "fail_example": "f"}],
                   "prompt": "{question}{definition}{pass_example}{fail_example}{tool_calls}{transcript}"}

    # vague judge says 8 to everything (the classic failure); good judge matches experts
    vague_llm = ScriptedChatModel(messages=iter(
        [AIMessage(content='{"score": 8, "reason": "fine"}')] * 4))
    good_verdicts = ["fail", "pass", "fail", "pass"]
    good_llm = ScriptedChatModel(messages=iter(
        [AIMessage(content=f'{{"evidence": [], "verdict": "{v}", "reason": "r"}}')
         for v in good_verdicts]))

    report = await compare_rubrics(vague_llm, good_llm, vague_rubric, good_rubric, traces, labels,
                                   concurrency=1)

    assert report["vague"]["overall"]["kappa"] == pytest.approx(0.0)
    assert report["good"]["overall"]["kappa"] == 1.0
    assert report["good"]["honest_about_actions"]["kappa"] == 1.0


@pytest.mark.filterwarnings("ignore:A single label:UserWarning")
@pytest.mark.filterwarnings("ignore:.*have only one label in common:sklearn.exceptions.UndefinedMetricWarning")
async def test_system_overall_also_requires_the_code_checks_to_pass(tmp_path):
    # honest_about_actions has one label on both sides here, so its kappa is undefined (nan)
    # by design. This test only checks the percent agreement of the overall scores.
    t_code_fail = trace("t1")
    t_code_fail.reward = 0
    t_ok = trace("t2")
    labels = {("t1", "overall"): "fail", ("t2", "overall"): "pass",
              ("t1", "honest_about_actions"): "pass", ("t2", "honest_about_actions"): "pass"}
    vague_rubric = {"pass_threshold": 7, "prompt": "{transcript}"}
    good_rubric = {"criteria": [{"id": "honest_about_actions", "question": "q", "definition": "d",
                                 "pass_example": "p", "fail_example": "f"}],
                   "prompt": "{question}{definition}{pass_example}{fail_example}{tool_calls}{transcript}"}
    vague_llm = ScriptedChatModel(messages=iter([AIMessage(content='{"score": 9, "reason": "x"}')] * 2))
    good_llm = ScriptedChatModel(messages=iter(
        [AIMessage(content='{"evidence": [], "verdict": "pass", "reason": "r"}')] * 2))

    report = await compare_rubrics(vague_llm, good_llm, vague_rubric, good_rubric,
                                   [t_code_fail, t_ok], labels, concurrency=1)

    assert report["good"]["overall"]["percent_agreement"] == 0.5      # judge alone misses t1
    assert report["good"]["system_overall"]["percent_agreement"] == 1.0  # code + judge catches it
