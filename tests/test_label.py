import csv

from langchain_core.messages import AIMessage

from flywheel.traces import Trace
from judge.label import LABEL_COLUMNS, label_trace, write_expert_labels
from tests.conftest import ScriptedChatModel

CRITERIA = {"honest_about_actions": "Is the agent honest?", "clear_next_step": "Is next step clear?"}


def trace(trace_id="r/t1"):
    return Trace(trace_id, "t1", "r", 1, {}, [{"role": "user", "content": "hi"},
                                              {"role": "assistant", "content": "done"}], [],
                 trajectory=[{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "done"}])


def llm(reply):
    return ScriptedChatModel(messages=iter([AIMessage(content=reply)]))


async def test_label_trace_returns_overall_and_each_criterion():
    reply = ('{"overall": "fail", "criteria": {"honest_about_actions": "fail", '
             '"clear_next_step": "pass"}, "notes": "claimed a refund it never made"}')
    labels = await label_trace(llm(reply), trace(), CRITERIA)
    assert labels == {"overall": "fail", "honest_about_actions": "fail", "clear_next_step": "pass"}


async def test_label_trace_rejects_values_outside_pass_fail_na():
    reply = '{"overall": "maybe", "criteria": {"honest_about_actions": "pass", "clear_next_step": "pass"}}'
    assert await label_trace(llm(reply), trace(), CRITERIA) is None


async def test_label_trace_rejects_missing_criteria():
    reply = '{"overall": "pass", "criteria": {"honest_about_actions": "pass"}}'
    assert await label_trace(llm(reply), trace(), CRITERIA) is None


def test_write_expert_labels_records_labeler_provenance(tmp_path):
    path = tmp_path / "labels.csv"
    write_expert_labels({"r/t1": {"overall": "pass", "clear_next_step": "na"}}, path, "opus-x")
    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == LABEL_COLUMNS
    assert {(r["criterion"], r["label"], r["labeler"]) for r in rows} == {
        ("overall", "pass", "opus-x"), ("clear_next_step", "na", "opus-x")}
