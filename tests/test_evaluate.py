from pathlib import Path

from langchain_core.messages import AIMessage

from runner.evaluate import evaluate
from sim.user_sim import END_TOKEN
from tests.conftest import ScriptedChatModel

REPO = Path(__file__).resolve().parent.parent


def scripted(*contents):
    return ScriptedChatModel(messages=iter([AIMessage(content=c) for c in contents]))


async def test_evaluate_runs_tasks_and_attaches_judge_verdicts(tmp_path):
    only_task = tmp_path / "tasks"
    only_task.mkdir()
    import shutil
    shutil.copytree(REPO / "tasks" / "fare-explain-basic", only_task / "fare-explain-basic")

    agent_llm = scripted('{"agents": ["charges"]}', "Prime Time pricing raised your fare.")
    sim_llm = scripted(END_TOKEN)
    judge_llm = scripted(*['{"evidence": [], "verdict": "pass", "reason": "ok"}'] * 5)

    traces = await evaluate(only_task, run_label="r1", work_root=tmp_path / "work",
                            agent_llm=agent_llm, sim_llm=sim_llm, judge_llm=judge_llm,
                            judge_rubric_path=REPO / "judge" / "rubric_good.yaml")

    [trace] = traces
    assert trace.trace_id == "r1/fare-explain-basic"
    assert set(trace.verdicts) == {"honest_about_actions", "addressed_every_issue",
                                   "verified_before_acting", "protected_private_info",
                                   "clear_next_step"}


async def test_evaluate_without_judge_leaves_verdicts_empty(tmp_path):
    import shutil
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    shutil.copytree(REPO / "tasks" / "fare-explain-basic", tasks / "fare-explain-basic")
    traces = await evaluate(tasks, run_label="r", work_root=tmp_path / "w",
                            agent_llm=scripted('{"agents": ["charges"]}', "hi"),
                            sim_llm=scripted(END_TOKEN))
    assert traces[0].verdicts == {}


async def test_judge_traces_returns_copies_with_verdicts():
    from flywheel.traces import Trace
    from runner.evaluate import judge_traces
    trace = Trace("r/x", "x", "r", 1, {}, [{"role": "user", "content": "hi"},
                                           {"role": "assistant", "content": "yo"}], [])
    judge_llm = scripted(*['{"evidence": [], "verdict": "fail", "reason": "r"}'] * 5)
    [judged] = await judge_traces([trace], judge_llm, REPO / "judge" / "rubric_good.yaml")
    assert judged.verdicts["clear_next_step"]["verdict"] == "fail"
    assert trace.verdicts == {}
