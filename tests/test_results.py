from runner.results import results_rows
from tests.test_flywheel import make_trace

JUDGE_FAILS_HONESTY = {"honest_about_actions": {"verdict": "fail"},
                       "clear_next_step": {"verdict": "pass"}}


def write_task(tasks_dir, task_id, extra_metadata=""):
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(f'[metadata]\narea = "charges"\n{extra_metadata}\n[agent]\n')


def row_for(trace, tasks_dir):
    [row] = results_rows([trace], tasks_dir)
    return row


def test_without_judge_verdicts_the_overall_result_is_the_code_result(tmp_path):
    write_task(tmp_path, "t")
    assert row_for(make_trace("a", task_id="t", reward=1), tmp_path)["overall pass"] is True
    assert row_for(make_trace("b", task_id="t", reward=0), tmp_path)["overall pass"] is False


def test_a_judge_failure_fails_a_task_whose_code_checks_all_passed(tmp_path):
    write_task(tmp_path, "t")
    row = row_for(make_trace("a", task_id="t", reward=1, verdicts=JUDGE_FAILS_HONESTY), tmp_path)
    assert (row["pass"], row["judge failed"], row["overall pass"]) == (True, "honest_about_actions", False)


def test_na_and_invalid_verdicts_do_not_fail_a_task(tmp_path):
    write_task(tmp_path, "t")
    verdicts = {"a": {"verdict": "na"}, "b": {"verdict": "invalid"}, "c": {"verdict": "pass"}}
    assert row_for(make_trace("a", task_id="t", reward=1, verdicts=verdicts), tmp_path)["overall pass"]


def test_a_failed_code_check_still_fails_the_task_when_the_judge_passes_everything(tmp_path):
    write_task(tmp_path, "t")
    verdicts = {"a": {"verdict": "pass"}}
    assert not row_for(make_trace("a", task_id="t", reward=0, verdicts=verdicts), tmp_path)["overall pass"]


def test_a_regression_task_only_looks_at_the_criteria_it_lists(tmp_path):
    write_task(tmp_path, "regression-t", 'judge_criteria = ["clear_next_step"]')
    row = row_for(make_trace("a", task_id="regression-t", reward=1, verdicts=JUDGE_FAILS_HONESTY),
                  tmp_path)
    assert (row["judge failed"], row["overall pass"]) == ("", True)


def test_a_regression_task_fails_when_a_listed_criterion_fails(tmp_path):
    write_task(tmp_path, "regression-t", 'judge_criteria = ["honest_about_actions"]')
    row = row_for(make_trace("a", task_id="regression-t", reward=1, verdicts=JUDGE_FAILS_HONESTY),
                  tmp_path)
    assert (row["judge failed"], row["overall pass"]) == ("honest_about_actions", False)


def test_a_task_with_no_folder_is_judged_on_every_criterion(tmp_path):
    row = row_for(make_trace("a", task_id="gone", reward=1, verdicts=JUDGE_FAILS_HONESTY), tmp_path)
    assert row["overall pass"] is False


def test_row_lists_the_failed_code_checks():
    trace = make_trace("a", reward=0, checks={"x": True, "must_not_call:issue_refund": False})
    [row] = results_rows([trace], "no-such-dir")
    assert row["failed checks"] == "must_not_call:issue_refund"
