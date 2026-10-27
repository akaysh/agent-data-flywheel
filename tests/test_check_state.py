from scripts.check_state import run_checks


def test_must_call_any_passes_when_one_alternative_was_called():
    checks = {"must_call_any": [["get_charges", "get_ride"]]}
    assert run_checks(checks, [{"tool": "get_ride", "args": {}}], [])["must_call_any:get_charges|get_ride"]


def test_must_call_any_fails_when_none_were_called():
    checks = {"must_call_any": [["get_charges", "get_ride"]]}
    assert not run_checks(checks, [{"tool": "issue_refund", "args": {}}], [])[
        "must_call_any:get_charges|get_ride"]
