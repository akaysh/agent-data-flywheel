import pytest

from llm.factory import build_chat_model


def test_roles_read_their_own_model_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OFFLINE", "0")
    monkeypatch.setenv("AGENT_MODEL", "small")
    monkeypatch.setenv("FIXED_AGENT_MODEL", "big")
    monkeypatch.setenv("JUDGE_MODEL", "judge")
    monkeypatch.setenv("LABELER_MODEL", "labeler")
    monkeypatch.delenv("SIM_MODEL", raising=False)
    names = {r: build_chat_model(r).model_name
             for r in ("agent", "fixed_agent", "simulator", "judge", "labeler")}
    assert names == {"agent": "small", "fixed_agent": "big", "simulator": "small",
                     "judge": "judge", "labeler": "labeler"}


def test_simulator_model_can_be_set_separately(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_MODEL", "small")
    monkeypatch.setenv("SIM_MODEL", "sim")
    assert build_chat_model("simulator").model_name == "sim"


def test_missing_model_without_offline_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OFFLINE", "0")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="AGENT_MODEL"):
        build_chat_model("agent")


def test_recording_saves_role_models_and_offline_reads_them_back(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OFFLINE", "0")
    monkeypatch.setenv("AGENT_MODEL", "small-recorded")
    monkeypatch.setenv("JUDGE_MODEL", "judge-recorded")
    build_chat_model("agent")
    build_chat_model("judge")

    monkeypatch.setenv("OFFLINE", "1")
    monkeypatch.delenv("AGENT_MODEL")
    monkeypatch.delenv("JUDGE_MODEL")
    assert build_chat_model("agent").model_name == "small-recorded"
    assert build_chat_model("judge").model_name == "judge-recorded"


@pytest.mark.parametrize("model, provider", [
    ("aigateway.claude-4.6-sonnet", "anthropic"),
    ("gemini-2.0-flash", "google"),
    ("llama-3.3-70b", "meta"),
    ("gpt-4o-mini", "openai"),
    ("something-unknown", "openai"),
])
def test_traces_report_the_models_real_provider(monkeypatch, tmp_path, model, provider):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_MODEL", model)
    ls_params = build_chat_model("agent")._get_ls_params()
    assert (ls_params["ls_provider"], ls_params["ls_model_name"]) == (provider, model)


def test_offline_without_a_recording_falls_back_to_a_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OFFLINE", "1")
    for name in ("AGENT_MODEL", "SIM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert build_chat_model("agent").model_name == "offline-agent"


def test_temperature_defaults_to_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_MODEL", "m")
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    assert build_chat_model("agent").temperature == 0


@pytest.mark.parametrize("raw, expected", [("none", None), ("None", None), ("0.3", 0.3)])
def test_temperature_can_be_overridden_or_omitted_for_reasoning_models(monkeypatch, tmp_path, raw, expected):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_MODEL", "m")
    monkeypatch.setenv("LLM_TEMPERATURE", raw)
    assert build_chat_model("agent").temperature == expected
