import pytest

from harbor_adapter.container_main import agent_role_from_env


def test_agent_role_defaults_to_the_shipped_agent(monkeypatch):
    monkeypatch.delenv("AGENT_ROLE", raising=False)
    assert agent_role_from_env() == "agent"


def test_agent_role_can_select_the_stronger_model(monkeypatch):
    monkeypatch.setenv("AGENT_ROLE", "fixed_agent")
    assert agent_role_from_env() == "fixed_agent"


def test_unknown_agent_role_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_ROLE", "judge")
    with pytest.raises(ValueError, match="AGENT_ROLE"):
        agent_role_from_env()
