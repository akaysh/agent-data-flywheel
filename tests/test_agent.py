import json

import pytest
from langchain_core.messages import AIMessage

from agent.mcp_tools import open_tools
from agent.graph import SupportAgent
from tests.conftest import ScriptedChatModel

FIXTURES = {
    "rides": [{"ride_id": "R1", "date": "2026-11-01", "pickup": "A", "dropoff": "B", "fare": 12.0}],
    "charges": [{"charge_id": "C1", "ride_id": "R1", "type": "airport_fee", "amount": 12.0,
                 "status": "settled"}],
    "account": {"email": "a@example.com", "phone": "+1-555-0100", "flags": []},
}


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def app_dir(tmp_path):
    (tmp_path / "fixtures.json").write_text(json.dumps(FIXTURES))
    return tmp_path


def calls(app_dir):
    path = app_dir / "tool_calls.jsonl"
    return [json.loads(line)["tool"] for line in path.read_text().splitlines()]


async def test_router_picks_agent_and_agent_calls_tools_then_replies(app_dir):
    llm = ScriptedChatModel(messages=iter([
        AIMessage(content='{"agents": ["charges"]}'),
        tool_call("get_charges", {"ride_id": "R1"}),
        tool_call("issue_refund", {"ride_id": "R1", "charge_id": "C1", "amount": 12.0}, "call_2"),
        AIMessage(content="Refunded your $12 airport fee."),
    ]))
    async with open_tools(app_dir) as tools:
        agent = SupportAgent(llm, tools)
        reply = await agent.reply("airport fee please refund")
    assert reply == "Refunded your $12 airport fee."
    assert calls(app_dir) == ["get_charges", "issue_refund"]


async def test_agent_only_gets_tools_for_its_area(app_dir):
    seen = {}

    class Spy(ScriptedChatModel):
        def bind_tools(self, tools, **kwargs):
            seen["names"] = {t.name for t in tools}
            return self

    llm = Spy(messages=iter([
        AIMessage(content='{"agents": ["account"]}'), AIMessage(content="ok")]))
    async with open_tools(app_dir) as tools:
        await SupportAgent(llm, tools).reply("change email")
    assert seen["names"] == {"get_account", "update_account", "escalate_to_human"}


async def test_v1_router_uses_only_first_agent(app_dir):
    llm = ScriptedChatModel(messages=iter([
        AIMessage(content='{"agents": ["lost_item", "charges"]}'),
        AIMessage(content="Opened a ticket.")]))
    async with open_tools(app_dir) as tools:
        reply = await SupportAgent(llm, tools, prompt_version="v1").reply("lost phone and a cleaning fee")
    assert reply == "Opened a ticket."


async def test_v2_router_hands_off_through_every_agent_in_order(app_dir):
    llm = ScriptedChatModel(messages=iter([
        AIMessage(content='{"agents": ["lost_item", "charges"]}'),
        tool_call("create_lost_item_ticket", {"ride_id": "R1", "item": "phone"}),
        AIMessage(content="ticket done"),                 # internal handoff note
        tool_call("get_charges", {"ride_id": "R1"}),
        AIMessage(content="Ticket opened and fee reviewed."),
    ]))
    async with open_tools(app_dir) as tools:
        reply = await SupportAgent(llm, tools, prompt_version="v2").reply("lost phone and a cleaning fee")
    assert reply == "Ticket opened and fee reviewed."
    assert calls(app_dir) == ["create_lost_item_ticket", "get_charges"]


async def test_unparseable_router_output_falls_back_to_charges(app_dir):
    llm = ScriptedChatModel(messages=iter([AIMessage(content="no idea"), AIMessage(content="hi")]))
    async with open_tools(app_dir) as tools:
        reply = await SupportAgent(llm, tools).reply("hello")
    assert reply == "hi"


async def test_tool_loop_stops_after_max_steps(app_dir):
    endless = (tool_call("list_recent_rides", {}, f"c{i}") for i in range(100))
    llm = ScriptedChatModel(messages=iter([AIMessage(content='{"agents": ["charges"]}'), *endless]))
    async with open_tools(app_dir) as tools:
        reply = await SupportAgent(llm, tools, max_tool_steps=3).reply("hi")
    assert len(calls(app_dir)) == 3
    assert reply


async def test_router_runs_on_the_first_turn_only_and_the_active_agent_keeps_the_conversation(app_dir):
    llm = ScriptedChatModel(messages=iter([
        AIMessage(content='{"agents": ["lost_item"]}'),
        AIMessage(content="Which item?"),
        AIMessage(content="Got it, ticket noted."),   # no router call before this
    ]))
    async with open_tools(app_dir) as tools:
        agent = SupportAgent(llm, tools)
        await agent.reply("I lost something")
        reply = await agent.reply("my phone, and also there is a cleaning fee")
    assert reply == "Got it, ticket noted."


async def test_non_final_agent_is_told_not_to_talk_to_the_customer(app_dir):
    prompts = []

    class Spy(ScriptedChatModel):
        async def _agenerate(self, messages, *a, **k):
            prompts.append(messages[0].content)
            return await super()._agenerate(messages, *a, **k)

    llm = Spy(messages=iter([
        AIMessage(content='{"agents": ["lost_item", "charges"]}'),
        AIMessage(content="ticket opened"), AIMessage(content="All done.")]))
    async with open_tools(app_dir) as tools:
        await SupportAgent(llm, tools, prompt_version="v2").reply("lost phone and a fee")
    assert "do not ask the customer" in prompts[1].lower()   # lost_item: not last
    assert "do not ask the customer" not in prompts[2].lower()  # charges: last


async def test_empty_model_reply_falls_back_instead_of_going_silent(app_dir):
    llm = ScriptedChatModel(messages=iter([
        AIMessage(content='{"agents": ["charges"]}'), AIMessage(content="")]))
    async with open_tools(app_dir) as tools:
        reply = await SupportAgent(llm, tools).reply("hi")
    assert reply.strip()


async def test_handoff_call_never_ends_on_an_assistant_message(app_dir):
    last_roles = []

    class Spy(ScriptedChatModel):
        async def _agenerate(self, messages, *a, **k):
            last_roles.append(messages[-1].type)
            return await super()._agenerate(messages, *a, **k)

    llm = Spy(messages=iter([
        AIMessage(content='{"agents": ["lost_item", "charges"]}'),
        AIMessage(content="ticket opened"), AIMessage(content="All done.")]))
    async with open_tools(app_dir) as tools:
        agent = SupportAgent(llm, tools, prompt_version="v2")
        await agent.reply("lost phone and a fee")
    assert last_roles == ["human", "human", "human"]
    assert not any("handoff" in str(m.content).lower() for m in agent.messages if m.type == "human")
