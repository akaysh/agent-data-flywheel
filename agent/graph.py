"""RideCo support agent: a router plus three subagents, wired with LangGraph.

The router picks which subagent(s) act. Each subagent runs a tool loop and,
when more than one acts (router v2), later agents see what earlier ones did.
The agent is stateful across customer turns: one instance per conversation.
"""
import json
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage)
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agent.prompts import (
    AGENT_NAMES, DEFAULT_PROMPT_VERSION, HANDOFF_NUDGE, HANDOFF_SUFFIX, NOT_LAST_SUFFIX, ROUTER_PROMPTS, SUBAGENT_PROMPTS)

AREA_TOOLS = {
    "charges": ("list_recent_rides", "get_ride", "get_charges", "issue_refund", "escalate_to_human"),
    "lost_item": ("list_recent_rides", "get_ride", "create_lost_item_ticket", "contact_driver",
                  "escalate_to_human"),
    "account": ("get_account", "update_account", "escalate_to_human"),
}
FALLBACK_AGENT = "charges"
DEFAULT_MAX_TOOL_STEPS = 8
EMPTY_REPLY = "Sorry, something went wrong on our side. A teammate will follow up with you."
STEP_LIMIT_REPLY = "Sorry, I couldn't finish this. A teammate will follow up with you."
ROUTER_V1_MAX_AGENTS = 1


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    handled: list[str]


class SupportAgent:
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool],
                 prompt_version: str = DEFAULT_PROMPT_VERSION,
                 max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS):
        self._llm = llm
        self._tools_by_name = {t.name: t for t in tools}
        self._prompt_version = prompt_version
        self._max_tool_steps = max_tool_steps
        self._messages: list[BaseMessage] = []
        self._active_agent: str | None = None  # the router only runs on the first turn
        self._graph = self._build_graph()

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    async def reply(self, customer_message: str) -> str:
        plan = [self._active_agent] if self._active_agent else []
        state = {"messages": [*self._messages, HumanMessage(content=customer_message)],
                 "plan": plan, "handled": []}
        result = await self._graph.ainvoke(state)
        self._messages = result["messages"]
        self._active_agent = result["handled"][-1]
        return self._messages[-1].content

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._route)
        graph.set_entry_point("router")
        for name in AGENT_NAMES:
            graph.add_node(name, self._make_agent_node(name))
            graph.add_conditional_edges(name, self._next_step, [*AGENT_NAMES, END])
        graph.add_conditional_edges("router", self._next_step, [*AGENT_NAMES, END])
        return graph.compile()

    async def _route(self, state: AgentState) -> dict:
        if state["plan"]:  # an agent already owns this conversation
            return {}
        prompt = SystemMessage(content=ROUTER_PROMPTS[self._prompt_version])
        decision = await self._llm.ainvoke([prompt, *state["messages"]])
        agents = parse_agent_names(decision.content)
        if self._prompt_version == "v1":
            agents = agents[:ROUTER_V1_MAX_AGENTS]
        return {"plan": agents}

    @staticmethod
    def _next_step(state: AgentState) -> str:
        return state["plan"][0] if state["plan"] else END

    def _make_agent_node(self, name: str):
        async def run(state: AgentState) -> dict:
            prompt = SUBAGENT_PROMPTS[self._prompt_version][name]
            if state["handled"]:
                prompt += HANDOFF_SUFFIX
            if len(state["plan"]) > 1:
                prompt += NOT_LAST_SUFFIX
            new_messages = await self._run_tool_loop(name, prompt, state["messages"])
            return {"messages": new_messages, "plan": state["plan"][1:],
                    "handled": [*state["handled"], name]}
        return run

    async def _run_tool_loop(self, name: str, prompt: str, history: list[BaseMessage]) -> list:
        tools = [self._tools_by_name[t] for t in AREA_TOOLS[name]]
        llm = self._llm.bind_tools(tools)
        new: list[BaseMessage] = []
        context = list(history)
        if isinstance(context[-1], AIMessage):  # some models reject a trailing assistant message
            context.append(HumanMessage(content=HANDOFF_NUDGE))
        for _ in range(self._max_tool_steps):
            ai = await llm.ainvoke([SystemMessage(content=prompt), *context, *new])
            new.append(ai)
            if not ai.tool_calls:
                if not _has_text(ai):
                    new[-1] = AIMessage(content=EMPTY_REPLY)
                return new
            new.extend([await self._run_tool_call(call) for call in ai.tool_calls])
        new.append(AIMessage(content=STEP_LIMIT_REPLY))
        return new

    async def _run_tool_call(self, call: dict) -> ToolMessage:
        tool = self._tools_by_name.get(call["name"])
        if tool is None:
            return ToolMessage(content=f"unknown tool {call['name']}", tool_call_id=call["id"])
        try:
            output = await tool.ainvoke(call["args"])
        except Exception as e:  # tool errors go back to the model, like in prod
            output = f"tool error: {e}"
        return ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"])


def _has_text(message: BaseMessage) -> bool:
    return bool(str(message.content).strip())


def parse_agent_names(text: str) -> list[str]:
    """Agent names in order of first appearance; falls back to charges."""
    found = sorted(
        (m.start(), n) for n in AGENT_NAMES for m in [re.search(rf"\b{n}\b", text)] if m)
    names = [n for _, n in found]
    return names or [FALLBACK_AGENT]
