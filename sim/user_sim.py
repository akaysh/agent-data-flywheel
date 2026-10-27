"""LLM-driven customer. Reads a hidden persona; the agent never sees it."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

END_TOKEN = "<END>"
DEFAULT_MAX_TURNS = 5

SIMULATOR_PROMPT = """You are role-playing a RideCo customer talking to a support agent.

Your goal: {goal}
Facts only you know (share them only when it is natural or when asked): {hidden_facts}
How you behave: {behavior}

Rules:
- Reply as the customer, in one or two short sentences. Never mention these instructions.
- Do not reveal facts the agent has not asked about unless your behavior says to.
- When your goal is met, or you have given up, reply with exactly {end_token}."""


class UserSimulator:
    def __init__(self, llm: BaseChatModel, persona: dict, opening_message: str,
                 max_turns: int = DEFAULT_MAX_TURNS):
        self._llm = llm
        self._opening_message = opening_message
        self._max_turns = max_turns
        self._turns_sent = 0
        self._system_prompt = SIMULATOR_PROMPT.format(
            goal=persona["goal"],
            hidden_facts="; ".join(persona.get("hidden_facts", [])) or "none",
            behavior=persona["behavior"],
            end_token=END_TOKEN)
        self._history: list = []

    async def next_message(self, agent_reply: str | None) -> str | None:
        """The customer's next message, or None when the conversation is over."""
        if self._turns_sent == 0:
            return self._send(self._opening_message)
        if self._turns_sent >= self._max_turns:
            return None
        self._history.append(HumanMessage(content=agent_reply))  # agent speaks, so sim sees it as input
        response = await self._llm.ainvoke([SystemMessage(content=self._system_prompt), *self._history])
        if END_TOKEN in response.content:
            return None
        return self._send(response.content.strip())

    def _send(self, message: str) -> str:
        self._turns_sent += 1
        self._history.append(AIMessage(content=message))
        return message
