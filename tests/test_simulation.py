import json

from langchain_core.messages import AIMessage

from sim.conversation import run_conversation
from sim.user_sim import END_TOKEN, UserSimulator
from tests.conftest import ScriptedChatModel

PERSONA = {"goal": "Get a refund.", "hidden_facts": ["It was ride R2."], "behavior": "Polite."}


class EchoAgent:
    def __init__(self):
        self.heard = []

    async def reply(self, message: str) -> str:
        self.heard.append(message)
        return f"agent heard: {message}"


def simulator(replies, max_turns=6):
    llm = ScriptedChatModel(messages=iter([AIMessage(content=r) for r in replies]))
    return UserSimulator(llm, PERSONA, opening_message="Refund me", max_turns=max_turns)


async def test_conversation_starts_with_opening_message_and_ends_on_end_token(tmp_path):
    agent = EchoAgent()
    transcript = await run_conversation(agent, simulator(["It was R2.", END_TOKEN]),
                                        tmp_path / "transcript.jsonl")
    assert agent.heard == ["Refund me", "It was R2."]
    assert [t["role"] for t in transcript] == ["user", "assistant", "user", "assistant"]


async def test_transcript_is_written_as_jsonl(tmp_path):
    path = tmp_path / "transcript.jsonl"
    await run_conversation(EchoAgent(), simulator([END_TOKEN]), path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [{"role": "user", "content": "Refund me"},
                     {"role": "assistant", "content": "agent heard: Refund me"}]


async def test_conversation_stops_at_max_turns(tmp_path):
    agent = EchoAgent()
    await run_conversation(agent, simulator(["a", "b", "c", "d"], max_turns=3),
                           tmp_path / "t.jsonl")
    assert len(agent.heard) == 3


async def test_end_token_with_extra_text_still_ends(tmp_path):
    agent = EchoAgent()
    await run_conversation(agent, simulator([f"thanks! {END_TOKEN}"]), tmp_path / "t.jsonl")
    assert len(agent.heard) == 1


async def test_simulator_prompt_contains_hidden_facts_but_agent_never_sees_them(tmp_path):
    seen = {}

    class Spy(ScriptedChatModel):
        async def _agenerate(self, messages, *args, **kwargs):
            seen["prompt"] = " ".join(str(m.content) for m in messages)
            return await super()._agenerate(messages, *args, **kwargs)

    sim = UserSimulator(Spy(messages=iter([AIMessage(content=END_TOKEN)])), PERSONA,
                        opening_message="Refund me")
    agent = EchoAgent()
    await run_conversation(agent, sim, tmp_path / "t.jsonl")
    assert "It was ride R2." in seen["prompt"]
    assert all("R2" not in h for h in agent.heard)
