import json
from pathlib import Path
from typing import Protocol

from sim.user_sim import UserSimulator


class Replier(Protocol):
    async def reply(self, customer_message: str) -> str: ...


async def run_conversation(agent: Replier, simulator: UserSimulator,
                           transcript_path: str | Path) -> list[dict]:
    """Alternates customer and agent until the simulator stops. Writes transcript.jsonl."""
    transcript: list[dict] = []
    agent_reply = None
    while (customer_message := await simulator.next_message(agent_reply)) is not None:
        agent_reply = await agent.reply(customer_message)
        transcript.append({"role": "user", "content": customer_message})
        transcript.append({"role": "assistant", "content": agent_reply})
    Path(transcript_path).write_text("".join(json.dumps(t) + "\n" for t in transcript))
    return transcript
