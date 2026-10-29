"""Other training formats built from judged traces, beside the DPO pairs.

SFT   good traces only, as {"messages": [...]} (TRL's conversational SFT format).
RLVR  a reward from the deterministic code checks alone. The judge is left out on purpose,
      because a judge can be argued with and a policy will learn to argue.
RLHF  needs no new file. A reward-model trainer reads the same prompt/chosen/rejected
      pairs as DPO (see preference_pairs.py).
"""
from flywheel.preference_pairs import is_good
from flywheel.traces import Trace


def build_sft_examples(traces: list[Trace], system_prompt: str | None = None) -> list[dict]:
    """The traces store the conversation without the agent's system prompt, so pass it in."""
    prefix = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return [{"messages": prefix + t.trajectory, "task_id": t.task_id, "trace_id": t.trace_id}
            for t in traces if t.trajectory and is_good(t)]


def build_verifiable_reward_records(traces: list[Trace]) -> list[dict]:
    return [_verifiable_reward_record(t) for t in traces]


def _verifiable_reward_record(trace: Trace) -> dict:
    passed = list(trace.checks.values())
    reward = sum(passed) / len(passed) if passed else float(trace.reward)
    return {"trace_id": trace.trace_id, "task_id": trace.task_id, "reward": reward,
            "checks": trace.checks, "messages": trace.trajectory}
