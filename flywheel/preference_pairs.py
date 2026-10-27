"""Judged traces become post-training data.

build_preference_pairs -> TRL DPO conversational format:
    {"prompt": [msgs], "chosen": [msgs], "rejected": [msgs]}
build_reward_records   -> one scalar reward per trace for reward modelling / RL.

A pair compares what two agents did from the same shared conversation prefix, up to
the customer's next message. That is the unit a policy can actually be trained on.
"""
import json
from collections import defaultdict
from pathlib import Path

from flywheel.regressions import failed_criteria
from flywheel.traces import Trace

IGNORED_VERDICTS = ("na", "invalid")
USER_ROLE = "user"


def is_good(trace: Trace) -> bool:
    failed_checks, failed_judge = failed_criteria(trace)
    return trace.reward == 1 and not failed_checks and not failed_judge


def build_preference_pairs(traces: list[Trace]) -> list[dict]:
    by_task: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        by_task[trace.task_id].append(trace)

    pairs = []
    for task_traces in by_task.values():
        good = [t for t in task_traces if is_good(t)]
        bad = [t for t in task_traces if not is_good(t)]
        for i, rejected in enumerate(bad if good else []):
            pair = _make_pair(good[i % len(good)], rejected)
            if pair:
                pairs.append(pair)
    return pairs


def build_reward_records(traces: list[Trace]) -> list[dict]:
    return [_reward_record(t) for t in traces]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records))


def _make_pair(chosen: Trace, rejected: Trace) -> dict | None:
    split = _shared_prefix_length(chosen.trajectory, rejected.trajectory)
    chosen_reply = _until_next_user_turn(chosen.trajectory[split:])
    rejected_reply = _until_next_user_turn(rejected.trajectory[split:])
    if not split or not chosen_reply or chosen_reply == rejected_reply:
        return None
    return {"prompt": chosen.trajectory[:split], "chosen": chosen_reply,
            "rejected": rejected_reply, "task_id": chosen.task_id,
            "chosen_trace_id": chosen.trace_id, "rejected_trace_id": rejected.trace_id}


def _shared_prefix_length(a: list[dict], b: list[dict]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _until_next_user_turn(messages: list[dict]) -> list[dict]:
    reply = []
    for message in messages:
        if message["role"] == USER_ROLE:
            break
        reply.append(message)
    return reply


def _reward_record(trace: Trace) -> dict:
    outcomes = list(trace.checks.values()) + [
        v["verdict"] == "pass" for v in trace.verdicts.values()
        if v["verdict"] not in IGNORED_VERDICTS]
    return {"trace_id": trace.trace_id, "task_id": trace.task_id,
            "reward": sum(outcomes) / len(outcomes) if outcomes else float(trace.reward),
            "passed_all": is_good(trace), "checks": trace.checks,
            "verdicts": {k: v["verdict"] for k, v in trace.verdicts.items()},
            "messages": trace.trajectory}
