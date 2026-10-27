"""A trace = one rollout plus everything we learned about it (verifier + judge)."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from runner.run_tasks import TaskResult


@dataclass
class Trace:
    trace_id: str
    task_id: str
    run_label: str
    reward: int
    checks: dict[str, bool]
    transcript: list[dict]
    tool_calls: list[dict]
    trajectory: list[dict] = field(default_factory=list)
    verdicts: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: TaskResult, run_label: str,
                    verdicts: dict[str, dict] | None = None) -> "Trace":
        return cls(trace_id=f"{run_label}/{result.task_id}", task_id=result.task_id,
                   run_label=run_label, reward=result.reward, checks=result.checks,
                   transcript=result.transcript, tool_calls=result.tool_calls,
                   trajectory=result.trajectory, verdicts=verdicts or {})


def save_traces(traces: list[Trace], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(asdict(t)) + "\n" for t in traces))


def load_traces(path: str | Path) -> list[Trace]:
    lines = Path(path).read_text().splitlines()
    return [Trace(**json.loads(line)) for line in lines if line.strip()]
