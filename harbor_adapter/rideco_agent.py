"""Harbor agent adapter: runs the RideCo support agent + simulated customer in the task container.

    harbor run -p tasks --agent-import-path harbor_adapter.rideco_agent:RideCoAgent

The agent loop runs inside the container next to /app, so the mock tools see the same
files the verifier reads. LLM settings are forwarded from the host environment.
"""
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_CODE_DIR = "/opt/rideco"
REMOTE_INSTRUCTION_PATH = "/opt/rideco-instruction.md"
CODE_PACKAGES = ("agent", "harbor_adapter", "llm", "runner", "sim", "tools")
FORWARDED_ENV = ("LLM_BASE_URL", "LLM_API_KEY", "AGENT_MODEL", "FIXED_AGENT_MODEL", "SIM_MODEL",
                 "JUDGE_MODEL", "OFFLINE", "PROMPT_VERSION", "AGENT_ROLE", "MAX_TURNS")
REMOTE_CACHE_DIR = f"{REMOTE_CODE_DIR}/cache"
INSTALL_TIMEOUT_SEC = 600
RUN_TIMEOUT_SEC = 280


class RideCoAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "rideco-support"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(f"mkdir -p {REMOTE_CODE_DIR}")
        for package in CODE_PACKAGES:
            await environment.upload_dir(REPO_ROOT / package, f"{REMOTE_CODE_DIR}/{package}")
        await environment.upload_file(REPO_ROOT / "requirements-agent.txt",
                                      f"{REMOTE_CODE_DIR}/requirements-agent.txt")
        if (REPO_ROOT / "cache").exists():
            await environment.upload_dir(REPO_ROOT / "cache", REMOTE_CACHE_DIR)
        install = await environment.exec(
            f"pip install --quiet -r {REMOTE_CODE_DIR}/requirements-agent.txt",
            timeout_sec=INSTALL_TIMEOUT_SEC)
        if install.return_code != 0:
            raise RuntimeError(f"agent dependency install failed: {install.stderr}")

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        instruction_file = self.logs_dir / "instruction.md"
        instruction_file.write_text(instruction)
        await environment.upload_file(instruction_file, REMOTE_INSTRUCTION_PATH)

        result = await environment.exec(
            f"cd {REMOTE_CODE_DIR} && python -m harbor_adapter.container_main",
            env=self._container_env(), timeout_sec=RUN_TIMEOUT_SEC)
        (self.logs_dir / "rollout.log").write_text(f"{result.stdout}\n{result.stderr}")
        if result.return_code != 0:
            raise RuntimeError(f"rollout failed:\n{result.stderr}")

    def _container_env(self) -> dict[str, str]:
        env = {name: value for name in FORWARDED_ENV if (value := self._get_env(name))}
        env.update(INSTRUCTION_PATH=REMOTE_INSTRUCTION_PATH, CACHE_DIR=REMOTE_CACHE_DIR,
                   PYTHONPATH=REMOTE_CODE_DIR)
        return env
