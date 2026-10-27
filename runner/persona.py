import json
from pathlib import Path

PERSONA_FILE = "persona.json"


def load_persona(app_dir: str | Path) -> dict:
    return json.loads((Path(app_dir) / PERSONA_FILE).read_text())
