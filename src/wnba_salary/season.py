"""Single source of truth for the production season."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "model_config.json"


def current_season() -> int:
    payload = json.loads(CONFIG_PATH.read_text())
    season = int(payload["current_season"])
    if not 2000 <= season <= 2100:
        raise ValueError(f"invalid current_season in {CONFIG_PATH}: {season}")
    return season


def set_current_season(value: int) -> None:
    """Persist a guarded rollover so later Actions runs use the new season."""
    value = int(value)
    if value != current_season() + 1:
        raise ValueError("season rollover must advance exactly one year")
    CONFIG_PATH.write_text(json.dumps({"current_season": value}, indent=2) + "\n")
