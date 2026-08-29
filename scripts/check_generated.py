#!/usr/bin/env python
"""Fail when committed web data or model status does not match the artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wnba_salary import export_web, value_history  # noqa: E402


def main() -> None:
    payload = export_web.build_payload()
    expected = {
        export_web.WEB_DIR / "players.js": export_web.render_js(payload),
        export_web.WEB_DIR / "model-status.md": export_web.render_status(payload),
    }
    stale = []
    for path, content in expected.items():
        if not path.exists() or path.read_text() != content:
            stale.append(path.relative_to(ROOT))

    errors = []
    processed = export_web.data.PROCESSED_DIR
    desc = json.loads((processed / "ratings_meta.json").read_text())
    forecast = json.loads((processed / "ratings_forecast_meta.json").read_text())
    for key in ("n_poss", "n_players", "target_season"):
        if desc[key] != forecast[key]:
            errors.append(f"rating configurations disagree on {key}")

    constants = json.loads((processed / "constants.json").read_text())
    values = pd.read_parquet(processed / "valuation.parquet")
    expected_war = float(constants["dollars_per_win"]["league_war"])
    if abs(values["war"].sum() - expected_war) / expected_war > 0.01:
        errors.append("summed WAR is more than 1% from the league identity")

    history = pd.read_parquet(processed / "history.parquet")
    current = history[history["season"] == desc["target_season"]]
    agreement = current[["athlete_id", "rating", "value"]].merge(
        values[["athlete_id", "rating", "value"]], on="athlete_id",
        suffixes=("_history", "_valuation"), validate="one_to_one")
    if len(agreement) != len(values):
        errors.append("target-season history and valuation player sets differ")
    for column in ("rating", "value"):
        gap = (agreement[f"{column}_history"] - agreement[f"{column}_valuation"]).abs().max()
        if gap > 1e-6:
            errors.append(f"target-season history and valuation {column} differ")

    raw_box = (export_web.data.RAW_DIR / "player_box" /
               f"player_box_{desc['target_season']}.parquet")
    if raw_box.exists() and export_web.raw_data_through() != export_web.data_through():
        errors.append("model snapshot date does not match the cached current-season input")

    snapshots = value_history.load(desc["target_season"])
    if snapshots.empty:
        errors.append("value history database has no current-season snapshots")
    else:
        latest_date = snapshots["as_of"].max()
        if latest_date != export_web.data_through():
            errors.append("latest value-history date does not match model snapshot")
        latest = snapshots[snapshots["as_of"].eq(latest_date)]
        values_check = values[["athlete_id", "value"]].merge(
            latest[["athlete_id", "value"]], on="athlete_id",
            suffixes=("_valuation", "_history"), validate="one_to_one")
        if len(values_check) != len(values):
            errors.append("latest value history and valuation player sets differ")
        elif (values_check["value_valuation"] - values_check["value_history"]).abs().max() > 1e-6:
            errors.append("latest value history and valuation values differ")

    if stale or errors:
        if errors:
            print("artifact invariants failed:")
            for error in errors:
                print(f"  {error}")
        if not stale:
            raise SystemExit(1)
        print("generated files are stale:")
        for path in stale:
            print(f"  {path}")
        print("regenerate with: ./.venv/bin/python -m src.wnba_salary.export_web")
        raise SystemExit(1)
    print("generated files and artifact invariants are current")


if __name__ == "__main__":
    main()
