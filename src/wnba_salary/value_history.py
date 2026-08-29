"""Daily player-value snapshots for change-over-time comparisons.

The website is static and never opens this database directly. Each successful
in-season rebuild upserts the current valuation table here; `export_web.py`
queries it and emits only the two compact comparisons used by player cards.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import data, valuation

DB_PATH = data.PROCESSED_DIR / "value_history.sqlite"

COLUMNS = {
    "season": "INTEGER NOT NULL",
    "as_of": "TEXT NOT NULL",
    "athlete_id": "INTEGER NOT NULL",
    "player_name": "TEXT NOT NULL",
    "team_id": "INTEGER",
    "team_games": "INTEGER",
    "games": "REAL",
    "minutes": "REAL",
    "mpg": "REAL",
    "availability": "REAL",
    "proj_minutes": "REAL",
    "rating": "REAL",
    "rating_se": "REAL",
    "war": "REAL",
    "value": "REAL",
    "market_value": "REAL",
    "salary": "REAL",
    "surplus": "REAL",
}


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    definitions = ",\n        ".join(f"{name} {kind}" for name, kind in COLUMNS.items())
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS player_values (
        {definitions},
        PRIMARY KEY (season, as_of, athlete_id)
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_player_values_lookup "
        "ON player_values (season, athlete_id, as_of)"
    )
    return con


def data_through(season: int = valuation.CURRENT_SEASON) -> str:
    box = data.load("player_box", [season])
    return pd.to_datetime(box["game_date"]).max().strftime("%Y-%m-%d")


def snapshot_frame(values: pd.DataFrame, as_of: str) -> pd.DataFrame:
    rename = {
        "athlete_display_name": "player_name",
        "g": "games",
        "mp": "minutes",
    }
    frame = values.rename(columns=rename).copy()
    frame["season"] = valuation.CURRENT_SEASON
    frame["as_of"] = as_of
    for column in COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame[list(COLUMNS)]


def upsert(values: pd.DataFrame, as_of: str, path: Path = DB_PATH) -> int:
    """Replace one date atomically, making reruns idempotent."""
    frame = snapshot_frame(values, as_of)
    rows = [tuple(None if pd.isna(v) else v for v in row)
            for row in frame.itertuples(index=False, name=None)]
    placeholders = ",".join("?" for _ in COLUMNS)
    columns = ",".join(COLUMNS)
    with connect(path) as con:
        con.execute("DELETE FROM player_values WHERE season = ? AND as_of = ?",
                    (valuation.CURRENT_SEASON, as_of))
        con.executemany(
            f"INSERT INTO player_values ({columns}) VALUES ({placeholders})", rows)
    return len(rows)


def load(season: int = valuation.CURRENT_SEASON,
         path: Path = DB_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    with connect(path) as con:
        return pd.read_sql_query(
            "SELECT * FROM player_values WHERE season = ? ORDER BY as_of, athlete_id",
            con, params=(season,))


def main() -> None:
    values = pd.read_parquet(data.PROCESSED_DIR / "valuation.parquet")
    as_of = data_through()
    n = upsert(values, as_of)
    history = load()
    print(f"wrote {n} player rows for {as_of} to {DB_PATH.name} "
          f"({history['as_of'].nunique()} snapshots)")


if __name__ == "__main__":
    main()
