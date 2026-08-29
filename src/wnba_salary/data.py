"""Data access for the WNBA salary model.

Everything here reads pre-built parquet from the `wehoop-wnba-data` repo, which
is the same source `wehoop::load_wnba_*()` reads in R. Going straight at the
parquet means Stage A needs no R and no `stats.wnba.com` (that host is slow and
intermittently blocks non-browser clients).

Files are cached under `data/raw/` on first fetch and read locally afterward.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/sportsdataverse/wehoop-wnba-data/main"

# dataset -> (subdirectory, filename stem)
DATASETS: dict[str, tuple[str, str]] = {
    "team_box": ("wnba/team_box/parquet", "team_box"),
    "player_box": ("wnba/player_box/parquet", "player_box"),
    "schedule": ("wnba/schedules/parquet", "wnba_schedule"),
    "pbp": ("wnba/pbp/parquet", "play_by_play"),
    "player_core": ("wnba/player_core/parquet", "player_core"),
}

# season_type codes in the ESPN-derived feed
REGULAR_SEASON = 2
POSTSEASON = 3

# ESPN labels the All-Star exhibition as a regular-season game. Filtering only
# on season_type therefore contaminates box priors, minutes, pace, and RAPM.
# These are event teams, never league clubs.
NON_LEAGUE_TEAM_ABBREVIATIONS = {
    "EAST", "WEST",             # 2017 conference format
    "DEL", "PAR", "WIL", "STE",  # captain-drafted teams, 2018–23
    "ALL", "USA", "WNBASTARS",    # WNBA vs USA, 2021/24
    "CLA", "COL", "SPO", "COOP", # captain-drafted teams, 2025–26
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _cache_path(dataset: str, season: int) -> Path:
    return RAW_DIR / dataset / f"{dataset}_{season}.parquet"


def fetch_season(dataset: str, season: int, *, refresh: bool = False) -> pd.DataFrame | None:
    """Fetch one season of one dataset. Returns None if the season doesn't exist.

    Missing seasons are expected — coverage differs by dataset and the league has
    expanded and contracted over its history — so a 404 is not an error here.
    """
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; have {sorted(DATASETS)}")

    path = _cache_path(dataset, season)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    subdir, stem = DATASETS[dataset]
    url = f"{BASE_URL}/{subdir}/{stem}_{season}.parquet"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    df = pd.read_parquet(io.BytesIO(resp.content))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def exhibition_game_ids(season: int) -> set[str]:
    """All-Star/event game IDs, derived from team identities in the feeds."""
    teams = fetch_season("team_box", season)
    if teams is not None and not teams.empty:
        mask = (
            teams["team_abbreviation"].isin(NON_LEAGUE_TEAM_ABBREVIATIONS)
            | teams["opponent_team_abbreviation"].isin(NON_LEAGUE_TEAM_ABBREVIATIONS)
        )
        return set(teams.loc[mask, "game_id"].astype(str))

    schedule = fetch_season("schedule", season)
    if schedule is None or schedule.empty:
        return set()
    mask = (
        schedule["home_abbreviation"].isin(NON_LEAGUE_TEAM_ABBREVIATIONS)
        | schedule["away_abbreviation"].isin(NON_LEAGUE_TEAM_ABBREVIATIONS)
    )
    return set(schedule.loc[mask, "game_id"].astype(str))


def exclude_exhibitions(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Remove event-team games even when ESPN calls them regular season."""
    if df.empty:
        return df
    if {"home_abbreviation", "away_abbreviation"}.issubset(df.columns):
        mask = (
            df["home_abbreviation"].isin(NON_LEAGUE_TEAM_ABBREVIATIONS)
            | df["away_abbreviation"].isin(NON_LEAGUE_TEAM_ABBREVIATIONS)
        )
        return df[~mask]
    if "game_id" not in df.columns:
        return df
    excluded = exhibition_game_ids(season)
    return df[~df["game_id"].astype(str).isin(excluded)] if excluded else df


def load(
    dataset: str,
    seasons: range | list[int],
    *,
    season_type: int | None = REGULAR_SEASON,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load and concatenate several seasons.

    `season_type=None` keeps postseason rows; the default keeps regular season
    only, which is what every constant in this model is defined against.
    """
    frames = []
    for season in seasons:
        df = fetch_season(dataset, season, refresh=refresh)
        if df is None or df.empty:
            continue
        if season_type is not None and "season_type" in df.columns:
            df = df[df["season_type"] == season_type]
        df = exclude_exhibitions(df, season)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"no data for {dataset} over {list(seasons)[:3]}...")
    return pd.concat(frames, ignore_index=True)


def available_seasons(dataset: str = "team_box") -> list[int]:
    """List seasons present in the remote repo for a dataset."""
    subdir, stem = DATASETS[dataset]
    url = "https://api.github.com/repos/sportsdataverse/wehoop-wnba-data/git/trees/main?recursive=1"
    tree = requests.get(url, timeout=60).json()["tree"]
    prefix = f"{subdir}/{stem}_"
    seasons = []
    for node in tree:
        p = node["path"]
        if p.startswith(prefix) and p.endswith(".parquet"):
            seasons.append(int(p[len(prefix) : -len(".parquet")]))
    return sorted(seasons)
