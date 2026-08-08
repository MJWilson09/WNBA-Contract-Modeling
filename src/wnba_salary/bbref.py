"""Basketball-Reference advanced-table scraper for NBA and WNBA.

Used for two things only:

1. The NBA training target — OBPM/DBPM are Basketball-Reference metrics and
   exist nowhere else in bulk.
2. Advanced rate stats for both leagues, on *identical definitions*. That
   consistency is the whole ballgame for transfer learning: if NBA TS% and WNBA
   TS% are computed differently, coefficients fitted on one and applied to the
   other are meaningless.

`pandas.read_html` mangles the WNBA table, so rows are parsed from `data-stat`
attributes instead. That is also stable against column reordering.

BBRef asks for at most ~20 requests/minute. RATE_LIMIT_SECONDS honours that, and
every season is cached to disk so a re-run costs nothing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import data

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
RATE_LIMIT_SECONDS = 3.5
CACHE_DIR = data.RAW_DIR / "bbref"

# BBRef data-stat -> our column name. Names differ slightly by league and era,
# so both variants are mapped onto one schema.
COLUMN_MAP = {
    "name_display": "player",
    "player": "player",
    "age": "age",
    "team_name_abbr": "team",
    "team": "team",
    "team_id": "team",
    "pos": "pos",
    "g": "g",
    "games": "g",
    "mp": "mp",
    "per": "per",
    "ts_pct": "ts_pct",
    "fg3a_per_fga_pct": "fg3a_rate",
    "fta_per_fga_pct": "ftr",
    "orb_pct": "orb_pct",
    "drb_pct": "drb_pct",
    "trb_pct": "trb_pct",
    "ast_pct": "ast_pct",
    "stl_pct": "stl_pct",
    "blk_pct": "blk_pct",
    "tov_pct": "tov_pct",
    "usg_pct": "usg_pct",
    "obpm": "obpm",
    "dbpm": "dbpm",
    "bpm": "bpm",
    "vorp": "vorp",
    "ws": "ws",
    "ws_per_48": "ws_per_48",
    "ws_per_40": "ws_per_40",
    "off_rtg": "off_rtg",
    "def_rtg": "def_rtg",
    # totals table
    "pf": "pf",
    "orb": "orb",
    "drb": "drb",
    "trb": "trb",
    "fga": "fga",
    "fta": "fta",
    "fg3a": "fg3a",
    "tov": "tov",
    "ast": "ast",
    "stl": "stl",
    "blk": "blk",
    "pts": "pts",
    # draft table
    "pick_overall": "pick",
    "seasons": "career_seasons",
}

NUMERIC = {
    "age", "g", "mp", "per", "ts_pct", "fg3a_rate", "ftr", "orb_pct", "drb_pct",
    "trb_pct", "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct", "obpm",
    "dbpm", "bpm", "vorp", "ws", "ws_per_48", "ws_per_40", "off_rtg", "def_rtg",
    "pf", "orb", "drb", "trb", "fga", "fta", "fg3a", "tov", "ast", "stl",
    "blk", "pts", "pick", "career_seasons",
}

_last_request = 0.0


def _url(league: str, season: int, kind: str = "advanced") -> str:
    if kind == "draft":
        if league != "wnba":
            raise ValueError("draft scraping is implemented for the WNBA only")
        return f"https://www.basketball-reference.com/wnba/draft/{season}.html"
    if kind not in ("advanced", "totals"):
        raise ValueError(f"kind must be 'advanced', 'totals' or 'draft', got {kind!r}")
    if league == "nba":
        return f"https://www.basketball-reference.com/leagues/NBA_{season}_{kind}.html"
    if league == "wnba":
        return f"https://www.basketball-reference.com/wnba/years/{season}_{kind}.html"
    raise ValueError(f"league must be 'nba' or 'wnba', got {league!r}")


def _throttle() -> None:
    global _last_request
    wait = RATE_LIMIT_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def _cell_text(cell) -> str:
    """Text belonging to this cell only, ignoring nested cells.

    BBRef's WNBA pages leave `<th data-stat="player">` unclosed, so lxml nests
    every subsequent `<td>` of the row inside it. A plain `get_text()` therefore
    returns the whole row concatenated onto the player's name. Numeric columns
    are unaffected (they are still found as descendants), so this only needs to
    guard text extraction.
    """
    parts = []
    for s in cell.strings:
        nested = False
        for parent in s.parents:
            if parent is cell:
                break
            if parent.name in ("td", "th"):
                nested = True
                break
        if not nested:
            parts.append(s)
    return "".join(parts).strip()


def _parse(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = tr.find_all(["td", "th"])
        rec = {}
        for c in cells:
            stat = c.get("data-stat")
            if stat in COLUMN_MAP:
                rec.setdefault(COLUMN_MAP[stat], _cell_text(c))
        # a real player row has a name plus either minutes (season tables) or a
        # draft pick (draft tables)
        if rec.get("player") and (rec.get("mp") or rec.get("pick")):
            rows.append(rec)

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_advanced(
    league: str, season: int, *, kind: str = "advanced", refresh: bool = False
) -> pd.DataFrame | None:
    """Fetch one season of one table. None if the season doesn't exist."""
    path = CACHE_DIR / f"{league}_{kind}_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    _throttle()
    resp = requests.get(
        _url(league, season, kind),
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=60,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    # BBRef serves UTF-8 without always declaring it; requests would otherwise
    # fall back to ISO-8859-1 and mangle accented names ("Schröder").
    resp.encoding = "utf-8"

    df = _parse(resp.text)
    if df.empty:
        return None

    df.insert(0, "season", season)
    df.insert(1, "league", league)
    df = dedupe_multi_team(df)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def dedupe_multi_team(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per player-season.

    Traded players get a combined row (team '2TM'/'TOT') plus one row per team.
    We want the combined row — season-long rate stats are what the model needs.
    """
    if "team" not in df.columns:
        return df
    combined_mask = df["team"].astype(str).str.upper().str.match(r"^(TOT|\dTM)$")
    combined = df[combined_mask]
    single = df[~combined_mask]
    single = single[~single["player"].isin(set(combined["player"]))]
    out = pd.concat([combined, single], ignore_index=True)
    return out.drop_duplicates(subset=["season", "league", "player"], keep="first")


def load_advanced(
    league: str, seasons: range | list[int], *, kind: str = "advanced", refresh: bool = False
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        df = fetch_advanced(league, season, kind=kind, refresh=refresh)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"no bbref {kind} for {league} over {list(seasons)[:3]}...")
    return pd.concat(frames, ignore_index=True)
