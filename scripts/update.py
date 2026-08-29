#!/usr/bin/env python
"""Refresh in-season data and rebuild everything downstream of it.

    ./.venv/bin/python scripts/update.py            # refresh + full rebuild
    ./.venv/bin/python scripts/update.py --check    # report staleness, change nothing
    ./.venv/bin/python scripts/update.py --no-history   # skip the slow history sweep

Why this exists
---------------
Every fetcher in this project caches to disk and, once a file is there, never
looks again — `data.fetch_season` takes a `refresh` flag that nothing passed.
That is right for finished seasons and wrong for the one in progress, and the
failure is silent: a re-run of the whole pipeline happily reproduces figures
built from weeks-old games. It went unnoticed until a player who had returned
from injury was still missing from the site.

So this script does the one thing the pipeline could not do for itself: force
the current season's inputs to refetch, drop the caches derived from them, and
then run the stages in order.

Two layers of staleness, only one of which is ours
--------------------------------------------------
`wehoop-wnba-data` is a mirror and lags live results by several days. This
script closes the gap between our cache and that mirror; it cannot close the gap
between the mirror and last night's box score. Both are printed, so the
distinction is visible rather than assumed.

What gets refreshed
-------------------
Current season only: the wehoop parquet feeds, the game rosters that
`espn_lineups` needs for starters, Basketball-Reference's WNBA tables, and the
Her Hoop Stats salary sheet. Finished seasons are left alone — they do not
change, and refetching them wastes a rate-limited BBRef budget.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.wnba_salary import (  # noqa: E402
    bbref, data, espn_lineups, rapm, salaries, season, valuation,
)

SEASON = valuation.CURRENT_SEASON
PY = str(ROOT / ".venv" / "bin" / "python")

# Stage order matters: each reads the previous stage's output.
STAGES = [
    ("constants",  "src.wnba_salary.constants",  "league constants from team box scores"),
    ("box_prior",  "src.wnba_salary.box_prior",  "transfer-learned box prior"),
    ("ratings",    "src.wnba_salary.ratings",    "RAPM, descriptive + forecast"),
    ("valuation",  "src.wnba_salary.valuation",  "ratings -> dollars"),
    ("history",    "src.wnba_salary.history",    "past seasons for the picker (slow)"),
    ("export_web", "src.wnba_salary.export_web", "docs/players.js"),
]


def cached_through() -> pd.Timestamp | None:
    """Newest game date in our local player_box cache."""
    p = data.RAW_DIR / "player_box" / f"player_box_{SEASON}.parquet"
    if not p.exists():
        return None
    return pd.to_datetime(pd.read_parquet(p)["game_date"]).max()


def _remote_dataset(dataset: str, target: int) -> pd.DataFrame | None:
    """Read a mirror parquet into memory without changing the disk cache."""
    import requests
    subdir, stem = data.DATASETS[dataset]
    url = f"{data.BASE_URL}/{subdir}/{stem}_{target}.parquet"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def upstream_through() -> tuple[pd.Timestamp | None, int]:
    """Newest game date the mirror is serving, WITHOUT touching the cache.

    `data.fetch_season(refresh=True)` writes as a side effect, so --check cannot
    use it: a command that advertises itself as read-only must not silently
    advance the very cache it is reporting on. Fetch into memory instead.
    """
    df = _remote_dataset("player_box", SEASON)
    if df is None:
        return None, 0
    if df.empty:
        return None, 0
    return pd.to_datetime(df["game_date"]).max(), int(df["game_id"].nunique())


def rollover_readiness(target: int) -> tuple[bool, list[str]]:
    """Read-only proof that a new season is complete enough for production.

    A mirror can publish its schedule weeks before it publishes usable game
    data. Require the first completed regular-season game to agree across all
    four game feeds, plus metadata, starters, salaries, and BBRef tables.
    """
    import requests

    reasons: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for ds in ("schedule", "team_box", "player_box", "pbp", "player_core"):
        try:
            frame = _remote_dataset(ds, target)
        except Exception as exc:
            reasons.append(f"{ds}: {type(exc).__name__}")
            continue
        if frame is None or frame.empty:
            reasons.append(f"{ds}: unavailable")
        else:
            frames[ds] = frame
    if len(frames) != 5:
        return False, reasons

    schedule = frames["schedule"]
    completed = schedule[
        schedule["season_type"].eq(data.REGULAR_SEASON)
        & schedule["status_type_completed"].fillna(False).astype(bool)
    ]
    completed_ids = set(completed["game_id"].astype(str))
    if not completed_ids:
        reasons.append("schedule: no completed regular-season game")
        return False, reasons

    # The complete schedule is also the expansion/team-count authority used by
    # constants.py. Do not roll on a mirror that has published only opening day.
    home = schedule[["home_id", "home_abbreviation"]].rename(
        columns={"home_id": "team_id", "home_abbreviation": "abbr"})
    away = schedule[["away_id", "away_abbreviation"]].rename(
        columns={"away_id": "team_id", "away_abbreviation": "abbr"})
    scheduled_teams = pd.concat([home, away], ignore_index=True)
    scheduled_teams = scheduled_teams[
        ~scheduled_teams["abbr"].isin(data.NON_LEAGUE_TEAM_ABBREVIATIONS)]
    old_constants = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    previous_n = int(old_constants["cba"]["n_teams"])
    if scheduled_teams["team_id"].nunique() < previous_n:
        reasons.append(f"schedule: fewer than {previous_n} league teams")
    scheduled_games = scheduled_teams.groupby("team_id").size()
    if scheduled_games.empty or scheduled_games.min() < valuation.games_in_season(target):
        reasons.append(f"schedule: not all teams have {valuation.games_in_season(target)} games")

    common = completed_ids
    for ds in ("team_box", "player_box", "pbp"):
        frame = frames[ds]
        regular = frame[frame["season_type"].eq(data.REGULAR_SEASON)]
        common &= set(regular["game_id"].astype(str))
    if not common:
        reasons.append("game feeds: no completed game present in every feed")

    roster_url = espn_lineups.ROSTER_URL.format(season=target)
    try:
        resp = requests.get(roster_url, timeout=180)
        if resp.status_code == 404:
            raise RuntimeError("unavailable")
        resp.raise_for_status()
        rosters = pd.read_parquet(io.BytesIO(resp.content))
        starters = rosters[rosters["starter"].eq(True)].copy()
        starters["game_id"] = starters["game_id"].astype(str)
        valid = (starters[starters["game_id"].isin(common)]
                 .groupby(["game_id", "team_id"])["athlete_id"].nunique())
        if valid.empty or not (valid.eq(5).groupby(level=0).sum() >= 2).any():
            reasons.append("game_rosters: no completed game with two valid starting fives")
    except Exception as exc:
        reasons.append(f"game_rosters: {type(exc).__name__}")

    try:
        resp = requests.get(
            salaries._url(target), headers={"User-Agent": salaries.USER_AGENT}, timeout=60)
        resp.raise_for_status()
        salary_rows = salaries.parse_salary_html(resp.text, target)
        if len(salary_rows) < 10:
            reasons.append("salaries: fewer than 10 populated rows")
    except Exception as exc:
        reasons.append(f"salaries: {type(exc).__name__}")

    for kind in ("advanced", "totals"):
        try:
            resp = requests.get(
                bbref._url("wnba", target, kind),
                headers={"User-Agent": bbref.USER_AGENT}, timeout=60)
            resp.raise_for_status()
            if resp.text.count('data-stat="player"') < 10:
                reasons.append(f"bbref {kind}: no populated player table")
        except Exception as exc:
            reasons.append(f"bbref {kind}: {type(exc).__name__}")

    return not reasons, reasons


def maybe_rollover(*, check_only: bool, requested: int | None = None) -> bool:
    """Advance the tracked config after the next season passes every guard."""
    candidate = requested or SEASON + 1
    if candidate != SEASON + 1:
        raise SystemExit(f"--season must be the next season ({SEASON + 1})")
    if requested is None and pd.Timestamp.now("America/Chicago").year < candidate:
        return False
    ready, reasons = rollover_readiness(candidate)
    if not ready:
        print(f"rollover {SEASON} -> {candidate}: not ready")
        for reason in reasons:
            print(f"  - {reason}")
        if requested is not None:
            raise SystemExit(1)
        return False
    print(f"rollover {SEASON} -> {candidate}: all required feeds ready")
    if check_only:
        return True
    season.set_current_season(candidate)
    print(f"  updated {season.CONFIG_PATH.relative_to(ROOT)}; restarting for {candidate}")
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                             *[a for a in sys.argv[1:] if a != "--season" and a != str(candidate)]])
    return True  # pragma: no cover - os.execv does not return


def report(before: pd.Timestamp | None, after: pd.Timestamp | None, games: int) -> None:
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    fmt = lambda t: "none" if t is None else t.strftime("%Y-%m-%d")
    print(f"  local cache was through : {fmt(before)}")
    print(f"  mirror now serves through: {fmt(after)}  ({games} games)")
    if after is not None:
        lag = (today - after.normalize()).days
        print(f"  mirror lag behind today  : {lag} day(s)"
              + ("   <- not something this script can close" if lag > 1 else ""))
    if before is not None and after is not None and after > before:
        print(f"  gained {(after - before).days} day(s) of games")
    elif before is not None and after is not None:
        print("  already current with the mirror")


def refresh_sources() -> None:
    """Refetch every current-season input and drop the caches derived from them."""
    for ds in ("player_box", "team_box", "pbp", "player_core", "schedule"):
        df = data.fetch_season(ds, SEASON, refresh=True)
        print(f"    {ds:<12} {'—' if df is None else f'{len(df):,} rows'}")

    # game_rosters is fetched by espn_lineups, not data.py, and has its own cache
    rp = data.RAW_DIR / "game_rosters" / f"game_rosters_{SEASON}.parquet"
    rp.unlink(missing_ok=True)
    print(f"    game_rosters {len(espn_lineups.load_rosters(SEASON)):,} rows")

    # BBRef: positions for the box prior, and the rate stats it is validated on
    for kind in ("advanced", "totals"):
        df = bbref.fetch_advanced("wnba", SEASON, kind=kind, refresh=True)
        print(f"    bbref {kind:<7}{'—' if df is None else f'{len(df):,} rows'}")

    df = salaries.fetch_salaries(SEASON, refresh=True)
    print(f"    salaries     {len(df):,} rows")

    # Possessions for this season are reconstructed from the pbp we just
    # replaced, so the cached frame is now wrong. Only this season's.
    pc = rapm.POSS_CACHE / f"poss_{SEASON}.parquet"
    if pc.exists():
        pc.unlink()
        print(f"    dropped poss_cache/{pc.name}")


def run_stage(module: str, label: str) -> float:
    t0 = time.time()
    r = subprocess.run([PY, "-W", "ignore", "-m", module],
                       cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"\n  FAILED after {dt:.0f}s: {module}\n")
        print(r.stdout[-2500:])
        print(r.stderr[-2500:])
        raise SystemExit(1)
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
    print(f"    {label:<12} {dt:6.0f}s   {tail[0].strip() if tail else ''}")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report staleness and exit without changing anything")
    ap.add_argument("--no-history", action="store_true",
                    help="skip the history sweep (~6 min); the season picker keeps old data")
    ap.add_argument("--season", type=int,
                    help="request a guarded rollover to exactly the next season")
    args = ap.parse_args()

    print(f"WNBA contract model — update for {SEASON}\n")
    before = cached_through()

    if args.check:
        rollover_due = maybe_rollover(check_only=True, requested=args.season)
        after, games = upstream_through()   # read-only; see the docstring
        report(before, after, games)
        stale = before is not None and after is not None and after > before
        needs_update = stale or rollover_due
        print("\n  " + ("STALE — run without --check to update"
                        if needs_update else "up to date with the mirror"))
        raise SystemExit(1 if needs_update else 0)

    maybe_rollover(check_only=False, requested=args.season)

    print("refreshing current-season sources")
    refresh_sources()
    after = cached_through()
    games = pd.read_parquet(
        data.RAW_DIR / "player_box" / f"player_box_{SEASON}.parquet")["game_id"].nunique()
    print()
    report(before, after, int(games))

    stages = [s for s in STAGES if not (args.no_history and s[0] == "history")]
    print(f"\nrebuilding ({len(stages)} stages)")
    total = sum(run_stage(mod, name) for name, mod, _ in stages)
    print(f"\ndone in {total/60:.1f} min")
    print(f"data through {after.strftime('%Y-%m-%d') if after is not None else 'unknown'}"
          " — commit data/processed/ and generated docs to publish")


if __name__ == "__main__":
    main()
