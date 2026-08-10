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
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.wnba_salary import bbref, data, espn_lineups, rapm, salaries, valuation  # noqa: E402

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


def upstream_through() -> tuple[pd.Timestamp | None, int]:
    """Newest game date the mirror is serving, WITHOUT touching the cache.

    `data.fetch_season(refresh=True)` writes as a side effect, so --check cannot
    use it: a command that advertises itself as read-only must not silently
    advance the very cache it is reporting on. Fetch into memory instead.
    """
    import io
    import requests
    subdir, stem = data.DATASETS["player_box"]
    url = f"{data.BASE_URL}/{subdir}/{stem}_{SEASON}.parquet"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None, 0
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    if df.empty:
        return None, 0
    return pd.to_datetime(df["game_date"]).max(), int(df["game_id"].nunique())


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
    args = ap.parse_args()

    print(f"WNBA contract model — update for {SEASON}\n")
    before = cached_through()

    if args.check:
        after, games = upstream_through()   # read-only; see the docstring
        report(before, after, games)
        stale = before is not None and after is not None and after > before
        print("\n  " + ("STALE — run without --check to update"
                        if stale else "up to date with the mirror"))
        raise SystemExit(1 if stale else 0)

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
          f" — commit data/processed/ and docs/players.js to publish")


if __name__ == "__main__":
    main()
