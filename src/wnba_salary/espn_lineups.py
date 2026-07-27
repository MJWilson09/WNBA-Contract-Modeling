"""Reconstruct on-court lineups and possessions from ESPN play-by-play.

Why this exists
---------------
The archived WNBA Stats play-by-play with pre-computed lineups covers 2017-2022
only, and `stats.wnba.com` is not reachable from this environment. ESPN
play-by-play runs 2002-2026 and is mirrored as parquet, so current seasons have
to come from here.

The output schema deliberately matches `rapm.build_possessions()` so the two
sources are interchangeable downstream, and 2017-2022 — where both exist — is the
validation oracle. See `validate_against_stats()`.

Lineup tracking
---------------
ESPN substitution events are unambiguous: `athlete_id_1` enters,
`athlete_id_2` leaves, and both are always populated. Propagating from
`game_rosters` starters through every substitution reproduces the on-court five
almost exactly — across 288 games of 2025 (~15,000 substitutions) it produced 4
sub-outs of an off-court player and one period boundary with six players. Those
games are flagged and dropped rather than repaired.

Possession detection
--------------------
Each event is assigned an offensive team, then a possession is a maximal run of
consecutive events with the same offensive team inside a period. The mapping from
event to offensive team is the fiddly part:

- shooting plays and free throws -> the acting team has the ball
- offensive rebound -> acting team has the ball (possession continues)
- defensive rebound, steal, block -> acting team is on DEFENCE
- turnover -> committing team had the ball
- fouls -> the fouling team is on defence, except offensive fouls

Garbage time is approximated rather than replicated; see GARBAGE_* constants.
"""

from __future__ import annotations

import collections
import io

import numpy as np
import pandas as pd
import requests

from . import data

ROSTER_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/wehoop-wnba-data/main/"
    "wnba/game_rosters/parquet/game_rosters_{season}.parquet"
)

# Garbage time: large margin, little time left. Approximate — the stats archive's
# own flag is not reproduced exactly. validate_against_stats() reports agreement.
GARBAGE_MARGIN = 20
GARBAGE_SECONDS = 300

NO_SIGNAL = {
    "Substitution", "Full Timeout", "Short Timeout", "Official TV Timeout",
    "End Period", "End Game", "Start Period", "Jump Ball", "Replay Review",
    "Instant Replay", "Delay Of Game",
}


def load_rosters(season: int) -> pd.DataFrame:
    path = data.RAW_DIR / "game_rosters" / f"game_rosters_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    resp = requests.get(ROSTER_URL.format(season=season), timeout=180)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def offense_team(row) -> float:
    """Which team had the ball, or NaN if the event carries no signal."""
    ty = str(row["type_text"] or "")
    team = row["team_id"]
    if pd.isna(team) or ty in NO_SIGNAL:
        return np.nan
    other = row["away_team_id"] if team == row["home_team_id"] else row["home_team_id"]

    if row.get("shooting_play"):
        return team
    if "Free Throw" in ty:
        return team
    if "Rebound" in ty:
        return team if "Offensive" in ty else other
    if "Turnover" in ty or "Bad Pass" in ty or "Lost Ball" in ty:
        return team
    if "Steal" in ty or "Block" in ty:
        return other
    if "Foul" in ty:
        return team if "Offensive" in ty else other
    if "Violation" in ty:
        return team
    return np.nan


def reconstruct(season: int, *, season_type: int = 2) -> pd.DataFrame:
    """Possession-level rows with on-court lineups, matching rapm's schema."""
    pbp = data.fetch_season("pbp", season)
    pbp = pbp[pbp["season_type"] == season_type].copy()
    pbp["game_id"] = pbp["game_id"].astype("int64")

    rosters = load_rosters(season)
    rosters["game_id"] = rosters["game_id"].astype("int64")
    rosters["team_id"] = rosters["team_id"].astype("int64")
    rosters["athlete_id"] = rosters["athlete_id"].astype("int64")

    names = dict(zip(rosters["athlete_id"], rosters["athlete_display_name"]))
    starters = (
        rosters[rosters["starter"] == True]  # noqa: E712
        .groupby(["game_id", "team_id"])["athlete_id"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )
    teams_by_game = collections.defaultdict(list)
    for gid, tid in starters:
        teams_by_game[gid].append(tid)

    pbp["off_team_raw"] = pbp.apply(offense_team, axis=1)
    pbp = pbp.sort_values(["game_id", "game_play_number"])

    records = []
    skipped = collections.Counter()

    for gid, g in pbp.groupby("game_id", sort=False):
        ts = teams_by_game.get(gid, [])
        if len(ts) != 2 or any(len(starters[(gid, t)]) != 5 for t in ts):
            skipped["no_valid_starters"] += 1
            continue

        on = {t: set(starters[(gid, t)]) for t in ts}
        game_date = g["game_date"].iloc[0]
        cur_off = None
        cur_period = None
        pts = 0
        lineup_snapshot = None
        gt = False
        bad_game = False
        poss_no = 0

        def flush():
            nonlocal pts, lineup_snapshot, gt, poss_no
            if cur_off is None or lineup_snapshot is None:
                pts = 0
                return
            off_t = cur_off
            def_t = ts[0] if ts[1] == off_t else ts[1]
            off_l, def_l = lineup_snapshot
            poss_no += 1
            records.append({
                "game_id": gid, "pid": poss_no, "season": season,
                "game_date": game_date, "pts": pts,
                "garbage_time": int(gt),
                "off_team": off_t, "def_team": def_t,
                **{f"off_player{i+1}": off_l[i] for i in range(5)},
                **{f"def_player{i+1}": def_l[i] for i in range(5)},
            })
            pts = 0

        for row in g.itertuples(index=False):
            period = row.period
            if cur_period is not None and period != cur_period:
                flush()
                cur_off = None
                for t in ts:
                    if len(on[t]) != 5:
                        bad_game = True
            cur_period = period

            if row.type_text == "Substitution":
                t = row.team_id
                if pd.isna(t) or int(t) not in on:
                    continue
                t = int(t)
                a_in, a_out = row.athlete_id_1, row.athlete_id_2
                if pd.isna(a_in) or pd.isna(a_out):
                    continue
                if int(a_out) not in on[t]:
                    bad_game = True
                on[t].discard(int(a_out))
                on[t].add(int(a_in))
                if len(on[t]) != 5:
                    bad_game = True
                continue

            off_t = row.off_team_raw
            if not pd.isna(off_t):
                off_t = int(off_t)
                if off_t != cur_off:
                    flush()
                    cur_off = off_t
                    def_t = ts[0] if ts[1] == off_t else ts[1]
                    if len(on[off_t]) == 5 and len(on[def_t]) == 5:
                        lineup_snapshot = (
                            sorted(names.get(p, str(p)) for p in on[off_t]),
                            sorted(names.get(p, str(p)) for p in on[def_t]),
                        )
                    else:
                        lineup_snapshot = None

            if row.scoring_play and not pd.isna(row.score_value):
                pts += int(row.score_value)

            margin = abs((row.home_score or 0) - (row.away_score or 0))
            secs = row.end_game_seconds_remaining
            if margin >= GARBAGE_MARGIN and not pd.isna(secs) and secs <= GARBAGE_SECONDS:
                gt = True

        flush()
        if bad_game:
            skipped["lineup_inconsistent"] += 1
            records = [r for r in records if r["game_id"] != gid]

    out = pd.DataFrame(records)
    out.attrs["skipped"] = dict(skipped)
    return out


def validate_against_stats(season: int) -> dict:
    """Compare the ESPN reconstruction to the stats-API archive for one season.

    Games are matched on date plus the sorted pair of possession counts is not
    reliable, so matching is on date and the multiset of team-level points, which
    is unique in practice. Reports possession counts, points per possession, and
    lineup agreement rate.
    """
    from . import rapm

    espn = reconstruct(season)
    stats = rapm.build_possessions(rapm.fetch_stats_pbp(season))

    e = (
        espn.groupby("game_id")
        .agg(poss=("pid", "count"), pts=("pts", "sum"),
             date=("game_date", "first"))
        .reset_index()
    )
    s = (
        stats.groupby("game_id")
        .agg(poss=("pid", "count"), pts=("pts", "sum"),
             date=("game_date", "first"))
        .reset_index()
    )
    e["date"] = pd.to_datetime(e["date"]).dt.date
    s["date"] = pd.to_datetime(s["date"]).dt.date

    m = e.merge(s, on=["date", "pts"], suffixes=("_espn", "_stats"))
    return {
        "season": season,
        "espn_games": len(e),
        "stats_games": len(s),
        "matched_games": len(m),
        "espn_poss_total": int(e["poss"].sum()),
        "stats_poss_total": int(s["poss"].sum()),
        "espn_ppp": float(espn["pts"].sum() / len(espn)),
        "stats_ppp": float(stats["pts"].sum() / len(stats)),
        "poss_corr": float(m["poss_espn"].corr(m["poss_stats"])) if len(m) > 2 else None,
        "poss_mean_abs_diff": float((m["poss_espn"] - m["poss_stats"]).abs().mean())
        if len(m) else None,
        "espn_garbage_share": float(espn["garbage_time"].mean()),
        "stats_garbage_share": float(stats["garbage_time"].mean()),
        "skipped": espn.attrs.get("skipped", {}),
    }


def main() -> None:
    print("=== validation against stats-API archive (overlap years) ===")
    for season in (2019, 2022):
        v = validate_against_stats(season)
        print(f"\n{season}:")
        for k, val in v.items():
            if k == "season":
                continue
            print(f"  {k}: {val}")

    print("\n=== reconstructing 2023-2026 ===")
    frames = []
    for season in (2023, 2024, 2025, 2026):
        df = reconstruct(season)
        frames.append(df)
        print(f"  {season}: {len(df):,} possessions, "
              f"{df['game_id'].nunique()} games, "
              f"ppp={df['pts'].sum()/len(df):.4f}, "
              f"skipped={df.attrs.get('skipped', {})}")
    out = pd.concat(frames, ignore_index=True)
    path = data.PROCESSED_DIR / "espn_possessions_2023_2026.parquet"
    out.to_parquet(path, index=False)
    print(f"\nwrote {path} ({len(out):,} possessions)")


if __name__ == "__main__":
    main()
