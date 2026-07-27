"""League-structural constants for the WNBA salary model.

These are the constants that do NOT depend on the player rating model. They are
the denominators everything else is estimated against, so they come first: if
points-per-win is wrong, every rating-scale constant fitted later is wrong in a
way that is invisible, because the error gets absorbed silently into dollars.

Estimated here
--------------
points_per_win      Points of season-long differential per marginal win.
poss_per_40         Pace, possessions per team per 40 minutes (OT-adjusted).
minutes_baseline    The analog of Noh's NBA `1475`. See below — it is DERIVED,
                    not fitted.

Assumed here (with sensitivity ranges, not estimates)
-----------------------------------------------------
replacement_win_pct Nobody fields a replacement-level team, so this cannot be
                    measured. It is a convention. Sensitivity-tested downstream.

The minutes baseline is derived, not fitted
-------------------------------------------
Noh's NBA formula is

    salary = (minutes / 1475) * (DARKO + 3) * 4.32

For `(minutes / B) * (rating + repl)` to equal wins above replacement, we need

    WAR = (rating + repl)/100 * possessions_played / points_per_win

and possessions_played = minutes * poss_per_min, so

    B = 100 * points_per_win / poss_per_min

That is a derived identity, not a free parameter. Sanity check against the NBA:
points_per_win ~= 30, poss_per_min = 100/48 = 2.083, giving B = 1440 — against
Noh's 1475. The identity reproduces his constant to within 2%, which is good
evidence it is the right way to pin the WNBA value rather than guessing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import data

# ---------------------------------------------------------------------------
# CBA / league structure, 2026 season.
#
# Source: WNBA-WNBPA CBA signed 2026, running through 2032.
#   https://www.wnba.com/news/wnba-wnbpa-tentative-cba-deal-2026
# NOTE: these are press-reported headline figures. Before this model is used for
# anything real, verify against the actual CBA text — reported caps often exclude
# benefits/bonuses that count against the real cap.
# ---------------------------------------------------------------------------
CBA_2026 = {
    "season": 2026,
    "n_teams": 15,             # expansion: Toronto Tempo + Portland Fire
    "games_per_team": 44,
    "salary_cap": 7_000_000,
    "max_salary": 1_400_000,
    "min_salary": 270_000,     # reported range 270k-300k by service time
    "roster_min": 12,
    "developmental_spots": 2,
    "minutes_per_game": 40,
}

# Seasons to exclude from pooled fits, with reasons.
EXCLUDED_SEASONS = {
    2020: "COVID bubble, 22-game season",
    2026: "season in progress as of this build",
}


@dataclass
class PointsPerWin:
    points_per_win: float
    ci_low: float
    ci_high: float
    slope_winpct_per_mov: float
    intercept: float
    r_squared: float
    n_team_seasons: int
    seasons: list[int] = field(default_factory=list)


@dataclass
class Pace:
    poss_per_game: float
    poss_per_40: float
    poss_per_min: float
    mean_game_minutes: float
    by_season: dict[int, float] = field(default_factory=dict)


def team_season_totals(team_box: pd.DataFrame) -> pd.DataFrame:
    """Collapse the team-game box to one row per team-season."""
    tb = team_box.copy()
    tb["win"] = tb["team_winner"].astype(bool).astype(int)
    tb["point_diff"] = tb["team_score"] - tb["opponent_team_score"]

    agg = (
        tb.groupby(["season", "team_id"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            wins=("win", "sum"),
            points_for=("team_score", "sum"),
            points_against=("opponent_team_score", "sum"),
            total_diff=("point_diff", "sum"),
            fga=("field_goals_attempted", "sum"),
            fta=("free_throws_attempted", "sum"),
            orb=("offensive_rebounds", "sum"),
            tov=("total_turnovers", "sum"),
        )
    )
    agg["win_pct"] = agg["wins"] / agg["games"]
    agg["mov"] = agg["total_diff"] / agg["games"]
    return agg


def estimate_points_per_win(
    team_seasons: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0
) -> PointsPerWin:
    """Points of season differential per marginal win.

    Fits `win_pct = a + b * MOV`. The slope `b` is season-length invariant:
    wins = G*a + b*total_diff, so d(wins)/d(total_diff) = b regardless of G.
    That lets us pool seasons of different lengths (the WNBA has run 32 to 44
    games) without rescaling. points_per_win = 1/b.

    The intercept is a free parameter rather than pinned at 0.5 precisely so it
    can be checked — if it drifts far from 0.5 the fit is misspecified.
    """
    df = team_seasons[~team_seasons["season"].isin(EXCLUDED_SEASONS)]
    x = df["mov"].to_numpy(float)
    y = df["win_pct"].to_numpy(float)

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b, _ = np.polyfit(x[idx], y[idx], 1)
        boot[i] = 1.0 / b
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return PointsPerWin(
        points_per_win=float(1.0 / slope),
        ci_low=float(lo),
        ci_high=float(hi),
        slope_winpct_per_mov=float(slope),
        intercept=float(intercept),
        r_squared=float(r2),
        n_team_seasons=int(n),
        seasons=sorted(int(s) for s in df["season"].unique()),
    )


def estimate_pace(team_box: pd.DataFrame, player_box: pd.DataFrame) -> Pace:
    """Possessions per team per 40 minutes.

    poss = FGA + 0.44*FTA - ORB + TOV, averaged across the two teams in a game
    (the standard symmetrisation — the two teams' estimates differ slightly and
    the truth is between them).

    Game length comes from summing player minutes and dividing by 5, which
    handles overtime properly. Using a flat 40 would inflate pace by roughly the
    OT rate.
    """
    tb = team_box.copy()
    tb["poss"] = (
        tb["field_goals_attempted"]
        + 0.44 * tb["free_throws_attempted"]
        - tb["offensive_rebounds"]
        + tb["total_turnovers"]
    )
    # symmetrise: mean of both teams' possession estimates within a game
    game_poss = tb.groupby(["season", "game_id"], as_index=False)["poss"].mean()

    pb = player_box[player_box["minutes"].notna()]
    team_minutes = pb.groupby(["season", "game_id", "team_id"], as_index=False)["minutes"].sum()
    team_minutes["game_minutes"] = team_minutes["minutes"] / 5.0
    game_minutes = (
        team_minutes.groupby(["season", "game_id"], as_index=False)["game_minutes"].mean()
    )

    merged = game_poss.merge(game_minutes, on=["season", "game_id"], how="inner")
    # Guard against malformed minute records (blowouts with unrecorded bench time)
    merged = merged[(merged["game_minutes"] > 35) & (merged["game_minutes"] < 75)]

    merged["poss_per_40"] = merged["poss"] * 40.0 / merged["game_minutes"]

    usable = merged[~merged["season"].isin(EXCLUDED_SEASONS)]
    poss_per_40 = float(usable["poss_per_40"].mean())

    return Pace(
        poss_per_game=float(usable["poss"].mean()),
        poss_per_40=poss_per_40,
        poss_per_min=poss_per_40 / 40.0,
        mean_game_minutes=float(usable["game_minutes"].mean()),
        by_season={
            int(s): float(g["poss_per_40"].mean())
            for s, g in merged.groupby("season")
        },
    )


def derive_minutes_baseline(points_per_win: float, poss_per_min: float) -> float:
    """B = 100 * points_per_win / poss_per_min. See module docstring."""
    return 100.0 * points_per_win / poss_per_min


def derive_replacement_level(
    minutes_baseline: float, cba: dict, replacement_win_pct: float
) -> float:
    """Replacement level in points per 100 possessions, as a positive magnitude.

    This does NOT require the player ratings. It falls out of an accounting
    identity: summed over every player, wins above replacement must equal the
    league's total wins above a replacement-level baseline.

        sum_players (minutes_i / B) * R  =  total_games * (1 - replacement_win_pct)

    Every player's minutes sum to total team-minutes, so

        R = total_games * (1 - replacement_win_pct) * B / total_team_minutes

    Estimating replacement from the ratings themselves (mean rating of
    minimum-salary signings) is then a *check* on this number rather than the
    primary method — and a weak one, since that population is small and noisy.
    """
    total_team_minutes = (
        cba["n_teams"] * cba["games_per_team"] * 5 * cba["minutes_per_game"]
    )
    total_games = cba["n_teams"] * cba["games_per_team"] / 2
    return total_games * (1 - replacement_win_pct) * minutes_baseline / total_team_minutes


def derive_dollars_per_win(cba: dict, replacement_win_pct: float) -> dict:
    """Dollars per win above replacement, priced out of *discretionary* cap space.

    Pricing against the full cap is wrong for the WNBA. A team must roster at
    least `roster_min` players and cannot pay any of them below the minimum, so
    that portion of the cap is not allocable. Only the remainder is money a team
    actually chooses how to spend.

    Using the full cap instead roughly doubles the implied price of a win and
    makes almost every rotation player look like surplus, which destroys the
    resolution you need in a league this salary-compressed.
    """
    committed = cba["roster_min"] * cba["min_salary"]
    discretionary_per_team = cba["salary_cap"] - committed
    league_discretionary = discretionary_per_team * cba["n_teams"]
    total_games = cba["n_teams"] * cba["games_per_team"] / 2
    league_war = total_games * (1 - replacement_win_pct)

    return {
        "committed_per_team": committed,
        "discretionary_per_team": discretionary_per_team,
        "league_discretionary": league_discretionary,
        "league_war": league_war,
        "dollars_per_win": league_discretionary / league_war,
        "naive_full_cap_dollars_per_win": (
            cba["salary_cap"] * cba["n_teams"] / league_war
        ),
    }


def minutes_profile(player_box: pd.DataFrame, season: int) -> dict:
    """Empirical minutes distribution, as a cross-check on the derived baseline.

    This does not feed the model. It exists so we can see whether the derived
    baseline lands somewhere sane relative to what players actually play — a
    baseline far above the league's busiest player would signal a broken
    derivation.
    """
    pb = player_box[(player_box["season"] == season) & player_box["minutes"].notna()]
    tot = pb.groupby("athlete_id")["minutes"].sum().sort_values(ascending=False)
    if tot.empty:
        return {}
    return {
        "season": season,
        "n_players": int(len(tot)),
        "max": float(tot.iloc[0]),
        "p90": float(tot.quantile(0.90)),
        "median": float(tot.median()),
        "mean_top_75": float(tot.head(75).mean()),   # ~5 starters x 15 teams
        "mean_top_180": float(tot.head(180).mean()),  # ~12 roster spots x 15
    }


def build() -> dict:
    seasons = range(2003, 2027)
    team_box = data.load("team_box", seasons)
    player_box = data.load("player_box", seasons)

    team_seasons = team_season_totals(team_box)
    ppw = estimate_points_per_win(team_seasons)
    pace = estimate_pace(team_box, player_box)
    baseline = derive_minutes_baseline(ppw.points_per_win, pace.poss_per_min)

    latest_complete = max(
        s for s in player_box["season"].unique() if s not in EXCLUDED_SEASONS
    )

    rwp = 0.25
    sensitivity = {
        f"{r:.2f}": {
            "replacement_level": derive_replacement_level(baseline, CBA_2026, r),
            "dollars_per_win": derive_dollars_per_win(CBA_2026, r)["dollars_per_win"],
        }
        for r in (0.20, 0.25, 0.30)
    }

    return {
        "cba": CBA_2026,
        "excluded_seasons": {str(k): v for k, v in EXCLUDED_SEASONS.items()},
        "points_per_win": asdict(ppw),
        "pace": asdict(pace),
        "minutes_baseline": {
            "value": baseline,
            "method": "derived: 100 * points_per_win / poss_per_min",
            "nba_reference": 1475,
            "note": (
                "Exceeds the busiest player's season minutes. That is expected, "
                "not a bug: B converts minutes to wins-per-unit-rating, it is not "
                "a share-of-season. The WNBA's 44-game season simply means nobody "
                "accumulates a full unit."
            ),
        },
        "minutes_profile": minutes_profile(player_box, int(latest_complete)),
        "replacement_win_pct": {
            "value": rwp,
            "method": "ASSUMPTION, not estimated - no replacement team exists",
            "sensitivity_range": [0.20, 0.30],
        },
        "replacement_level": {
            "value": derive_replacement_level(baseline, CBA_2026, rwp),
            "units": "points per 100 possessions below average",
            "method": "derived from league WAR accounting identity; needs no ratings",
            "nba_reference": 3.0,
        },
        "dollars_per_win": derive_dollars_per_win(CBA_2026, rwp),
        "sensitivity": sensitivity,
    }


def main() -> None:
    consts = build()
    data.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = data.PROCESSED_DIR / "constants.json"
    out.write_text(json.dumps(consts, indent=2))

    ppw = consts["points_per_win"]
    pace = consts["pace"]
    print(f"wrote {out}\n")
    print(f"points_per_win    {ppw['points_per_win']:.2f}  "
          f"[{ppw['ci_low']:.2f}, {ppw['ci_high']:.2f}]  "
          f"R2={ppw['r_squared']:.3f}  n={ppw['n_team_seasons']}")
    print(f"  intercept       {ppw['intercept']:.4f}   (expect ~0.500)")
    print(f"pace              {pace['poss_per_40']:.2f} poss/40min  "
          f"(game len {pace['mean_game_minutes']:.1f} min)")
    print(f"minutes_baseline  {consts['minutes_baseline']['value']:.0f}  "
          f"(NBA analog 1475)")
    print(f"replacement_level {consts['replacement_level']['value']:.2f} pts/100  "
          f"(DARKO/NBA reference 3.00)")

    dpw = consts["dollars_per_win"]
    print(f"dollars_per_win   ${dpw['dollars_per_win']:,.0f}  "
          f"(naive full-cap: ${dpw['naive_full_cap_dollars_per_win']:,.0f})")
    print(f"  discretionary   ${dpw['discretionary_per_team']:,.0f}/team of "
          f"${consts['cba']['salary_cap']:,} cap")

    print("\nsensitivity to replacement_win_pct:")
    for k, v in consts["sensitivity"].items():
        print(f"  {k}  repl={v['replacement_level']:.2f} pts/100   "
              f"${v['dollars_per_win']:,.0f}/win")


if __name__ == "__main__":
    main()
