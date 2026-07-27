"""Advanced rate stats computed from wehoop box scores.

Why compute these when Basketball-Reference publishes them? Because BBRef's WNBA
advanced table has no `DRB%` — it carries ORB% and TRB% only — and the 412
defensive model needs it. Rather than bodge DRB% out of TRB%, we compute the
whole family from raw box scores.

That raises a consistency risk: the NBA training set uses BBRef's definitions,
so if our formulas differ even slightly, coefficients fitted on one and applied
to the other are measuring different things. `validate_against_bbref()` closes
that gap directly — it recomputes the columns BBRef *does* publish for the WNBA
and checks they agree. If ORB% reproduces, the identical code path computing
DRB% is trustworthy.

Formulas follow Basketball-Reference's published definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_STATS = {
    "field_goals_made": "fg",
    "field_goals_attempted": "fga",
    "free_throws_attempted": "fta",
    "three_point_field_goals_attempted": "fg3a",
    "offensive_rebounds": "orb",
    "defensive_rebounds": "drb",
    "total_turnovers": "tov",
}


def team_season_context(team_box: pd.DataFrame, player_box: pd.DataFrame) -> pd.DataFrame:
    """Team and opponent season aggregates needed by the rate formulas."""
    tb = team_box.rename(columns=TEAM_STATS)
    keep = ["season", "game_id", "team_id", *TEAM_STATS.values()]
    tb = tb[keep]

    # opponent row = the other team in the same game
    opp = tb.rename(columns={c: f"opp_{c}" for c in TEAM_STATS.values()})
    opp = opp.rename(columns={"team_id": "opp_team_id"})
    merged = tb.merge(opp, on=["season", "game_id"])
    merged = merged[merged["team_id"] != merged["opp_team_id"]]

    agg_cols = list(TEAM_STATS.values()) + [f"opp_{c}" for c in TEAM_STATS.values()]
    team = merged.groupby(["season", "team_id"], as_index=False)[agg_cols].sum()

    # team minutes from player box: handles overtime without special-casing
    pm = player_box[player_box["minutes"].notna()]
    tm_mp = pm.groupby(["season", "team_id"], as_index=False)["minutes"].sum()
    tm_mp = tm_mp.rename(columns={"minutes": "tm_mp"})
    team = team.merge(tm_mp, on=["season", "team_id"], how="left")

    # possessions, for STL%
    team["opp_poss"] = (
        team["opp_fga"] + 0.44 * team["opp_fta"] - team["opp_orb"] + team["opp_tov"]
    )
    return team


def player_season_totals(player_box: pd.DataFrame) -> pd.DataFrame:
    """Season totals per player, with a minutes-weighted team context.

    Players traded mid-season played under more than one team context. BBRef
    computes each stint separately and combines; we approximate with a
    minutes-weighted blend of the team contexts, which is within rounding for
    all but heavily-traded players.
    """
    pb = player_box[player_box["minutes"].notna()].copy()
    pb["fg3a"] = pb["three_point_field_goals_attempted"]

    stats = {
        "minutes": "mp", "points": "pts", "field_goals_made": "fg",
        "field_goals_attempted": "fga", "free_throws_attempted": "fta",
        "fg3a": "fg3a", "offensive_rebounds": "orb", "defensive_rebounds": "drb",
        "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
        "fouls": "pf",
    }
    pb = pb.rename(columns=stats)

    by_team = (
        pb.groupby(["season", "athlete_id", "athlete_display_name", "team_id"], as_index=False)
        .agg({v: "sum" for v in stats.values()} | {"game_id": "nunique"})
        .rename(columns={"game_id": "g"})
    )
    return by_team


def compute_rates(
    player_box: pd.DataFrame, team_box: pd.DataFrame
) -> pd.DataFrame:
    """Player-season advanced rate stats, Basketball-Reference definitions."""
    team = team_season_context(team_box, player_box)
    stints = player_season_totals(player_box)

    df = stints.merge(team, on=["season", "team_id"], how="left", suffixes=("", "_tm"))

    # team-context columns carry team totals; disambiguate explicitly
    tm = {c: f"tm_{c}" for c in TEAM_STATS.values()}
    df = df.rename(columns={f"{v}_tm": f"tm_{v}" for v in TEAM_STATS.values()})

    mp = df["mp"].replace(0, np.nan)
    tm_mp5 = df["tm_mp"] / 5.0

    df["ts_pct"] = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"]))
    df["fg3a_rate"] = df["fg3a"] / df["fga"].replace(0, np.nan)
    df["ftr"] = df["fta"] / df["fga"].replace(0, np.nan)

    df["orb_pct"] = 100 * (df["orb"] * tm_mp5) / (mp * (df["tm_orb"] + df["opp_drb"]))
    df["drb_pct"] = 100 * (df["drb"] * tm_mp5) / (mp * (df["tm_drb"] + df["opp_orb"]))
    df["trb_pct"] = (
        100 * ((df["orb"] + df["drb"]) * tm_mp5)
        / (mp * (df["tm_orb"] + df["tm_drb"] + df["opp_orb"] + df["opp_drb"]))
    )

    df["ast_pct"] = 100 * df["ast"] / (
        ((mp / tm_mp5) * df["tm_fg"]) - df["fg"]
    )
    df["stl_pct"] = 100 * (df["stl"] * tm_mp5) / (mp * df["opp_poss"])
    df["blk_pct"] = 100 * (df["blk"] * tm_mp5) / (
        mp * (df["opp_fga"] - df["opp_fg3a"])
    )
    df["tov_pct"] = 100 * df["tov"] / (df["fga"] + 0.44 * df["fta"] + df["tov"])
    df["usg_pct"] = 100 * (
        (df["fga"] + 0.44 * df["fta"] + df["tov"]) * tm_mp5
    ) / (mp * (df["tm_fga"] + 0.44 * df["tm_fta"] + df["tm_tov"]))

    df["mpg"] = df["mp"] / df["g"].replace(0, np.nan)
    df["pf_per_40"] = 40 * df["pf"] / mp

    return combine_stints(df)


def combine_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-team seasons into one row per player-season.

    Counting stats sum; rate stats blend by minutes.
    """
    rate_cols = [
        "ts_pct", "fg3a_rate", "ftr", "orb_pct", "drb_pct", "trb_pct", "ast_pct",
        "stl_pct", "blk_pct", "tov_pct", "usg_pct", "mpg", "pf_per_40",
    ]
    count_cols = ["mp", "g", "pts", "fga", "fta", "orb", "drb", "ast", "stl", "blk", "tov", "pf"]

    def _combine(group: pd.DataFrame) -> pd.Series:
        w = group["mp"]
        out = {c: group[c].sum() for c in count_cols}
        for c in rate_cols:
            vals = group[c]
            mask = vals.notna()
            out[c] = np.average(vals[mask], weights=w[mask]) if mask.any() and w[mask].sum() > 0 else np.nan
        out["n_teams"] = group["team_id"].nunique()
        return pd.Series(out)

    combined = (
        df.groupby(["season", "athlete_id", "athlete_display_name"])
        .apply(_combine, include_groups=False)
        .reset_index()
    )
    return combined


def validate_against_bbref(
    computed: pd.DataFrame, bbref_wnba: pd.DataFrame, season: int
) -> pd.DataFrame:
    """Compare our computed rates to BBRef's published WNBA values.

    This is the check that licenses using our DRB% in a model whose other
    predictors come from BBRef. Correlations should be ~0.99+; systematic
    offsets indicate a formula mismatch.
    """
    from .box_prior import normalize_name

    c = computed[computed["season"] == season].copy()
    b = bbref_wnba[bbref_wnba["season"] == season].copy()
    c["key"] = c["athlete_display_name"].map(normalize_name)
    b["key"] = b["player"].map(normalize_name)

    m = c.merge(b, on="key", suffixes=("_ours", "_bbref"))
    m = m[m["mp_ours"] >= 100]

    rows = []
    for col in ["ts_pct", "orb_pct", "trb_pct", "ast_pct", "stl_pct",
                "blk_pct", "tov_pct", "usg_pct", "fg3a_rate", "ftr"]:
        a, bb = f"{col}_ours", f"{col}_bbref"
        if a not in m or bb not in m:
            continue
        sub = m[[a, bb]].dropna()
        if len(sub) < 10:
            continue
        rows.append({
            "stat": col,
            "n": len(sub),
            "corr": sub[a].corr(sub[bb]),
            "mean_ours": sub[a].mean(),
            "mean_bbref": sub[bb].mean(),
            "mean_abs_diff": (sub[a] - sub[bb]).abs().mean(),
        })
    return pd.DataFrame(rows)
