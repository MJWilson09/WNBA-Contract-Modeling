"""Season-by-season ratings, back to the first season with possession data.

The rest of the pipeline answers one question about one season: what is a player
worth in 2026. This stage runs the *same* rating recipe over every earlier season
so the site's league table can be read backwards — 2017 is as far as possessions
go (`rapm.AVAILABLE_SEASONS` starts there; before it the stats archive has no
lineups and ESPN substitutions are unreliable, AGENTS.md trap 21).

What moves with the season and what does not
--------------------------------------------
`ratings.build(target=T)` slides the whole recipe back: the pooled window
(T-3..T, truncated at 2017), the recency weights on both the box prior and the
possessions, and the minutes the level is pinned against. λ and the half-life are
held at the descriptive optimum — they were tuned on 2017-2022 holdouts, so
re-tuning them per season would be fitting the tuning set.

The *scale* constants do not move, and that is not an approximation:

  * `minutes_baseline` and `points_per_win` are pooled across 2003-2026 already.
  * `replacement_level` looks season-dependent but is not. Expanding the identity
    in `constants.derive_replacement_level`, team count and games cancel:
    R = (n·G/2)·(1-w)·B / (n·G·5·40) = 0.75·B/400. Same for every season.
  * The pin therefore reduces to setting the minutes-weighted mean rating to
    zero, which needs no CBA figures at all — only that season's minutes.

So ratings, and WAR per unit of minutes, are directly comparable across seasons.

Two normalisations, both deliberate
-----------------------------------
**Dollars are always 2026 dollars.** We have the 2026 CBA and no other. Pricing
2019 against the 2014 CBA would need a cap schedule we have not loaded, and
pricing it against the 2026 cap without saying so would be worse. Every season's
`value` answers "what would this season's play be worth under today's CBA",
which is also the only comparison anyone actually wants.

**Minutes are normalised to a 44-game season.** The league has played 34, 22 and
44-game seasons in this window, so raw seasonal WAR is mostly a season-length
artifact. `proj_minutes = mpg × 44 × availability` is the same formula the
current season already uses, so 2026 is unchanged and earlier seasons become
comparable to it. The consequence, stated plainly: summed WAR in a historical
season will not equal that season's actual league wins.

Salaries are 2026-only (AGENTS.md §7 item 4), so `salary`, `surplus` and
`signing` are null for every earlier season rather than guessed at.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import box_prior, data, ratings, valuation

SEASONS = list(range(ratings.FIRST_POSS_SEASON, valuation.CURRENT_SEASON + 1))
PRICING_SEASON = valuation.CURRENT_SEASON
MAX_WORKERS = min(4, os.cpu_count() or 1)

# All-Star rosters appear in the team box alongside real clubs.
NON_TEAMS = {"SPO", "COOP", "WEST", "EAST", "USA"}

OUT_NAME = "history"


def team_lookup(season: int) -> dict[int, str]:
    tb = data.load("team_box", [season])
    pairs = tb[["team_id", "team_abbreviation"]].drop_duplicates()
    return {
        int(r.team_id): r.team_abbreviation
        for r in pairs.itertuples()
        if r.team_abbreviation not in NON_TEAMS
    }


def ages_at_season(season: int) -> pd.Series:
    """Age during `season`, keyed on athlete_id.

    player_core's own `age` column is the player's age *now*, not during the
    season the file is named for — Sue Bird is listed at 45 in `player_core_2017`
    — so reading it directly would silently age every historical player by however
    many years have passed. It is recomputed from `date_of_birth` against July 1,
    roughly mid-season, and floored to whole years.
    """
    core = data.fetch_season("player_core", season, refresh=False)
    if core is None or "date_of_birth" not in core:
        return pd.Series(dtype=float)
    dob = pd.to_datetime(core["date_of_birth"], format="mixed", utc=True)
    ref = pd.Timestamp(f"{season}-07-01", tz="UTC")
    age = np.floor((ref - dob).dt.days / 365.25)
    return pd.Series(age.to_numpy(), index=core["athlete_id"].to_numpy()).dropna()


def valuation_frame(target: int, r: pd.DataFrame, consts: dict) -> pd.DataFrame:
    """One season's table: rated players, normalised minutes, 2026-dollar value.

    Mirrors `valuation.build` for the parts that are season-agnostic. It does not
    reuse it because that function is hardwired to the current season in the
    places that matter — salaries, the supermax test, the forward projection —
    and none of those have a historical analogue.
    """
    bp = pd.read_parquet(data.PROCESSED_DIR / "box_prior.parquet")
    # No minutes filter here: inclusion is decided by `valuation.include_mask`
    # after the ratings merge, so past seasons and the current one use one rule.
    cur = bp[bp["season"] == target].copy()
    cur["key"] = cur["athlete_display_name"].map(box_prior.normalize_name)

    # RAPM where the player cleared the possession floor, box prior otherwise —
    # the same fallback, and the same disclosure, as the current season.
    cur = cur.merge(
        r[["player", "rapm", "o_rapm", "d_rapm", "prior", "rating_se"]],
        left_on="key", right_on="player", how="left")
    cur["rating"] = cur["rapm"].where(cur["rapm"].notna(), cur["bpm"])
    cur["rating_source"] = np.where(cur["rapm"].notna(), "rapm", "box_prior")
    cur = cur[valuation.include_mask(cur)].copy()

    pbx = data.load("player_box", [target])
    player_team = (
        pbx[pbx["minutes"].notna()]
        .groupby("athlete_id")
        .agg(team_id=("team_id", "last"))
        .reset_index()
    )
    cur = cur.merge(player_team, on="athlete_id", how="left")
    cur = cur.merge(valuation.team_games_played(target), on="team_id", how="left")
    teams = team_lookup(target)
    cur["team"] = [teams.get(int(t), "—") if not pd.isna(t) else "—"
                   for t in cur["team_id"]]

    cur["availability"] = (cur["g"] / cur["team_games"]).clip(upper=1.0)
    cur["proj_minutes"] = (
        cur["mpg"] * consts["cba"]["games_per_team"] * cur["availability"])

    cur["age"] = cur["athlete_id"].map(ages_at_season(target))

    B = consts["minutes_baseline"]["value"]
    R = consts["replacement_level"]["value"]
    dpw = valuation.dollars_per_win(consts, PRICING_SEASON)
    sched = valuation.cba_schedule(PRICING_SEASON)

    cur["war"] = (cur["proj_minutes"] / B) * (cur["rating"] + R)
    cur["value"] = sched["min_salary"] + cur["war"] * dpw
    cur["value_se"] = (cur["proj_minutes"] / B) * dpw * cur["rating_se"]
    cur["season"] = target

    cols = ["season", "athlete_id", "athlete_display_name", "team", "pos", "age",
            "g", "mp", "mpg", "availability", "proj_minutes", "rating",
            "rating_source", "o_rapm", "d_rapm", "prior", "rating_se",
            "war", "value", "value_se"]
    return cur[cols].sort_values("value", ascending=False).reset_index(drop=True)


def build_season(target: int) -> tuple[pd.DataFrame, dict]:
    """Rate and value one season. Runs in a worker process."""
    consts = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    res = ratings.build(target=target)
    r = res["ratings"]
    frame = valuation_frame(target, r, consts)

    rated = frame[frame["rating_source"].eq("rapm")]
    meta = {
        "season": target,
        "pooled_seasons": res["seasons"],
        "n_poss": res["n_poss"],
        "n_players_fit": res["n_players"],
        "prior_matched": res["prior_matched"],
        "pin_offset": res["pin"]["offset"],
        "minutes_share": res["pin"]["minutes_share"],
        "n_table": int(len(frame)),
        "n_rapm": int(len(rated)),
        "rating_sd": float(frame["rating"].std()),
        "rating_se_median": float(frame["rating_se"].median()),
        "summed_war": float(frame["war"].sum()),
    }
    return frame, meta


def main() -> None:
    print(f"rating {len(SEASONS)} seasons ({SEASONS[0]}–{SEASONS[-1]}) "
          f"on {MAX_WORKERS} workers")
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        out = list(ex.map(build_season, SEASONS))

    frames = pd.concat([f for f, _ in out], ignore_index=True)
    metas = [m for _, m in out]

    frames.to_parquet(data.PROCESSED_DIR / f"{OUT_NAME}.parquet", index=False)
    (data.PROCESSED_DIR / f"{OUT_NAME}_meta.json").write_text(json.dumps(
        {"seasons": metas, "pricing_season": PRICING_SEASON,
         "lambda": ratings.LAMBDA, "half_life": ratings.HALF_LIFE,
         "window": ratings.WINDOW}, indent=2))

    md = pd.DataFrame(metas)
    md["pooled"] = md["pooled_seasons"].map(len)
    print()
    print(md[["season", "pooled", "n_poss", "n_players_fit", "n_table", "n_rapm",
              "pin_offset", "rating_sd", "rating_se_median", "summed_war"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # End-to-end check: the current season is produced twice by two different
    # code paths, so it must agree with the production artifact.
    prod = pd.read_parquet(data.PROCESSED_DIR / "valuation.parquet")
    mine = frames[frames["season"] == PRICING_SEASON]
    j = prod[["athlete_id", "rating", "war", "value"]].merge(
        mine[["athlete_id", "rating", "war", "value"]], on="athlete_id",
        suffixes=("_prod", "_hist"))
    print(f"\nvs valuation.parquet ({PRICING_SEASON}): {len(j)}/{len(prod)} matched, "
          f"max |Δrating| {(j.rating_prod - j.rating_hist).abs().max():.4f}, "
          f"max |Δvalue| ${(j.value_prod - j.value_hist).abs().max():,.0f}")
    print(f"wrote {OUT_NAME}.parquet ({len(frames)} player-seasons)")


if __name__ == "__main__":
    main()
