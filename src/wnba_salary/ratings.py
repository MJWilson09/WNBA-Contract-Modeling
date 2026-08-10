"""Current-season player ratings: RAPM on ESPN possessions, shrunk to the box prior.

This is the production rating pipeline — 412 stages 1-3 applied to the seasons the
salary model actually needs. `rapm.py` holds the solver and `rapm_validation.py`
the leak-free λ tuning; this module wires them to ESPN possessions and emits a
single ratings table for `valuation.py`.

Why the level needs pinning
---------------------------
The RAPM design has an exact degeneracy. Predictions are
`sum(off) - sum(def) + intercept`, so adding a constant `c` to *every* offensive
coefficient and to *every* defensive coefficient changes the offensive sum by
`+5c` and the defensive contribution by `-5c`, leaving every prediction
identical. The level is therefore not estimable from possessions at all — ridge
inherits it from whatever level the prior carries.

The pooled prior is a recency-weighted blend across four seasons, so its level is
not the target season's league average. Left alone this inflated summed WAR to
259.1 against a league-wide 247.5, when the rated subset should total ~223.

Fixing it means choosing a constraint. We use the same accounting identity that
`constants.py` uses to derive replacement level: summed player WAR must equal the
league's wins above replacement, pro-rated to the share of league minutes the
rated players cover. Because the level is unidentified, imposing a known
constraint on it is the correct resolution rather than a fudge.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import box_prior, data, rapm, rapm_validation

CURRENT_SEASONS = [2023, 2024, 2025, 2026]
TARGET_SEASON = 2026

# The pooled window is the target season plus the three before it. `history.py`
# slides the same window back over earlier seasons; `CURRENT_SEASONS` is just
# `pooled_window(2026)` written out.
WINDOW = 4
FIRST_POSS_SEASON = 2017        # earliest season with possession-level data

# Descriptive config — "what is she worth right now". λ is the interior optimum
# from the within-season chronological holdout (rapm_validation).
HALF_LIFE = 1.5
LAMBDA = 1500.0

# Forecast config — used only for the multi-year projection columns. Tuned on the
# FORWARD metric (forecast_validation --sweep): fit on seasons < T, predict season
# T's game margins. Optimal shrinkage is ~4x heavier than the descriptive optimum,
# because a model tuned to describe the current season over-trusts noisy
# possession data when asked to forecast. Interior on both axes; λ does most of
# the work (1500->6000 is +0.086 RMSE) while the half-life surface is shallow
# (0.5/0.75/1.5 all within 0.011).
FORECAST_HALF_LIFE = 0.75
FORECAST_LAMBDA = 6000.0

MIN_POSS = 200


def recency_weight(season: np.ndarray | pd.Series, target: int = TARGET_SEASON,
                   half_life: float = HALF_LIFE) -> np.ndarray:
    return 0.5 ** ((target - np.asarray(season, dtype=float)) / half_life)


def pooled_window(target: int, window: int = WINDOW) -> list[int]:
    """The seasons pooled to rate `target`: itself plus the `window - 1` before.

    Truncated at `FIRST_POSS_SEASON`, so the earliest targets pool fewer seasons
    and their ratings are correspondingly noisier — visible in `rating_se`.
    """
    return list(range(max(FIRST_POSS_SEASON, target - window + 1), target + 1))


def pooled_prior(seasons: list[int], players: list[str],
                 half_life: float = HALF_LIFE,
                 target: int = TARGET_SEASON) -> tuple[np.ndarray, int, pd.DataFrame]:
    """Recency-weighted box prior aligned to the design's player order."""
    off_fit, def_fit = box_prior.fit_nba_models()
    positions = box_prior.wnba_positions(seasons)
    meta = json.loads((data.PROCESSED_DIR / "box_prior_fit.json").read_text())

    prior_df = box_prior.prior_from_box(
        data.load("player_box", seasons),
        data.load("team_box", seasons),
        off_fit, def_fit, positions,
        targets=meta["replacement_targets"],
        off_k=meta["shrinkage"]["offense"]["k"],
        def_k=meta["shrinkage"]["defense"]["k"],
    )
    prior_df["w"] = prior_df["mp"] * recency_weight(prior_df["season"],
                                                    target=target,
                                                    half_life=half_life)

    agg = (
        prior_df.dropna(subset=["obpm", "dbpm"])
        .groupby("key")
        .apply(
            lambda g: pd.Series({
                "obpm": np.average(g["obpm"], weights=g["w"]),
                "dbpm": np.average(g["dbpm"], weights=g["w"]),
            }),
            include_groups=False,
        )
    )
    off = np.nan_to_num(np.array([agg["obpm"].get(k, 0.0) for k in players], dtype=float))
    dfn = np.nan_to_num(np.array([agg["dbpm"].get(k, 0.0) for k in players], dtype=float))
    matched = sum(k in agg.index for k in players)
    return np.concatenate([off, dfn, [0.0]]), matched, prior_df


def target_minutes(season: int = TARGET_SEASON) -> pd.Series:
    """Target-season minutes per player, keyed on normalised name."""
    pb = data.load("player_box", [season])
    pb = pb[pb["minutes"].notna()].copy()
    pb["key"] = pb["athlete_display_name"].map(box_prior.normalize_name)
    return pb.groupby("key")["minutes"].sum()


def pin_level(ratings: pd.DataFrame, minutes: pd.Series, consts: dict) -> dict:
    """Shift ratings so summed WAR satisfies the league accounting identity.

    Solves for the single offset `c` in

        sum_i (mp_i / B) * (rating_i + c + R) = target_WAR

    where target_WAR is league wins above replacement scaled by the rated
    players' share of league minutes. Applied equally to offence and defence
    (half each) so the split is preserved.
    """
    B = consts["minutes_baseline"]["value"]
    R = consts["replacement_level"]["value"]
    cba = consts["cba"]
    league_minutes = cba["n_teams"] * cba["games_per_team"] * 5 * cba["minutes_per_game"]
    league_war = (cba["n_teams"] * cba["games_per_team"] / 2) * (
        1 - consts["replacement_win_pct"]["value"]
    )

    mp = ratings["player"].map(minutes).to_numpy(float)
    ok = ~np.isnan(mp) & (mp > 0)
    mp = np.where(ok, mp, 0.0)
    rating = ratings["rapm"].to_numpy(float)

    covered = mp.sum() / league_minutes
    target_war = league_war * covered

    offset = (target_war * B - float((mp * rating).sum())) / mp.sum() - R
    return {
        "offset": float(offset),
        "minutes_share": float(covered),
        "target_war": float(target_war),
        "rated_minutes": float(mp.sum()),
    }


def build(lam: float = LAMBDA, half_life: float = HALF_LIFE, *,
          target: int = TARGET_SEASON, seasons: list[int] | None = None) -> dict:
    """Rate `target` from the pooled window ending at it.

    `target` is a parameter rather than a constant so the same recipe can be run
    over earlier seasons (`history.py`). Everything that anchors to "now" —
    the recency weights on both the prior and the possessions, and the minutes
    the level is pinned against — moves with it.
    """
    seasons = list(seasons) if seasons is not None else pooled_window(target)
    poss = pd.concat(
        [rapm.season_possessions(s) for s in seasons], ignore_index=True
    )
    design = rapm.build_design(poss, min_poss=MIN_POSS)
    players = design["players"]
    n = len(players)

    prior, matched, _ = pooled_prior(seasons, players, half_life=half_life,
                                     target=target)

    kept = poss[poss["garbage_time"] == 0]
    offn = np.column_stack([kept[c].map(box_prior.normalize_name).to_numpy()
                            for c in rapm.OFF_NAME_COLS])
    dfnn = np.column_stack([kept[c].map(box_prior.normalize_name).to_numpy()
                            for c in rapm.DEF_NAME_COLS])
    keep = set(players)
    ok = np.array([
        all(p in keep for p in offn[i]) and all(p in keep for p in dfnn[i])
        for i in range(len(kept))
    ])
    rows = kept[ok].reset_index(drop=True)

    weights = recency_weight(rows["season"].to_numpy(), target=target,
                             half_life=half_life)
    b = rapm.fit_ridge_prior(design["A"], design["y"], lam, prior, weights=weights)
    se = rapm.posterior_se(design["A"], design["y"], lam, b, n, weights=weights)

    r = rapm.ratings_frame(players, b, design)
    r["o_se"] = se["o_se"]
    r["d_se"] = se["d_se"]
    r["rating_se"] = se["rating_se"]
    r["o_prior"] = prior[:n]
    r["d_prior"] = prior[n:2 * n]
    r["prior"] = r["o_prior"] + r["d_prior"]
    r["rapm_unpinned"] = r["rapm"]

    consts = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    minutes = target_minutes(target)
    pin = pin_level(r, minutes, consts)

    # split the offset evenly across offence and defence to preserve the split
    r["o_rapm"] = r["o_rapm"] + pin["offset"] / 2
    r["d_rapm"] = r["d_rapm"] + pin["offset"] / 2
    r["rapm"] = r["o_rapm"] + r["d_rapm"]
    r["d_delta"] = r["d_rapm"] - r["d_prior"]

    r["target_minutes"] = r["player"].map(minutes)
    return {"ratings": r, "pin": pin, "prior_matched": matched,
            "n_players": n, "n_poss": design["n_poss"], "seasons": seasons,
            "target": target,
            "lambda": lam, "half_life": half_life, "sigma2": se["sigma2"]}


def report(res: dict, label: str, out_name: str) -> pd.DataFrame:
    """Print diagnostics for one config and write its artifacts."""
    r, pin = res["ratings"], res["pin"]
    consts = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    B = consts["minutes_baseline"]["value"]
    R = consts["replacement_level"]["value"]

    def summed_war(col: str) -> float:
        mp = r["target_minutes"].to_numpy(float)
        v = r[col].to_numpy(float)
        m = ~np.isnan(mp) & (mp > 0)
        return float(((mp[m] / B) * (v[m] + R)).sum())

    print(f"=== {label} ===")
    print(f"possessions {res['n_poss']:,}  players {res['n_players']}  "
          f"prior matched {res['prior_matched']}/{res['n_players']}")
    print(f"lambda {res['lambda']:,.0f}  half-life {res['half_life']} seasons")
    print(f"  rated players cover {pin['minutes_share']*100:.1f}% of league minutes")
    print(f"  offset applied      {pin['offset']:+.3f} pts/100")
    print(f"  summed WAR  {summed_war('rapm_unpinned'):.1f} -> {summed_war('rapm'):.1f}"
          f"  (target {pin['target_war']:.1f})")

    mp = r["target_minutes"].to_numpy(float)
    m = ~np.isnan(mp) & (mp > 0)
    print(f"  minutes-wtd mean    {np.average(r['rapm'].to_numpy(float)[m], weights=mp[m]):+.3f}")
    print(f"  rating sd           {r['rapm'].std():.2f}")
    print(f"  rating_se  median {r['rating_se'].median():.2f}  "
          f"p10 {r['rating_se'].quantile(.10):.2f}  p90 {r['rating_se'].quantile(.90):.2f}")

    r.to_parquet(data.PROCESSED_DIR / f"{out_name}.parquet", index=False)
    (data.PROCESSED_DIR / f"{out_name}_meta.json").write_text(json.dumps(
        {"lambda": res["lambda"], "half_life": res["half_life"],
         "seasons": res["seasons"], "target_season": res["target"], "pin": pin,
         "n_players": res["n_players"], "n_poss": res["n_poss"]}, indent=2))
    return r


def main() -> None:
    res = build()
    r = report(res, "descriptive (current-season value)", "ratings")

    fc = build(lam=FORECAST_LAMBDA, half_life=FORECAST_HALF_LIFE)
    print()
    fr = report(fc, "forecast (multi-year projections)", "ratings_forecast")

    merged = r[["player", "rapm"]].merge(
        fc["ratings"][["player", "rapm"]], on="player", suffixes=("_desc", "_fc"))
    print(f"\ndescriptive vs forecast: corr {merged['rapm_desc'].corr(merged['rapm_fc']):.4f}"
          f"  mean|diff| {(merged['rapm_desc']-merged['rapm_fc']).abs().mean():.3f}")

    cur = r[r["target_minutes"].notna() & (r["target_minutes"] >= 100)]
    print(f"\nTop 12 by descriptive rating, {TARGET_SEASON} ({len(cur)} qualified):")
    print(cur.nlargest(12, "rapm")[
        ["player", "o_rapm", "d_rapm", "rapm", "prior", "target_minutes"]
    ].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"\nwrote ratings.parquet, ratings_forecast.parquet (+ _meta.json each)")


if __name__ == "__main__":
    main()
