"""Forward validation: do the ratings predict NEXT season's games?

Everything else in this project is tuned descriptively — λ and the recency
half-life were chosen to predict held-out games *within* the fitted window. A
contract model prices future seasons, so the question that matters is forward:
fit on seasons ≤ t, predict season t+1.

Protocol
--------
For each test season T (2018–2026), fit candidate rating vectors on data from
seasons < T only, then predict every game of season T at the possession level
and score **per-game point margin RMSE**. Margin is the right target because it
is level-invariant: RAPM's overall level is unidentified (AGENTS.md trap 8), and
a constant added to every rating cancels in the margin, so candidates with
different implicit levels compare fairly. Players unseen in training get 0
(league average) — the realistic treatment of rookies.

Candidates
----------
zero        no player information (predicts ~0 margin) — the floor
box         season T-1 box prior only (shrunk obpm/dbpm)
rapm1       RAPM-to-prior fit on season T-1 only, λ=1500
pooled      RAPM-to-prior on all seasons < T, recency half-life 1.5 anchored at
            T-1, λ=1500 — the production recipe applied forward
pooled_age  pooled, plus one year of aging applied to every rating

Parallelism
-----------
Test seasons are independent, so they run in a ProcessPoolExecutor. macOS uses
spawn, so workers share nothing — possession frames are therefore cached to
parquet (data/processed/poss_cache/) and workers read them from disk rather
than rebuilding. All players are retained in every fit: RAPM is a joint
regression and the scored target is a full-game margin, so subsampling players
would bias the kept coefficients and mostly measure the imputation of the
dropped ones. Runtime scales with possessions, not players.

Caveats, disclosed: the box prior's NBA transfer coefficients and shrinkage
constants were fit once on all seasons (second-order leakage, as in
rapm_validation); 2020 is the 22-game COVID bubble; 2023 crosses the
stats→ESPN source boundary (names align via normalize_name, validated at
r=0.994 in 2022).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import box_prior, data, espn_lineups, rapm, valuation

TEST_SEASONS = list(range(2018, 2027))
FIRST_SEASON = 2017
LAMBDA = 1500.0
HALF_LIFE = 1.5
LAMBDA_SWEEP = [1500.0, 6000.0, 24000.0]   # pooled candidate only
REPLACEMENT = 2.98

POSS_CACHE = data.PROCESSED_DIR / "poss_cache"
MAX_WORKERS = min(4, os.cpu_count() or 1)


def cached_season_poss(season: int) -> pd.DataFrame:
    """Season possessions, cached to parquet so spawned workers can read them."""
    path = POSS_CACHE / f"poss_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if season <= 2022:
        df = rapm.build_possessions(rapm.fetch_stats_pbp(season))
    else:
        df = espn_lineups.reconstruct(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def warm_cache(season: int) -> int:
    cached_season_poss(season)
    return season


def norm_cols(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    off = np.column_stack([df[c].map(box_prior.normalize_name).to_numpy()
                           for c in rapm.OFF_NAME_COLS])
    dfn = np.column_stack([df[c].map(box_prior.normalize_name).to_numpy()
                           for c in rapm.DEF_NAME_COLS])
    return off, dfn


def kept_rows(poss: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    """Reproduce build_design's row filter so metadata aligns with A's rows."""
    kept = poss[poss["garbage_time"] == 0]
    off, dfn = norm_cols(kept)
    keep = set(players)
    ok = np.array([all(p in keep for p in off[i]) and all(p in keep for p in dfn[i])
                   for i in range(len(kept))])
    return kept[ok].reset_index(drop=True)


def fit_rapm(design: dict, rows: pd.DataFrame, prior_frame: pd.DataFrame,
             anchor: int, lam: float, *, weighted: bool) -> dict[str, tuple[float, float]]:
    """Fit ridge-to-prior on a prebuilt design; return name -> (o, d).

    The design is passed in rather than built here so the λ sweep can reuse it —
    building it is the expensive step (row filtering over every possession) and
    is identical across λ values.
    """
    players = design["players"]
    n = len(players)

    agg = (prior_frame.dropna(subset=["obpm", "dbpm"])
           .groupby("key")
           .apply(lambda g: pd.Series({
               "obpm": np.average(g["obpm"], weights=g["w"]),
               "dbpm": np.average(g["dbpm"], weights=g["w"])}),
               include_groups=False))
    off = np.nan_to_num(np.array([agg["obpm"].get(k, 0.0) for k in players], dtype=float))
    dfn = np.nan_to_num(np.array([agg["dbpm"].get(k, 0.0) for k in players], dtype=float))
    prior = np.concatenate([off, dfn, [0.0]])

    weights = None
    if weighted:
        weights = 0.5 ** ((anchor - rows["season"].to_numpy(float)) / HALF_LIFE)

    b = rapm.fit_ridge_prior(design["A"], design["y"], lam, prior, weights=weights)
    return {p: (b[i], b[n + i]) for i, p in enumerate(players)}


def margin_rmse(test_poss: pd.DataFrame, rating: dict[str, tuple[float, float]],
                c0: float) -> tuple[float, int, float]:
    """Per-game margin RMSE, n games, share of player-slots unseen in training."""
    rows = test_poss[test_poss["garbage_time"] == 0].reset_index(drop=True)
    off, dfn = norm_cols(rows)

    o = np.vectorize(lambda p: rating.get(p, (0.0, 0.0))[0])(off).sum(axis=1)
    d = np.vectorize(lambda p: rating.get(p, (0.0, 0.0))[1])(dfn).sum(axis=1)
    yhat = o - d + c0
    unseen = np.mean([p not in rating for p in np.concatenate([off.ravel(), dfn.ravel()])])

    t = pd.DataFrame({"g": rows["game_id"], "team": rows["off_team"].astype(str),
                      "y": rows["pts"].to_numpy(float) * 100.0, "p": yhat})
    s = t.groupby(["g", "team"], as_index=False).agg(y=("y", "sum"), p=("p", "sum"))
    s = s.sort_values(["g", "team"])
    first = s.groupby("g").nth(0).set_index("g")
    second = s.groupby("g").nth(1).set_index("g")
    m = first.join(second, lsuffix="_a", rsuffix="_b").dropna()
    err = (m["y_a"] - m["y_b"]) / 100.0 - (m["p_a"] - m["p_b"]) / 100.0
    return float(np.sqrt((err ** 2).mean())), len(m), float(unseen)


def evaluate_season(T: int) -> tuple[dict, dict]:
    """Everything for one test season. Runs in a worker process."""
    train = pd.concat([cached_season_poss(s) for s in range(FIRST_SEASON, T)],
                      ignore_index=True)
    test = cached_season_poss(T)
    c0 = float(train[train["garbage_time"] == 0]["pts"].mean() * 100)

    bp = pd.read_parquet(data.PROCESSED_DIR / "box_prior.parquet")
    bp["key"] = bp["athlete_display_name"].map(box_prior.normalize_name)
    prev = bp[bp["season"] == T - 1].copy()
    prev["w"] = prev["mp"]
    pooled_pf = bp[bp["season"] < T].copy()
    pooled_pf["w"] = pooled_pf["mp"] * 0.5 ** ((T - 1 - pooled_pf["season"]) / HALF_LIFE)

    # build each design once per test season; reused across candidates and λ
    train_design = rapm.build_design(train, min_poss=200)
    train_rows = kept_rows(train, train_design["players"])
    prev_poss = cached_season_poss(T - 1)
    prev_design = rapm.build_design(prev_poss, min_poss=200)
    prev_rows = kept_rows(prev_poss, prev_design["players"])

    cands = {
        "zero": {},
        "box": {r.key: (r.obpm, r.dbpm) for r in prev.itertuples()
                if not (pd.isna(r.obpm) or pd.isna(r.dbpm))},
        "rapm1": fit_rapm(prev_design, prev_rows, prev, T - 1, LAMBDA, weighted=False),
        "pooled": fit_rapm(train_design, train_rows, pooled_pf, T - 1, LAMBDA, weighted=True),
    }

    curve = valuation.load_aging_curve()
    core = data.fetch_season("player_core", T - 1)
    age_map = {} if core is None else dict(
        zip(core["display_name"].map(box_prior.normalize_name), core["age"]))
    aged = {}
    for p, (o, d) in cands["pooled"].items():
        a = age_map.get(p)
        if a is None or pd.isna(a):
            aged[p] = (o, d)
            continue
        new_total = valuation.project_rating(o + d, float(a), 1, curve, REPLACEMENT)
        delta = (new_total - (o + d)) / 2 if not np.isnan(new_total) else 0.0
        aged[p] = (o + delta, d + delta)
    cands["pooled_age"] = aged

    row = {"T": T}
    for name, vec in cands.items():
        rmse, n, unseen = margin_rmse(test, vec, c0)
        row[name] = rmse
        if name == "pooled":
            row["n_games"], row["unseen"] = n, unseen

    sweep_row = {}
    for lam in LAMBDA_SWEEP:
        vec = cands["pooled"] if lam == LAMBDA else fit_rapm(
            train_design, train_rows, pooled_pf, T - 1, lam, weighted=True)
        sweep_row[lam] = margin_rmse(test, vec, c0)[0]
    return row, sweep_row


def main() -> None:
    seasons = list(range(FIRST_SEASON, 2027))
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        list(ex.map(warm_cache, seasons))
    print(f"possession cache warm ({len(seasons)} seasons); "
          f"evaluating {len(TEST_SEASONS)} test seasons on {MAX_WORKERS} workers")

    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        out = list(ex.map(evaluate_season, TEST_SEASONS))

    df = pd.DataFrame([r for r, _ in out]).sort_values("T")
    print("\nper-game MARGIN RMSE by test season (lower = better forecast):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    pooled_mean = df.drop(columns=["T", "n_games", "unseen"]).mean()
    print("\nmean across test seasons:")
    print(pooled_mean.to_string(float_format=lambda x: f"{x:.4f}"))
    sweep = {lam: [s[lam] for _, s in out] for lam in LAMBDA_SWEEP}
    print("\npooled λ sweep (mean margin RMSE): " +
          "  ".join(f"λ={int(l)}: {np.mean(v):.4f}" for l, v in sweep.items()))
    print(f"mean unseen player-slot share: {df['unseen'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
