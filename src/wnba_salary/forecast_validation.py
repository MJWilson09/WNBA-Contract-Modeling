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

from . import box_prior, data, draft_prior, espn_lineups, rapm, valuation

TEST_SEASONS = list(range(2018, 2027))
FIRST_SEASON = 2017
LAMBDA = 1500.0
HALF_LIFE = 1.5
LAMBDA_SWEEP = [1500.0, 6000.0, 24000.0]   # pooled candidate only
REPLACEMENT = 2.98

# Task 1 grid (PLAN.md). Extend either axis if the optimum lands on an edge.
LAMBDA_GRID = [1500, 3000, 6000, 12000, 24000]
HALF_LIFE_GRID = [0.25, 0.5, 0.75, 1.5, 3.0]

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


def prior_vector(players: list[str], prior_frame: pd.DataFrame) -> np.ndarray:
    """[obpm..., dbpm..., 0] aligned to `players`, weighted by `prior_frame.w`.

    Separate from `fit_rapm` because a (λ, half-life) sweep rebuilds the prior
    once per half-life but solves once per (λ, half-life) pair.
    """
    agg = (prior_frame.dropna(subset=["obpm", "dbpm"])
           .groupby("key")
           .apply(lambda g: pd.Series({
               "obpm": np.average(g["obpm"], weights=g["w"]),
               "dbpm": np.average(g["dbpm"], weights=g["w"])}),
               include_groups=False))
    off = np.nan_to_num(np.array([agg["obpm"].get(k, 0.0) for k in players], dtype=float))
    dfn = np.nan_to_num(np.array([agg["dbpm"].get(k, 0.0) for k in players], dtype=float))
    return np.concatenate([off, dfn, [0.0]])


def fit_rapm(design: dict, rows: pd.DataFrame, prior_frame: pd.DataFrame,
             anchor: int, lam: float, *, weighted: bool,
             half_life: float = HALF_LIFE) -> dict[str, tuple[float, float]]:
    """Fit ridge-to-prior on a prebuilt design; return name -> (o, d).

    The design is passed in rather than built here so the λ sweep can reuse it —
    building it is the expensive step (row filtering over every possession) and
    is identical across λ values.
    """
    players = design["players"]
    n = len(players)
    prior = prior_vector(players, prior_frame)

    weights = None
    if weighted:
        weights = 0.5 ** ((anchor - rows["season"].to_numpy(float)) / half_life)

    b = rapm.fit_ridge_prior(design["A"], design["y"], lam, prior, weights=weights)
    return {p: (b[i], b[n + i]) for i, p in enumerate(players)}


def prep_test(test_poss: pd.DataFrame, players: list[str]) -> dict:
    """Precompute index arrays for a test season.

    A grid sweep scores the same test possessions dozens of times against
    different coefficient vectors over the *same* player set, so the name lookup
    is hoisted out: each on-court slot becomes an integer index into `players`,
    or -1 for a player unseen in training (imputed league-average).
    """
    rows = test_poss[test_poss["garbage_time"] == 0].reset_index(drop=True)
    off, dfn = norm_cols(rows)
    idx = {p: i for i, p in enumerate(players)}
    lut = np.vectorize(lambda p: idx.get(p, -1), otypes=[np.int64])
    off_i, def_i = lut(off), lut(dfn)
    return {
        "off_i": off_i, "def_i": def_i,
        "game": rows["game_id"].to_numpy(),
        "team": rows["off_team"].astype(str).to_numpy(),
        "y": rows["pts"].to_numpy(float) * 100.0,
        "unseen": float(np.mean(np.concatenate([off_i.ravel(), def_i.ravel()]) < 0)),
    }


def score(prep: dict, o_vec: np.ndarray, d_vec: np.ndarray, c0: float) -> tuple[float, int]:
    """Per-game margin RMSE against precomputed indices. Unseen slots score 0."""
    o_pad = np.append(o_vec, 0.0)
    d_pad = np.append(d_vec, 0.0)
    yhat = o_pad[prep["off_i"]].sum(axis=1) - d_pad[prep["def_i"]].sum(axis=1) + c0

    t = pd.DataFrame({"g": prep["game"], "team": prep["team"],
                      "y": prep["y"], "p": yhat})
    s = t.groupby(["g", "team"], as_index=False).agg(y=("y", "sum"), p=("p", "sum"))
    s = s.sort_values(["g", "team"])
    first = s.groupby("g").nth(0).set_index("g")
    second = s.groupby("g").nth(1).set_index("g")
    m = first.join(second, lsuffix="_a", rsuffix="_b").dropna()
    err = (m["y_a"] - m["y_b"]) / 100.0 - (m["p_a"] - m["p_b"]) / 100.0
    return float(np.sqrt((err ** 2).mean())), len(m)


def margin_rmse(test_poss: pd.DataFrame, rating: dict[str, tuple[float, float]],
                c0: float) -> tuple[float, int, float]:
    """Per-game margin RMSE, n games, share of player-slots unseen in training."""
    players = list(rating)
    prep = prep_test(test_poss, players)
    o_vec = np.array([rating[p][0] for p in players], dtype=float) if players else np.zeros(0)
    d_vec = np.array([rating[p][1] for p in players], dtype=float) if players else np.zeros(0)
    rmse, n = score(prep, o_vec, d_vec, c0)
    return rmse, n, prep["unseen"]


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

    # Draft-slot priors for unseen rookies. Fitted strictly on seasons < T so the
    # curve never sees the ratings it is being scored against.
    dcoefs = draft_prior.fit(max_season=T)
    cands["pooled_draft"] = draft_prior.apply_to(cands["pooled"], T, dcoefs)

    row = {"T": T}
    for name, vec in cands.items():
        rmse, n, unseen = margin_rmse(test, vec, c0)
        row[name] = rmse
        if name == "pooled":
            row["n_games"], row["unseen"] = n, unseen
        if name == "pooled_draft":
            row["unseen_after"] = unseen

    sweep_row = {}
    for lam in LAMBDA_SWEEP:
        vec = cands["pooled"] if lam == LAMBDA else fit_rapm(
            train_design, train_rows, pooled_pf, T - 1, lam, weighted=True)
        sweep_row[lam] = margin_rmse(test, vec, c0)[0]
    return row, sweep_row


def sweep_season(T: int) -> tuple[int, dict[tuple[float, float], float]]:
    """Grid-sweep (λ, half-life) for the pooled candidate on one test season."""
    train = pd.concat([cached_season_poss(s) for s in range(FIRST_SEASON, T)],
                      ignore_index=True)
    test = cached_season_poss(T)
    c0 = float(train[train["garbage_time"] == 0]["pts"].mean() * 100)

    bp = pd.read_parquet(data.PROCESSED_DIR / "box_prior.parquet")
    bp["key"] = bp["athlete_display_name"].map(box_prior.normalize_name)
    base = bp[bp["season"] < T].copy()

    design = rapm.build_design(train, min_poss=200)
    rows = kept_rows(train, design["players"])
    players = design["players"]
    n = len(players)
    prep = prep_test(test, players)
    age = (T - 1) - rows["season"].to_numpy(float)

    out = {}
    for hl in HALF_LIFE_GRID:
        pf = base.copy()
        pf["w"] = pf["mp"] * 0.5 ** ((T - 1 - pf["season"]) / hl)
        prior = prior_vector(players, pf)
        weights = 0.5 ** (age / hl)
        for lam in LAMBDA_GRID:
            b = rapm.fit_ridge_prior(design["A"], design["y"], lam, prior,
                                     weights=weights)
            out[(lam, hl)] = score(prep, b[:n], b[n:2 * n], c0)[0]
    return T, out


def main_sweep() -> None:
    seasons = list(range(FIRST_SEASON, 2027))
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        list(ex.map(warm_cache, seasons))
    print(f"(λ × half-life) sweep: {len(LAMBDA_GRID)}×{len(HALF_LIFE_GRID)} "
          f"= {len(LAMBDA_GRID)*len(HALF_LIFE_GRID)} configs × "
          f"{len(TEST_SEASONS)} test seasons, {MAX_WORKERS} workers")

    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        out = list(ex.map(sweep_season, TEST_SEASONS))

    grid = pd.DataFrame(
        [[np.mean([d[(lam, hl)] for _, d in out]) for hl in HALF_LIFE_GRID]
         for lam in LAMBDA_GRID],
        index=[f"λ={int(l)}" for l in LAMBDA_GRID],
        columns=[f"HL={hl}" for hl in HALF_LIFE_GRID],
    )
    print("\nmean per-game margin RMSE (test seasons 2018-2026):")
    print(grid.to_string(float_format=lambda x: f"{x:.4f}"))

    flat = grid.stack()
    best = flat.idxmin()
    li, hi = LAMBDA_GRID.index(int(best[0].split("=")[1])), \
        HALF_LIFE_GRID.index(float(best[1].split("=")[1]))
    interior = (0 < li < len(LAMBDA_GRID) - 1) and (0 < hi < len(HALF_LIFE_GRID) - 1)
    print(f"\nbest: {best[0]}, {best[1]} -> {flat.min():.4f}")
    print(f"  interior on both axes: {interior}"
          f"{'' if interior else '  <-- EXTEND THE GRID (trap 4)'}")
    print(f"  vs production λ=1500/HL=1.5: {grid.loc['λ=1500','HL=1.5']:.4f}"
          f"  (gain {grid.loc['λ=1500','HL=1.5'] - flat.min():+.4f})")


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
    import sys
    main_sweep() if "--sweep" in sys.argv else main()
