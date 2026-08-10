"""Possession-level RAPM, shrunk toward the box prior (412 stages 2-3).

The method:

    min_b ||Ab - y||^2 + lambda * ||b - b_box||^2
    =>  (A'WA + lambda*I) b = A'Wy + lambda*b_box

`A` has one row per possession and 2N+1 columns: +1 for each of the five
offensive players, -1 for each of the five defensive players, plus an
unpenalised intercept. `y` is points per 100 possessions. Under the -1 encoding
a larger defensive coefficient means better defence, matching DBPM's sign
convention, so the prior vector is simply [obpm..., dbpm...].

Shrinking toward the box prior rather than toward zero is the entire point.
WNBA single-season RAPM has roughly 34,000 possessions against ~300 parameters
— about half the NBA's data per parameter, against a response variable that is
nearly pure noise at the possession level. Vanilla ridge would be unusable.

Data source
-----------
The WNBA Stats API play-by-play archived in `sportsdataverse/wehoop-data`, which
ships pre-computed on-court lineups, possession flags and a garbage-time flag —
the same structure `wehoop::wnba_possession_lineups()` produces, and the same
thing the 412 author obtained privately. It requires no live API call.

**Coverage is 2017-2022 only.** Later seasons need lineups reconstructed from
ESPN play-by-play; the overlap years are the natural validation oracle for that
work.

Player identity
---------------
The stats feed uses WNBA person IDs while the box prior uses ESPN athlete IDs,
so players are keyed on normalised names throughout.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from . import box_prior, data, espn_lineups

STATS_BASE = "https://raw.githubusercontent.com/sportsdataverse/wehoop-data/main"
AVAILABLE_SEASONS = range(2017, 2023)

OFF_NAME_COLS = [f"off_player{i}" for i in range(1, 6)]
DEF_NAME_COLS = [f"def_player{i}" for i in range(1, 6)]

MAX_POSS_POINTS = 6   # and-one plus a technical; anything above is a data error

POSS_CACHE = data.PROCESSED_DIR / "poss_cache"


def fetch_stats_pbp(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Archived WNBA Stats play-by-play for one season."""
    path = data.RAW_DIR / "stats_pbp" / f"stats_pbp_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    url = f"{STATS_BASE}/wnba_stats/pbp/parquet/play_by_play_{season}.parquet"
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def build_possessions(pbp: pd.DataFrame) -> pd.DataFrame:
    """Collapse event-level play-by-play to one row per possession.

    A possession is a maximal run of events with the same offensive team inside a
    period. That is the definition, and it reconciles: it captures 99.8% of the
    season's points at 101.4 per 100 for 2022, matching the actual league
    offensive rating.

    The `possession` column in this feed is NOT a usable basis for grouping —
    building possession ids from it loses ~13-40% of points depending on how
    points are attributed, because its boundaries do not align with changes of
    offensive team. It is ignored here.

    Two other details that matter:
      * Period-start rows are duplicated ten times (once per player on court), so
        events are deduplicated on (game_id, number_event) first. Left in, they
        would multiply any points landing on those rows.
      * Points are taken from the offensive team's own score column after the
        possession's offensive team is known, not per-event.
    """
    df = pbp.sort_values(["game_id", "number_event"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["game_id", "number_event"], keep="first")

    df["off_ff"] = df.groupby("game_id")["off_slug_team"].ffill()
    changed = (
        df["off_ff"] != df.groupby("game_id")["off_ff"].shift(1)
    ) | (df["period"] != df.groupby("game_id")["period"].shift(1))
    df["pid"] = changed.groupby(df["game_id"]).cumsum()

    df["pts_home"] = df["shot_pts_home"].fillna(0)
    df["pts_away"] = df["shot_pts_away"].fillna(0)

    def first_valid(s):
        v = s.dropna()
        return v.iloc[0] if len(v) else np.nan

    agg = {
        "pts_home": ("pts_home", "sum"), "pts_away": ("pts_away", "sum"),
        "garbage_time": ("garbage_time", "max"),
        "season": ("season", "first"), "game_date": ("game_date", "first"),
        "period": ("period", "first"),
        "team_home": ("team_home", first_valid),
        "off_team": ("off_ff", "first"),
        "def_team": ("def_slug_team", first_valid),
    }
    for col in OFF_NAME_COLS + DEF_NAME_COLS:
        agg[col] = (col, first_valid)

    poss = df.groupby(["game_id", "pid"], as_index=False).agg(**agg)
    poss["pts"] = np.where(
        poss["off_team"] == poss["team_home"], poss["pts_home"], poss["pts_away"]
    )
    poss = poss.drop(columns=["pts_home", "pts_away", "team_home"])
    poss = poss.dropna(subset=OFF_NAME_COLS + DEF_NAME_COLS + ["off_team"])

    bad = poss["pts"] > MAX_POSS_POINTS
    if bad.any():
        poss = poss[~bad]
    poss.attrs["dropped_bad_points"] = int(bad.sum())
    return poss.reset_index(drop=True)


def season_possessions(season: int) -> pd.DataFrame:
    """One season of possessions, from whichever source covers it, cached to disk.

    The two sources are not interchangeable code paths but they produce the same
    frame: the archived stats feed through 2022, ESPN reconstruction afterwards.
    The 2022 overlap is the oracle that licenses the switch (r=0.9937 on ratings;
    `espn_lineups.validate_against_stats`).

    Cached to `data/processed/poss_cache/` because several callers rebuild the
    same seasons — the multi-season history sweep and `forecast_validation`'s
    spawned workers, which inherit nothing on macOS. Delete the directory to
    force a rebuild after changing `build_possessions` or `reconstruct`.
    """
    path = POSS_CACHE / f"poss_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = (build_possessions(fetch_stats_pbp(season)) if season in AVAILABLE_SEASONS
          else espn_lineups.reconstruct(season))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def build_design(
    poss: pd.DataFrame, *, exclude_garbage: bool = True, min_poss: int = 50
) -> dict:
    """Sparse design matrix keyed on normalised player names."""
    df = poss[poss["garbage_time"] == 0] if exclude_garbage else poss

    off = np.column_stack([df[c].map(box_prior.normalize_name).to_numpy() for c in OFF_NAME_COLS])
    dfn = np.column_stack([df[c].map(box_prior.normalize_name).to_numpy() for c in DEF_NAME_COLS])

    # drop players below a possession floor; they cannot be identified at all
    counts = pd.Series(np.concatenate([off.ravel(), dfn.ravel()])).value_counts()
    keep = set(counts[counts >= min_poss].index)
    row_ok = np.array([
        all(p in keep for p in off[i]) and all(p in keep for p in dfn[i])
        for i in range(len(df))
    ])
    df, off, dfn = df[row_ok], off[row_ok], dfn[row_ok]

    players = sorted(keep)
    idx = {p: i for i, p in enumerate(players)}
    n, p = len(df), len(players)

    rows = np.repeat(np.arange(n), 10)
    cols = np.empty(n * 10, dtype=np.int64)
    vals = np.empty(n * 10, dtype=np.float64)
    for k in range(5):
        cols[k::10] = [idx[x] for x in off[:, k]]
        vals[k::10] = 1.0
    for k in range(5):
        cols[5 + k::10] = [p + idx[x] for x in dfn[:, k]]
        vals[5 + k::10] = -1.0

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, 2 * p))
    intercept = sparse.csr_matrix(np.ones((n, 1)))
    A = sparse.hstack([A, intercept], format="csr")

    return {
        "A": A,
        "y": df["pts"].to_numpy(float) * 100.0,
        "players": players,
        "dates": pd.to_datetime(df["game_date"]).to_numpy(),
        "n_poss": n,
    }


def prior_vector(players: list[str], prior_df: pd.DataFrame, season: int) -> np.ndarray:
    """[obpm..., dbpm..., 0] aligned to `players`; league mean where unmatched."""
    p = prior_df[prior_df["season"] == season].copy()
    p["key"] = p["athlete_display_name"].map(box_prior.normalize_name)
    p = p.drop_duplicates("key").set_index("key")

    off = np.zeros(len(players))
    dfn = np.zeros(len(players))
    matched = 0
    for i, name in enumerate(players):
        if name in p.index:
            off[i] = p.at[name, "obpm"]
            dfn[i] = p.at[name, "dbpm"]
            matched += 1
    # A player can match by name yet carry a NaN rating (undefined rate stats on
    # a tiny sample). Any NaN in the prior propagates through the solve and
    # silently returns an all-NaN coefficient vector, so it is zeroed here —
    # zero being the league mean on this scale.
    vec = np.nan_to_num(np.concatenate([off, dfn, [0.0]]), nan=0.0)
    return vec, matched


def fit_ridge_prior(
    A: sparse.csr_matrix, y: np.ndarray, lam: float, prior: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve (A'WA + lambda*I) b = A'Wy + lambda*b_prior, intercept unpenalised."""
    if weights is None:
        AtA = (A.T @ A).toarray()
        Aty = A.T @ y
    else:
        W = sparse.diags(weights)
        AtA = (A.T @ W @ A).toarray()
        Aty = A.T @ (weights * y)

    k = A.shape[1]
    pen = np.full(k, lam)
    pen[-1] = 0.0                      # never penalise the intercept
    AtA[np.diag_indices(k)] += pen
    rhs = Aty + pen * prior
    return np.linalg.solve(AtA, rhs)


def select_lambda(
    design: dict, prior: np.ndarray, grid: np.ndarray, *, holdout_frac: float = 0.25
) -> dict:
    """Choose lambda on a chronological holdout.

    The split is by date, not random: the box prior is built from full-season box
    scores, so a random split would leak the validation possessions into the
    prior that is being tuned against them.
    """
    order = np.argsort(design["dates"])
    cut = int(len(order) * (1 - holdout_frac))
    tr, te = order[:cut], order[cut:]

    A, y = design["A"], design["y"]
    A_tr, y_tr, A_te, y_te = A[tr], y[tr], A[te], y[te]

    results = []
    for lam in grid:
        b = fit_ridge_prior(A_tr, y_tr, lam, prior)
        resid = y_te - A_te @ b
        results.append({"lambda": float(lam), "rmse": float(np.sqrt((resid**2).mean()))})

    best = min(results, key=lambda r: r["rmse"])
    return {"best_lambda": best["lambda"], "best_rmse": best["rmse"],
            "curve": results, "n_train": len(tr), "n_test": len(te)}


def posterior_se(
    A: sparse.csr_matrix, y: np.ndarray, lam: float, b: np.ndarray,
    n_players: int, weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-player standard errors from the ridge posterior.

        V = sigma^2 * (A'WA + lambda*I)^-1

    This is the Bayesian posterior covariance under a N(b_box, sigma^2/lambda)
    prior — i.e. it **treats the box prior as truth**. It is therefore an
    approximate credible interval, not a frequentist standard error, and it is
    optimistic for players whose prior is itself poorly determined (few minutes,
    or a defensive specialist the box score cannot see). Read it as "how much do
    the possessions pin this down, given the prior", not as a full accounting of
    uncertainty.

    The total rating is `o + d`, so its variance needs the cross term:
    `var(o) + var(d) + 2cov(o, d)`. That is why the full inverse is kept rather
    than just the diagonal.
    """
    resid = y - A @ b
    if weights is None:
        sigma2 = float((resid ** 2).mean())
        AtA = (A.T @ A).toarray()
    else:
        sigma2 = float(np.average(resid ** 2, weights=weights))
        W = sparse.diags(weights)
        AtA = (A.T @ W @ A).toarray()

    k = A.shape[1]
    pen = np.full(k, lam)
    pen[-1] = 0.0                      # intercept is unpenalised
    AtA[np.diag_indices(k)] += pen
    cov = sigma2 * np.linalg.inv(AtA)

    p = n_players
    idx = np.arange(p)
    var_o = np.diag(cov)[:p]
    var_d = np.diag(cov)[p:2 * p]
    cov_od = cov[idx, p + idx]
    var_total = np.clip(var_o + var_d + 2 * cov_od, 0.0, None)
    return {
        "o_se": np.sqrt(np.clip(var_o, 0.0, None)),
        "d_se": np.sqrt(np.clip(var_d, 0.0, None)),
        "rating_se": np.sqrt(var_total),
        "sigma2": sigma2,
    }


def ratings_frame(players: list[str], b: np.ndarray, design: dict) -> pd.DataFrame:
    p = len(players)
    A = design["A"]
    off_poss = np.asarray((A[:, :p] > 0).sum(axis=0)).ravel()
    def_poss = np.asarray((A[:, p:2 * p] < 0).sum(axis=0)).ravel()
    return pd.DataFrame({
        "player": players,
        "o_rapm": b[:p],
        "d_rapm": b[p:2 * p],
        "rapm": b[:p] + b[p:2 * p],
        "off_poss": off_poss,
        "def_poss": def_poss,
    })


def build(season: int = 2022) -> dict:
    pbp = fetch_stats_pbp(season)
    poss = build_possessions(pbp)
    design = build_design(poss)
    prior_df = pd.read_parquet(data.PROCESSED_DIR / "box_prior.parquet")
    prior, matched = prior_vector(design["players"], prior_df, season)

    grid = np.array([100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000], float)
    tuned = select_lambda(design, prior, grid)
    lam = tuned["best_lambda"]

    b_prior_shrunk = fit_ridge_prior(design["A"], design["y"], lam, prior)
    b_zero_shrunk = fit_ridge_prior(design["A"], design["y"], lam, np.zeros_like(prior))

    return {
        "season": season,
        "poss": poss,
        "design": design,
        "prior": prior,
        "prior_matched": matched,
        "tuned": tuned,
        "ratings": ratings_frame(design["players"], b_prior_shrunk, design),
        "ratings_zero": ratings_frame(design["players"], b_zero_shrunk, design),
    }


def main() -> None:
    res = build(2022)
    design, tuned = res["design"], res["tuned"]
    n_players = len(design["players"])

    print(f"WNBA {res['season']} possession RAPM")
    print(f"  possessions used : {design['n_poss']:,} "
          f"(garbage time excluded, {res['poss'].attrs.get('dropped_bad_points',0)} bad-points rows dropped)")
    print(f"  players          : {n_players}  -> {design['A'].shape[1]} columns")
    print(f"  obs per parameter: {design['n_poss']/design['A'].shape[1]:.0f}")
    print(f"  prior matched    : {res['prior_matched']}/{n_players}")
    print(f"  lambda           : {tuned['best_lambda']:,.0f}  "
          f"(holdout RMSE {tuned['best_rmse']:.2f}, chronological split)")

    print("\n  lambda curve:")
    for r in tuned["curve"]:
        mark = "  <-- best" if r["lambda"] == tuned["best_lambda"] else ""
        print(f"    {r['lambda']:>7,.0f}  RMSE {r['rmse']:.3f}{mark}")

    rt = res["ratings"].merge(
        res["ratings_zero"][["player", "rapm"]], on="player", suffixes=("", "_zero")
    )
    print(f"\n  dispersion: sd(rapm to prior)={rt['rapm'].std():.2f}  "
          f"sd(rapm to zero)={rt['rapm_zero'].std():.2f}")

    print(f"\nTop 15, {res['season']} (shrunk to box prior):")
    top = rt.nlargest(15, "rapm")[
        ["player", "o_rapm", "d_rapm", "rapm", "rapm_zero", "off_poss"]
    ]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    out = data.PROCESSED_DIR / f"rapm_{res['season']}.parquet"
    rt.to_parquet(out, index=False)
    (data.PROCESSED_DIR / f"rapm_{res['season']}_fit.json").write_text(
        json.dumps({k: v for k, v in tuned.items()}, indent=2)
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
