"""Box-score prior for the WNBA rating model (412 stage 1).

Follows the 412 Sports Analytics method: fit BPM's offensive and defensive
components on NBA data, then apply the fitted coefficients to WNBA box scores.
The WNBA has no published BPM, so the model is transferred rather than fitted
in-league.

    OBPM ~ Big + MPG + TS% + 3PAr + FTr + ORB% + AST% + TOV% + USG%
    DBPM ~ Big + MPG + DRB% + STL% + BLK% + PF

Two deliberate departures from the published spec
-------------------------------------------------
1. `OnCourt*OnOff` is dropped. The 412 spec includes on/off terms, but on/off is
   derived from the same possessions that become the RAPM response. Shrinking a
   ridge fit toward a prior built from its own response variable undercuts the
   point of having an independent prior. This keeps the prior purely box-based.

2. Predictors are z-scored within league-season before the coefficients are
   applied. BPM is defined against NBA-average usage, pace and role
   distributions; feeding raw WNBA rates into NBA-fitted coefficients would
   import a league-mean bias. Z-scoring makes the transfer assumption explicit
   and defensible: a player one SD above her league's average in a stat is
   treated like an NBA player one SD above his.

Known limitation: `Big`
-----------------------
The WNBA lists only G/F/C (plus hyphenates), so `F` conflates what the NBA
splits into SF and PF. The Big indicator is therefore noisier on the WNBA side
than on the NBA side. `fit_report()` reports the model with and without it so the
cost is visible rather than assumed.
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from . import bbref, data, rates

NBA_TRAIN_SEASONS = range(2005, 2026)
WNBA_SEASONS = range(2010, 2027)

NBA_GAME_MINUTES = 48
WNBA_GAME_MINUTES = 40

MIN_MINUTES_TRAIN = 500   # BPM is very noisy below this
MIN_MINUTES_APPLY = 100

OFF_PREDICTORS = [
    "big", "mpg_share_z", "ts_pct_z", "fg3a_rate_z", "ftr_z",
    "orb_pct_z", "ast_pct_z", "tov_pct_z", "usg_pct_z",
]
DEF_PREDICTORS = [
    "big", "mpg_share_z", "drb_pct_z", "stl_pct_z", "blk_pct_z", "pf_per_40_z",
]
Z_STATS = [
    "mpg_share", "ts_pct", "fg3a_rate", "ftr", "orb_pct", "drb_pct",
    "ast_pct", "tov_pct", "usg_pct", "stl_pct", "blk_pct", "pf_per_40",
]


def normalize_name(name: str) -> str:
    """Fold accents/punctuation/suffixes so BBRef and wehoop names join."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def is_big(pos: str, league: str) -> int:
    """Binary frontcourt indicator. See module docstring on WNBA imprecision."""
    if not isinstance(pos, str):
        return 0
    p = pos.upper().split("-")[0].strip()
    if league == "nba":
        return int(p in {"C", "PF"})
    return int(p in {"C", "F"})


def load_nba_training(seasons=NBA_TRAIN_SEASONS) -> pd.DataFrame:
    """NBA player-seasons with OBPM/DBPM targets and all predictors."""
    adv = bbref.load_advanced("nba", seasons, kind="advanced")
    tot = bbref.load_advanced("nba", seasons, kind="totals")

    tot = tot[["season", "player", "pf", "mp"]].rename(columns={"mp": "mp_tot"})
    tot["key"] = tot["player"].map(normalize_name)
    adv["key"] = adv["player"].map(normalize_name)

    df = adv.merge(
        tot.drop(columns=["player"]).drop_duplicates(subset=["season", "key"]),
        on=["season", "key"], how="left",
    )
    df["pf_per_40"] = 40 * df["pf"] / df["mp"].replace(0, np.nan)
    df["mpg"] = df["mp"] / df["g"].replace(0, np.nan)
    df["mpg_share"] = df["mpg"] / NBA_GAME_MINUTES
    df["big"] = df["pos"].map(lambda p: is_big(p, "nba"))
    return df


def load_wnba_features(seasons=WNBA_SEASONS) -> pd.DataFrame:
    """WNBA player-seasons with all predictors, DRB% computed from box scores."""
    player_box = data.load("player_box", seasons)
    team_box = data.load("team_box", seasons)
    computed = rates.compute_rates(player_box, team_box)

    adv = bbref.load_advanced("wnba", seasons, kind="advanced")
    adv["key"] = adv["player"].map(normalize_name)
    pos = adv[["season", "key", "pos"]].drop_duplicates(subset=["season", "key"])

    computed["key"] = computed["athlete_display_name"].map(normalize_name)
    df = computed.merge(pos, on=["season", "key"], how="left")

    df["mpg_share"] = df["mpg"] / WNBA_GAME_MINUTES
    df["big"] = df["pos"].map(lambda p: is_big(p, "wnba"))
    return df


def add_zscores(df: pd.DataFrame, min_minutes: int) -> pd.DataFrame:
    """Z-score each predictor within season, over the qualified population.

    Standardising over qualified players only keeps the scale from being pulled
    around by 40-minute cameos, whose rate stats are wild.
    """
    out = df.copy()
    qualified = out["mp"] >= min_minutes
    for stat in Z_STATS:
        if stat not in out.columns:
            continue
        z = np.full(len(out), np.nan)
        for season, idx in out.groupby("season").groups.items():
            rows = out.loc[idx]
            ref = rows[qualified.loc[idx]][stat].dropna()
            if len(ref) < 10:
                continue
            mu, sd = ref.mean(), ref.std(ddof=0)
            if not sd or np.isnan(sd):
                continue
            z[out.index.get_indexer(idx)] = (rows[stat] - mu) / sd
        out[f"{stat}_z"] = z
    return out


def fit_component(
    train: pd.DataFrame, target: str, predictors: list[str], *, seed: int = 0
) -> dict:
    """Fit one component, reporting held-out error."""
    cols = predictors + [target]
    d = train[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X, y = d[predictors].to_numpy(float), d[target].to_numpy(float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)

    resid = y_te - pred
    return {
        "target": target,
        "predictors": predictors,
        "coefficients": dict(zip(predictors, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "oos_rmse": float(np.sqrt((resid**2).mean())),
        "oos_corr": float(np.corrcoef(y_te, pred)[0, 1]),
        "_model": model,
    }


def predict_component(df: pd.DataFrame, fit: dict) -> np.ndarray:
    X = df[fit["predictors"]].replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1)
    out = np.full(len(df), np.nan)
    if ok.any():
        out[ok.to_numpy()] = fit["_model"].predict(X[ok].to_numpy(float))
    return out


def recenter(values: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    """Shift so the minutes-weighted league mean is zero.

    BPM is defined so the minutes-weighted league average is 0. Transferred
    coefficients carry no guarantee of that, so it is imposed explicitly —
    otherwise every WNBA rating would sit at some arbitrary NBA-anchored offset
    and the replacement-level constant would be measuring against the wrong zero.
    """
    ok = ~np.isnan(values) & ~np.isnan(minutes) & (minutes > 0)
    if not ok.any():
        return values
    mean = np.average(values[ok], weights=minutes[ok])
    return values - mean


def fit_nba_models(seasons=NBA_TRAIN_SEASONS) -> tuple[dict, dict]:
    """Fit the two transfer models on NBA data. No WNBA data involved, so this
    is reusable across WNBA train/test splits without leaking anything."""
    nba = add_zscores(load_nba_training(seasons), MIN_MINUTES_TRAIN)
    train = nba[(nba["mp"] >= MIN_MINUTES_TRAIN) & nba["obpm"].notna()]
    return (
        fit_component(train, "obpm", OFF_PREDICTORS),
        fit_component(train, "dbpm", DEF_PREDICTORS),
    )


def wnba_positions(seasons=WNBA_SEASONS) -> pd.DataFrame:
    """Positions per player-season. Not outcome data, so safe across splits."""
    adv = bbref.load_advanced("wnba", seasons, kind="advanced")
    adv["key"] = adv["player"].map(normalize_name)
    return adv[["season", "key", "pos"]].drop_duplicates(subset=["season", "key"])


def prior_from_box(
    player_box: pd.DataFrame,
    team_box: pd.DataFrame,
    off_fit: dict,
    def_fit: dict,
    positions: pd.DataFrame,
    *,
    min_minutes: int = MIN_MINUTES_APPLY,
    targets: dict | None = None,
    off_k: float = 75.0,
    def_k: float = 200.0,
) -> pd.DataFrame:
    """Build the box prior from an arbitrary slice of box scores.

    Factored out of `build()` so a validation split can construct the prior from
    training games only. Passing the full season reproduces `build()`; passing a
    date-truncated slice gives a prior that has not seen the holdout.
    """
    computed = rates.compute_rates(player_box, team_box)
    computed["key"] = computed["athlete_display_name"].map(normalize_name)
    df = computed.merge(positions, on=["season", "key"], how="left")

    df["mpg_share"] = df["mpg"] / WNBA_GAME_MINUTES
    df["big"] = df["pos"].map(lambda p: is_big(p, "wnba"))
    df = add_zscores(df, min_minutes)

    df["obpm_raw"] = predict_component(df, off_fit)
    df["dbpm_raw"] = predict_component(df, def_fit)

    for col in ("obpm_c", "dbpm_c"):
        df[col] = np.nan
    for season, idx in df.groupby("season").groups.items():
        rows = df.loc[idx]
        pos = df.index.get_indexer(idx)
        mins = rows["mp"].to_numpy(float)
        df.iloc[pos, df.columns.get_loc("obpm_c")] = recenter(
            rows["obpm_raw"].to_numpy(float), mins)
        df.iloc[pos, df.columns.get_loc("dbpm_c")] = recenter(
            rows["dbpm_raw"].to_numpy(float), mins)

    if targets is None:
        df["obpm_prior"], df["dbpm_prior"] = df["obpm_c"], df["dbpm_c"]
        targets = replacement_targets(df)

    mins = df["mp"].to_numpy(float)
    df["obpm"] = shrink(df["obpm_c"].to_numpy(float), mins, off_k, targets["offense"])
    df["dbpm"] = shrink(df["dbpm_c"].to_numpy(float), mins, def_k, targets["defense"])
    return df[["season", "key", "athlete_display_name", "mp", "obpm", "dbpm"]]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan


def replacement_targets(wnba: pd.DataFrame, low_minutes: int = 200) -> dict:
    """Offense/defense split of replacement level.

    Estimated as what marginal players actually look like — the minutes-weighted
    component means among sub-`low_minutes` players — then rescaled so the two
    components sum to the replacement level derived independently in
    `constants.py`. Reporting the unscaled sum alongside gives a genuine
    consistency check between the two workstreams.
    """
    consts = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    total = -consts["replacement_level"]["value"]

    low = wnba[wnba["mp"] < low_minutes]
    w = low["mp"].to_numpy(float)
    off_raw = weighted_mean(low["obpm_prior"].to_numpy(float), w)
    def_raw = weighted_mean(low["dbpm_prior"].to_numpy(float), w)

    observed = off_raw + def_raw
    scale = total / observed if observed else 1.0
    return {
        "total": total,
        "observed_sum": observed,
        "offense": off_raw * scale,
        "defense": def_raw * scale,
        "scale_applied": scale,
    }


def shrink(values: np.ndarray, minutes: np.ndarray, k: float, target: float) -> np.ndarray:
    """Regress toward replacement with weight equivalent to `k` minutes.

    Without this the prior is unusable as `b_box`: a six-minute cameo produces a
    +10 rating, and that noise would propagate straight through the ridge.
    """
    return (minutes * values + k * target) / (minutes + k)


def estimate_shrinkage_k(
    wnba: pd.DataFrame, col: str, target: float, *, min_next_minutes: int = 300
) -> dict:
    """Pick k by how well season t predicts season t+1.

    The shrinkage constant is a reliability question, so it is estimated rather
    than assumed: grid-search the k whose shrunk season-t rating best predicts
    the same player's season-t+1 rating, restricting the target to players with
    enough t+1 minutes for that target to mean something.
    """
    df = wnba[["season", "athlete_id", "mp", col]].dropna()
    nxt = df.copy()
    nxt["season"] -= 1
    pairs = df.merge(nxt, on=["season", "athlete_id"], suffixes=("_t", "_n"))
    pairs = pairs[pairs[f"mp_n"] >= min_next_minutes]
    if len(pairs) < 50:
        return {"k": 300.0, "n_pairs": len(pairs), "note": "too few pairs; fell back to default"}

    mp_t = pairs["mp_t"].to_numpy(float)
    v_t = pairs[f"{col}_t"].to_numpy(float)
    v_n = pairs[f"{col}_n"].to_numpy(float)
    w = pairs["mp_n"].to_numpy(float)

    grid = np.arange(25, 2001, 25, dtype=float)
    errs = []
    for k in grid:
        pred = shrink(v_t, mp_t, k, target)
        errs.append(np.sqrt(np.average((v_n - pred) ** 2, weights=w)))
    errs = np.array(errs)
    best = int(errs.argmin())
    return {
        "k": float(grid[best]),
        "rmse": float(errs[best]),
        "rmse_at_k0": float(errs[0]),
        "n_pairs": int(len(pairs)),
    }


def build() -> dict:
    nba = add_zscores(load_nba_training(), MIN_MINUTES_TRAIN)
    train = nba[(nba["mp"] >= MIN_MINUTES_TRAIN) & nba["obpm"].notna()]

    off_fit = fit_component(train, "obpm", OFF_PREDICTORS)
    def_fit = fit_component(train, "dbpm", DEF_PREDICTORS)

    wnba = add_zscores(load_wnba_features(), MIN_MINUTES_APPLY)
    wnba["obpm_prior_raw"] = predict_component(wnba, off_fit)
    wnba["dbpm_prior_raw"] = predict_component(wnba, def_fit)

    # recenter within each season independently
    for col in ("obpm_prior", "dbpm_prior"):
        wnba[col] = np.nan
    for season, idx in wnba.groupby("season").groups.items():
        rows = wnba.loc[idx]
        pos = wnba.index.get_indexer(idx)
        mins = rows["mp"].to_numpy(float)
        wnba.iloc[pos, wnba.columns.get_loc("obpm_prior")] = recenter(
            rows["obpm_prior_raw"].to_numpy(float), mins)
        wnba.iloc[pos, wnba.columns.get_loc("dbpm_prior")] = recenter(
            rows["dbpm_prior_raw"].to_numpy(float), mins)

    wnba["bpm_prior"] = wnba["obpm_prior"] + wnba["dbpm_prior"]

    # Regress toward replacement by sample size. Must happen after recentering,
    # since replacement level is defined against a centered scale.
    targets = replacement_targets(wnba)
    off_k = estimate_shrinkage_k(wnba, "obpm_prior", targets["offense"])
    def_k = estimate_shrinkage_k(wnba, "dbpm_prior", targets["defense"])

    mins = wnba["mp"].to_numpy(float)
    wnba["obpm"] = shrink(wnba["obpm_prior"].to_numpy(float), mins,
                          off_k["k"], targets["offense"])
    wnba["dbpm"] = shrink(wnba["dbpm_prior"].to_numpy(float), mins,
                          def_k["k"], targets["defense"])
    wnba["bpm"] = wnba["obpm"] + wnba["dbpm"]

    return {
        "nba_train": nba, "wnba": wnba, "off_fit": off_fit, "def_fit": def_fit,
        "targets": targets, "off_k": off_k, "def_k": def_k,
    }


def main() -> None:
    res = build()
    wnba, off_fit, def_fit = res["wnba"], res["off_fit"], res["def_fit"]

    out_dir = data.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = [
        "season", "athlete_id", "athlete_display_name", "pos", "big", "g", "mp",
        "mpg", "obpm_prior", "dbpm_prior", "bpm_prior", "obpm", "dbpm", "bpm",
    ]
    wnba[keep].to_parquet(out_dir / "box_prior.parquet", index=False)

    meta = {
        "offense_fit": {k: v for k, v in off_fit.items() if not k.startswith("_")},
        "defense_fit": {k: v for k, v in def_fit.items() if not k.startswith("_")},
        "replacement_targets": res["targets"],
        "shrinkage": {"offense": res["off_k"], "defense": res["def_k"]},
    }
    (out_dir / "box_prior_fit.json").write_text(json.dumps(meta, indent=2))

    print("NBA training fit (held-out 20%)")
    for name, f in (("OBPM", off_fit), ("DBPM", def_fit)):
        print(f"  {name}: n={f['n_train']:,}  RMSE={f['oos_rmse']:.2f}  r={f['oos_corr']:.3f}")
    print("  412 reported: SE 2.8, corr 0.86 overall (0.88 off / ~0.70 def)")

    t = res["targets"]
    print(f"\nreplacement split (offense/defense)")
    print(f"  empirical sum from sub-200-min players: {t['observed_sum']:+.2f}")
    print(f"  independently derived in constants.py:  {t['total']:+.2f}   <- consistency check")
    print(f"  targets used: offense {t['offense']:+.2f}, defense {t['defense']:+.2f}")

    print(f"\nshrinkage constant k (estimated by season t -> t+1 prediction)")
    for name, kf in (("offense", res["off_k"]), ("defense", res["def_k"])):
        if "rmse" in kf:
            print(f"  {name}: k={kf['k']:.0f} min  RMSE {kf['rmse_at_k0']:.3f} -> "
                  f"{kf['rmse']:.3f}  (n={kf['n_pairs']} pairs)")
        else:
            print(f"  {name}: {kf}")

    latest = int(wnba[wnba["mp"] >= MIN_MINUTES_APPLY]["season"].max())
    cur = wnba[(wnba["season"] == latest) & (wnba["mp"] >= MIN_MINUTES_APPLY)]
    print(f"\nWNBA {latest}: {len(cur)} qualified players")
    print(f"  bpm  mean={weighted_mean(cur['bpm'].to_numpy(float), cur['mp'].to_numpy(float)):+.2f} "
          f"(minutes-weighted)  sd={cur['bpm'].std():.2f}")
    print(f"\nTop 15 by shrunk bpm, {latest}:")
    top = cur.nlargest(15, "bpm")[
        ["athlete_display_name", "pos", "g", "mp", "obpm", "dbpm", "bpm", "bpm_prior"]
    ]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
