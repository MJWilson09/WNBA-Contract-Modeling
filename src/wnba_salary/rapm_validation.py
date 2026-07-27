"""Leak-free comparison of RAPM-shrunk-to-prior against the prior alone.

The problem being fixed
----------------------
The first Stage B sweep built the box prior from *full-season* box scores and
then held out the last 25% of games. The prior therefore already knew the
box-score outcome of the games RAPM was being scored on. RAPM was competing
against an opponent holding the test set, and unsurprisingly lost.

What this does
--------------
Splits each season at its 75th-percentile game date. The prior is rebuilt from
training-window box scores only; RAPM is fitted on training possessions only;
both are scored on the same held-out games. The full-season (leaky) prior is
carried through as a reference so the size of the leakage is measurable rather
than assumed.

Scoring is at game level. Possession-level RMSE has sd ~111 and its whole range
across λ is under 0.4%, which is too noisy to choose on.

Residual leakage, disclosed
---------------------------
The shrinkage constants and the offense/defense replacement split are reused
from the full-sample fit rather than re-estimated per split. Both are scalars
fitted across 17 seasons, so the contamination is small, but it is not zero.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import box_prior, data, rapm

SEASONS = range(2017, 2023)
TRAIN_FRAC = 0.75


def season_cutoffs(poss: pd.DataFrame, frac: float = TRAIN_FRAC) -> dict[int, pd.Timestamp]:
    """Per-season date splitting games `frac` of the way through."""
    out = {}
    for season, g in poss.groupby("season"):
        dates = pd.to_datetime(g["game_date"])
        first = dates.groupby(g["game_id"]).min().sort_values()
        out[int(season)] = first.iloc[int(len(first) * frac)]
    return out


def aggregate_prior(prior_df: pd.DataFrame, players: list[str]) -> np.ndarray:
    """Minutes-weighted prior per player, aligned to the design's player order."""
    d = prior_df.dropna(subset=["obpm", "dbpm"])
    agg = d.groupby("key").apply(
        lambda g: pd.Series({
            "obpm": np.average(g["obpm"], weights=g["mp"]),
            "dbpm": np.average(g["dbpm"], weights=g["mp"]),
        }),
        include_groups=False,
    )
    off = np.array([agg["obpm"].get(k, 0.0) for k in players], dtype=float)
    dfn = np.array([agg["dbpm"].get(k, 0.0) for k in players], dtype=float)
    matched = sum(k in agg.index for k in players)
    return np.nan_to_num(np.concatenate([off, dfn, [0.0]])), matched


def run() -> dict:
    poss = pd.concat(
        [rapm.build_possessions(rapm.fetch_stats_pbp(s)) for s in SEASONS],
        ignore_index=True,
    )
    cutoffs = season_cutoffs(poss)
    print("season cutoffs (train = games before):")
    for s, c in sorted(cutoffs.items()):
        print(f"  {s}: {pd.Timestamp(c).date()}")

    design = rapm.build_design(poss, min_poss=200)
    players = design["players"]
    A, y = design["A"], design["y"]

    # rebuild the row filter that build_design applied, so rows align
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

    row_date = pd.to_datetime(rows["game_date"])
    row_cut = rows["season"].map(cutoffs)
    is_train = (row_date < pd.to_datetime(row_cut)).to_numpy()
    print(f"\npossessions: {is_train.sum():,} train / {(~is_train).sum():,} test")

    # ---- box data, split by the same dates ---------------------------------
    pbx = data.load("player_box", SEASONS)
    tbx = data.load("team_box", SEASONS)
    pbx_date = pd.to_datetime(pbx["game_date"])
    tbx_date = pd.to_datetime(tbx["game_date"])
    pb_train = pbx[pbx_date < pd.to_datetime(pbx["season"].map(cutoffs))]
    tb_train = tbx[tbx_date < pd.to_datetime(tbx["season"].map(cutoffs))]
    print(f"box scores : {len(pb_train):,} of {len(pbx):,} player-games in train window")

    off_fit, def_fit = box_prior.fit_nba_models()
    positions = box_prior.wnba_positions(SEASONS)
    fitmeta = json.loads((data.PROCESSED_DIR / "box_prior_fit.json").read_text())
    targets = fitmeta["replacement_targets"]
    off_k = fitmeta["shrinkage"]["offense"]["k"]
    def_k = fitmeta["shrinkage"]["defense"]["k"]

    clean_df = box_prior.prior_from_box(
        pb_train, tb_train, off_fit, def_fit, positions,
        min_minutes=int(box_prior.MIN_MINUTES_APPLY * TRAIN_FRAC),
        targets=targets, off_k=off_k, def_k=def_k,
    )
    leaky_df = box_prior.prior_from_box(
        pbx, tbx, off_fit, def_fit, positions,
        targets=targets, off_k=off_k, def_k=def_k,
    )

    clean_prior, clean_matched = aggregate_prior(clean_df, players)
    leaky_prior, leaky_matched = aggregate_prior(leaky_df, players)
    print(f"prior match: clean {clean_matched}/{len(players)}, "
          f"leaky {leaky_matched}/{len(players)}")

    # ---- game-level scorer -------------------------------------------------
    test_game = rows.loc[~is_train, "game_id"].to_numpy()
    test_off = rows.loc[~is_train, "off_team"].to_numpy()
    A_te, y_te = A[~is_train], y[~is_train]
    A_tr, y_tr = A[is_train], y[is_train]

    def game_rmse(b: np.ndarray) -> float:
        t = pd.DataFrame({
            "g": test_game, "ot": test_off,
            "a": y_te / 100.0, "p": (A_te @ b) / 100.0,
        })
        s = t.groupby(["g", "ot"], as_index=False).agg(
            actual=("a", "sum"), predicted=("p", "sum"))
        return float(np.sqrt(((s["actual"] - s["predicted"]) ** 2).mean()))

    def prior_only(vec: np.ndarray) -> np.ndarray:
        b = vec.copy()
        b[-1] = y_tr.mean()
        return b

    grid = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    results = {"clean": [], "leaky": [], "zero": []}
    for lam in grid:
        for tag, vec in (("clean", clean_prior), ("leaky", leaky_prior),
                         ("zero", np.zeros_like(clean_prior))):
            b = rapm.fit_ridge_prior(A_tr, y_tr, lam, vec)
            r = b[:len(players)] + b[len(players):2 * len(players)]
            results[tag].append({
                "lambda": lam,
                "game_rmse": game_rmse(b),
                "sd": float(np.std(r)),
            })

    baselines = {
        "prior_only_clean": game_rmse(prior_only(clean_prior)),
        "prior_only_leaky": game_rmse(prior_only(leaky_prior)),
    }
    return {"grid": grid, "results": results, "baselines": baselines,
            "n_players": len(players),
            "n_train": int(is_train.sum()), "n_test": int((~is_train).sum())}


def main() -> None:
    res = run()
    r = res["results"]
    b = res["baselines"]

    print(f"\n{'':>9}  {'CLEAN prior':>22}  {'leaky prior':>22}  {'zero':>14}")
    print(f"{'lambda':>9}  {'game RMSE':>11}{'sd':>11}  {'game RMSE':>11}{'sd':>11}"
          f"  {'game RMSE':>14}")
    for i, lam in enumerate(res["grid"]):
        c, l, z = r["clean"][i], r["leaky"][i], r["zero"][i]
        print(f"{lam:>9,}  {c['game_rmse']:>11.4f}{c['sd']:>11.2f}"
              f"  {l['game_rmse']:>11.4f}{l['sd']:>11.2f}  {z['game_rmse']:>14.4f}")

    print(f"\nbaselines (no possession data at all):")
    print(f"  prior only, CLEAN : {b['prior_only_clean']:.4f}")
    print(f"  prior only, leaky : {b['prior_only_leaky']:.4f}"
          f"   <- leakage worth {b['prior_only_clean']-b['prior_only_leaky']:.4f}")

    best_c = min(r["clean"], key=lambda x: x["game_rmse"])
    best_l = min(r["leaky"], key=lambda x: x["game_rmse"])
    print(f"\nCLEAN: best lambda {best_c['lambda']:,} -> {best_c['game_rmse']:.4f} "
          f"vs prior-only {b['prior_only_clean']:.4f} "
          f"(gain {b['prior_only_clean']-best_c['game_rmse']:+.4f})")
    print(f"leaky: best lambda {best_l['lambda']:,} -> {best_l['game_rmse']:.4f} "
          f"vs prior-only {b['prior_only_leaky']:.4f} "
          f"(gain {b['prior_only_leaky']-best_l['game_rmse']:+.4f})")

    out = data.PROCESSED_DIR / "rapm_validation.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
