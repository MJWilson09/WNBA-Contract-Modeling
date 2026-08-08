"""Draft-slot priors for players with no possession history.

The problem this addresses: `forecast_validation` imputes league-average (0, 0)
for any player unseen in training, and 14.4% of on-court slots in a test season
are such players — ~21% in expansion years, which are the worst-forecast seasons.
Rookies are not league-average. Minutes-weighted, they run about -1.2 points/100,
and the spread by draft slot is large: top-two picks average +1.2 while picks 25+
average -3.0.

DARKO handles this by initialising rookies from a common prior and learning from
there. This is the cheap WNBA analogue: a prior indexed by draft slot.

Model
-----
`obpm ~ a·log(pick) + b`, and the same for `dbpm`, fitted on rookie seasons since
2010 and **weighted by minutes played**.

Two deliberate choices:

* **Minutes weighting.** The prior is consumed per on-court slot, so the relevant
  expectation is over slots, not over players. A rookie who plays 700 minutes
  occupies far more slots than one who plays 40. Unweighted, the fit is dragged
  down by fringe rookies and is too pessimistic exactly where it matters most
  (weighted wR² 0.412 on offence versus 0.335 unweighted).
* **Linear in log(pick), not quadratic.** The quadratic buys ~0.005 wR² and
  overshoots at the top of the draft (+1.79 fitted at pick 1 against +1.16
  observed). Fewer parameters, less edge extrapolation.

Offence carries nearly all the signal (wR² 0.412 versus 0.041 for defence). That
is the box prior's known inability to resolve defence, not a claim that draft
position fails to predict defensive quality.

On survivorship
---------------
Only ~56% of drafted players ever record a rated rookie season, so the fit
conditions on having played. That is normally a bias — but here it is the
correct conditioning: the prior is only ever applied to a player who is *on the
floor* in the test season, so "given she plays" is exactly the question. The
95.6% play rate for top-four picks also confirms the name join is sound; a
systematic join failure would depress that number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import bbref, box_prior, data

DRAFT_SEASONS = range(2010, 2027)
MIN_FIT_ROOKIES = 40      # below this, fall back to the pooled rookie mean
UNDRAFTED_PICK = 40.0     # slot assigned to undrafted free agents


def load_draft(seasons=DRAFT_SEASONS) -> pd.DataFrame:
    """Draft picks with normalised names. One row per pick."""
    dr = bbref.load_advanced("wnba", seasons, kind="draft")
    dr = dr[dr["pick"].notna()].copy()
    dr["key"] = dr["player"].map(box_prior.normalize_name)
    return dr[["season", "pick", "player", "key"]].drop_duplicates(["season", "key"])


def rookie_ratings(draft: pd.DataFrame, prior_df: pd.DataFrame) -> pd.DataFrame:
    """Join each pick to that player's rookie-season box prior."""
    bp = prior_df.copy()
    if "key" not in bp.columns:
        bp["key"] = bp["athlete_display_name"].map(box_prior.normalize_name)
    m = draft.merge(bp[["key", "season", "mp", "obpm", "dbpm"]],
                    on=["key", "season"], how="left")
    return m


def fit(max_season: int | None = None, prior_df: pd.DataFrame | None = None) -> dict:
    """Fit the draft curve on rookie seasons strictly before `max_season`.

    `max_season` keeps the fit leak-free inside forward validation: predicting
    season T must not use season-T rookies' realised ratings.
    """
    if prior_df is None:
        prior_df = pd.read_parquet(data.PROCESSED_DIR / "box_prior.parquet")
    draft = load_draft()
    if max_season is not None:
        draft = draft[draft["season"] < max_season]

    m = rookie_ratings(draft, prior_df).dropna(subset=["obpm", "dbpm", "mp"])
    m = m[m["mp"] > 0]

    n_drafted = len(draft)
    played_rate = len(m) / n_drafted if n_drafted else 0.0

    if len(m) < MIN_FIT_ROOKIES:
        return {"ok": False, "n": len(m), "played_rate": played_rate,
                "o": (0.0, -1.0), "d": (0.0, -0.2)}

    lp = np.log(m["pick"].to_numpy(float))
    w = np.sqrt(m["mp"].to_numpy(float))
    o = np.polyfit(lp, m["obpm"].to_numpy(float), 1, w=w)
    d = np.polyfit(lp, m["dbpm"].to_numpy(float), 1, w=w)
    return {"ok": True, "n": len(m), "n_drafted": n_drafted,
            "played_rate": played_rate,
            "o": (float(o[0]), float(o[1])), "d": (float(d[0]), float(d[1]))}


def predict(pick: float, coefs: dict) -> tuple[float, float]:
    """(offensive, defensive) prior for a given draft slot."""
    lp = np.log(max(float(pick), 1.0))
    ao, bo = coefs["o"]
    ad, bd = coefs["d"]
    return float(ao * lp + bo), float(ad * lp + bd)


def rookie_priors(season: int, coefs: dict, *, lookback: int = 1) -> dict[str, tuple[float, float]]:
    """name -> (o, d) for players drafted into `season` (and `lookback` before).

    The one-season lookback catches players drafted the previous year who never
    cleared the possession threshold and so are still unseen. The curve is fitted
    on rookie seasons, so stretching it further than that would be applying it to
    a population it was not estimated on.
    """
    draft = load_draft()
    window = range(season - lookback, season + 1)
    rookies = draft[draft["season"].isin(window)]
    return {r.key: predict(r.pick, coefs) for r in rookies.itertuples()}


def apply_to(rating: dict[str, tuple[float, float]], season: int, coefs: dict,
             *, lookback: int = 1) -> dict[str, tuple[float, float]]:
    """Overlay draft priors for rookies absent from `rating`. Never overwrites."""
    out = dict(rating)
    for key, od in rookie_priors(season, coefs, lookback=lookback).items():
        out.setdefault(key, od)
    return out


def main() -> None:
    coefs = fit()
    print(f"draft prior fitted on {coefs['n']} rated rookie seasons "
          f"of {coefs['n_drafted']} picks ({coefs['played_rate']*100:.1f}% played)")
    print(f"  obpm = {coefs['o'][0]:+.3f}·log(pick) {coefs['o'][1]:+.3f}")
    print(f"  dbpm = {coefs['d'][0]:+.3f}·log(pick) {coefs['d'][1]:+.3f}")
    print(f"\n{'pick':>5}{'off':>8}{'def':>8}{'total':>8}")
    for pk in (1, 2, 3, 5, 8, 12, 20, 30, 36):
        o, d = predict(pk, coefs)
        print(f"{pk:>5}{o:>8.2f}{d:>8.2f}{o + d:>8.2f}")
    print(f"\nreplacement level is -2.98; league average 0.00")


if __name__ == "__main__":
    main()
