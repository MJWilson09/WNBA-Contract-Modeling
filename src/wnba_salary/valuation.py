"""Salary layer: ratings + constants -> dollars.

Mirrors the chain in Noh's NBA model, but every constant is re-derived for the
WNBA in `constants.py` rather than ported, and dollars are priced out of
discretionary cap space rather than the full cap.

    WAR   = (minutes / minutes_baseline) * (rating + replacement_level)
    value = min_salary + WAR * dollars_per_win

Two WNBA-specific wrinkles
--------------------------
**The max binds constantly.** Under the 2026 CBA the max is $1.4M against a
$270K minimum — a 5.2x band, versus roughly 43x in the NBA. A large share of
starters price above the max, so `value` (unconstrained) and `market_value`
(clipped to the CBA band) are both reported. The gap between them is the
interesting quantity: it is the surplus a team captures purely because the CBA
forbids paying what a player is worth.

**Aging is multiplicative on value above replacement, not on the rating.** The
bundled WNBA curve is a `rel_value` multiplier. Applying it to a raw rating
would be wrong — scaling -1.0 by 0.9 makes a below-average player *better*.
Applying it to (rating + replacement) is coherent: a replacement-level player
stays replacement-level, and a star declines proportionally.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests

from . import box_prior, data, salaries

AGING_CURVE_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/sportsdataverse-py/main/"
    "sportsdataverse/nba/models/wnba_aging_curve.json"
)

PROJECTION_YEARS = 3        # WNBA deals are mostly 1-3 years; 5 is extrapolation
CURRENT_SEASON = 2026
MIN_MINUTES = 100

# ---------------------------------------------------------------------------
# Verified against the 2026 CBA itself (data/raw/cba/wnba_cba.pdf, gitignored;
# fetched from wnbpa.com). Article and section numbers are given so each figure
# can be re-checked. This replaced a set of press-reported numbers and an
# invented year-by-year schedule; see README for what changed.
# ---------------------------------------------------------------------------

CBA_FIRST_YEAR = 2026
BASE_SALARY_CAP = 7_000_000        # Art. VII §1(a)(i) — exact, not a projection

# Art. V §8(a)-(b): the maximum is a share of the cap, not a dollar figure.
SUPERMAX_SHARE = 0.20              # qualifying 5+ yr vets re-signing, Core players,
STANDARD_MAX_SHARE = 0.17          # rookie-scale extensions; everyone else gets 17%
MIN_TEAM_SALARY_SHARE = 0.85       # Art. VII §1(c) — floor on total team spend

# Art. VII §1(a)(ii): the cap after 2026 is revenue-determined (a share of SBR),
# NOT a published schedule. It is only bounded: 2027 may move at most 13% from
# 2026, and each year after that at most 10%. Any projection is therefore an
# assumption; this one tracks the league's public "$11M by 2032" trajectory,
# which works out to ~7.9%/yr and sits comfortably inside the ceiling.
CAP_GROWTH = 0.079
CAP_GROWTH_CEILING = {2027: 0.13}  # 0.10 for every later year
DEFAULT_CAP_CEILING = 0.10

# Art. V §7(a): the minimum is an exact table by Years of Service. Reproduced
# for the seasons this model can project into. The 0-year column is used for the
# discretionary-pool arithmetic in constants.py; see the note there.
MIN_SALARY_TABLE = {                      # season: {years_of_service: minimum}
    2026: {0: 270_000, 1: 277_500, 4: 285_000, 7: 292_500, 10: 300_000},
    2027: {0: 280_800, 1: 288_600, 4: 296_400, 7: 304_200, 10: 312_000},
    2028: {0: 292_000, 1: 300_100, 4: 308_300, 7: 316_400, 10: 324_500},
    2029: {0: 303_700, 1: 312_100, 4: 320_600, 7: 329_000, 10: 337_500},
    2030: {0: 315_900, 1: 324_600, 4: 333_400, 7: 342_200, 10: 351_000},
    2031: {0: 328_500, 1: 337_600, 4: 346_700, 7: 355_900, 10: 365_000},
    2032: {0: 341_600, 1: 351_100, 4: 360_600, 7: 370_100, 10: 379_600},
}

# Art. XI: the regular season lengthens over the agreement. The model previously
# assumed 44 games in every projected year, which understated league-wide wins
# (and so overstated dollars per win) from 2027 on.
GAMES_PER_SEASON = {2026: 44, 2027: 50, 2028: 50}
GAMES_FROM_2029 = 52


def games_in_season(season: int) -> int:
    return GAMES_PER_SEASON.get(season, GAMES_FROM_2029 if season >= 2029 else 44)


def salary_cap(season: int) -> float:
    """Projected cap. 2026 is contractual; later years are an assumption."""
    if season <= CBA_FIRST_YEAR:
        return float(BASE_SALARY_CAP)
    cap = float(BASE_SALARY_CAP)
    for yr in range(CBA_FIRST_YEAR + 1, season + 1):
        ceiling = CAP_GROWTH_CEILING.get(yr, DEFAULT_CAP_CEILING)
        cap *= 1 + min(CAP_GROWTH, ceiling)
    return cap


def minimum_salary(season: int, years_of_service: int = 0) -> float:
    """Minimum for a service level. Falls back to the last tabulated season."""
    table = MIN_SALARY_TABLE.get(season) or MIN_SALARY_TABLE[max(MIN_SALARY_TABLE)]
    tier = max(k for k in table if k <= max(int(years_of_service or 0), 0))
    return float(table[tier])


def cba_schedule(season: int, years_of_service: int = 0) -> dict:
    """Cap, both maxima, and the applicable minimum for a season."""
    cap = salary_cap(season)
    return {
        "salary_cap": cap,
        "supermax_salary": cap * SUPERMAX_SHARE,
        "standard_max_salary": cap * STANDARD_MAX_SHARE,
        # `max_salary` keeps the old key name and means the *standard* maximum,
        # which is what limits most contracts.
        "max_salary": cap * STANDARD_MAX_SHARE,
        "min_salary": minimum_salary(season, years_of_service),
        "min_team_salary": cap * MIN_TEAM_SALARY_SHARE,
    }


def load_aging_curve() -> pd.DataFrame:
    """WNBA aging curve (peak age 29), from sportsdataverse-py's bundled fit."""
    path = data.RAW_DIR / "wnba_aging_curve.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        resp = requests.get(AGING_CURVE_URL, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    return pd.DataFrame({"age": payload["age"], "rel_value": payload["rel_value"]})


def age_multiplier(curve: pd.DataFrame, age: float) -> float:
    """rel_value at an arbitrary age, linearly interpolated and clamped."""
    if age is None or np.isnan(age):
        return np.nan
    return float(np.interp(age, curve["age"], curve["rel_value"]))


def dollars_per_win(consts: dict, season: int) -> float:
    """$/win for a season, scaled by that season's discretionary pool.

    As the cap grows faster than the minimum, discretionary money grows faster
    than the cap itself, so wins inflate faster than headline salaries.
    """
    sched = cba_schedule(season)
    cba = consts["cba"]
    discretionary = sched["salary_cap"] - cba["roster_min"] * sched["min_salary"]
    league_disc = discretionary * cba["n_teams"]
    # The season lengthens under this CBA (44 -> 50 -> 52), so league-wide wins
    # grow with it. Holding 44 fixed overstated dollars per win from 2027 on.
    league_war = (cba["n_teams"] * games_in_season(season) / 2) * (
        1 - consts["replacement_win_pct"]["value"]
    )
    return league_disc / league_war


def value_from_rating(
    minutes: float, rating: float, consts: dict, season: int
) -> tuple[float, float]:
    """(WAR, unconstrained dollar value)."""
    baseline = consts["minutes_baseline"]["value"]
    replacement = consts["replacement_level"]["value"]
    war = (minutes / baseline) * (rating + replacement)
    sched = cba_schedule(season)
    return war, sched["min_salary"] + war * dollars_per_win(consts, season)


def project_rating(
    rating: float, age: float, years_ahead: int, curve: pd.DataFrame, replacement: float
) -> float:
    """Age a rating forward. Multiplicative on value above replacement."""
    now = age_multiplier(curve, age)
    later = age_multiplier(curve, age + years_ahead)
    if not now or np.isnan(now) or np.isnan(later):
        return np.nan
    return (rating + replacement) * (later / now) - replacement


def load_ages(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Player ages from wehoop player_core."""
    core = data.fetch_season("player_core", season, refresh=False)
    if core is None:
        raise RuntimeError(f"no player_core for {season}")
    core = core[["athlete_id", "display_name", "age", "experience_years",
                 "draft_year", "position_abbreviation"]]
    return core.rename(columns={"display_name": "player_core_name"})


def team_games_played(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Games each team has played, for prorating a partial season."""
    tb = data.load("team_box", [season])
    return (
        tb.groupby("team_id", as_index=False)["game_id"]
        .nunique()
        .rename(columns={"game_id": "team_games"})
    )


def build() -> pd.DataFrame:
    consts = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    curve = load_aging_curve()
    replacement = consts["replacement_level"]["value"]
    cba = consts["cba"]

    ratings = pd.read_parquet(data.PROCESSED_DIR / "box_prior.parquet")
    cur = ratings[
        (ratings["season"] == CURRENT_SEASON) & (ratings["mp"] >= MIN_MINUTES)
    ].copy()
    cur["key"] = cur["athlete_display_name"].map(box_prior.normalize_name)

    # ---- rating source: RAPM where available, box prior otherwise -----------
    # RAPM needs MIN_POSS possessions in the pooled fit, so a handful of
    # low-usage players fall back to the box prior. `rating_source` records which
    # applied, since the two are not interchangeable in quality — the box prior
    # cannot see defence (see README).
    rapm_path = data.PROCESSED_DIR / "ratings.parquet"
    if rapm_path.exists():
        rr = pd.read_parquet(rapm_path)[
            ["player", "rapm", "o_rapm", "d_rapm", "prior", "rating_se"]]
        cur = cur.merge(rr, left_on="key", right_on="player", how="left")
        cur["rating"] = cur["rapm"].where(cur["rapm"].notna(), cur["bpm"])
        cur["rating_source"] = np.where(cur["rapm"].notna(), "rapm", "box_prior")
    else:
        cur["rating"] = cur["bpm"]
        cur["rating_source"] = "box_prior"

    # Forward-looking rating for the projection years only. Tuned on the forward
    # metric, where optimal shrinkage is ~4x heavier — a rating tuned to describe
    # the current season over-trusts noisy possession data when extrapolated.
    # Year 0 keeps the descriptive rating: "is she worth her contract now" is a
    # descriptive question.
    fc_path = data.PROCESSED_DIR / "ratings_forecast.parquet"
    if fc_path.exists():
        fc = pd.read_parquet(fc_path)[["player", "rapm"]].rename(
            columns={"rapm": "rating_forecast", "player": "player_fc"})
        cur = cur.merge(fc, left_on="key", right_on="player_fc", how="left")
        cur = cur.drop(columns=["player_fc"])
        cur["rating_forecast"] = cur["rating_forecast"].where(
            cur["rating_forecast"].notna(), cur["rating"])
    else:
        cur["rating_forecast"] = cur["rating"]

    # ---- prorate a partial season to a full 44 games -----------------------
    pbx = data.load("player_box", [CURRENT_SEASON])
    player_team = (
        pbx[pbx["minutes"].notna()]
        .groupby("athlete_id")
        .agg(team_id=("team_id", "last"))
        .reset_index()
    )
    cur = cur.merge(player_team, on="athlete_id", how="left")
    cur = cur.merge(team_games_played(), on="team_id", how="left")

    cur["availability"] = (cur["g"] / cur["team_games"]).clip(upper=1.0)
    cur["proj_minutes"] = cur["mpg"] * cba["games_per_team"] * cur["availability"]
    cur["full_health_minutes"] = cur["mpg"] * cba["games_per_team"]

    # ---- join ages ---------------------------------------------------------
    ages = load_ages()
    cur = cur.merge(ages, on="athlete_id", how="left")

    # ---- join salaries by name --------------------------------------------
    sal = salaries.fetch_salaries(CURRENT_SEASON)
    sal["key"] = sal["player"].map(box_prior.normalize_name)
    cur = cur.merge(
        sal[["key", "salary", "signing"]].drop_duplicates("key"), on="key", how="left"
    )

    # ---- current-season value ---------------------------------------------
    war, value = [], []
    for _, r in cur.iterrows():
        w, v = value_from_rating(r["proj_minutes"], r["rating"], consts, CURRENT_SEASON)
        war.append(w)
        value.append(v)
    cur["war"] = war
    cur["value"] = value

    # Uncertainty band. Value is linear in rating, so the SE propagates directly:
    # d(value)/d(rating) = (minutes / B) * $/win. See rapm.posterior_se for what
    # this interval does and does not cover.
    baseline = consts["minutes_baseline"]["value"]
    dpw = dollars_per_win(consts, CURRENT_SEASON)
    cur["value_se"] = (cur["proj_minutes"] / baseline) * dpw * cur["rating_se"]
    cur["value_lo"] = cur["value"] - cur["value_se"]
    cur["value_hi"] = cur["value"] + cur["value_se"]

    sched = cba_schedule(CURRENT_SEASON)

    # Art. V §8: most contracts are limited to the Standard Maximum (17% of the
    # cap). The Supermax (20%) is available only to Core players, to qualifying
    # veterans with 5+ years re-signing with their prior team, and to rookie-scale
    # extensions. Whether a free agent is re-signing with her *prior* team is not
    # in any data we have, so 5+ years of service is treated as eligible — that
    # makes this an upper bound on eligibility. A salary already above the
    # standard maximum is taken as revealed eligibility.
    exp = cur["experience_years"].fillna(0)
    cur["supermax_eligible"] = (
        cur["signing"].eq("Core")
        | (exp >= 5)
        | (cur["salary"].fillna(0) > sched["standard_max_salary"] + 1)
    )
    cur["applicable_max"] = np.where(
        cur["supermax_eligible"], sched["supermax_salary"], sched["standard_max_salary"])
    cur["applicable_min"] = [
        minimum_salary(CURRENT_SEASON, e) for e in exp
    ]

    cur["market_value"] = np.minimum(
        np.maximum(cur["value"], cur["applicable_min"]), cur["applicable_max"])
    cur["capped_by_max"] = cur["value"] > cur["applicable_max"]
    cur["surplus"] = cur["market_value"] - cur["salary"]
    cur["surplus_uncapped"] = cur["value"] - cur["salary"]

    # ---- forward projection ------------------------------------------------
    for k in range(1, PROJECTION_YEARS):
        season = CURRENT_SEASON + k
        s = cba_schedule(season)
        rt, vl, mv = [], [], []
        for _, r in cur.iterrows():
            proj = project_rating(r["rating_forecast"], r["age"], k, curve, replacement)
            if np.isnan(proj):
                rt.append(np.nan); vl.append(np.nan); mv.append(np.nan)
                continue
            _, v = value_from_rating(r["proj_minutes"], proj, consts, season)
            hi = (s["supermax_salary"] if r["supermax_eligible"]
                  else s["standard_max_salary"])
            lo = minimum_salary(season, r["experience_years"] or 0)
            rt.append(proj); vl.append(v)
            mv.append(min(max(v, lo), hi))
        cur[f"rating_{season}"] = rt
        cur[f"value_{season}"] = vl
        cur[f"market_value_{season}"] = mv

    return cur


def main() -> None:
    df = build()
    out_dir = data.PROCESSED_DIR
    df.to_parquet(out_dir / "valuation.parquet", index=False)

    matched = df["salary"].notna().sum()
    print(f"{len(df)} rated players, {matched} matched to a 2026 salary "
          f"({matched/len(df)*100:.0f}%)")
    unmatched = df[df["salary"].isna()].nlargest(5, "mp")["athlete_display_name"].tolist()
    if unmatched:
        print(f"  highest-minute unmatched: {', '.join(unmatched)}")

    sched = cba_schedule(CURRENT_SEASON)
    capped = int(df["capped_by_max"].sum())
    print(f"\n{capped} of {len(df)} players price above the ${sched['max_salary']:,.0f} max "
          f"({capped/len(df)*100:.0f}%)")

    have = df[df["salary"].notna()].copy()
    print(f"\nTop 15 by 2026 value:")
    cols = ["athlete_display_name", "age", "mpg", "rating", "war", "value",
            "market_value", "salary", "surplus", "signing"]
    top = have.nlargest(15, "value")[cols]
    print(top.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print(f"\nBiggest surplus (value over contract):")
    print(have.nlargest(10, "surplus")[
        ["athlete_display_name", "age", "rating", "market_value", "salary", "surplus", "signing"]
    ].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    print(f"\nBiggest deficit (contract over value):")
    print(have.nsmallest(10, "surplus")[
        ["athlete_display_name", "age", "rating", "market_value", "salary", "surplus", "signing"]
    ].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))


if __name__ == "__main__":
    main()
