"""Emit the static web UI's data file.

Writes `docs/players.js`, the data file for the static site. `docs/` is the
GitHub Pages publish directory (Settings -> Pages -> Deploy from branch,
main /docs), which is why it is not called `web/`.

The site is hand-written HTML plus one shared stylesheet — `index.html`
(the model), `about.html`, `assets/site.css`. Only the data file is generated;
no build step.

The JS recomputes value from scratch whenever the user moves a slider, so every
constant the formula needs is embedded here rather than baked into precomputed
numbers. The JS implementations of `computeWar`, `computeValue` and
`projectRating` mirror `valuation.py` exactly — if you change one, change both.

"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import data, history, valuation
from .history import team_lookup

WEB_DIR = data.PROJECT_ROOT / "docs"

PROJECTION_SEASONS = [
    valuation.CURRENT_SEASON + k for k in range(valuation.PROJECTION_YEARS)
]


def build_payload() -> dict:
    consts = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    df = pd.read_parquet(data.PROCESSED_DIR / "valuation.parquet")
    curve = valuation.load_aging_curve()
    teams = team_lookup(valuation.CURRENT_SEASON)

    schedule = {}
    for season in PROJECTION_SEASONS:
        s = valuation.cba_schedule(season)
        schedule[str(season)] = {
            "min_salary": round(s["min_salary"]),
            "max_salary": round(s["max_salary"]),          # standard maximum
            "supermax_salary": round(s["supermax_salary"]),
            "salary_cap": round(s["salary_cap"]),
            "games": valuation.games_in_season(season),
            # Art. V §7(a) minimum by Years of Service, so the page can floor
            # each player at the minimum that actually applies to her.
            "min_tiers": {str(k): round(v) for k, v in
                          sorted(valuation.MIN_SALARY_TABLE.get(
                              season, valuation.MIN_SALARY_TABLE[
                                  max(valuation.MIN_SALARY_TABLE)]).items())},
            "dollars_per_win": valuation.dollars_per_win(consts, season),
        }

    players = []
    for r in df.sort_values("value", ascending=False).itertuples():
        players.append({
            "id": int(r.athlete_id),
            "name": r.athlete_display_name,
            "team": teams.get(int(r.team_id), "—") if not pd.isna(r.team_id) else "—",
            "pos": r.pos if isinstance(r.pos, str) else "—",
            "age": None if pd.isna(r.age) else int(r.age),
            "g": int(r.g),
            "mpg": round(float(r.mpg), 3),
            "availability": round(float(r.availability), 4),
            "projMinutes": round(float(r.proj_minutes), 2),
            "rating": round(float(r.rating), 3),
            "ratingForecast": round(float(r.rating_forecast), 3),
            "ratingSe": None if pd.isna(r.rating_se) else round(float(r.rating_se), 3),
            "oRapm": None if pd.isna(r.o_rapm) else round(float(r.o_rapm), 3),
            "dRapm": None if pd.isna(r.d_rapm) else round(float(r.d_rapm), 3),
            "prior": None if pd.isna(r.prior) else round(float(r.prior), 3),
            "source": r.rating_source,
            "salary": None if pd.isna(r.salary) else round(float(r.salary)),
            "signing": r.signing if isinstance(r.signing, str) else "—",
            "exp": None if pd.isna(r.experience_years) else int(r.experience_years),
            "supermax": bool(r.supermax_eligible),
        })

    hist_rows, hist_meta = history_payload()

    return {
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "season": valuation.CURRENT_SEASON,
        "constants": {
            "minutes_baseline": consts["minutes_baseline"]["value"],
            "replacement_level": consts["replacement_level"]["value"],
            "points_per_win": consts["points_per_win"]["points_per_win"],
            "games_per_team": consts["cba"]["games_per_team"],
            "game_minutes": consts["cba"]["minutes_per_game"],
            "n_teams": consts["cba"]["n_teams"],
            "roster_min": consts["cba"]["roster_min"],
        },
        "schedule": schedule,
        "seasons": PROJECTION_SEASONS,
        "agingCurve": {
            "age": [int(a) for a in curve["age"]],
            "relValue": [float(v) for v in curve["rel_value"]],
        },
        "players": players,
        "history": hist_rows,
        "historyMeta": hist_meta,
    }


def history_payload() -> tuple[dict, dict]:
    """Past seasons for the league table's season picker.

    Only seasons *before* the current one are emitted. The current season keeps
    coming from `players`, which carries the contract fields history has no
    analogue for — so the default view of the page is byte-for-byte what it was
    before the picker existed, and there is one source of truth for it.

    Rows carry the same key names as `players` so the page's `derive()` runs on
    them unchanged. `salary`/`signing` are null because salaries are loaded for
    2026 only; `supermax` is false because years of service are not reconstructed
    historically, so every past season is priced against the standard maximum.
    """
    path = data.PROCESSED_DIR / f"{history.OUT_NAME}.parquet"
    if not path.exists():
        return {}, {}

    hist = pd.read_parquet(path)
    meta = json.loads(
        (data.PROCESSED_DIR / f"{history.OUT_NAME}_meta.json").read_text())
    by_season = {m["season"]: m for m in meta["seasons"]}

    rows, out_meta = {}, {}
    for season, g in hist[hist["season"] < valuation.CURRENT_SEASON].groupby("season"):
        season = int(season)
        rows[str(season)] = [
            {
                "id": int(r.athlete_id),
                "name": r.athlete_display_name,
                "team": r.team,
                "pos": r.pos if isinstance(r.pos, str) else "—",
                "age": None if pd.isna(r.age) else int(r.age),
                "g": int(r.g),
                "mpg": round(float(r.mpg), 3),
                "availability": round(float(r.availability), 4),
                "projMinutes": round(float(r.proj_minutes), 2),
                "rating": round(float(r.rating), 3),
                "ratingSe": None if pd.isna(r.rating_se) else round(float(r.rating_se), 3),
                "oRapm": None if pd.isna(r.o_rapm) else round(float(r.o_rapm), 3),
                "dRapm": None if pd.isna(r.d_rapm) else round(float(r.d_rapm), 3),
                "source": r.rating_source,
                "salary": None,
                "signing": "—",
                "exp": None,
                "supermax": False,
            }
            for r in g.sort_values("value", ascending=False).itertuples()
        ]
        m = by_season.get(season, {})
        pooled = m.get("pooled_seasons") or [season]
        out_meta[str(season)] = {
            "pooled": [int(pooled[0]), int(pooled[-1])],
            "nPoss": int(m.get("n_poss", 0)),
            # Games each team actually played that season — 34, 22 (bubble) and
            # 40 all appear in this window, which is why minutes are normalised.
            "games": int(valuation.team_games_played(season)["team_games"].max()),
        }
    return rows, out_meta


def dump_js(payload: dict) -> str:
    """JSON with the big row lists collapsed to one player per line.

    The payload is mostly small nested config that reads best pretty-printed, but
    the row lists are ~1,400 flat records; at indent=2 those alone run to 25,000
    lines. Emitting each record compactly on its own line keeps the file about a
    tenth the size while still giving a one-row-per-line diff when it changes.
    """
    marks: dict[str, list] = {}

    def stash(rows: list) -> str:
        token = f"@@ROWS{len(marks)}@@"
        marks[token] = rows
        return token

    skeleton = dict(payload)
    skeleton["players"] = stash(payload["players"])
    skeleton["history"] = {k: stash(v) for k, v in payload["history"].items()}

    text = json.dumps(skeleton, indent=2)
    for token, rows in marks.items():
        pad = re.search(rf'^([ ]*)"[^"]+": "{token}"', text, re.M).group(1)
        body = ",\n".join(f"{pad}  " + json.dumps(r, separators=(",", ":"))
                          for r in rows)
        text = text.replace(f'"{token}"',
                            f"[\n{body}\n{pad}]" if rows else "[]")
    return text


def main() -> None:
    payload = build_payload()
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    js = (
        "// Generated by src/wnba_salary/export_web.py — do not edit by hand.\n"
        f"const MODEL = {dump_js(payload)};\n"
    )
    (WEB_DIR / "players.js").write_text(js)

    n_capped = sum(1 for p in payload["players"] if p["salary"] is not None)
    hist = payload["history"]
    print(f"wrote {WEB_DIR / 'players.js'}")
    print(f"  {len(payload['players'])} players, {n_capped} with contracts")
    print(f"  projection seasons: {payload['seasons']}")
    if hist:
        past = sorted(int(s) for s in hist)
        print(f"  table seasons: {past[0]}–{payload['season']} "
              f"({sum(len(v) for v in hist.values())} historical rows)")


if __name__ == "__main__":
    main()
