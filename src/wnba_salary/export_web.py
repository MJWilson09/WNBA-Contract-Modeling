"""Emit the static web UI's data file.

Writes `docs/players.js` and `docs/model-status.md`. `docs/` is the
GitHub Pages publish directory (Settings -> Pages -> Deploy from branch,
main /docs), which is why it is not called `web/`. Both generated files come
from the same artifacts so the site and human-readable status cannot drift.

The site is hand-written HTML plus one shared stylesheet — `index.html`
(the model), `about.html`, `assets/site.css`. The model data and status report
are generated; there is no frontend build step.

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

from . import data, history, valuation, value_history
from .history import team_lookup

WEB_DIR = data.PROJECT_ROOT / "docs"
SNAPSHOT_PATH = data.PROCESSED_DIR / "model_snapshot.json"

PROJECTION_SEASONS = [
    valuation.CURRENT_SEASON + k for k in range(valuation.PROJECTION_YEARS)
]


def build_payload() -> dict:
    consts = json.loads((data.PROCESSED_DIR / "constants.json").read_text())
    df = pd.read_parquet(data.PROCESSED_DIR / "valuation.parquet")
    snapshot = json.loads(SNAPSHOT_PATH.read_text()) if SNAPSHOT_PATH.exists() else {}
    if "aging_curve" in snapshot:
        curve = pd.DataFrame(snapshot["aging_curve"])
    else:
        curve = valuation.load_aging_curve()
    if "teams" in snapshot:
        teams = {int(k): v for k, v in snapshot["teams"].items()}
    else:
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

    trends = player_value_trends(df)
    players = []
    for r in df.sort_values("value", ascending=False).itertuples():
        trend = trends.get(int(r.athlete_id), {})
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
            "valueTrend": trend,
        })

    hist_rows, hist_meta = history_payload()

    return {
        "generated": data_through(),
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


def player_value_trends(current: pd.DataFrame) -> dict[int, dict]:
    """Value deltas for cards, based on persisted full-model snapshots."""
    snapshots = value_history.load(valuation.CURRENT_SEASON)
    if snapshots.empty:
        return {}
    start_date = snapshots["as_of"].min()
    current_date = data_through()
    older = snapshots[snapshots["as_of"] < current_date]
    out: dict[int, dict] = {}

    for row in current.itertuples():
        athlete_id = int(row.athlete_id)
        player_history = older[older["athlete_id"].eq(athlete_id)].copy()
        trend: dict[str, dict | None] = {"last10": None, "season": None}

        if not player_history.empty and not pd.isna(row.team_games):
            player_history["game_gap"] = (
                int(row.team_games) - player_history["team_games"])
            # A daily snapshot may skip several no-game days, so use the closest
            # stored point in a narrow 10–14 game band. Never fall all the way
            # back to opening day and call a 35-game span "last 10".
            eligible = player_history[player_history["game_gap"].between(10, 14)]
            if not eligible.empty:
                baseline = eligible.sort_values(
                    ["game_gap", "as_of"], ascending=[True, False]).iloc[0]
                trend["last10"] = {
                    "delta": round(float(row.value - baseline["value"])),
                    "from": baseline["as_of"],
                    "games": int(baseline["game_gap"]),
                }

        opening = player_history[player_history["as_of"].eq(start_date)]
        if not opening.empty:
            baseline = opening.iloc[0]
            trend["season"] = {
                "delta": round(float(row.value - baseline["value"])),
                "from": start_date,
            }
        out[athlete_id] = trend
    return out


def data_through() -> str:
    """Latest game date represented by the current-season model inputs."""
    if SNAPSHOT_PATH.exists():
        return json.loads(SNAPSHOT_PATH.read_text())["data_through"]
    return raw_data_through()


def raw_data_through() -> str:
    """Latest cached game date, used to stamp a newly exported snapshot."""
    path = data.RAW_DIR / "player_box" / f"player_box_{valuation.CURRENT_SEASON}.parquet"
    if not path.exists():
        return "unknown"
    dates = pd.read_parquet(path, columns=["game_date"])["game_date"]
    if dates.empty:
        return "unknown"
    return pd.to_datetime(dates).max().strftime("%Y-%m-%d")


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
    snapshot = json.loads(SNAPSHOT_PATH.read_text()) if SNAPSHOT_PATH.exists() else {}
    games_by_season = snapshot.get("games_by_season", {})

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
        games = games_by_season.get(str(season))
        if games is None:
            games = int(valuation.team_games_played(season)["team_games"].max())
        out_meta[str(season)] = {
            "pooled": [int(pooled[0]), int(pooled[-1])],
            "nPoss": int(m.get("n_poss", 0)),
            # Games each team actually played that season — 34, 22 (bubble) and
            # 40 all appear in this window, which is why minutes are normalised.
            "games": int(games),
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


def render_js(payload: dict) -> str:
    return (
        "// Generated by src/wnba_salary/export_web.py — do not edit by hand.\n"
        f"const MODEL = {dump_js(payload)};\n"
    )


def render_status(payload: dict) -> str:
    """Render the authoritative report for values that change on rebuild."""
    processed = data.PROCESSED_DIR
    desc = json.loads((processed / "ratings_meta.json").read_text())
    forecast = json.loads((processed / "ratings_forecast_meta.json").read_text())
    hist = pd.read_parquet(processed / f"{history.OUT_NAME}.parquet")
    values = pd.read_parquet(processed / "valuation.parquet")
    consts = json.loads((processed / "constants.json").read_text())
    prior_fit = json.loads((processed / "box_prior_fit.json").read_text())

    rating_rows = []
    for name, meta in (("Descriptive", desc), ("Forecast", forecast)):
        rating_rows.append(
            f"| {name} | {int(meta['n_poss']):,} | {int(meta['n_players']):,} | "
            f"{meta['lambda']:,.0f} | {meta['half_life']:.2f} | "
            f"{meta['pin']['offset']:+.3f} |"
        )
    expected_war = float(consts["dollars_per_win"]["league_war"])
    salary_matches = int(values["salary"].notna().sum())
    capped = int((values["value"] > values["market_value"] + 1).sum())

    return "\n".join([
        "# Current model status",
        "",
        "<!-- Generated by src.wnba_salary.export_web; do not edit by hand. -->",
        "",
        f"Data through **{payload['generated']}** · target season **{payload['season']}**.",
        "",
        "This is the canonical source for values that change when the model is rebuilt.",
        "Stable methodology lives in [`../README.md`](../README.md); operational",
        "instructions and tolerances live in [`../AGENTS.md`](../AGENTS.md).",
        "",
        "## Structural constants and box prior",
        "",
        f"- Points per win: **{consts['points_per_win']['points_per_win']:.2f}**",
        f"- Pace: **{consts['pace']['poss_per_40']:.2f} possessions per 40 minutes**",
        f"- Minutes baseline: **{consts['minutes_baseline']['value']:.2f}**",
        f"- Replacement level: **−{consts['replacement_level']['value']:.2f} points per 100**",
        f"- Dollars per win: **${consts['dollars_per_win']['dollars_per_win']:,.0f}**",
        f"- Box-prior offense OOS correlation/RMSE: **{prior_fit['offense_fit']['oos_corr']:.3f} / {prior_fit['offense_fit']['oos_rmse']:.2f}**",
        f"- Box-prior defense OOS correlation/RMSE: **{prior_fit['defense_fit']['oos_corr']:.3f} / {prior_fit['defense_fit']['oos_rmse']:.2f}**",
        f"- Shrinkage k, offense/defense: **{prior_fit['shrinkage']['offense']['k']:.0f} / {prior_fit['shrinkage']['defense']['k']:.0f}**",
        "",
        "## Ratings",
        "",
        "| Configuration | Possessions | Players | λ | Half-life | Pin offset |",
        "|---|---:|---:|---:|---:|---:|",
        *rating_rows,
        "",
        "Both configurations use the same possession and player universe.",
        "",
        "## Valuation",
        "",
        f"- Table players: **{len(values):,}**",
        f"- Salary matches: **{salary_matches:,}**",
        f"- Summed WAR: **{values['war'].sum():,.2f}** (league identity: **{expected_war:,.2f}**)",
        f"- Summed clipped market value: **${values['market_value'].sum():,.0f}**",
        f"- Players above their applicable CBA maximum: **{capped:,}**",
        "",
        "## History",
        "",
        f"- Coverage: **{int(hist['season'].min())}–{int(hist['season'].max())}**",
        f"- Player-seasons: **{len(hist):,}**",
        "- The target-season history slice is independently checked against valuation.",
        "",
        "## Refresh and verification",
        "",
        "```bash",
        "./.venv/bin/python scripts/update.py --check",
        "./.venv/bin/python -m src.wnba_salary.export_web",
        "./.venv/bin/python scripts/check_generated.py",
        "```",
        "",
    ])


def main() -> None:
    # Commit the semantic snapshot date alongside processed artifacts. The raw
    # inputs are gitignored, so checks in a fresh clone must not depend on them.
    curve = valuation.load_aging_curve()
    SNAPSHOT_PATH.write_text(json.dumps({
        "season": valuation.CURRENT_SEASON,
        "data_through": raw_data_through(),
        "aging_curve": {
            "age": [int(v) for v in curve["age"]],
            "rel_value": [float(v) for v in curve["rel_value"]],
        },
        "teams": {str(k): v for k, v in
                  sorted(team_lookup(valuation.CURRENT_SEASON).items())},
        "games_by_season": {
            str(season): int(valuation.team_games_played(season)["team_games"].max())
            for season in history.SEASONS
        },
    }, indent=2) + "\n")
    payload = build_payload()
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    (WEB_DIR / "players.js").write_text(render_js(payload))
    (WEB_DIR / "model-status.md").write_text(render_status(payload))

    n_capped = sum(1 for p in payload["players"] if p["salary"] is not None)
    hist = payload["history"]
    print(f"wrote {WEB_DIR / 'players.js'}")
    print(f"wrote {WEB_DIR / 'model-status.md'}")
    print(f"  {len(payload['players'])} players, {n_capped} with contracts")
    print(f"  projection seasons: {payload['seasons']}")
    if hist:
        past = sorted(int(s) for s in hist)
        print(f"  table seasons: {past[0]}–{payload['season']} "
              f"({sum(len(v) for v in hist.values())} historical rows)")


if __name__ == "__main__":
    main()
