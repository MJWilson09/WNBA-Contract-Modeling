# AGENTS.md

Operational notes for an agent picking this up cold. **`README.md` explains the
model — what it computes and why. This file covers how to run it, what must not
break, and mistakes already made.** Read the traps section before touching the
rating pipeline; several are silent failures that produce plausible wrong numbers.

---

## 1. Environment

Python only. `.venv` at the repo root; **always invoke it explicitly.**

```bash
./.venv/bin/python -m src.wnba_salary.constants
```

- **System `pip` is blocked** by PEP 668 (externally-managed). Install into the
  venv: `./.venv/bin/python -m pip install -r requirements.txt`.
- **R is installed (4.6.1) but is not needed and does not help.** See trap 1.
- Python is 3.14; pandas 3.x. `groupby.apply` needs `include_groups=False`.

## 2. Pipeline

Run in order. Each stage reads the previous stage's output from `data/processed/`.

| # | Command (`./.venv/bin/python -m …`) | Reads | Writes |
|---|---|---|---|
| 1 | `src.wnba_salary.constants` | box scores | `constants.json` |
| 2 | `src.wnba_salary.box_prior` | BBRef, box scores | `box_prior.parquet`, `box_prior_fit.json` |
| 3 | `src.wnba_salary.ratings` | `constants.json`, `box_prior_fit.json`, ESPN pbp | `ratings.parquet`, `ratings_meta.json` |
| 4 | `src.wnba_salary.valuation` | `constants.json`, `box_prior.parquet`, `ratings.parquet`, HHS | `valuation.parquet` |
| 5 | `src.wnba_salary.export_web` | `valuation.parquet`, `constants.json` | `web/players.js`, `web/standalone.html` |

Stage 2 is the slow one (~4 min cold: ~60 BBRef requests at 3.5s). Stage 3 is
~3 min. Both are fully cached afterwards.

`rapm.py` and `rapm_validation.py` are **not** in the production path. They hold
the solver and the λ-tuning experiment against the 2017–2022 oracle. Re-run
`rapm_validation` only if you change the prior or the solver.

## 3. Data

`data/raw/` is **gitignored** — a fresh clone fetches ~54 MB on first run.
Everything is cached to disk after the first fetch; deleting a subdirectory forces
a refetch.

| Source | Coverage | Notes |
|---|---|---|
| `wehoop-wnba-data` (parquet) | 2002–2026 | box scores, ESPN pbp, rosters, player_core |
| `wehoop-data` `wnba_stats/` | **2017–2022 only** | pre-solved lineups + possessions + garbage flag |
| Basketball-Reference | NBA 2005–2025, WNBA 2010–2026 | rate limit **3.5s**, honour it |
| Her Hoop Stats | 2026 salaries | rate limit 4s |

## 4. Invariants

These are verified numbers. If a change moves one materially, that is a
regression until proven otherwise.

**constants.json**
- `points_per_win` 31.77, **intercept 0.5000** (R²=0.778, n=311) — the intercept
  is a free parameter left free precisely so it can be checked. It must fit ~0.5.
- `pace` 79.92 poss/40min · `minutes_baseline` 1590 · `replacement_level` −2.98
- `dollars_per_win` $227,879 (discretionary pool, **not** full cap — full cap
  gives $424,242 and is wrong)

**box_prior_fit.json**
- OBPM r=0.941 / RMSE 0.82 · DBPM r=0.698 / RMSE 0.80
- shrinkage k: offense 75, defense 200
- replacement split: off −2.80, def −0.18 (empirical sum −3.74 vs derived −2.98)

**ratings_meta.json**
- λ=1500, half-life 1.5, 143,272 possessions, 270 players, pin offset −0.131
- minutes-weighted mean of `rapm` must be **0.000**

**valuation.parquet**
- 164 players, 163 matched to a salary
- **summed WAR 248.3** vs league-wide 247.5 — nothing is fitted to this, so it is
  a real end-to-end check
- summed market value $100.7M vs $105M cap · 15 above the $1.4M max · rating sd 3.03

**Cross-source checks**
- computed rate stats vs BBRef published: r = 0.998–0.9999
  (`rates.validate_against_bbref`)
- ESPN-derived vs stats-API RAPM, 2022: **rapm r=0.9937**, o 0.9956, d 0.9795,
  per-player possessions 0.9984 (`espn_lineups.validate_against_stats`)
- stats-API possessions retain **99.7%** of season points at ~101.4 per 100
- web UI vs Python: worst value gap **$117** across 164 players × 3 seasons
- leak-free λ sweep (`rapm_validation`, 2017–2022): optimum **λ=1,000**,
  clean-prior game RMSE **9.2274** vs prior-only **9.8398** (+0.61); interior
  optimum, 500 and 2,000 both worse

## 5. Traps

Numbered so they can be referenced. Most of these were live bugs; several
produce believable wrong answers rather than errors.

1. **`stats.wnba.com` is unreachable and R will not help.** The socket opens in
   0.13s then returns zero bytes for 90s+ — Akamai bot mitigation, not a network
   fault (`www.wnba.com` 403s instantly from the same host). R's curl has the
   same TLS fingerprint problem, so `wehoop::wnba_possession_lineups()` hangs
   identically. Do not retry, do not install R packages to work around it. This
   is also why the archive stops at 2022. Current seasons come from ESPN pbp.
2. **Never build possession ids from the stats feed's `possession` column.** Its
   boundaries do not align with changes of offensive team; doing so silently
   loses 13–40% of points depending on how you then attribute them. Group by
   **runs of `off_slug_team` within a period** — that reconciles to 99.7%.
3. **Dedupe stats pbp on `(game_id, number_event)`.** Period-start rows are
   duplicated ten times (once per player on court).
4. **λ boundary artifacts — this happened twice.** Always extend the grid until
   the optimum is strictly interior. A first sweep picked 32,000 (grid ceiling)
   and the ESPN sweep initially picked 1,000 (its grid floor at the time; the
   true ESPN optimum was 1,500). Both were artifacts. Note the stats-side
   leak-free optimum *is* legitimately 1,000 — but with 500 and 2,000 both tested
   and worse.
5. **Do not select λ on possession-level RMSE.** Response sd is ~111 and the
   entire range across λ is under 0.4%. Score at **game level** (aggregate
   predicted vs actual points per game per offensive team).
6. **Split design changes the conclusion more than leakage does.** Splitting
   chronologically *across* pooled seasons makes possession ratings stale
   relative to the test window and hands the win to a contemporaneous prior —
   this produced a false "RAPM adds nothing" result. Split **within** each season
   (`rapm_validation.season_cutoffs`).
7. **The box prior leaks if built from full-season box scores.** Use
   `box_prior.prior_from_box()` with a date-truncated slice. Leakage was worth
   0.139 game RMSE.
8. **RAPM's level is not identifiable.** Adding a constant to every offensive
   *and* every defensive coefficient leaves all predictions unchanged, so ridge
   inherits the level from the prior. `ratings.pin_level()` fixes it via the
   accounting identity. Skipping this inflates summed WAR ~4%.
9. **A NaN anywhere in the prior vector returns an all-NaN solution.** It fails
   silently. `nan_to_num` the prior (a player can match by name yet carry a NaN
   rating from undefined rate stats).
10. **Her Hoop Stats' URL inner-joins salary season × stats season.** They must
    match. `salary_2026/stats_2025/` silently drops every 2026 rookie — 155
    players instead of 217.
11. **BBRef's WNBA pages leave `<th data-stat="player">` unclosed.** lxml nests
    the whole row inside it, so `get_text()` returns the row concatenated onto
    the name. Numeric columns are unaffected, so this fails *silently* into
    garbage names. Use `bbref._cell_text()`.
12. **BBRef serves UTF-8 without declaring it.** Set `resp.encoding = "utf-8"`
    or names mangle (`Dennis SchrÃ¶der`).
13. **BBRef's WNBA advanced table has no `drb_pct`.** All rate stats are
    recomputed from box scores instead; `rates.validate_against_bbref()` is what
    licenses trusting the computed value. Re-run it if you touch `rates.py`.
14. **wehoop sources disagree on dtypes.** `game_id` is `str` in game_rosters and
    `int32` in pbp; `team_id` is `int32` vs `float64`. Cast before joining or you
    get zero matches with no error.
15. **The web UI duplicates the model formula in JS.** `computeWar`,
    `computeValue`, `projectRating` in `web/index.html` mirror `valuation.py`.
    **Change one, change both**, then re-verify the $117 invariant.
16. **Players are keyed on normalised names throughout**
    (`box_prior.normalize_name`). The stats feed uses WNBA person IDs, wehoop
    uses ESPN athlete IDs — they do not interoperate. Report match rates whenever
    you add a join.

## 6. Looks wrong, is correct

Do not "fix" these:

- **`minutes_baseline` (1590) exceeds the busiest player's season minutes.** It
  converts minutes to wins-per-unit-rating; it is not a share of a season.
- **Three players have negative unconstrained `value`** (Kiah Stokes, Zia Cooke,
  Diamond Miller). They are far below replacement; `market_value` clamps them to
  the $270K minimum. Below-replacement minutes cost wins.
- **The offense/defense replacement split is lopsided** (−2.80 / −0.18). That is
  the box score's inability to resolve defence, not a claim about basketball.
- **Box prior OBPM r=0.941 is not a triumph.** BPM is itself a linear function of
  box stats, so fitting box stats to BPM largely recovers its own formula. The
  prior adds little beyond BPM; the edge is in the RAPM stage.
- **2026 is a partial season** (~62% complete as of the last run). Minutes are
  prorated to 44 games via `availability`.

## 7. Open work

Ordered by value.

1. **Align the ESPN garbage-time rule.** It flags 2.4% against the archive's
   5.9%. Demonstrably harmless to ratings (r=0.994) but the cleanest remaining
   discrepancy. Tune `espn_lineups.GARBAGE_*` against the oracle.
2. **Verify CBA figures against the actual agreement text.** All cap/max/min
   numbers are press-reported. Intermediate years 2027–2031 are geometrically
   interpolated in `valuation.CBA_ENDPOINTS`; the real schedule is not public.
   The minimum is a $270–300K range by service time and 270K is used throughout.
3. **Re-estimate shrinkage constants per split** in `rapm_validation`. They are
   currently reused from the full-sample fit — small residual leakage, disclosed.
4. **`replacement_win_pct` (0.25) is a convention, not an estimate.** Nobody
   fields a replacement team. Sensitivity is reported across 0.20–0.30; treat any
   dollar figure as carrying that band.
5. **Multi-year contract detail.** Only 2026 salaries are loaded. Future contract
   years would need Spotrac or HHS team pages (Spotrac is client-rendered and
   returns no table to a plain fetch).
6. **Do not extend ESPN-based RAPM before ~2020 without new work.** The 2019
   ESPN reconstruction drops 72 of 204 games for lineup inconsistency — ESPN
   substitution data is unreliable that far back. 2023–2026 skips are 0–3 games.

---

*Claude Code reads `CLAUDE.md`. If you want it to load this file specifically,
add a one-line `CLAUDE.md` containing `See @AGENTS.md`.*
