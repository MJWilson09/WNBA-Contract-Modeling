# PLAN.md — forecast upgrade tranche

Implementation plan for the next round of model work. Written for a cold Opus
session: read `AGENTS.md` first (traps are referenced by number below), skim
`README.md` §"Stage B" for context. Everything here builds on the forward
validation in `src/wnba_salary/forecast_validation.py`, which is the measuring
stick for the whole tranche.

## Baseline to beat

`./.venv/bin/python -m src.wnba_salary.forecast_validation` (~75s, 4 workers)
produced, as mean per-game margin RMSE over test seasons 2018–2026:

| candidate | RMSE |
|---|---|
| zero (no player info) | 13.5702 |
| box prior (T−1) | 12.8405 |
| single-season RAPM | 12.7746 |
| pooled + decay, λ=1500 (production) | 12.6040 |
| pooled + 1yr aging | 12.5948 |
| pooled, λ=6000 | **12.5192** |

Mean unseen player-slot share 14.4% (≈21% in expansion years 2018/2026, the
worst-forecast seasons). Aging adds ~nothing at a 1-year horizon (finding, not a
bug). Any change claiming forecast improvement must move these numbers, on this
harness, not a new metric.

## Design decision already made (do not re-litigate)

Two named rating configs, not one:

- **descriptive** — the existing production recipe (λ=1500, HL=1.5), unchanged.
  Feeds current-season value: "is she worth her contract right now."
- **forecast** — forward-tuned (λ, HL) from Task 1. Feeds only the multi-year
  projection columns (`rating_2027`, `value_2027`, …) in `valuation.py` and the
  projection table in the web UI.

Current-season invariants (AGENTS §4: summed WAR 248.3, $100.7M, 15 above max)
must be unchanged by this entire tranche, because the descriptive path is
untouched. Re-verify after each task.

---

## Task 1 — forward-tune (λ, half-life); ship the forecast config  ✅ DONE

1. Extend `forecast_validation.py` with a grid sweep for the pooled candidate:
   λ ∈ {1500, 3000, 6000, 12000, 24000} × HL ∈ {0.75, 1.5, 3.0, 5.0}.
   Reuse the per-T design (already cached in `evaluate_season`); only weights
   and prior aggregation change with HL. Expect ~10–20 min wall. Extend either
   axis if the optimum lands on an edge (trap 4).
2. Add the winning pair as `FORECAST_LAMBDA` / `FORECAST_HALF_LIFE` in
   `ratings.py`; emit a second artifact `ratings_forecast.parquet` using the
   same pipeline (same pinning via `pin_level` — trap 8 applies to both configs).
3. `valuation.py`: projections start from the forecast rating; year-0 value
   keeps the descriptive rating. Add a `rating_forecast` column so the parquet
   records both.
4. `export_web.py` + `web/index.html`: projection rows use the forecast base
   rating. The JS mirrors `valuation.py` (trap 15) — change both, then re-verify
   the ≤ ~$120 JS-vs-Python agreement check.

Acceptance: sweep optimum is interior on both axes; forecast config beats
12.6040 on the harness; descriptive invariants unchanged; README table updated.

## Task 2 — rookie priors from draft slot

Attacks the 14.4% unseen share directly; biggest headroom in expansion years.

1. Data: `wnba/draft/parquet` in `sportsdataverse/wehoop-wnba-data` (add a
   `draft` entry to `data.DATASETS`; inspect the actual filename layout first —
   the repo tree shows a single parquet plus a csv index, so it may be one file
   for all seasons rather than per-season).
2. Fit: first-season rating (use the shrunk box prior from
   `box_prior.parquet`, which exists back to 2010) regressed on draft slot —
   log(pick) or a monotone spline; undrafted/UDFA pooled as slot ~40. Split the
   prediction into (o, d) using the league-average rookie o/d split. Watch
   survivorship: players who never logged 100 minutes have no rating row, so
   fit on picks, not on rated players only — treat missing as "below
   replacement" via a censored/two-stage approach, or at minimum document the
   bias.
3. Apply in `forecast_validation.margin_rmse`: unseen players who are incoming
   rookies get their draft-slot (o, d) instead of (0, 0); unseen veterans keep
   0. Measure the delta — acceptance is improvement concentrated in 2018/2026.
4. If (3) wins, apply in production: rookies' `b_box` entries in
   `ratings.py`'s prior use the draft prior blended with their observed
   box prior by minutes (the existing `shrink()` machinery fits this).
5. Name joins are the risk (trap 16): report the draft-name → possession-name
   match rate; expect misses on transliterated international names.

Acceptance: harness improvement (mean and expansion seasons); match rate
reported; no change to descriptive invariants.

## Task 3 — per-player uncertainty bands

1. In `rapm.py`, add a function returning per-player variance from the ridge
   posterior: `V = σ̂² (A'WA + λI)⁻¹` with σ̂² from training residuals. The
   system is ~540×540 — invert directly. Total-rating variance needs
   `var(o) + var(d) + 2cov(o, d)` (off-diagonal terms, so keep the full
   inverse). Document the caveat: this treats the prior as truth, so bands are
   approximate credible intervals, not frequentist SEs — they will be too
   narrow for low-minute players.
2. Thread `rating_se` through `ratings.parquet` → `valuation.parquet`
   (`value_lo`/`value_hi` at ±1 SE through the same dollar formula) →
   `export_web.py` → one line in the web card ("value $X ± $Y"). Keep the UI
   change minimal; remember trap 15.

Acceptance: SE larger for low-possession players (sanity: rank-correlate SE
against 1/√possessions); UI shows bands; JS-Python agreement check still passes.

## Task 4 — record the findings (do first, it is 10 minutes)  ✅ DONE

1. README: add a "Forward validation" subsection under Stage B with the
   baseline table above and the three findings (forward λ heavier than
   descriptive; aging ≈ 0 at 1yr; unseen-share/expansion effect).
2. AGENTS §2: add `forecast_validation.py` to the module table. §3: note the
   possession cache (`data/processed/poss_cache/`, delete to force rebuild).
   §4: add the baseline invariant line (zero 13.5702 / pooled 12.6040 /
   λ=6000 12.5192 / unseen 14.4%).

## Non-goals

- No per-stat Kalman filtering, no daily updates, no gradient-boost stat→impact
  mapping (DARKO proper). Closed-source model, wrong granularity for an annual
  contract cycle, and Tasks 1–2 capture the cheap majority of the headroom.
- No player subsampling in validation — invalid for a joint regression and
  saves nothing (runtime scales with possessions).
- No re-tuning of the descriptive config; its λ=1500 stays.

## Working notes

- Environment: always `./.venv/bin/python`; system pip is PEP-668 blocked.
- Runtime: harness ~75s; possession cache warm. `stats.wnba.com` is
  unreachable — do not retry it (trap 1).
- Order: Task 4 → 1 → 2 → 3. Tasks 2 and 3 are independent of each other.
- **Status: Tasks 4 and 1 complete** (λ=6000 / HL=0.75 shipped as the
  forecast config, +0.0954 forward RMSE). Tasks 2 and 3 remain.
- Commit per task when the user asks; they handle pushes.
