# Roadmap

This is the single source of truth for work state. Model behavior belongs in
`README.md`, current generated results in `docs/model-status.md`, and operational
instructions and traps in `AGENTS.md`.

## In progress

None.

## Backlog

Ordered by expected value. Nothing here blocks the production pipeline.

1. **Align the ESPN garbage-time rule.** It flags 2.4% against the archive's
   5.9%. Ratings still agree at r=0.994, but this is the cleanest remaining
   cross-source discrepancy. Tune `espn_lineups.GARBAGE_*` against the oracle.
2. **Re-estimate shrinkage constants per validation split.** They are currently
   reused from the full-sample fit, creating small disclosed residual leakage.
3. **Clarify replacement-level uncertainty.** `replacement_win_pct = 0.25` is a
   convention rather than an estimate. Retain and foreground the 0.20–0.30
   sensitivity band for dollar figures.
4. **Add multi-year contract detail.** Only current-season salaries are loaded.
   Future years require Spotrac or Her Hoop Stats team pages.
5. **Add historical salaries and CBA schedules.** Historical ratings currently
   price every season in current-CBA dollars. Past caps, maxima, minima, and
   salary seasons would enable actual historical surplus analysis.

## Recently completed

| Work | Commits | Outcome |
|---|---|---|
| Documentation consolidation | — | One roadmap, generated model status, and drift check |
| In-season updater + certainty gate | `a33dbe1`, `67a67d6` | `scripts/update.py`; inclusion moved to `rating_se` |
| Season picker, 2017–2026 | `166a8a6` | Historical rating and valuation views |
| CBA verification | `41405fa` | Two maxima, tiered minimums, real game counts |
| Site polish S1–S5 | `832d58d` | Mobile table, inline bars, uncertainty band |
| Forecast tranche | `cb24ceb`, `3a57d40`, `b652250` | Forecast config, draft priors, uncertainty |

## Non-goals

- No full site redesign or design-system tooling.
- No per-stat Kalman filtering or daily DARKO-style model.
- No player subsampling in validation; the rating is a joint regression.
- No re-tuning of the descriptive λ=1,500 configuration.

## Working rules

- Always use `./.venv/bin/python`; system pip is blocked by PEP 668.
- Read `AGENTS.md` before rating-pipeline work.
- In season, use `scripts/update.py`, not manual stage runs.
- After editing `site.css`, run `scripts/stamp_css_version.py`.
- Change valuation formulas in both Python and `docs/index.html`.
- Regenerate web data and status with
  `./.venv/bin/python -m src.wnba_salary.export_web`.
- Verify generated files with `./.venv/bin/python scripts/check_generated.py`.
