# PLAN.md — active work

## How work is tracked

Three files, three jobs. Do not duplicate items between them.

| File | Holds | Lifetime |
|---|---|---|
| **PLAN.md** (this file) | the **current tranche** — a batch of related work with acceptance criteria | rewritten when a tranche completes |
| **AGENTS.md §7** | the **standing backlog** — everything unstarted, ordered by value | permanent, items move out as they are done |
| **README.md** | what the model *is* and what it currently produces | permanent |

An item lives in AGENTS §7 until it is pulled into a tranche here. When a tranche
finishes, its findings go into README/AGENTS and this file is rewritten — the
commit history is the record of what was done, not this file.

**Previously completed here:** the forecast tranche (forward-tuned config, rookie
draft priors, uncertainty bands) — commits `cb24ceb`, `3a57d40`, `b652250`. Its
results are in README §"Forward validation" and AGENTS §4.

---

## Current tranche: none

Recently completed, newest first:

| Work | Commits | Outcome |
|---|---|---|
| In-season updater + certainty gate | `a33dbe1`, `67a67d6` | `scripts/update.py`; inclusion now on `rating_se`, 167 → 187 players |
| Season picker, 2017–2026 | `166a8a6` | `history.py`, 1,507 player-seasons |
| CBA verified against the agreement | `41405fa` | two maxima, tiered minimums, real game counts |
| Site polish S1–S5 | `832d58d` | mobile table, inline bars, uncertainty band |
| Forecast tranche | `cb24ceb`, `3a57d40`, `b652250` | forecast config, draft priors, uncertainty |

**Pick the next tranche from `AGENTS.md` §7.** The old standing recommendation
(verify the CBA) is done; there is no longer one item that dominates the rest.

Two things worth knowing before choosing:

* **A data-freshness line on the page is unbuilt and cheap.** The site is
  roughly a week behind live results and says so nowhere, so a reader comparing
  against last night's box score has no way to know the window. One line in
  `index.html` fed by `MODEL.generated`.
* **Nothing in §7 is blocking.** The pipeline runs end to end with every
  invariant holding. Stopping here is a legitimate choice.

---

## Non-goals

- No full redesign. The site is a few pages and should stay that way.
- No design-system tooling (`DesignSync` / claude.ai/design). There is no
  component library to sync — two pages and one stylesheet.
- No per-stat Kalman filtering or daily updates (DARKO proper). Closed model,
  wrong granularity for an annual contract cycle. See the About page.
- No player subsampling in validation — invalid for a joint regression.
- No re-tuning of the descriptive config; λ=1500 stays.

## Working notes

- Environment: always `./.venv/bin/python`; system pip is PEP-668 blocked.
- Read `AGENTS.md` first. Traps are numbered and referenced by number.
- After any `site.css` edit run `./.venv/bin/python scripts/stamp_css_version.py`.
  A stale stylesheet presents as fonts silently falling back and new rules doing
  nothing, which is easy to misdiagnose as broken CSS.
- After any change to the value formula, change it in `valuation.py` **and**
  `docs/index.html`, then re-run the agreement check (worst gap ~$113).
- In season, `scripts/update.py` rather than the stages by hand; `--check`
  first if you only want to know whether anything moved.
- Verify claims against the artifacts before writing them down. Several
  numbers in these files have gone stale after a rebuild and been quoted
  onward; a two-line pandas check is cheaper than a wrong invariant.
- Commit per item; the user handles pushes.
