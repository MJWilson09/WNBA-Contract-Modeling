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

The site-polish tranche (S1-S5) is complete — commit `bb94ee2`. Results:

| | Item | Outcome |
|---|---|---|
| S1 | Reduce the table on narrow screens | Player/Rating/Value/Surplus fit 341px in a 341px container; `all columns` toggle restores the other nine |
| S2 | Visual encoding | Inline bars behind Rating and Surplus, centre-baselined so sign reads without reading the number |
| S3 | Draw uncertainty | Hatched ±1 SE band on the value bar |
| S4 | Landscape and tablet | 375 / 768 / 812x375 / 1280 all clear, no page overflow |
| S5 | Tidy | Unused `.placeholder` removed; cache-buster now a content hash via `scripts/stamp_css_version.py` |

Pick the next tranche from **AGENTS.md §7**. The standing recommendation is
unchanged: verifying the CBA figures outranks everything else, because every
published dollar rests on press-reported numbers and an interpolated 2027-31
schedule.

---

## Not in this tranche, but higher consequence than any of it

**Verifying the CBA figures** (AGENTS §7 item 2) matters more than everything
above combined. Every dollar the site publishes rests on press-reported cap, max
and minimum numbers, and the 2027–2031 schedule is an interpolation invented in
`valuation.CBA_ENDPOINTS`. Nothing on this list can make a number wrong the way
that can. It is not here only because it is research rather than code.

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
- After any `site.css` edit, bump `?v=N` in **both** HTML files. A stale
  stylesheet presents as fonts silently falling back and new rules doing nothing.
- After any change to the value formula, change it in `valuation.py` **and**
  `docs/index.html`, then re-run the agreement check (worst gap ~$112).
- Commit per item; the user handles pushes.
