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

## Current tranche: site polish

The site went up in `704853d`…`a382367` (multi-page, GitHub Pages, typography and
colour, mobile fixes). What follows is the unfinished part of that work. None of
it blocks anything; the site is live and correct.

### S1 — Reduce the league table on narrow screens
The table has 13 columns and scrolls horizontally on a phone. That is honest but
awkward: the columns a reader most wants (Value, Salary, Surplus) are the ones
off-screen. At `max-width: 640px`, show Player / Rating / Value / Surplus and put
the rest behind a per-row expand, or a "show all columns" toggle above the table.

*Acceptance:* no horizontal scroll needed for the four core columns at 375px; all
13 still reachable; sorting still works on hidden columns.

### S2 — Visual encoding in the league table
The table is 13 columns of undifferentiated figures. An inline bar behind Rating
and Surplus — scaled to the column's own range, drawn as a CSS gradient so it
costs no markup — would let a reader scan rather than read. This is the single
biggest legibility gain available and needs no new data.

*Acceptance:* bars read correctly in both themes; negative values bar leftward
from a centre baseline; no change to the numbers themselves.

### S3 — Draw uncertainty instead of printing it
Task 3 produced per-player error bands and the card currently renders them as the
string `± $557,011`. A band drawn on the max-overflow bar would make the point the
number cannot: that many players' ranges overlap, so the ranking is softer than a
sorted table implies. Load the `dataviz` skill before drawing anything.

*Acceptance:* band visible on the card; reads in both themes; the existing
JS/Python agreement check still passes.

### S4 — Test landscape and tablet
Only 375px and desktop have been checked. 768px and landscape phone are unseen.

### S5 — Tidy
- `.placeholder` in `site.css` is now unused (the bio replaced it). Remove, or
  keep deliberately for future draft sections — decide, don't leave it ambiguous.
- The stylesheet cache-buster (`site.css?v=N`) is bumped by hand and is currently
  at `v=4`. Easy to forget; see AGENTS trap 15. Consider having `export_web.py`
  stamp it, which would make it automatic but couples a data script to the HTML.

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
