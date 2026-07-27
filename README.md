# WNBA Contract / Salary Model

A WNBA analog of [StephenNoh/nbasalarymodel](https://github.com/StephenNoh/nbasalarymodel),
which prices NBA players as `(minutes / 1475) × (DARKO + 3) × 4.32`.

The WNBA has no DARKO, so the rating engine is instead the method from
[412 Sports Analytics](https://412sportsanalytics.wordpress.com/2022/08/18/a-detailed-guide-for-developing-player-ratings-for-wnba-and-other-leagues/):
a transfer-learned box-score prior, possession-level RAPM, and a ridge that
shrinks RAPM toward the prior rather than toward zero.

## Status

| Stage | What | State |
|---|---|---|
| A1 | League-structural constants | **done** |
| A2 | Box-score prior (412 stage 1) | **done** |
| A3 | Salary layer | **done** |
| B  | Possession RAPM, ridge-to-prior (412 stages 2-3) | **done, wired in** |
| A4 | Web UI | **done** |

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
```

```bash
./.venv/bin/python -m src.wnba_salary.constants
```

```bash
./.venv/bin/python -m src.wnba_salary.box_prior
```

```bash
./.venv/bin/python -m src.wnba_salary.ratings
```

```bash
./.venv/bin/python -m src.wnba_salary.valuation
```

```bash
./.venv/bin/python -m src.wnba_salary.export_web
```

Run in that order — each stage reads the previous stage's output from
`data/processed/`. Cold, `box_prior` takes ~4 minutes (~60 Basketball-Reference
requests at a 3.5s rate limit) and `ratings` ~3 minutes; both are cached
afterwards.

`data/raw/` is gitignored, so a fresh clone downloads ~54 MB on first run.
Everything caches to disk — delete a subdirectory to force a refetch.

**Working on this with an agent?** See [AGENTS.md](AGENTS.md) for the
environment traps, the invariants worth re-verifying after a change, and a
numbered list of mistakes already made in building this. Several of them are
silent failures that produce believable wrong numbers.

## Modules

- `src/wnba_salary/data.py` — cached parquet from `wehoop-wnba-data` (same source
  `wehoop::load_wnba_*()` reads in R, minus the R).
- `src/wnba_salary/bbref.py` — Basketball-Reference scraper, NBA + WNBA.
- `src/wnba_salary/rates.py` — advanced rate stats from box scores, with a
  validation harness against BBRef's published values.
- `src/wnba_salary/constants.py` — league-structural constants.
- `src/wnba_salary/box_prior.py` — the transfer-learned prior (412 stage 1).
- `src/wnba_salary/rapm.py` — possession building, sparse design, ridge-to-prior
  solver, λ selection (412 stages 2–3).
- `src/wnba_salary/rapm_validation.py` — leak-free λ tuning against the oracle.
- `src/wnba_salary/espn_lineups.py` — lineup/possession reconstruction from ESPN
  play-by-play for seasons the archive doesn't cover.
- `src/wnba_salary/ratings.py` — production ratings: RAPM on ESPN possessions,
  level pinned to the accounting identity.
- `src/wnba_salary/salaries.py` — contracts from Her Hoop Stats.
- `src/wnba_salary/valuation.py` — ratings + constants → dollars, aging, projections.
- `src/wnba_salary/export_web.py` — emits `web/players.js` and `web/standalone.html`.

## Constants (2026)

| Constant | Value | How |
|---|---|---|
| `points_per_win` | 31.77 | regression, `win_pct ~ MOV`, intercept fits 0.5000 |
| `pace` | 79.92 poss/40min | box score, OT-adjusted via player minutes |
| `minutes_baseline` | 1590 | **derived**: `100 × ppw / poss_per_min` |
| `replacement_level` | −2.98 pts/100 | **derived** from league WAR identity |
| `dollars_per_win` | $227,879 | discretionary cap pool |

Two of these are identities rather than fits:

**`minutes_baseline`** — for `(minutes/B) × (rating + repl)` to equal wins above
replacement, `B = 100 × points_per_win / poss_per_min`. Feeding NBA inputs to
that identity returns 1440 against Noh's 1475, which is where his constant comes
from.

**`replacement_level`** — summed player WAR must equal league wins above
replacement, so `R = games × (1 − repl_win_pct) × B / total_team_minutes`. It
needs no ratings. It lands at −2.98, essentially DARKO's NBA −3.00, but derived
from WNBA structure rather than borrowed.

`replacement_win_pct` (0.25) is an assumption, not an estimate — nobody fields a
replacement-level team. Sensitivity is reported across 0.20–0.30.

**`dollars_per_win`** is priced out of *discretionary* cap space (cap minus 12
minimum salaries), not the full cap. Pricing on the full cap gives $424,242 —
1.9× higher — and makes nearly every rotation player look like surplus, which
destroys resolution in a league this salary-compressed.

## Box prior

Transfer learning per the 412 method, with two deliberate departures:

1. **`OnCourt*OnOff` dropped.** On/off derives from the same possessions that
   become the RAPM response; shrinking a ridge toward a prior built from its own
   response variable undercuts the point of an independent prior.
2. **Predictors z-scored within league-season.** BPM is defined against NBA
   usage/pace distributions, so raw WNBA rates through NBA coefficients would
   import a league-mean bias.

`DRB%` is missing from BBRef's WNBA advanced table, so all rate stats are
recomputed from box scores. `rates.validate_against_bbref()` checks the
recomputation against BBRef's published WNBA columns — correlations are
0.998–0.9999, which is what licenses trusting the computed `DRB%`.

Ratings are then regressed toward replacement by sample size, with the shrinkage
constant estimated (season *t* → *t+1*) rather than assumed.

## Salary layer

```
WAR   = (minutes / 1590) × (rating + 2.98)
value = min_salary + WAR × dollars_per_win
```

Contracts come from Her Hoop Stats (server-rendered; Spotrac's table is
client-rendered and returns nothing to a plain fetch). The stats season in the
URL **must** match the salary season — the page inner-joins the two, so pairing
2026 salaries with 2025 stats silently drops every rookie (155 players instead
of 217).

Aging uses sportsdataverse-py's bundled WNBA curve (**peak age 29**, notably
later than NBA intuition). It is applied multiplicatively to *value above
replacement*, not to the raw rating — scaling a −1.0 rating by 0.9 would make a
below-average player better.

Cap inflation follows the CBA's 2026→2032 endpoints ($7M→$11M cap,
$1.4M→$2.4M max, $270K→$340K min). Intermediate years are geometrically
interpolated; the real year-by-year schedule is not public.

Both `value` (unconstrained) and `market_value` (clipped to the CBA band) are
reported. The gap is the surplus a team captures purely because the CBA forbids
paying what a player is worth.

### Aggregate validation

| | model | expected |
|---|---|---|
| summed WAR | 248.3 | 247.5 league-wide |
| summed market value | $100.7M | $105M league cap |

Nothing is fitted to either target, so these are genuine out-of-sample checks on
the whole chain. Ratings come from `ratings.parquet` (RAPM) where available and
fall back to the box prior otherwise; `rating_source` records which applied. For
2026 all 164 qualified players have RAPM coverage.

## Stage B: possession RAPM

`rapm.py` implements the 412 solve — `(A'WA + λI)b = A'Wy + λb_box` — on
possession data with on-court lineups, garbage time excluded, λ tuned on a
chronological holdout.

Two possession sources. The WNBA Stats play-by-play archived in
`sportsdataverse/wehoop-data` ships pre-computed lineups, possession flags and a
garbage-time flag, but **covers 2017–2022 only** — it is used to tune λ and as the
validation oracle. Current seasons come from ESPN play-by-play via
`espn_lineups.py`, validated against that oracle at r=0.994 (below).

`ratings.py` is the production pipeline: ESPN possessions 2023–2026, recency
weighted, λ=1,500, level pinned, written to `ratings.parquet` for the salary
layer.

### The finding

Pooling 2017–2022 gives 167,579 possessions across 537 parameters (312
obs/param, versus 101 single-season). `rapm_validation.py` splits each season at
its 75th-percentile game date, rebuilds the prior from training-window box
scores only, fits RAPM on training possessions only, and scores both on the same
held-out games. Scoring is at game level — possession-level RMSE has sd ~111 and
its entire range across λ is under 0.4%, far too noisy to choose on.

| λ | game RMSE (clean prior) | sd of ratings |
|---|---|---|
| 250 | 9.5287 | 4.77 |
| 2,000 | 9.3641 | 2.78 |
| **4,000** | **9.3557** | **2.44** |
| 16,000 | 9.4019 | 2.11 |
| 128,000 | 9.5044 | 2.00 |
| prior only | 9.7130 | 2.16 |

**RAPM shrunk to the prior beats the prior alone by 0.357 game RMSE (3.7%)**, at
λ=4,000, with a clean interior optimum — error rises on both sides. Refitting on
all possessions at that λ gives ratings correlating 0.911 with the prior (not
0.998) and sd 2.63 versus the prior's 2.16, so the compression problem partly
resolves too.

Defence is where the gain concentrates: `corr(d_rapm, d_prior)` is 0.687. The
players RAPM most upgrades defensively are Natasha Howard (2019 DPOY), Alyssa
Thomas, Candace Parker and Jonquel Jones; the ones it downgrades include Kelsey
Plum and Aerial Powers. That is the pattern you would want, and it is exactly
what the box score cannot see.

### What the first attempt got wrong

An earlier sweep concluded possession data added essentially nothing (optimal
λ=64,000, ratings correlating 0.998 with the prior). Two things were wrong, and
the larger one was **not** the leakage:

1. **Split design (dominant).** The first attempt split chronologically across
   the pooled six seasons, training on 2017–2020 and testing on 2021–22. That
   makes possession-based ratings stale relative to the test period — roster
   turnover and aging intervene — so shrinking to a contemporaneous prior won.
   Splitting *within* each season keeps ratings contemporaneous with the games
   they are scored on, which is the right design for a descriptive rating.
   Holding the leaky prior fixed and changing only the split moves optimal λ from
   64,000 to 4,000.
2. **Prior leakage (secondary).** Building the prior from full-season box scores
   let it see the held-out games. Fixing it moves prior-only RMSE from 9.5737 to
   9.7130 — the leakage was worth 0.139 — which widens RAPM's margin from +0.247
   to +0.357 but does not move optimal λ.

### Extending to 2023-2026 from ESPN play-by-play

`espn_lineups.py` reconstructs lineups and possessions from ESPN play-by-play,
which runs 2002-2026 and is mirrored as parquet.

**Why not stats.wnba.com?** It would be strictly better — same data, lineups and
possessions already solved. It is blocked by bot mitigation, demonstrated rather
than assumed: DNS resolves, the TCP socket opens in 0.13s, then the request hangs
90s returning zero bytes, while `www.wnba.com` 403s instantly from the same host.
Requests reach Akamai and are evaluated; `stats.wnba.com` chooses to hang. R would
hit the identical wall — this is a TLS-fingerprint problem, not connectivity — and
it likely explains why the archive stops at 2022 (no releases, nothing later in
the tree).

ESPN substitution events are unambiguous (`athlete_id_1` enters, `athlete_id_2`
leaves, both always populated). Propagating from `game_rosters` starters through
every substitution across 288 games of 2025 (~15,000 subs) produced 4 sub-outs of
an off-court player and one period boundary with six players. Those games are
dropped, not repaired.

### ESPN validated against the stats-API oracle

2022, both sources, same prior, same λ:

| | correlation | mean abs diff |
|---|---|---|
| `o_rapm` | 0.9956 | 0.161 |
| `d_rapm` | 0.9795 | 0.176 |
| `rapm` | **0.9937** | 0.222 |
| per-player possessions | 0.9984 | — |

ESPN is 3% short on total possessions (ppp 105.5 vs 101.4) and its garbage-time
rule flags 2.4% against the archive's 5.9%, yet ratings agree at r=0.994. Those
gaps are a near-uniform scaling — they move the intercept, not relative player
coefficients. Per-player possession counts and lineup composition are what RAPM
actually depends on, and those agree at 0.9984. Possession-count parity was the
wrong thing to chase.

### Current ratings (2023-2026, ESPN)

151,517 possessions over 973 games, 265 obs/param, recency-weighted with a
1.5-season half-life. λ=**1,500** by within-season chronological holdout, a
genuine interior optimum (worse at both 1,000 and 2,000, and much worse at 100).

Game RMSE 9.3336 against 9.7103 for prior-only — a 3.9% gain. Rating sd rises to
3.11 from the prior's 2.24, correlation with the prior falls to 0.872.

Defence is where it lands, and the corrections are credible: Leonie Fiebich
−0.43 → +3.28 defensively, Napheesa Collier +1.29 → +4.40, Ezi Magbegor
+1.00 → +3.69. All three are reputationally strong defenders the box score could
not see.

Effect on the salary model: players priced above the $1.4M max goes from 6 to
**15**, and Alanna Smith — flagged earlier as the clearest box-prior failure —
moves from −3.41 to +0.10.

### Pinning the level

RAPM's level is not estimable. Predictions are `sum(off) - sum(def) + intercept`,
so adding a constant to *every* offensive and *every* defensive coefficient shifts
the offensive sum by `+5c` and the defensive contribution by `-5c`, leaving every
prediction identical. Ridge inherits the level from the prior, and the pooled
prior is a recency-weighted blend across four seasons — so its level is not the
target season's league average. Unpinned, this inflated summed WAR by ~4%.

`ratings.py` solves for the single offset that satisfies the same accounting
identity `constants.py` uses for replacement level: summed player WAR equals
league wins above replacement, pro-rated to the rated players' share of league
minutes. Because the level is unidentified, imposing a known constraint is the
correct resolution rather than a fudge. The offset applied is −0.131 pts/100 and
the resulting minutes-weighted mean is 0.000.

### Residual leakage, disclosed

The shrinkage constants and the offense/defense replacement split are reused from
the full-sample fit rather than re-estimated per split. Both are scalars fitted
across 17 seasons, so contamination is small but not zero.

## Known limitations

- **The prior largely re-derives BPM.** BPM is itself a linear function of box
  stats, so fitting box stats to BPM recovers its formula (OBPM r=0.94). The
  prior inherits BPM's blind spots wholesale and adds nothing beyond it. The
  actual edge arrives in Stage B.
- **Defense is weakly identified.** DBPM r=0.698 — the box cannot see off-ball
  defense. The offense/defense replacement split (−2.80/−0.18) mostly reflects
  that compression, not a basketball fact.
- **`Big` is noisier for the WNBA.** The league lists only G/F/C, so `F`
  conflates what the NBA splits into SF and PF.
- **CBA figures are press-reported.** Verify against the actual CBA text before
  relying on any dollar output; the minimum is a $270–300K range by service time,
  and 270K is used throughout.
- **Rating dispersion (resolved).** With RAPM wired in, rating sd is 3.03 (was
  2.10 on the box prior) and 15 of 164 players price above the $1.4M max (was 6).
- **Defensive valuation (largely resolved).** Alanna Smith moved from −3.41 to
  −0.03 and Leonie Fiebich from −1.30 to +3.41. Remaining large negatives on
  known defenders should still be treated with suspicion, but they are no longer
  systematic artifacts.
- **Minutes are prorated from a partial 2026 season** at each player's current
  rate, adjusted for games missed. Late-season role changes aren't captured.

## Web UI

`web/index.html` + `web/players.js`, mirroring Noh's shape — static, no build step,
no dependencies. `web/standalone.html` is the same page with data inlined, for
opening or sharing as one file.

Open `web/index.html` directly, or regenerate with `export_web`.

The page recomputes value client-side from the embedded constants rather than
displaying precomputed numbers, so the sliders (games, minutes per game, rating
adjustment) actually re-run the model. `computeWar`, `computeValue` and
`projectRating` in the page mirror `valuation.py` — **change one, change both.**
Agreement is verified: worst discrepancy across 164 players × 3 seasons is $117
(0.005%), entirely decimal rounding in the export.

Two details worth knowing:

- **Default minutes use the pipeline's prorated value, not `games × mpg`.** The
  sliders force `games` to an integer, so `round(44 × availability) × mpg` drifts
  ~0.6% from `valuation.py`'s continuous arithmetic — enough that the page would
  disagree with the parquet it was generated from. Once either minutes slider is
  touched, it switches to the product.
- **Unconstrained value can go negative.** Three players (Kiah Stokes, Zia Cooke,
  Diamond Miller) grade far enough below replacement that `value` is negative;
  `market_value` correctly clamps them to the $270K minimum. That is the formula
  working, not a bug — below-replacement minutes cost wins.

The max-overflow bar is the point of the page: for the 15 players priced above
$1.4M it shows how much value the CBA hands the team for free.
