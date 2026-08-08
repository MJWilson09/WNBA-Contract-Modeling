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
- `src/wnba_salary/draft_prior.py` — draft-slot priors for players with no
  possession history (forward/preseason use; not in the production path).
- `src/wnba_salary/forecast_validation.py` — forward validation harness; the
  measuring stick for any predictive claim. `--sweep` runs the (λ, HL) grid.
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

Pooling 2017–2022 gives 169,439 possessions across 539 parameters (314
obs/param, versus ~100 single-season). `rapm_validation.py` splits each season at
its 75th-percentile game date, rebuilds the prior from training-window box
scores only, fits RAPM on training possessions only, and scores both on the same
held-out games. Scoring is at game level — possession-level RMSE has sd ~111 and
its entire range across λ is under 0.4%, far too noisy to choose on.

| λ | game RMSE (clean prior) | sd of ratings |
|---|---|---|
| 250 | 9.2860 | 5.20 |
| 500 | 9.2464 | 4.28 |
| **1,000** | **9.2274** | **3.53** |
| 2,000 | 9.2377 | 2.96 |
| 16,000 | 9.4172 | 2.16 |
| 128,000 | 9.6012 | 2.01 |
| prior only | 9.8398 | 2.16 |

**RAPM shrunk to the prior beats the prior alone by 0.612 game RMSE (6.2%)**, at
λ=1,000, with a clean interior optimum — error rises on both sides. That optimum
also sits consistently next to the λ=1,500 tuned independently on the ESPN side.
Refitting on all possessions at λ=1,000 gives ratings correlating 0.759 with the
prior and sd 3.78 versus the prior's 2.16, so the compression problem largely
resolves too.

Defence is where the gain concentrates: `corr(d_rapm, d_prior)` is 0.48. The
players RAPM most upgrades defensively (>4,000 possessions) are Jonquel Jones,
Jewell Loyd, Candace Parker and Alyssa Thomas; the ones it most downgrades are
late-career Sue Bird and Diana Taurasi. That is the pattern you would want, and
it is exactly what the box score cannot see.

### What earlier attempts got wrong

The numbers above are the third version of this experiment. Each earlier version
contained a defect that changed the conclusion, worth recording because all three
failed silently:

1. **Split design (dominant).** The first sweep split chronologically across the
   pooled six seasons — train 2017–2020, test 2021–22 — which makes
   possession-based ratings stale relative to the test window, so shrinking
   almost entirely to a contemporaneous prior won (optimal λ=64,000, ratings
   correlating 0.998 with the prior, "RAPM adds nothing"). Splitting *within*
   each season keeps ratings contemporaneous with the games they are scored on,
   which is the right design for a descriptive rating, and moved the optimum by
   more than an order of magnitude on its own.
2. **Prior leakage (secondary).** Building the prior from full-season box scores
   let it see the held-out games, flattering the prior-only baseline. Fixing it
   is worth 0.164 game RMSE on that baseline (9.6753 → 9.8398) and widens RAPM's
   margin, but does not move the optimal λ.
3. **Possession response variable (found last).** The original possession builder
   grouped on the stats feed's `possession` flag and attributed points per-event,
   silently losing ~13% of points (AGENTS.md trap 2). The sweep was re-run after
   the fix; the finding survived and strengthened — RAPM's margin roughly doubled
   (+0.36 → +0.61) and the optimum moved from 4,000 to 1,000.

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

### Forward validation

Everything above is tuned *descriptively* — λ chosen to predict held-out games
inside the fitted window. A contract model prices future seasons, so
`forecast_validation.py` asks the forward question instead: fit on seasons < T,
predict season T's games, score per-game **margin** RMSE (margin is
level-invariant, so candidates with different implicit levels compare fairly).
Nine test seasons, 2018–2026, ~75s on four workers.

| candidate | mean margin RMSE |
|---|---|
| zero (no player information) | 13.5702 |
| box prior, season T−1 only | 12.8405 |
| RAPM, single season | 12.7746 |
| **pooled + recency decay (production recipe)** | **12.6040** |
| pooled + one year of aging | 12.5948 |

The production recipe is the best of the set and beats the no-information floor
by 7.1% in every one of the nine seasons, so the ratings do carry genuine
predictive value — recency-weighted pooling is a steady-state approximation of
what DARKO's Kalman filter does, and it survives the forward test.

Three findings came out of this:

1. **Forward-optimal shrinkage is heavier than descriptive.** λ=6,000 scores
   12.5192 against λ=1,500's 12.6040, with 24,000 worse again. A model tuned to
   describe the current season trusts noisy possession data more than a
   forecaster should.
2. **Aging contributes almost nothing at a one-year horizon** — 0.009 RMSE,
   within noise. It still matters for multi-year *dollar* projections, where it
   compounds against a rising cap, but not as a rating adjustment.
3. **14.4% of player-slots are unseen in training** and imputed at league
   average, rising to ~21% in expansion years (2018, 2026) — visibly the
   worst-forecast seasons. This is the clearest structural gap against DARKO,
   which initialises rookies from a common prior.

### Two rating configs

Acting on finding 1, the model now carries two configs rather than one. Both run
through the identical pipeline and are pinned to the same accounting identity;
they differ only in (λ, half-life).

| | λ | half-life | rating sd | used for |
|---|---|---|---|---|
| **descriptive** | 1,500 | 1.5 | 3.11 | current-season value — "is she worth her contract now" |
| **forecast** | 6,000 | 0.75 | 2.46 | projection years only (`rating_2027`, `value_2028`, …) |

The forecast pair comes from a 5×5 grid on the forward metric
(`forecast_validation --sweep`, ~26s), interior on both axes:

|  | HL=0.25 | HL=0.5 | HL=0.75 | HL=1.5 | HL=3.0 |
|---|---|---|---|---|---|
| λ=1500 | 12.6620 | 12.6059 | 12.5947 | 12.6040 | 12.6285 |
| λ=3000 | 12.5605 | 12.5225 | 12.5160 | 12.5273 | 12.5496 |
| **λ=6000** | 12.5414 | 12.5126 | **12.5085** | 12.5192 | 12.5380 |
| λ=12000 | 12.5617 | 12.5384 | 12.5375 | 12.5505 | 12.5689 |
| λ=24000 | 12.5972 | 12.5781 | 12.5809 | 12.5992 | 12.6205 |

λ does nearly all the work — 1,500→6,000 is worth 0.086 — while the half-life
surface is shallow (0.5/0.75/1.5 all within 0.011), so don't over-read the
half-life choice. Total gain over the production pair: **+0.0954**.

The two configs correlate 0.945 (mean absolute difference 0.95 pts/100). The
forecast rating is more conservative — A'ja Wilson 8.12 → 7.10, Breanna Stewart
6.05 → 4.13 — which is the point: extrapolating a descriptively-tuned rating
three years forward over-trusts one season of possessions.

Current-season outputs are unchanged by this split (summed WAR 248.3, $100.7M,
15 above the max), because the descriptive path was not touched.

### Rookie draft priors

Acting on finding 3. Imputing league-average for a player with no history is
badly wrong for rookies: minutes-weighted they run about −1.2 points/100, and the
spread by draft slot is wide. `draft_prior.py` fits `obpm ~ a·log(pick) + b`
(and the same for defence) on rookie seasons since 2010, **weighted by minutes**
— the prior is consumed per on-court slot, so the relevant expectation is over
slots, not players. Unweighted it is dragged down by fringe rookies and is too
pessimistic exactly where it matters (weighted wR² 0.412 on offence vs 0.335).

| pick | 1 | 3 | 8 | 20 | 36 |
|---|---|---|---|---|---|
| prior | +1.49 | −0.14 | −1.59 | −2.95 | −3.82 |

Effect on the forward harness — the two upgrades are complementary and stack:

| | production λ=1500/HL=1.5 | forecast λ=6000/HL=0.75 |
|---|---|---|
| no draft prior | 12.6040 | 12.5085 |
| **with draft prior** | 12.4951 | **12.3686** |

Total **+0.2353** over the original production recipe, improving in all nine
test seasons. Unseen player-slot share falls from 14.4% to 4.7%. The draft gain
is *larger* on the forecast config (+0.140 vs +0.109), so the two are not
substitutes.

**Scope limit, stated plainly.** This does not change the shipped 2026
valuation. In production the RAPM design includes the current season, so every
rated 2026 player already has possessions and none is "unseen" — all 164 carry
`rating_source = rapm`. The draft prior is validated infrastructure for the
*preseason* case (rating a season before it is played), which is precisely what
forward validation simulates. It is deliberately not wired into `ratings.py`.

Two things were tried and rejected on the evidence:

- **Rookie draft slots as the box prior's shrinkage target** (instead of generic
  replacement). Intuitively appealing — Sabrina Ionescu's injured 80-minute
  rookie year moves from +1.03 to +3.55, which is obviously more sensible — but
  it made the harness *worse* (12.3686 → 12.3803). The harness scores game
  margins, which low-minute players barely affect, so it is insensitive to the
  thing this fixes; absent evidence, it was reverted rather than shipped on
  intuition.
- **The expansion-year hypothesis.** Gains were predicted to concentrate in 2018
  and 2026 (highest unseen share). They did not: the largest gains are 2020
  (+0.29) and 2022 (+0.20), and 2018 is the one season that got marginally
  worse.

`PLAN.md` has the remaining item (uncertainty bands).

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
