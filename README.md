# gta6-trophy-rarity-forecast-01

Forecast each Grand Theft Auto VI trophy's Exophase earn rate at
2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%), scored by
pinball loss on the logit scale.

## Usage

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Offline and deterministic: two runs on
the same case write identical bytes. Reads every `<id>.in` in the input
directory and writes `<id>.out` for each.

## Model

A trophy's earn rate is `earned / players_tracked`, and the two move very
differently: Exophase refreshes its counts in jumps, hours apart — players
tracked can multiply ~6x in one refresh while earned counts barely move, so
every rate falls at once. A model that extrapolates the rate itself misses
that. This entry decomposes the rate and forecasts each part:

- `t` = hours since launch (2026-11-19T00:00:00Z), `x = ln(1+t)`.
- **Players:** recency-weighted least squares of `ln P` on `x`, fitted only
  on distinct reads (a read counts only when its value changed from the
  previous kept read, so flat reprints between refreshes are not counted
  twice).
- **Earned, per trophy:** the same weighted fit of `ln E` on `x`, where
  `E = rate * players_tracked / 100` on that read (floored at 0.5). Each
  trophy's slope is shrunk toward its tier's median slope (bronze/silver/
  gold/platinum, 2 pseudo-observations); with fewer than 2 distinct points
  the slope is the tier median and the level is anchored on the last
  observed earned count.
- **Median:** `100 * E_pred / P_pred` at the target, clamped to
  [0.01, 99.99]. When the history never refreshed, the median is instead
  `last_rate / 2` — one refresh is expected before the target, players
  roughly doubling while earned counts stay put. The median is also never
  allowed to move more than 4 logit units from the last observed rate.
- **Spread** (logit scale): fit residuals of E and P (floored at 0.25 and
  0.10), an extrapolation-leverage term (0.35 per log-time unit past the
  last read), a random-walk term (0.5 per sqrt(day) of horizon), and a
  refresh prior (0.9 when the history is flat, 0.25 otherwise), combined in
  quadrature.
- **Quantiles:** `inv_logit(logit(median) + z_tau * sigma)` with
  z = [-1.645, -0.674, 0, 0.674, 1.645], rounded to 4 decimals (half up),
  clamped to [0, 100] and non-decreasing.

## Practice validation

Local practice score (cases p00/p01 vs the verified practice answer):
**~0.51** mean pinball loss, vs **1.51** for the starter (persistence)
baseline — the flat-history branch and the refresh prior do the work, since
both practice cases straddle Exophase's first refresh.
