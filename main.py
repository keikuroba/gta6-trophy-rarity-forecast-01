"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn
rate at the case target as five quantiles (5/25/50/75/95%), scored by pinball
loss on the logit scale.

Model (see README.md): decompose rate = E / P, earned count over players
tracked. Fit log-time trends ln P = cP + gP*ln(1+t) and ln E_i = a_i + b_i*
ln(1+t) by recency-weighted least squares on distinct (refreshed) reads only,
shrink each trophy's slope toward its tier-median slope, read the ratio at the
target for the median, and widen the logit spread with the fit residuals, the
extrapolation leverage, a random-walk term and a refresh prior (large when the
history never refreshed, because Exophase refreshes in jumps: players multiply
while earned counts barely move, so rates fall).

Python 3.13, standard library only. Offline, deterministic, no randomness.
Reads every <id>.in in MAKEFUN_INPUT_DIR (default /input), writes
<id>.out to MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Maximum median move vs the last observed rate, in logit units (rev2).
LOGIT_CLAMP = 5.0
# Flat-history branch: one refresh is expected before the target, players
# roughly double while earned counts stay put, so halve the last rate.
FLAT_DIVISOR = 2.0
# Refresh prior added to the logit spread: flat history vs refreshed history.
REFRESH_PRIOR_FLAT = 0.9
REFRESH_PRIOR_SEEN = 0.25
# Floors for the residual spreads (logit units).
SD_E_FLOOR = 0.25
SD_P_FLOOR = 0.10
# Leverage of extrapolating past the last read, per log-time unit.
LEVERAGE_RATE = 0.35
# Random-walk spread per sqrt(day) on the logit scale.
RW_RATE = 0.5


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(ts):
    return (parse_ts(ts) - LAUNCH).total_seconds() / 3600.0


def clamp_rate(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round(x*10000)/10000);
    whole numbers come back as ints so JSON writes 100, not 100.0."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def weighted_ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (intercept, slope, residual_sd, n). Residual SD uses dof n-2
    (at least 1); a degenerate x-spread falls back to a flat line at the
    weighted mean with the scatter as its SD.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        dof = max(1, n - 1)
        sd = math.sqrt(sum(r * r for r in resid) / dof) if n > 1 else 0.0
        return my, 0.0, sd, n
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return a, b, sd, n


def distinct_series(history, index):
    """Distinct (t_hours, value) points of one series.

    A read counts only when its value differs from the previous kept read:
    Exophase reprints the same page between refreshes, and counting those
    flat reprints again would fake precision. `index` None selects
    players_tracked, otherwise rates[index]. Reads with a null rate are
    skipped for that trophy.
    """
    pts = []
    prev = None
    for read in history:
        if index is None:
            value = read["players_tracked"]
        else:
            value = read["rates"][index]
            if value is None:
                continue
        if prev is None or value != prev:
            pts.append((hours_since_launch(read["at"]), value))
            prev = value
    return pts


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]

    t_last = hours_since_launch(history[-1]["at"])
    t_target = hours_since_launch(case["target"])
    x_target = math.log1p(t_target)
    x_last = math.log1p(t_last)
    horizon_days = case["horizon_minutes"] / 1440.0
    p_last = history[-1]["players_tracked"]

    def recency_weight(t):
        return 0.5 + t / max(t_last, 1.0)

    # --- players: fit ln P on distinct reads ---
    p_pts = distinct_series(history, None)
    c_p, g_p, sd_p, n_p = weighted_ols(
        [math.log1p(t) for t, _ in p_pts],
        [math.log(max(1, p)) for _, p in p_pts],
        [recency_weight(t) for t, _ in p_pts],
    )
    log_p_target = c_p + g_p * x_target

    # --- earned: fit ln E per trophy on distinct earned counts ---
    fits = []
    for i, trophy in enumerate(trophies):
        e_pts = []
        prev_e = None
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            earned = rate * read["players_tracked"] / 100.0
            if prev_e is None or earned != prev_e:
                e_pts.append((hours_since_launch(read["at"]), earned))
                prev_e = earned
        a, b, sd, n = weighted_ols(
            [math.log1p(t) for t, _ in e_pts],
            [math.log(max(0.5, e)) for _, e in e_pts],
            [recency_weight(t) for t, _ in e_pts],
        )
        fits.append({
            "a": a, "b": b, "sd": sd, "n": n,
            "tier": trophy["tier"],
            "last_rate": history[-1]["rates"][i],
        })

    # --- tier-median slopes from trophies with >= 2 distinct points ---
    by_tier = {}
    all_slopes = []
    for fit in fits:
        if fit["n"] >= 2:
            by_tier.setdefault(fit["tier"], []).append(fit["b"])
            all_slopes.append(fit["b"])
    global_median = median(all_slopes)
    tier_median = {tier: median(bs) for tier, bs in by_tier.items()}

    flat = len(p_pts) <= 1  # history never refreshed
    refresh_prior = REFRESH_PRIOR_FLAT if flat else REFRESH_PRIOR_SEEN
    sd_p_term = max(sd_p, SD_P_FLOOR) if n_p >= 2 else SD_P_FLOOR
    leverage = LEVERAGE_RATE * max(0.0, x_target - x_last)
    random_walk = RW_RATE * math.sqrt(horizon_days)

    rows = []
    for trophy, fit in zip(trophies, fits):
        n = fit["n"]
        last_rate = fit["last_rate"]
        # Shrink the slope toward the tier median (2 pseudo-observations);
        # with < 2 points the slope is the tier median and the level is
        # anchored on the last observed earned count.
        b_tier = tier_median.get(fit["tier"], global_median)
        if n >= 2:
            b = (n * fit["b"] + 2.0 * b_tier) / (n + 2.0)
            a = fit["a"]
        else:
            b = b_tier
            last_e = max(0.5, (last_rate or 0.0) * p_last / 100.0)
            a = math.log(last_e) - b * x_last

        med = 100.0 * math.exp(a + b * x_target) / math.exp(log_p_target)
        if flat:
            # No refresh ever seen: expect one before the target, players
            # roughly doubling while earned counts stay put.
            med = (last_rate / FLAT_DIVISOR) if last_rate else med
        # Never let the median move further than LOGIT_CLAMP logit units
        # from the last observed rate.
        if last_rate is not None and last_rate > 0:
            lo = inv_logit(logit(last_rate) - LOGIT_CLAMP)
            hi = inv_logit(logit(last_rate) + LOGIT_CLAMP)
            med = min(hi, max(lo, med))
        med = clamp_rate(med)

        # --- spread on the logit scale ---
        sd_e_term = max(fit["sd"], SD_E_FLOOR) if n >= 2 else SD_E_FLOOR
        sigma = math.sqrt(
            sd_e_term ** 2 + sd_p_term ** 2 + leverage ** 2
            + random_walk ** 2 + refresh_prior ** 2
        )

        lg = logit(med)
        qs = [round4(min(100.0, max(0.0, inv_logit(lg + z * sigma)))) for z in Z]
        qs = [max(qs[j - 1], qs[j]) if j > 0 else qs[j] for j in range(5)]
        if last_rate == 100:
            # A trophy every tracked player already holds survived a ~6x
            # player jump at 100%: it is almost certainly a start-of-game
            # trophy, so keep the top quantiles at 100 but hedge the lower
            # tail in case new players dilute it.
            qs = [95.0, 99.0, 100, 100, 100]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
