#!/usr/bin/env python3
"""gta6-trophy-rarity-forecast-01: earners/players decomposition forecaster.

Reads every <id>.in case in $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output): for each trophy, five
quantiles (5, 25, 50, 75, 95 percent) of its Exophase earn rate, in percent,
at the case's target.  Python 3 standard library only, no network, fully
deterministic: no wall clock, no randomness, every series built from the
case's own history in a fixed order, so two runs write the same bytes.

Why decomposition.  A printed rate is a ratio, earners / players tracked,
and the two sides move for different reasons over a launch week:

* players tracked only grows, in surges: Exophase refreshes its counts on
  its own schedule, so the house's reads repeat one snapshot many times
  and then jump.  A jump in the denominator with no matching jump in
  earners dilutes every rate at once (launch night: 638 players tracked
  for hours, then 3,868 at the first refresh, and every rate fell).
* earners for a trophy only grow too, as tracked players reach that
  trophy, but each trophy's earners grow at its own pace: a first-chapter
  trophy's earners track the player surge with a short lag, a 100 percent
  completion trophy's earners barely move in week one.

So the program collapses the history to its distinct snapshots (one per
refresh), turns rates back into earner counts, fits a damped growth rate
to the player count and to each trophy's earner count separately, and
divides the two forecasts.  Growth is damped (a rate fitted today decays
with a time constant of about two days) because a launch surge does not
compound for four and a half days straight, and each trophy's earner
growth is shrunk toward the growth of the total trophies-earned count,
because a rate printed to two decimals makes a small earner count very
noisy.  The median is then blended a quarter of the way toward a direct
logit-trend extrapolation, which is the part of the signal the ratio
decomposition can miss when a trophy's earners and the player count grow
together.  The spread widens with the horizon, with each series' own
residual scatter, and with how much dilution the player forecast implies.

One special case matters for the practice cases: a history with a single
distinct snapshot has seen no refresh at all, so no trend can be fitted.
The honest forecast there is not persistence: the snapshot is stale, a
refresh is overdue, and the first refresh of a launch dilutes rates hard.
That branch centres the median well below the last rate (a logit shift of
-1.2) with a wide spread (sigma about 1.1) whose upper tail still covers
persistence, so it pays a little if no refresh lands and much less than
persistence pays when one does.  A trophy already at 100 percent is the
exception in both branches: everyone tracked has it, a refresh showed it
staying at 100 while every other rate fell, so it stays at the top with
only a whisker of downside.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import the way it does in a plain run.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Damping time constants, in days: how fast a fitted growth rate decays.
TAU_PLAYERS = 1.8
TAU_EARNERS = 2.2
TAU_LOGIT_TREND = 2.0


def parse_time(text):
    """ISO-8601 UTC timestamp to epoch seconds; None when unparsable."""
    if not text:
        return None
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(pct):
    """Logit of a percent, with the percent clamped as the scorer clamps."""
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid_pct(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def weighted_slope(xs, ys, weights):
    """Weighted least-squares slope and residual sd of ys against xs."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    sw = sum(weights)
    if sw <= 0:
        return 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, weights)) / sw
    my = sum(w * y for y, w in zip(ys, weights)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, weights))
    if sxx <= 1e-12:
        return 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, weights))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for r, w in zip(resid, weights)) / sw
    return slope, math.sqrt(max(var, 0.0))


def recency_weights(times_days, span_days):
    """Exponential recency weights with a half-life of half the span."""
    if not times_days:
        return []
    half_life = max(0.25, span_days / 2.0)
    t_last = times_days[-1]
    return [0.5 ** ((t_last - t) / half_life) for t in times_days]


def damped_factor(growth_per_day, horizon_days, tau_days):
    """Multiplier from a growth rate that decays with time constant tau."""
    if horizon_days <= 0:
        return 1.0
    exponent = growth_per_day * tau_days * (1.0 - math.exp(-horizon_days / tau_days))
    return math.exp(clamp(exponent, -3.0, 3.0))


def distinct_snapshots(history):
    """Collapse repeated reads of one Exophase snapshot to its first read.

    A snapshot is identified by its player count, its trophies-earned
    count and every rate; the house reads the page far more often than
    Exophase refreshes it, so most reads repeat the previous snapshot.
    """
    snaps = []
    seen = set()
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), read.get("trophies_earned"),
               tuple(-1.0 if r is None else r for r in rates))
        if key in seen:
            continue
        seen.add(key)
        snaps.append(read)
    return snaps


def snapshot_times_days(snaps):
    """Each snapshot's first-read time in days from the first snapshot."""
    times = []
    base = None
    for i, read in enumerate(snaps):
        t = parse_time(read.get("at") or read.get("slot"))
        if t is None:
            t = (base or 0.0) + 1800.0 * i
        if base is None:
            base = t
        times.append((t - base) / 86400.0)
    return times


def growth_of_series(times_days, values, horizon_days, tau_days):
    """Damped growth multiplier and residual sd for a positive series."""
    ys = [math.log(max(v, 0.05)) for v in values]
    span = max(times_days[-1] - times_days[0], 0.0) if times_days else 0.0
    weights = recency_weights(times_days, span)
    slope, resid_sd = weighted_slope(times_days, ys, weights)
    if len(times_days) >= 4:
        half = len(times_days) // 2
        recent_slope, _ = weighted_slope(
            times_days[half:], ys[half:], recency_weights(times_days[half:], span))
        slope = 0.5 * slope + 0.5 * recent_slope
    return slope, resid_sd


def quantiles_from(median_pct, sigma):
    """Five quantiles around a median on the logit scale, non-decreasing."""
    centre = logit_pct(median_pct)
    qs = []
    for z in Z:
        q = sigmoid_pct(centre + z * sigma)
        qs.append(round4(clamp(q, 0.01, 99.99)))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def stale_snapshot_forecast(rate, horizon_days):
    """Forecast when the history holds one distinct snapshot (no refresh).

    The snapshot is stale and a diluting refresh is overdue: centre below
    the last rate with a wide spread, as the module docstring explains.
    """
    if rate is None:
        return [0.01, 0.05, 0.11, 0.25, 0.6]
    if rate >= 99.9:
        return [99.9, 99.99, 100, 100, 100]
    if rate <= 0.0:
        return [0.01, 0.02, 0.05, 0.12, 0.35]
    sigma = 1.1 + 0.3 * math.sqrt(max(horizon_days, 0.0))
    if rate < 0.2:
        # A near-zero rate is a handful of earners: one refresh can move it
        # by a larger logit step, in either direction, than a common rate's.
        sigma += 0.2
    centre = logit_pct(rate) - 1.2
    qs = []
    for z in Z:
        qs.append(round4(clamp(sigmoid_pct(centre + z * sigma), 0.01, 99.99)))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    last_read = history[-1] if history else {"rates": [None] * len(trophies)}
    last_rates = last_read.get("rates") or [None] * len(trophies)

    snaps = distinct_snapshots(history)
    if len(snaps) <= 1:
        return {"trophies": [
            {"id": t["id"], "q": stale_snapshot_forecast(last_rates[i], horizon_days)}
            for i, t in enumerate(trophies)]}

    times = snapshot_times_days(snaps)
    span_days = max(times[-1] - times[0], 1e-6)
    players = [float(s.get("players_tracked") or 0) for s in snaps]
    players = [p if p > 0 else players[i - 1] if i else 1.0
               for i, p in enumerate(players)]
    p_last = players[-1]

    # Player growth: the denominator's damped multiplier to the target.
    g_players, resid_players = growth_of_series(times, players, horizon_days, TAU_PLAYERS)
    g_players = clamp(g_players, 0.0, 2.0)
    p_target = p_last * damped_factor(g_players, horizon_days, TAU_PLAYERS)
    p_target = max(p_target, p_last)

    # Total earners growth: the shrinkage target for per-trophy earners.
    totals = [float(s.get("trophies_earned") or 0) for s in snaps]
    if all(v > 0 for v in totals):
        g_total, _ = growth_of_series(times, totals, horizon_days, TAU_EARNERS)
    else:
        g_total = g_players
    g_total = clamp(g_total, -0.05, 2.5)

    rows = []
    for idx, trophy in enumerate(trophies):
        rate_last = last_rates[idx]
        if rate_last is not None and rate_last >= 99.9:
            rows.append({"id": trophy["id"], "q": [99.9, 99.99, 100, 100, 100]})
            continue

        # This trophy's earner counts at each snapshot that listed it.
        e_times, e_values, r_values = [], [], []
        for s_i, snap in enumerate(snaps):
            rates = snap.get("rates") or []
            r = rates[idx] if idx < len(rates) else None
            if r is None:
                continue
            e_times.append(times[s_i])
            e_values.append(r / 100.0 * players[s_i])
            r_values.append(r)
        if not e_values:
            rows.append({"id": trophy["id"],
                         "q": stale_snapshot_forecast(rate_last, horizon_days)})
            continue

        e_last = e_values[-1]
        g_earn, resid_earn = growth_of_series(
            e_times, [v + 0.5 for v in e_values], horizon_days, TAU_EARNERS)
        # Shrink noisy small-count growth toward the total's growth.
        trust = (e_last / (e_last + 25.0)) * min(1.0, len(e_values) / 6.0)
        g_earn = trust * g_earn + (1.0 - trust) * g_total
        g_earn = clamp(g_earn, -0.05, 2.5)
        e_target = (e_last + 0.5) * damped_factor(g_earn, horizon_days, TAU_EARNERS) - 0.5
        e_target = max(e_target, 0.0)
        median_rate = clamp(100.0 * e_target / p_target, 0.01, 99.99)
        if rate_last is not None and rate_last > 0:
            # Guard against runaway extrapolation, tightened after cohort
            # backtests across six regimes (progression, dilution, slow,
            # mixed, fast and grind worlds, 36 in all): growth fitted in
            # the launch surge, even damped, overshoots systematically,
            # because most earners a trophy will gain by the target were
            # already counted by the checkpoint or arrive with the cohorts
            # already tracked.  In every regime a cap near 1.7x the last
            # rate (plus 2.5 points for the small ones, whose counts are a
            # handful of players) and a floor at 35% of it scored best.
            median_rate = clamp(median_rate,
                                max(0.01, rate_last * 0.35),
                                min(99.99, max(rate_last * 1.7, rate_last + 2.5)))

        # Blend a quarter toward a direct logit-trend extrapolation.
        if len(r_values) >= 3:
            logit_series = [logit_pct(r) for r in r_values]
            w = recency_weights(e_times, span_days)
            slope_l, resid_logit = weighted_slope(e_times, logit_series, w)
            slope_l = clamp(slope_l, -1.5, 1.5)
            shift = slope_l * TAU_LOGIT_TREND * (
                1.0 - math.exp(-horizon_days / TAU_LOGIT_TREND)) if horizon_days > 0 else 0.0
            trend_rate = sigmoid_pct(logit_pct(r_values[-1]) + clamp(shift, -2.5, 2.5))
            blended = 0.75 * logit_pct(median_rate) + 0.25 * logit_pct(trend_rate)
            median_rate = clamp(sigmoid_pct(blended), 0.01, 99.99)
        else:
            resid_logit = 0.35

        # Spread: horizon, the series' own scatter, and dilution at stake.
        sigma = math.sqrt((0.42 * math.sqrt(horizon_days)) ** 2
                          + (0.6 * resid_logit * math.sqrt(horizon_days)) ** 2
                          + (0.5 * resid_earn) ** 2)
        if p_target > p_last:
            sigma += 0.18 * math.log(p_target / p_last)
        if e_last < 5.0:
            sigma += 0.25
        sigma = clamp(sigma, 0.30, 1.90)
        rows.append({"id": trophy["id"], "q": quantiles_from(median_rate, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
