"""Physics-based pressure estimation and slow-leak prognostics.

Pressure estimate (indirect TPMS principle): the wheel-hop frequency obeys
f^2 ~ (ks + kt) / mu. With a reference frequency f0 learned on the same
vehicle when its tyres were correctly inflated,

    kt / kt0 = (f / f0)^2 (1 + r) - r,      r = ks / kt0,

and the tyre model of :mod:`scenarios` turns kt / kt0 into a pressure ratio.
``r`` is not measured: a single nominal value is assumed for every vehicle.

Prognostics: a slow leak makes the pressure decay as p(t) = p0 exp(-t / tau).
After each drive the pressure estimate is updated; a robust exponential trend
fitted on the estimates so far predicts when p will cross the alarm level,
hence the remaining useful life (RUL) before the tyre becomes underinflated.
"""
from __future__ import annotations

import numpy as np

from .features import wheel_hop
from .scenarios import (Condition, pressure_from_stiffness_ratio, sample_condition,
                        sample_vehicle, simulate_run)

R_NOMINAL = 22_000 / 200_000   # assumed ks / kt0 for every vehicle
ALARM = 0.75                  # pressure ratio defining "underinflated"


def pressure_ratio(f_hop, f_ref, r=R_NOMINAL):
    kt_ratio = (np.asarray(f_hop) / f_ref) ** 2 * (1 + r) - r
    return pressure_from_stiffness_ratio(kt_ratio)


def baseline_frequency(car, rng, n_runs=5):
    """Reference wheel-hop frequency: median over healthy drives."""
    f = []
    for _ in range(n_runs):
        run = simulate_run(car, sample_condition("healthy", rng), rng)
        f.append(wheel_hop(run.hub_acc, run.fs, run.wheel_rot_hz)[0])
    return float(np.median(f))


def pressure_study(n_vehicles=60, runs=6, seed=1):
    """Estimated vs true pressure ratio on new drives of each vehicle,
    pressure drawn uniformly between 40 % and 105 % of nominal."""
    rng = np.random.default_rng(seed)
    true, est, vid = [], [], []
    for v in range(n_vehicles):
        car = sample_vehicle(rng)
        f0 = baseline_frequency(car, rng)
        for _ in range(runs):
            cond = sample_condition("healthy", rng)
            cond.pressure_ratio = rng.uniform(0.40, 1.05)
            run = simulate_run(car, cond, rng)
            true.append(cond.pressure_ratio)
            est.append(float(pressure_ratio(wheel_hop(run.hub_acc, run.fs, run.wheel_rot_hz)[0], f0)))
            vid.append(v)
    return np.array(true), np.array(est), np.array(vid)


def fit_decay(t, p_hat):
    """Robust fit of log p = a - t / tau (Theil-Sen slope). Returns (a, b)."""
    t = np.asarray(t, float)
    lp = np.log(np.clip(p_hat, 0.05, None))
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 0
    b = float(np.median((lp[j] - lp[i])[ok] / dt[ok])) if ok.any() else 0.0
    a = float(np.median(lp - b * t))
    return a, b


def predict_crossing(t, p_hat, level=ALARM):
    """Predicted time at which the fitted trend reaches ``level`` (inf if not decreasing)."""
    a, b = fit_decay(t, p_hat)
    if b >= 0:
        return np.inf
    return (np.log(level) - a) / b


def leak_scenario(car, rng, tau_days, p0, f0, dt_days=0.5, horizon=None, smooth=3):
    """Simulate one drive every ``dt_days`` while the tyre leaks.

    Returns a dict with the times, true and estimated pressures, the alarm
    time (median of the last ``smooth`` estimates below ``ALARM``), the true
    crossing time and the RUL predictions made along the way."""
    t_cross = tau_days * np.log(p0 / ALARM)
    horizon = horizon or t_cross * 1.4
    times = np.arange(0, horizon, dt_days)
    p_true = p0 * np.exp(-times / tau_days)
    p_hat = []
    for p in p_true:
        cond = sample_condition("healthy", rng)
        cond.pressure_ratio = float(p)
        run = simulate_run(car, cond, rng)
        p_hat.append(float(pressure_ratio(wheel_hop(run.hub_acc, run.fs, run.wheel_rot_hz)[0], f0)))
    p_hat = np.array(p_hat)
    alarm = np.inf
    for k in range(smooth - 1, len(times)):
        if np.median(p_hat[k - smooth + 1:k + 1]) < ALARM:
            alarm = times[k]
            break
    rul_pred = np.full(len(times), np.nan)
    for k in range(4, len(times)):
        rul_pred[k] = predict_crossing(times[:k + 1], p_hat[:k + 1]) - times[k]
    return {"times": times, "p_true": p_true, "p_hat": p_hat, "alarm": alarm,
            "t_cross": t_cross, "rul_true": t_cross - times, "rul_pred": rul_pred}


def prognostics_study(n_vehicles=40, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_vehicles):
        car = sample_vehicle(rng)
        f0 = baseline_frequency(car, rng)
        tau = rng.uniform(15, 60)          # days; p loses 25 % in ~4-17 days
        p0 = rng.uniform(0.97, 1.03)
        out.append(leak_scenario(car, rng, tau, p0, f0))
    return out
