"""Generation of the simulated fleet dataset (features + spectrograms)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .features import extract
from .scenarios import CLASSES, sample_condition, sample_vehicle, simulate_run

SPEC_NPERSEG = 256
SPEC_FMAX = 250.0


def log_spectrogram(x, fs):
    """Log-power spectrogram of one channel, 0-250 Hz, normalised per image."""
    f, t, S = signal.spectrogram(x, fs, nperseg=SPEC_NPERSEG, noverlap=0)
    S = np.log10(S[f < SPEC_FMAX] + 1e-12)
    return ((S - S.mean()) / (S.std() + 1e-12)).astype(np.float32)


def generate(n_vehicles=160, runs_per_class=3, seed=0, classes=CLASSES,
             with_spectrograms=True):
    """Simulate ``n_vehicles`` vehicles; each runs every condition
    ``runs_per_class`` times (new speed, road and load each time).

    Returns (features DataFrame, spectrograms array or None). The DataFrame
    keeps the ground truth (true pressure ratio, true wheel-hop frequency…)
    in columns prefixed with ``true_``; they must never be used as inputs.
    """
    rng = np.random.default_rng(seed)
    rows, specs = [], []
    for vid in range(n_vehicles):
        car = sample_vehicle(rng)
        for label in classes:
            for _ in range(runs_per_class):
                cond = sample_condition(label, rng)
                run = simulate_run(car, cond, rng, vehicle_id=vid)
                feats = extract(run.body_acc, run.hub_acc, run.fs, run.speed, run.wheel_rot_hz)
                fb, fh = run.car.natural_frequencies()
                feats.update(
                    vehicle_id=vid, label=label, road_class=run.road_class,
                    true_pressure_ratio=cond.pressure_ratio,
                    true_damper_ratio=cond.damper_ratio,
                    true_imbalance_g=cond.imbalance_kgm / 0.20 * 1000,
                    true_bulge_mm=cond.bulge_m * 1000,
                    true_hop_freq=fh, true_body_freq=fb,
                    true_susp_zeta=run.car.cs / (2 * np.sqrt(run.car.ks * run.car.ms)),
                )
                rows.append(feats)
                if with_spectrograms:
                    specs.append(log_spectrogram(run.hub_acc, run.fs))
    df = pd.DataFrame(rows)
    return df, (np.stack(specs) if with_spectrograms else None)


def feature_columns(df):
    """Input columns: everything measured, nothing from the ground truth."""
    drop = {"vehicle_id", "label", "road_class"}
    return [c for c in df.columns if c not in drop and not c.startswith("true_")]
