"""Vehicles, operating conditions and fault scenarios.

Every number below is a modelling assumption, chosen to be plausible for a
passenger car; none of it is measured. The point is variability: each
simulated vehicle has its own masses and stiffnesses, each run its own speed,
road and load, so a classifier cannot succeed by memorising one curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import QuarterCar, bulge_profile, road_profile, simulate

CLASSES = ("healthy", "underinflated", "imbalance", "bulge", "worn_damper")

FS = 1000.0          # sampling rate of the simulated accelerometers (Hz)
DURATION = 16.0      # recorded duration (s)
SETTLE = 3.0         # simulated before recording, discarded (s)
RIM_RADIUS = 0.20    # radius at which balancing masses sit (m)
PATCH_LENGTH = 0.12  # contact patch length (m)
CARCASS_SHARE = 0.25  # share of tyre stiffness that does not come from air pressure


def tyre_stiffness(kt_nominal: float, pressure_ratio: float) -> float:
    """Vertical tyre stiffness at a fraction of the nominal inflation pressure.
    Linear in pressure, with a carcass contribution at zero pressure."""
    return kt_nominal * (CARCASS_SHARE + (1 - CARCASS_SHARE) * pressure_ratio)


def pressure_from_stiffness_ratio(kt_ratio: np.ndarray) -> np.ndarray:
    """Inverse of :func:`tyre_stiffness` (ratio kt / kt_nominal -> pressure ratio)."""
    return (np.asarray(kt_ratio) - CARCASS_SHARE) / (1 - CARCASS_SHARE)


def sample_vehicle(rng: np.random.Generator) -> QuarterCar:
    return QuarterCar(
        ms=rng.uniform(250, 450),
        mu=rng.uniform(32, 50),
        ks=rng.uniform(16_000, 30_000),
        cs=rng.uniform(1_300, 2_600),
        kt=rng.uniform(170_000, 240_000),
        ct=rng.uniform(80, 250),
        wheel_radius=rng.uniform(0.29, 0.34),
    )


@dataclass
class Condition:
    """State of the wheel during one run."""
    label: str = "healthy"
    pressure_ratio: float = 1.0      # inflation pressure / nominal
    damper_ratio: float = 1.0        # damper coefficient / nominal
    imbalance_kgm: float = 0.0       # rotating imbalance m*e (kg m)
    runout_m: float = 0.0            # radial run-out amplitude (m), 1 per rev
    bulge_m: float = 0.0             # local defect height (m)
    extra: dict = field(default_factory=dict)


def sample_condition(label: str, rng: np.random.Generator) -> Condition:
    """Healthy tyres still carry small imperfections (pressure within spec,
    residual imbalance, run-out); each fault pushes one of them out of range."""
    c = Condition(
        label=label,
        pressure_ratio=rng.uniform(0.92, 1.05),
        damper_ratio=rng.uniform(0.85, 1.05),
        imbalance_kgm=rng.uniform(0, 10e-3) * RIM_RADIUS,   # 0-10 g at the rim
        runout_m=rng.uniform(0, 0.3e-3),
    )
    if label == "underinflated":
        c.pressure_ratio = rng.uniform(0.45, 0.75)         # 25-55 % pressure loss
    elif label == "imbalance":
        c.imbalance_kgm = rng.uniform(40e-3, 120e-3) * RIM_RADIUS   # 40-120 g
    elif label == "bulge":
        c.bulge_m = rng.uniform(1.5e-3, 4e-3)
    elif label == "worn_damper":
        c.damper_ratio = rng.uniform(0.25, 0.55)
    elif label != "healthy":
        raise ValueError(label)
    return c


@dataclass
class Run:
    vehicle_id: int
    label: str
    speed: float
    wheel_rot_hz: float      # wheel rotation frequency (ABS wheel-speed sensor)
    road_class: str
    body_acc: np.ndarray
    hub_acc: np.ndarray
    fs: float
    car: QuarterCar          # parameters actually used (after fault and load)
    condition: Condition


def simulate_run(car: QuarterCar, condition: Condition, rng: np.random.Generator,
                 vehicle_id: int = 0, speed: float | None = None,
                 road_class: str | None = None, duration: float = DURATION,
                 fs: float = FS, noise_std: float = 0.03) -> Run:
    """One record of body and hub acceleration at constant speed."""
    speed = rng.uniform(12, 30) if speed is None else speed
    road_class = rng.choice(["A", "B", "C"], p=[0.3, 0.5, 0.2]) if road_class is None else road_class
    load = rng.uniform(0.95, 1.20)  # passengers, luggage
    used = car.with_(
        ms=car.ms * load,
        kt=tyre_stiffness(car.kt, condition.pressure_ratio),
        ct=car.ct * (1 + 0.5 * (1 - condition.pressure_ratio)),
        cs=car.cs * condition.damper_ratio,
    )
    total = duration + SETTLE
    N = int(round(total * fs))
    t = np.arange(N) / fs
    omega = speed / used.wheel_radius                    # wheel rotation (rad/s)
    zr = road_profile(speed, total, fs, road_class, rng, gd_scale=rng.lognormal(0, 0.3))
    zr = zr + condition.runout_m * np.sin(omega * t + rng.uniform(0, 2 * np.pi))
    zr = zr + bulge_profile(t, speed, used.wheel_radius, condition.bulge_m,
                            PATCH_LENGTH, rng.uniform())
    force = condition.imbalance_kgm * omega ** 2 * np.sin(omega * t + rng.uniform(0, 2 * np.pi))
    acc = simulate(used, zr, force, fs)
    keep = slice(int(SETTLE * fs), None)
    gain = rng.uniform(0.97, 1.03, size=2)               # sensor calibration error
    body = acc[0, keep] * gain[0] + rng.normal(0, noise_std, N)[keep]
    hub = acc[1, keep] * gain[1] + rng.normal(0, noise_std, N)[keep]
    return Run(vehicle_id, condition.label, float(speed), float(omega / (2 * np.pi)),
               str(road_class),
               body, hub, fs, used, condition)
