"""Quarter-car vibration model with tyre faults.

Two degrees of freedom: the sprung mass (a quarter of the body, ``ms``) on the
suspension (``ks``, ``cs``), and the unsprung mass (wheel, hub, brake, ``mu``)
on the tyre (``kt``, ``ct``). The road profile ``zr`` excites the tyre, and an
optional force ``F`` acts on the wheel (rotating imbalance).

    ms zs'' = -ks (zs - zu) - cs (zs' - zu')
    mu zu'' =  ks (zs - zu) + cs (zs' - zu') - kt (zu - zr) - ct (zu' - zr') + F

The outputs are the two accelerations that an accelerometer on the body and
one on the wheel hub would measure.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import signal

# ISO 8608 road classes: displacement PSD Gd(n0) at n0 = 0.1 cycle/m (m^3),
# geometric mean of each class.
ISO8608_GD0 = {"A": 16e-6, "B": 64e-6, "C": 256e-6, "D": 1024e-6}


@dataclass(frozen=True)
class QuarterCar:
    ms: float = 350.0          # sprung mass (kg)
    mu: float = 40.0           # unsprung mass (kg)
    ks: float = 22_000.0       # suspension stiffness (N/m)
    cs: float = 1_800.0        # damper coefficient (N s/m)
    kt: float = 200_000.0      # tyre vertical stiffness (N/m)
    ct: float = 150.0          # tyre damping (N s/m)
    wheel_radius: float = 0.31  # rolling radius (m)

    def with_(self, **kw) -> "QuarterCar":
        return replace(self, **kw)

    def matrices(self):
        """Continuous state space. States [zs, zu, zs', zu'],
        inputs [zr, zr', F], outputs [zs'', zu'']."""
        ms, mu, ks, cs, kt, ct = self.ms, self.mu, self.ks, self.cs, self.kt, self.ct
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-ks / ms, ks / ms, -cs / ms, cs / ms],
            [ks / mu, -(ks + kt) / mu, cs / mu, -(cs + ct) / mu],
        ])
        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [kt / mu, ct / mu, 1 / mu],
        ])
        C = A[2:].copy()
        D = B[2:].copy()
        return A, B, C, D

    def natural_frequencies(self) -> tuple[float, float]:
        """Undamped natural frequencies (Hz): body mode, wheel-hop mode."""
        M = np.diag([self.ms, self.mu])
        K = np.array([[self.ks, -self.ks], [-self.ks, self.ks + self.kt]])
        w2 = np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real)
        return tuple(float(np.sqrt(x) / (2 * np.pi)) for x in w2)

    def frequency_response(self, f: np.ndarray) -> np.ndarray:
        """Complex response of [body acc, hub acc] to road displacement zr.
        Shape (2, len(f))."""
        A, B, C, D = self.matrices()
        out = np.empty((2, len(f)), complex)
        for i, fi in enumerate(f):
            s = 2j * np.pi * fi
            u = np.array([1.0, s, 0.0])  # zr = 1, zr' = s
            x = np.linalg.solve(s * np.eye(4) - A, B @ u)
            out[:, i] = C @ x + D @ u
        return out


def road_profile(v: float, duration: float, fs: float, road_class: str,
                 rng: np.random.Generator, gd_scale: float = 1.0,
                 n_min: float = 0.011, n_max: float = 10.0) -> np.ndarray:
    """Random road elevation seen by a tyre at constant speed ``v`` (m/s).

    Spectral synthesis of the ISO 8608 displacement PSD
    Gd(n) = Gd(n0) (n / n0)^-2 between ``n_min`` and ``n_max`` (cycle/m),
    random phases. Returns zr(t) in metres, ``int(duration * fs)`` samples.
    """
    N = int(round(duration * fs))
    df = fs / N
    f = np.arange(N // 2 + 1) * df
    n = f / v
    gd0 = ISO8608_GD0[road_class] * gd_scale
    band = (n >= n_min) & (n <= n_max)
    Gz = np.zeros_like(f)
    Gz[band] = gd0 * (n[band] / 0.1) ** -2 / v      # temporal one-sided PSD (m^2/Hz)
    amp = np.sqrt(2 * Gz * df)                      # sinusoid amplitude per bin
    phase = rng.uniform(0, 2 * np.pi, len(f))
    X = amp * np.exp(1j * phase) * N / 2
    return np.fft.irfft(X, n=N)


def bulge_profile(t: np.ndarray, v: float, wheel_radius: float, height: float,
                  patch_length: float, phase: float) -> np.ndarray:
    """Extra radial input of a local tyre defect (bulge, tread separation).

    Once per revolution the defect crosses the contact patch and lifts the
    wheel by ``height`` (m) with a raised-cosine shape lasting
    ``patch_length / v`` seconds.
    """
    if height <= 0:
        return np.zeros_like(t)
    T_rev = 2 * np.pi * wheel_radius / v
    tau = patch_length / v
    local = np.mod(t - phase * T_rev, T_rev)
    out = np.where(local < tau, 0.5 * (1 - np.cos(2 * np.pi * local / tau)), 0.0)
    return height * out


def simulate(car: QuarterCar, zr: np.ndarray, force: np.ndarray, fs: float) -> np.ndarray:
    """Linear response to road input ``zr`` (m) and wheel force ``force`` (N).

    First-order-hold discretisation, then one IIR filter per input and output
    (scipy ``lfilter``). Returns accelerations, shape (2, N): [body, hub].
    """
    A, B, C, D = car.matrices()
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), 1 / fs, method="foh")
    zr_dot = np.gradient(zr, 1 / fs)
    inputs = (zr, zr_dot, force)
    y = np.zeros((2, len(zr)))
    for i, u in enumerate(inputs):
        if not np.any(u):
            continue
        num, den = signal.ss2tf(Ad, Bd, Cd, Dd, input=i)
        for j in range(2):
            y[j] += signal.lfilter(num[j], den, u)
    return y
