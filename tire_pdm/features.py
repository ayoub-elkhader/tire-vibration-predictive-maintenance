"""Signal-processing features of one run (body and hub accelerations).

Four families, each tied to a physical mechanism:

* wheel-hop resonance: the hub acceleration PSD divided by f^2 is
  proportional to |zu/zr|^2 for an ISO 8608 road (displacement PSD ~ f^-2), so
  its peak tracks the wheel-hop frequency sqrt((ks + kt)/mu)/2pi, which drops
  when the tyre loses pressure (the principle of indirect TPMS);
* suspension transfer: body/hub acceleration ratio estimated with the H1
  cross-spectral estimator. It depends only on ms, ks and cs, not on the road
  or the tyre, and a model fit gives the suspension damping ratio;
* wheel orders: with the wheel speed known (ABS sensor), the amplitude at the wheel rotation
  frequency (1x) and its harmonics; time-synchronous averaging (TSA) over the
  revolutions separates what repeats every turn (imbalance, run-out, bulge)
  from the random road;
* global statistics: RMS, kurtosis, crest factor, relative band energies.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, signal, stats

HUB_BANDS = [(0.5, 3), (3, 8), (8, 15), (15, 30), (30, 60), (60, 120), (120, 250)]
BODY_BANDS = [(0.5, 2), (2, 5), (5, 10), (10, 30)]
TSA_POINTS = 256
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz


def _band_energy(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(_trapz(P[m], f[m])) if m.sum() > 1 else 0.0


def _hop_model(f, fn, zeta):
    r = f / fn
    return 1.0 / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)


def wheel_hop(hub, fs, f_rot=None, band=(4.0, 30.0), n_orders=30):
    """Wheel-hop natural frequency (Hz) and damping ratio.

    Above the body mode the body barely moves, so the wheel behaves as a
    1-DOF oscillator on ks + kt and |zu/zr|^2 ~ 1 / ((1 - r^2)^2 + (2 zeta r)^2),
    r = f / fn. The whitened hub PSD (PSD / f^2) is fitted with that shape
    plus a gain, in log scale. Bins near the wheel orders (k x ``f_rot``)
    are left out: imbalance, run-out and defects put lines there that are
    not part of the road response.
    """
    f, P = signal.welch(hub, fs, nperseg=4096)
    m = (f >= band[0]) & (f <= band[1])
    if f_rot:
        for k in range(1, n_orders + 1):
            m &= np.abs(f - k * f_rot) > 0.5
    fm = f[m]
    lw = np.log(P[m] / fm ** 2)
    W = P[m] / fm ** 2
    f0 = float(fm[np.argmax(np.convolve(W, np.ones(5) / 5, mode="same"))])

    def resid(p):
        fn, zeta = p
        model = np.log(_hop_model(fm, fn, zeta))
        return model - lw - np.mean(model - lw)   # the gain is profiled out

    fit = optimize.least_squares(resid, x0=[f0, 0.3], bounds=([4.0, 0.03], [30.0, 1.5]))
    return float(fit.x[0]), float(fit.x[1])


def _susp_model(f, fn, zeta):
    w = 2 * np.pi * f
    wn = 2 * np.pi * fn
    # body/hub acceleration ratio of a base-excited 1-DOF oscillator
    return np.abs((wn ** 2 + 2j * zeta * wn * w) / (wn ** 2 - w ** 2 + 2j * zeta * wn * w))


def suspension_transfer(body, hub, fs, fmax=8.0):
    """Fit of |body/hub| (H1 estimator) with a base-excited 1-DOF model.
    Returns (body natural frequency Hz, damping ratio, peak gain)."""
    f, Pxy = signal.csd(hub, body, fs, nperseg=8192)
    _, Pxx = signal.welch(hub, fs, nperseg=8192)
    m = (f >= 0.4) & (f <= fmax)
    H = np.abs(Pxy[m] / Pxx[m])
    fit = optimize.least_squares(
        lambda p: np.log(_susp_model(f[m], p[0], p[1])) - np.log(H),
        x0=[1.2, 0.3], bounds=([0.5, 0.03], [3.0, 1.5]))
    fn, zeta = fit.x
    return float(fn), float(zeta), float(H.max())


def tsa(hub, fs, f_rot, points=TSA_POINTS):
    """Time-synchronous average over wheel revolutions.

    ``f_rot`` is the wheel rotation frequency (Hz), as given by the ABS
    wheel-speed sensor of the same wheel. Returns (average revolution,
    number of revolutions)."""
    n = len(hub)
    T = 1 / f_rot
    revs = int(n / fs / T)
    t = np.arange(n) / fs
    tq = (np.arange(revs * points) / points) * T
    resampled = np.interp(tq, t, hub).reshape(revs, points)
    return resampled.mean(axis=0), revs


def order_amplitudes(hub, fs, f_rot, n_orders=10):
    """Amplitude (m/s^2) at each wheel order and the local noise floor."""
    n = len(hub)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(hub * win)) * 2 / win.sum()
    freqs = np.fft.rfftfreq(n, 1 / fs)
    df = freqs[1]
    amps, floors = [], []
    for k in range(1, n_orders + 1):
        fk = k * f_rot
        i = int(round(fk / df))
        amps.append(spec[max(i - 2, 0):i + 3].max())
        ring = np.r_[spec[max(i - 40, 0):max(i - 5, 0)], spec[i + 6:i + 41]]
        floors.append(np.median(ring) if len(ring) else 1e-9)
    return np.array(amps), np.array(floors)


def extract(body, hub, fs, speed, f_rot):
    """All features of one run, as a flat dict of floats."""
    feats = {"speed": speed}
    for name, x in (("hub", hub), ("body", body)):
        x = x - x.mean()
        feats[f"{name}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        feats[f"{name}_kurtosis"] = float(stats.kurtosis(x, fisher=False))
        feats[f"{name}_crest"] = float(np.abs(x).max() / feats[f"{name}_rms"])
    f, P = signal.welch(hub, fs, nperseg=2048)
    tot = _band_energy(f, P, 0.5, 250)
    for lo, hi in HUB_BANDS:
        feats[f"hub_band_{lo}_{hi}"] = _band_energy(f, P, lo, hi) / tot
    f, P = signal.welch(body, fs, nperseg=4096)
    tot = _band_energy(f, P, 0.5, 30)
    for lo, hi in BODY_BANDS:
        feats[f"body_band_{lo}_{hi}"] = _band_energy(f, P, lo, hi) / tot

    feats["hop_freq"], feats["hop_zeta"] = wheel_hop(hub, fs, f_rot)
    feats["susp_fn"], feats["susp_zeta"], feats["susp_peak"] = suspension_transfer(body, hub, fs)

    avg, revs = tsa(hub, fs, f_rot)
    hub_c = hub - hub.mean()
    avg_c = avg - avg.mean()
    feats["tsa_ratio"] = float(np.sqrt(np.mean(avg_c ** 2)) / np.sqrt(np.mean(hub_c ** 2)))
    feats["tsa_kurtosis"] = float(stats.kurtosis(avg_c, fisher=False))
    H = np.abs(np.fft.rfft(avg_c))
    e = H[1:].astype(float) ** 2
    feats["tsa_order1_share"] = float(e[0] / e.sum())
    feats["tsa_high_orders_share"] = float(e[3:20].sum() / e.sum())

    amps, floors = order_amplitudes(hub, fs, f_rot)
    feats["order1_snr_db"] = float(20 * np.log10(amps[0] / floors[0]))
    feats["orders2_10_snr_db"] = float(20 * np.log10(np.mean(amps[1:] / floors[1:])))
    feats["order1_amp_per_w2"] = float(amps[0] / (2 * np.pi * f_rot) ** 2)
    return feats
