import numpy as np
from scipy import signal

from tire_pdm.physics import ISO8608_GD0, QuarterCar, road_profile, simulate


def test_natural_frequencies_match_closed_form_limits():
    car = QuarterCar()
    fb, fh = car.natural_frequencies()
    # classic approximations: body on series(ks, kt), wheel on ks + kt
    k_eq = car.ks * car.kt / (car.ks + car.kt)
    assert abs(fb - np.sqrt(k_eq / car.ms) / (2 * np.pi)) / fb < 0.05
    assert abs(fh - np.sqrt((car.ks + car.kt) / car.mu) / (2 * np.pi)) / fh < 0.05


def test_time_simulation_matches_frequency_response():
    car = QuarterCar()
    fs = 1000.0
    t = np.arange(20000) / fs
    for f0 in (1.2, 5.0, 11.0, 30.0):
        zr = 1e-3 * np.sin(2 * np.pi * f0 * t)
        y = simulate(car, zr, np.zeros_like(zr), fs)[:, 10000:]
        amp = np.abs(np.fft.rfft(y, axis=1))[:, int(round(f0 * 10))] * 2 / 10000
        ref = 1e-3 * np.abs(car.frequency_response(np.array([f0])))[:, 0]
        assert np.allclose(amp, ref, rtol=0.01), (f0, amp, ref)


def test_road_profile_follows_iso8608():
    rng = np.random.default_rng(0)
    v, fs = 20.0, 200.0
    z = road_profile(v, 200.0, fs, "B", rng)
    f, P = signal.welch(z, fs, nperseg=4096)
    band = (f > 1) & (f < 50)
    expected = ISO8608_GD0["B"] * (f[band] / v / 0.1) ** -2 / v
    ratio = np.median(P[band] / expected)
    assert 0.8 < ratio < 1.25
